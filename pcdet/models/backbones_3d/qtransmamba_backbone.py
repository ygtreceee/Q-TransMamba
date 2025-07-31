import os
import math
import copy
import csv
import time
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from functools import partial
from collections import defaultdict
import torch.utils.checkpoint as cp
from torch.nn import functional as F 
from collections import namedtuple, defaultdict
from typing import List, Tuple, Dict, Any, Union, Optional, Iterable


from pcdet.utils.spconv_utils import replace_feature, spconv

from pcdet.models.backbones_3d.qdefmamba.basic_utils import PointEmbedding, debug_check_croods_repeated, \
                _init_weights, manhattan_kernel_template_fast, XFormersCrossAttention, CrossTransformer
from pcdet.models.backbones_3d.qdefmamba.query_forward_branch import QueryFFN

from pcdet.models.backbones_3d.qdefmamba.window_sparseconv_tensor import WindowSparseConvTensor


from pcdet.models.backbones_3d.qdefmamba.visualizer import GridPosVisualizer, visualize_query_pos
from pcdet.models.backbones_3d.qdefmamba.qdfmamba_4d_layer \
    import QueryDeformableMambaLayer_v5 as QueryDeformableMambaLayer
# import pcdet.utils.loss_utils_qdefmamba as loss_utils_qdefmamba




class QueryDeformableMamba_v5(nn.Module):
    def __init__(self, model_cfg, detector_type, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.detector_type = detector_type

        self.norm1d_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm2d_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.layernorm_fn = partial(nn.LayerNorm, eps=1e-5)
    
        ### for general config ###
        general_cfg = self.model_cfg.general_cfg
        self.dim = general_cfg.dim
        self.num_stage = general_cfg.num_stage
        self.total_queries = general_cfg.total_queries
        self.device = general_cfg.device
        self.sparse_shape = general_cfg.sparse_shape
        self.force_layernorm = general_cfg.force_layernorm
        # self.query_mode = True if general_cfg.total_queries is not -1 else False
        # general_cfg['query_mode'] = self.query_mode
        if self.force_layernorm == True:
            self.norm1d_fn = partial(nn.LayerNorm, eps=1e-5)
            self.norm2d_fn = partial(nn.LayerNorm, eps=1e-5)

        # Build Hilbert tempalte 
        if general_cfg['space_filing_curve'] == 'hilbert':
            INPUT_LAYER = general_cfg.INPUT_LAYER
            self.curve_template = {}
            self.hilbert_spatial_size = {}
            for rank, path in INPUT_LAYER.items():
                self.load_template(path, int(rank[-1]))
            general_cfg['curve_template'] = self.curve_template
            general_cfg['hilbert_spatial_size'] = self.hilbert_spatial_size
            general_cfg['dtype'] = torch.float32 # NOTE: for mamba setting

        ### for self-modeling ###
        # self_modeling_cfg = self.model_cfg.self_modeling_cfg 

        self.window_pos_embed = general_cfg.get('window_pos_embed', False)
        if self.window_pos_embed == True:
            self.pos_embed =  nn.Sequential(
                nn.Linear(9, self.dim),
                nn.BatchNorm1d(self.dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim, self.dim),
            )
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(3, self.dim),
                nn.LayerNorm(self.dim), # If use BatchNorm, it would cause BN Error (bs should > 1) when batch_size=1
                nn.ReLU(inplace=True),
                nn.Linear(self.dim, self.dim),
            )
        

        ### for QSSM ###
        qssm_cfg = self.model_cfg.qssm_cfg
        
        ### for Fusion Model ###
        fusion_cfg = self.model_cfg.fusion_cfg

        ### for query-ffn ###
        query_ffn_cfg = self.model_cfg.query_ffn_cfg

        if self.total_queries > 0:
            self.hidden_queries = nn.Embedding(self.total_queries, self.dim, device=self.device)
        
            # hidden_queries = torch.empty(self.total_queries, self.dim)  # 未初始化的张量
            # nn.init.xavier_uniform_(hidden_queries)  # Xavier/Glorot 初始化（适用于 Transformer 等）
            # self.register_buffer('hidden_queries', hidden_queries)

            ### for Query Feedback ###
            self.wholy_residual = self.model_cfg.query_tranformer_cfg['wholy_residual']
            self.hybrid_gate = self.model_cfg.query_tranformer_cfg['hybrid_gate']
            self.update_rate_learnable = self.model_cfg.query_tranformer_cfg.get('update_rate_learnable', False)
            if self.wholy_residual == True:
                if self.update_rate_learnable == True:
                    if self.hybrid_gate == True:
                        self.update_rate = nn.Linear(self.dim * 2, 1, device=self.device, dtype=torch.float32)
                    else:
                        self.update_rate = nn.Linear(self.dim, 1, device=self.device, dtype=torch.float32)
                    # self.update_rate_p = nn.Parameter(torch.tensor([0.0], device=self.device, dtype=torch.float32))
                    # self.register_parameter('update_rate_p', self.update_rate)
                else:
                    self.update_rate = self.model_cfg.query_tranformer_cfg['update_rate']
                # self.norm_out = self.norm1d_fn(self.dim)
                self.norm_out = self.layernorm_fn(self.dim)

            self.query_transformer = CrossTransformer(self.dim, norm_fn=self.layernorm_fn, **self.model_cfg.query_tranformer_cfg)
            # self.query_transformer = XFormersCrossAttention(
            #     self.dim, attn_type='flash', 
            #     proj_type='linear',
            #     res_norm_fn=self.layernorm_fn, # or BN?
            #     residual=True, # 必须要残差
            #     pre_norm=False, # NOTE: output = Norm(Attn(x)) + x, when True
            #     dropout=0.0, # TODO 需斟酌 
            #     attn_cfg=flash_attn_cfg
            # )
        else:
            self.hidden_queries = None

            
        ### for GLSSM ###
        glssm_cfg = self.model_cfg.glssm_cfg

        ### for FFN ###
        ffn_cfg = self.model_cfg.ffn_cfg
        

        ### list of QueryDeformableMambaBlock ###
        self.block_list = nn.ModuleList()
        for i, num_s in enumerate(self.num_stage):
            for j in range(num_s):
                configs = [
                    cfg.copy() for cfg in [
                        # self_modeling_cfg, 
                        qssm_cfg, 
                        fusion_cfg, 
                        query_ffn_cfg,
                        ffn_cfg,
                        glssm_cfg,
                        ]]
                general_cfg['num_stage'], general_cfg['num_block'] = i, j
                for cfg in configs:
                    cfg.update({
                        k: (v[i] if isinstance(v, list) and isinstance(v[0], list) else v)
                        for k, v in general_cfg.items()
                        if k not in cfg
                    })
                self.block_list.append(
                    QueryDeformableMambaLayer(
                        # self_modeling_cfg=configs[0],
                        qssm_cfg=configs[0],
                        query_ffn_cfg=configs[2],
                        ffn_cfg=configs[3],
                        **configs[4],
                    ))

        ### for downsampling z between stages (half per stage) ### 
        downZ_list = []
        # for i in range(0, 3):  # for debug
        for i in range(len(self.num_stage)):
            downZ_list.append(
                spconv.SparseSequential(
                spconv.SparseConv3d(self.dim, self.dim, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key=f'downz_{i}'),
                self.norm1d_fn(self.dim),
                nn.ReLU(),)
            )
        self.downZ_list = nn.ModuleList(downZ_list)
        self.conv_out = spconv.SparseSequential(
                spconv.SparseConv3d(self.dim, self.dim, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key=f'final_conv_out'),
                self.norm1d_fn(self.dim),
                nn.ReLU(),)

        ### for initialization ###
        initializer_cfg = None
        self._reset_parameters()
        self.apply(
            partial(
                _init_weights,
                n_layer=sum(self.num_stage),
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        # For detector3d_template
        self.num_point_features = self.dim

        # For recording memory usage
        self.memory_log_path = "info_log.csv"

        # self.grid_visualizer = GridPosVisualizer(
        #     output_dir='/raid5/ygtrece/LION/visualizations', 
        #     mode='2d&3d', show=False, sample=1, 
        #     point_cloud_range=self.model_cfg.get('POINT_CLOUD_RANGE', [-74.88, -74.88, -2, 74.88, 74.88, 4.0]),
        #     sparse_shape=self.sparse_shape[0][0])


    @staticmethod
    @torch.no_grad()
    def _record_info(
            # self,
            memory_log_path,
            stage_name,
            info_before=None,
            feats_after=None,
            feats_before=None,
            newline=False,
            execute=True,
        ):
        if not os.path.exists(memory_log_path):
            with open(memory_log_path, "w", newline="") as f:
                f.write("Time,  Stage,  Delta Time(s),  Memory-Delta(MB),  Memory-Delta(GB),  NEVoxel-Delta,  Memory-Before(MB),  Memory-After(MB)\n")
        if not execute:
            return None
        if stage_name is None:
            with open(memory_log_path, "a", newline="") as f:
                log_line = (
                    f"\n"
                )
                f.write(log_line)  
            return {}

        

        current_time = time.time() # for r
        dt = datetime.fromtimestamp(current_time)
        format_dt = dt.strftime('%Y-%m-%d %H:%M:%S')
        memory_after = torch.cuda.memory_allocated() / (1024 ** 2)
        if info_before is not None:
            memory_before = info_before['memory']
            time_before = info_before['time']
            # feats_after = info_before['feats']
        else:
            memory_before = memory_after
            time_before = current_time


        delta_mb = memory_after - memory_before
        delta_gb = delta_mb / 1024
        delta_time = current_time - time_before
        # ne_voxel_before = feats_before.shape[0].item() if feats_before is not None else None
        # ne_voxel_after = feats_after.shape[0] if feats_after is not None else None
        # delta_ne_voxel_num = ne_voxel_after - ne_voxel_before

        

        if stage_name.startswith("QDefMambav5_") or stage_name.startswith("Single"):

            char_num = 30 if stage_name == "QDefMambav5_" else 15
            banner = "=" * 15 + f" MEMORY TRACKING: {stage_name} " + "=" * char_num
            with open(memory_log_path, "a", newline="") as f:
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}, {banner}\n")
        
        
        with open(memory_log_path, "a", newline="") as f:
            log_line = (
                f"{format_dt:<19}, "
                f"{stage_name:<20}, "
                f"{delta_time:>10.3f} s"
                f"{delta_mb:>10.2f} Mb, "
                f"{delta_gb:>8.4f} Gb, "
                # f"{(ne_voxel_before if ne_voxel_before else None):>10}, " 
                # f"{ne_voxel_after:>10}, "
                f"{memory_before:>10.2f}, "
                f"{memory_after:>10.2f}, "
                f"\n"
            ) + ("\n" if newline else '')
            f.write(log_line)        

        info_current = {}
        info_current['time'] = current_time
        info_current['memory'] = memory_after
        return info_current


    def forward(self, batch_dict):
        debug = True if self.training and batch_dict['train_dict']['debug_mode'] else False
        locked = False | self.training and batch_dict['train_dict']['debug_mode'] # False when gt-training

        if self.training:
            train_dict = batch_dict['train_dict']
        batch_size = batch_dict['voxel_coords'][:, 0].max().item() + 1
        feat_3d = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']

        if debug and False:
            self.grid_visualizer.visualize_voxels_3d(
                voxel_coords.detach().cpu().numpy(),
                sample_idx=0,
                stage_idx=0,
                block_idx=0
            )
        
        # initialize query to be averagely distributed in the spatial position
        if self.total_queries > 0:
            hidden_queries = self.hidden_queries.weight.to(self.device)  # [total_queries, 2]
            # hidden_queries = self.hidden_queries.to(self.device)           # [total_queries, 2]

            # NOTE: revise to solve bs problem temporary
            # hidden_queries = hidden_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [bs, total_queries, dim]
            hidden_queries = hidden_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # [bs, total_queries, dim]
        else:
            hidden_queries = None
        
        if debug:
            QueryDeformableMambaLayer._record_info("info_log.csv", None)
        info_start = self._record_info(self.memory_log_path, f"Single Sample Start.....", execute=debug)

        idx = 0
        for i, num_s in enumerate(self.num_stage):
            # iterate through each block in the current stage
            # stage_results = [] 
            for ns in range(num_s):
                info_current = self._record_info(self.memory_log_path, f"QMamba_v5_Stage{i}", execute=debug)

                feat_3d, voxel_coords, hidden_queries, info_current = self.block_list[idx](
                    feat_3d, voxel_coords, hidden_queries, self.pos_embed, idx, debug)

                info_current = self._record_info(self.memory_log_path, f"v5_Stage{i}_1", info_current, execute=debug)
                
                idx += 1

            ### down-sample z axis additionaly ###
            xd = spconv.SparseConvTensor(
                features=feat_3d,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape[i][0],
                batch_size=batch_size
            )

            if i == (len(self.num_stage) - 1):
                xd = self.conv_out(xd)

            xd = self.downZ_list[i](xd)
            feat_3d = xd.features
            voxel_coords = xd.indices

            info_current = self._record_info(self.memory_log_path, f"v5_Stage{i}_2", info_current, execute=debug)


        ### Query Feedback ###
        if self.total_queries > 0:
            if self.wholy_residual == True:
                res_feat_3d = feat_3d.clone()
            # TODO 07.16 for bs problem: 应当用 zero_likes 接收最终特征, 而不是列表接收再 cat
            # out_feats_list = []
            # for b in range(batch_size):
            #     bm = (voxel_coords[:, 0] == b)
            #     kv = hidden_queries[b].unsqueeze(0)
            #     out_feats_list.append(self.query_transformer(Q=feat_3d[bm].unsqueeze(0), K=kv, V=kv.clone()).squeeze(0))
            # out_feat_3d = torch.cat(out_feats_list, dim=0)
            out_feat_3d = torch.zeros_like(feat_3d)
            for b in range(batch_size):
                bm = (voxel_coords[:, 0] == b)
                kv = hidden_queries[b].unsqueeze(0)
                out_feat_3d[bm] = self.query_transformer(Q=feat_3d[bm].unsqueeze(0), K=kv, V=kv.clone()).squeeze(0).to(dtype=feat_3d.dtype)

            if self.wholy_residual == True:
                if self.update_rate_learnable == True:
                    if self.hybrid_gate == True:
                        update_scale = torch.sigmoid(self.update_rate(torch.cat([res_feat_3d, out_feat_3d], dim=1))) # [N, 1]
                    else:
                        update_scale = torch.sigmoid(self.update_rate(res_feat_3d)) # [N, 1]
                    # update_scale = update_scale.clamp(min=0.01, max=0.99)
                    
                    # update_scale = torch.sigmoid(self.update_rate_p.reshape(1, 1)).to(dev
                    # ice=self.device) # [N, 1]
                    # update_scale = self.update_gate(self.update_rate) + feat_3d.mean() * 0 # For Test
                    feat_3d = update_scale * out_feat_3d + (1 - update_scale) * res_feat_3d
                    # print(f"scale={update_scale.mean().item()}")
                else:
                    feat_3d = self.update_rate * out_feat_3d + (1 - self.update_rate) * res_feat_3d
                feat_3d = self.norm_out(feat_3d)
            else:
                feat_3d = out_feat_3d
            info_current = self._record_info(self.memory_log_path, f"Transformer", info_current, execute=debug)

            if self.training and torch.isnan(hidden_queries).any().item():
                hidden_queries = torch.nan_to_num(hidden_queries, nan=0.0)

        # Detect and handle NaN (Not a Number) values ​​in feat_3d, usually used 
        # for numerical stability protection during training (From VoxelMamba)
        if self.training and torch.isnan(feat_3d).any().item():
            feat_3d = torch.nan_to_num(feat_3d, nan=0.0)

        if self.detector_type == 'CenterPoint':
            # MAP_TO_BEV: PointPillarScatter3d 
            # Following Voxel-Mamba
            batch_dict.update({
                'pillar_features': feat_3d,
                'voxel_features': feat_3d,
                'voxel_coords': voxel_coords
            })
        elif self.detector_type == 'TransFusion':
            # xd = xd.replace_feature(feat_3d + hidden_queries.mean()) # NOTE: for debug
            xd = xd.replace_feature(feat_3d)
            # MAP_TO_BEV: HeightCompression
            batch_dict.update({
                'encoded_spconv_tensor': xd,
                'encoded_spconv_tensor_stride': 1,
                # 'hidden_queries': hidden_queries
            })
            # batch_dict.update({'spatial_features_2d': x_new})
        
        ### for Query Forward Process Loss ###
        # if self.training and self.query_position_loss:
        #     pred_dicts = {
        #         'window_x_list': immediate_results, # S * window_x
        #     }
        #     target_dicts = {
        #         'gt_boxes': batch_dict['gt_boxes'], # [bs, num_boxes, 8]
        #     }
        #     # batch_dict['query_pos_loss'] = self.get_loss(pred_dicts, target_dicts)
        #     self.forward_ret_dict['pred_dicts'] = pred_dicts
        #     self.forward_ret_dict['target_dicts'] = target_dicts

        info_current = self._record_info(self.memory_log_path, f"Single Sample Time", info_start, execute=debug)
        return batch_dict


    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def load_template(self, path, rank):
        template = torch.load(path)
        if isinstance(template, dict):
            self.curve_template[f'curve_template_rank{rank}'] = template['data'].reshape(-1)
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = template['size'] 
        else:
            self.curve_template[f'curve_template_rank{rank}'] = template.reshape(-1)
            spatial_size = 2 ** rank
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = (1, spatial_size, spatial_size) # [z, y, x]
