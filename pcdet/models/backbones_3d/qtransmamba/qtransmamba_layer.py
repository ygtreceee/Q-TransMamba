import os
import math
import copy
import csv
import time
import torch
import numpy as np
import torch_scatter
import torch.nn as nn
from datetime import datetime
from functools import partial
import torch.utils.checkpoint as cp
from torch.nn import functional as F 
from collections import namedtuple, defaultdict
from typing import List, Tuple, Dict, Any, Union, Optional, Iterable

from mamba_ssm.models.mixer_seq_simple import create_block

from pcdet.utils.spconv_utils import replace_feature, spconv

from pcdet.models.backbones_3d.qdefmamba.basic_utils import Sparse1ConvBlock, \
                SparseBasicBlock3D, post_act_block, get_hilbert_index_3d_mamba_lite, SimpleFFN, QueryFFN_M
# from pcdet.models.backbones_3d.qdefmamba.fusion_module import FusionModule4_4 as FusionModule
from pcdet.models.backbones_3d.qdefmamba.fusion_module import FusionModule4_3 as FusionModule
# from pcdet.models.backbones_3d.qdefmamba.fusion_module import FusionModule4_2 as FusionModule
# from pcdet.models.backbones_3d.qdefmamba.fusion_module import FusionModule4_1 as FusionModule
# from pcdet.models.backbones_3d.qdefmamba.fusion_module import FusionModule3_2 as FusionModule
# from pcdet.models.backbones_3d.qdefmamba.fusion_module import FusionModule3_1 as FusionModule

from pcdet.models.backbones_3d.qdefmamba.deformable_sampler_4d_module import DeformableSampler

from pcdet.models.backbones_3d.qdefmamba.window_sparseconv_tensor import WindowSparseConvTensor
# from pcdet.models.backbones_3d.qdefmamba.fusion_module import SimpleFFN

from pcdet.models.backbones_3d.qdefmamba.qssm_4d_module import QSSM, QSSM_v5
from pcdet.models.backbones_3d.qdefmamba.gssm_module import GSSM_v1, GSSM_v2_v3, GSSM_v4, GSSM_v5, Bi_GSSM_v5
from pcdet.models.backbones_3d.qdefmamba.query_forward_branch import QueryFFN
from pcdet.models.backbones_3d.qdefmamba.qdefssm_module import QDefSSM

from spconv.pytorch import ops


class QueryDeformableMambaLayer_v4(nn.Module):
    def __init__(
            self,
            self_modeling_cfg: Dict,  # ?
            gssm_cfg: Dict,
            qdefssm_cfg: Dict,
            fusion_cfg: Dict, # ?
            num_stage: int,
            num_block: int,
            **kwargs
        ):
        super().__init__()

        self.dim = self_modeling_cfg['dim']
        self.sparse_shape = self_modeling_cfg['sparse_shape']

        norm = {
            "norm1d_fn": partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
            "norm2d_fn": partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01),
            "layernorm_fn": partial(nn.LayerNorm, eps=1e-5)
        }

        ### for self-modeling ###
        dim = self_modeling_cfg['dim']
        num_selfmodeling_layer = self_modeling_cfg['num_selfmodeling_layers'][num_stage][num_block]  # 2
        norm_fn = norm[self_modeling_cfg['norm_fn']]
        indice_key = f"stage_{num_stage}"
        self_modeling_blocks = []
        for i in range(num_selfmodeling_layer):
            self_modeling_blocks.append(
                Sparse1ConvBlock(dim, dim, norm_fn=norm_fn, indice_key=f'{indice_key}_selfmodeling'))
        self.self_modeling_blocks = spconv.SparseSequential(*self_modeling_blocks)

        self.gssm = GSSM_v4(**gssm_cfg)

        self.ffn_0 = SimpleFFN(dim)

        # self.qdefssm = QDefSSM(**qdefssm_cfg)
        self.qdefssm = QSSM_v5(**qdefssm_cfg) # TODO: test
        

        self.ffn_1 = SimpleFFN(dim)

        # For recording memory usage
        self.memory_log_path = "info_log.csv"

        
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
        ne_voxel_num = feats_after.shape[0] if feats_after is not None else 0
        # ne_voxel_before = feats_before.shape[0].item() if feats_before is not None else None
        # ne_voxel_after = feats_after.shape[0] if feats_after is not None else None
        # delta_ne_voxel_num = ne_voxel_after - ne_voxel_before



        if stage_name.startswith("Stage"):

            char_num = 30 if stage_name == "Stage_0_Initial" else 15
            banner = "=" * 15 + f" MEMORY TRACKING: {stage_name} " + "=" * char_num
            with open(memory_log_path, "a", newline="") as f:
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}, {banner}\n")
        
        
        with open(memory_log_path, "a", newline="") as f:
            log_line = (
                f"{dt.strftime('%Y-%m-%d %H:%M:%S'):<19}, "
                f"{stage_name:<20}, "
                f"{delta_time:>10.3f} s"
                f"{delta_mb:>10.2f} Mb, "
                f"{ne_voxel_num:>10f} "
                f"{delta_gb:>8.4f} Gb, "
                f"{memory_before:>10.2f}, "
                f"{memory_after:>10.2f}, "
                f"\n"
            ) + ("\n" if newline else '')
            f.write(log_line)        

        info_current = {}
        info_current['time'] = current_time
        info_current['memory'] = memory_after
        return info_current
        
        
        # return memory_after


    def forward(self,
            feats: torch.Tensor,
            coords: torch.Tensor,
            hidden_queries: torch.Tensor,
            pos_embed=None,
            num_stage=None, # current stage index
            debug=False
            ):
        # debug = debug
        # self.qdefssm.warmup()
        info_current = self._record_info(self.memory_log_path, f"Stage_{num_stage}_Initial", feats_after=feats, execute=debug)

        x = spconv.SparseConvTensor(
            feats, coords, self.sparse_shape, coords[:, 0].max().item() + 1) 
        x = self.self_modeling_blocks(x)
        info_current = self._record_info(self.memory_log_path, "After SelfModel", info_current, feats_after=feats, execute=debug)
        
        x = self.gssm(x, pos_embed, debug)
        feats = x.features
        coords = x.indices
        info_current = self._record_info(self.memory_log_path, "After GSSM", info_current, feats_after=feats, execute=debug)

        feats = self.ffn_0(feats)
        info_current = self._record_info(self.memory_log_path, "After FFN", info_current, feats_after=feats, execute=debug)

        feats, updated_hidden_queries = self.qdefssm(feats, coords, hidden_queries, pos_embed, debug=debug)
        info_current = self._record_info(self.memory_log_path, "After QSSM", info_current, feats_after=feats, execute=debug)

        feats = self.ffn_1(feats)
        info_current = self._record_info(self.memory_log_path, "After FFN", info_current, feats_after=feats, execute=debug)

        return feats, coords, updated_hidden_queries, info_current


class QueryDeformableMambaLayer_v3(nn.Module):
    '''
    parallel version,
                 --------- GSSM ---------->
    Selfmodeling                            Fusion
                 -> Def-Sampler -> QSSM -->
    '''
    def __init__(
            self,
            self_modeling_cfg: Dict,
            def_sampler_cfg: Dict,
            qssm_cfg: Dict,
            gssm_cfg: Dict,
            fusion_module_cfg: Dict,
            query_ffn_cfg: Dict,
            num_stage: int,
            num_block: int,
            **kwargs
        ):
        super().__init__()

        norm = {
            "norm1d_fn": partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
            "norm2d_fn": partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01),
            "layernorm_fn": partial(nn.LayerNorm, eps=1e-5)
        }
        
        ### for self-modeling ###
        dim = self_modeling_cfg['dim']
        num_selfmodeling_layer = self_modeling_cfg['num_selfmodeling_layers'][num_stage][num_block]  # 2
        norm_fn = norm[self_modeling_cfg['norm_fn']]
        indice_key = f"stage_{num_stage}"
        self_modeling_blocks = []
        for i in range(num_selfmodeling_layer):
            self_modeling_blocks.append(
                Sparse1ConvBlock(dim, dim, norm_fn=norm_fn, indice_key=f'{indice_key}_selfmodeling'))
        self.self_modeling_blocks = spconv.SparseSequential(*self_modeling_blocks)

        ### initialize deformable sampler ###
        self.deformable_sampler = DeformableSampler(**def_sampler_cfg)
        
        ### initialize QSSM ###
        self.qssm = QSSM(**qssm_cfg)

        ### initialize GSSM ###
        self.gssm = GSSM_v4(**gssm_cfg)
        
        ### initialize Query FFN ###
        self.query_ffn = QueryFFN(**query_ffn_cfg)
        self.curve_template = query_ffn_cfg['curve_template']
        self.hilbert_spatial_size = query_ffn_cfg['hilbert_spatial_size']

        ### initialize Fusion Module ###
        self.fusion_module = FusionModule(**fusion_module_cfg)


        # For recording memory usage
        self.memory_log_path = "info_log.csv"
        if not os.path.exists(self.memory_log_path):
            with open(self.memory_log_path, "w", newline="") as f:
                f.write("Time,  Stage,  Memory-Delta(MB),  Memory-Delta(GB),  Memory-Before(MB),  Memory-After(MB)\n")


    @staticmethod
    def _record_info(
            # self, 
            memory_log_path,
            stage_name, 
            memory_before=None, 
            feats_after=None,
            feats_before=None,
            newline=False,
            execute=True,
        ):
        if not execute:
            return None

        current_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

        memory_after = torch.cuda.memory_allocated() / (1024 ** 2)
        memory_before = memory_before if memory_before else memory_after
        delta_mb = memory_after - memory_before
        delta_gb = delta_mb / 1024
        # ne_voxel_before = feats_before.shape[0].item() if feats_before is not None else None
        ne_voxel_after = feats_after.shape[0] if feats_after is not None else None
        # delta_ne_voxel_num = ne_voxel_after - ne_voxel_before
        
        if stage_name is None:
            with open(memory_log_path, "a", newline="") as f:
                log_line = (
                    f"\n"
                )
                f.write(log_line)  
            return None

        if stage_name.startswith("Stage"):

            char_num = 30 if stage_name == "Stage_0_Initial" else 15
            banner = "=" * 15 + f" MEMORY TRACKING: {stage_name} " + "=" * char_num
            with open(memory_log_path, "a", newline="") as f:
                f.write(f"{current_time}, {banner}\n")
        
        
        with open(memory_log_path, "a", newline="") as f:
            log_line = (
                f"{current_time:<19}, "
                f"{stage_name:<20}, "
                f"{delta_mb:>10.2f} Mb, "
                f"{delta_gb:>8.4f} Gb, "
                # f"{(ne_voxel_before if ne_voxel_before else None):>10}, " 
                f"{ne_voxel_after:>10}, "
                f"{memory_before:>10.2f}, "
                f"{memory_after:>10.2f}, "
                f"\n"
            ) + ("\n" if newline else '')
            f.write(log_line)        
        
        return memory_after


    def forward(self,
            window_x : WindowSparseConvTensor,
            qssm_pos_embed,
            gssm_pos_embed,
            num_stage  # current stage index
            ):
        # debug
        debug = True
        
        current_memory = self._record_info(self.memory_log_path, f"Stage_{num_stage}_Initial", feats_after=window_x.sparse_tensor.features, execute=debug)
        # current_memory = self._record_info(f"Stage_{num_stage}_Initial", feats_after=window_x.unify_sparse_tensor.features)

        query_pos_embed, query_embed_res = self.query_ffn(query=window_x.window_pos)

        # NOTE: 06.20 先进行自建模, 然后将 self-modeling 的输出作为输入传入后续模块
        F_sm = self.self_modeling_blocks(window_x.unify_sparse_tensor)
        window_x.update_with_sparse(F_sm, activate=True)
        current_memory = self._record_info(self.memory_log_path, f"After SelfModling", feats_after=window_x.sparse_tensor.features, execute=debug)

        # 1 G
        F_g = self.gssm(
            window_x.unify_sparse_tensor, 
            gssm_pos_embed
        )
        current_memory = self._record_info(self.memory_log_path, "After GSSM", current_memory, feats_after=F_g.features, execute=debug)


        # 0.65 G
        window_sample_x, window_enhanced_main_x, window_sample_pos_global = self.deformable_sampler(
            window_x, sparse_grid_sample=True)
        current_memory = self._record_info(self.memory_log_path, "After Def-Sampler", current_memory, feats_after=window_enhanced_main_x.sparse_tensor.features, execute=debug)
        # current_memory = self._record_info("After Def-Sampler", current_memory, feats_after=window_enhanced_main_x.unify_sparse_tensor.features)


        # 1.2 G 
        window_F_q, enhanced_query_pos_embed = self.qssm(
                window_sample_x, window_enhanced_main_x, window_sample_pos_global, 
                query_pos_embed, qssm_pos_embed)
        current_memory = self._record_info(self.memory_log_path, "After QSSM", current_memory, feats_after=window_F_q.sparse_tensor.features, execute=debug)
        # current_memory = self._record_info("After QSSM", current_memory, feats_after=window_F_q.unify_sparse_tensor.features)
        

        # < 0.1 GB
        F_q_sp = window_F_q.merge_all_window(merge_type='mean') 
        current_memory = self._record_info(self.memory_log_path, "After Merge", current_memory, feats_after=window_F_q.sparse_tensor.features, execute=debug)
        # current_memory = self._record_info("After Merge", current_memory, feats_after=window_F_q.unify_sparse_tensor.features)

        updated_query_pos = self.query_ffn(query=window_x.window_pos, query_embed_res=query_embed_res, # TODO: The residual of `query_embed_res` is just possible
                                        hidden_query_updated=enhanced_query_pos_embed)

        # NOTE: update new query sequence to sparse_tensor (for future)
        window_F_q.window_pos = updated_query_pos
        window_F_q.query_sort(self.curve_template, self.hilbert_spatial_size, updated_query_pos)
        window_F_q.update_with_points()

        # < 0.1 GB
        fm_output = self.fusion_module(F_g, F_q_sp)
        F_down, coords_down = fm_output[0], fm_output[1]
        current_memory = self._record_info(self.memory_log_path, "After Fusion", current_memory, feats_after=F_down, newline=False, execute=debug)
        
        return F_down, coords_down, window_F_q
    

class QueryDeformableMambaLayer_v2(nn.Module):
    '''
    sequential version
    GSSM -> Def-Sampler -> QSSM -> Fusion
    '''
    def __init__(
            self,
            self_modeling_cfg: Dict,
            def_sampler_cfg: Dict,
            qssm_cfg: Dict,
            gssm_cfg: Dict,
            fusion_module_cfg: Dict,
            query_ffn_cfg: Dict,
            num_stage: int,
            num_block: int,
            **kwargs
        ):
        super().__init__()

        norm = {
            "norm1d_fn": partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
            "norm2d_fn": partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01),
            "layernorm_fn": partial(nn.LayerNorm, eps=1e-5)
        }
        
        ### for self-modeling ###
        dim = self_modeling_cfg['dim']
        num_selfmodeling_layer = self_modeling_cfg['num_selfmodeling_layers'][num_stage][num_block]  # 2
        norm_fn = norm[self_modeling_cfg['norm_fn']]
        indice_key = f"stage_{num_stage}"

        # NOTE 06.24
        # self_modeling_blocks = []
        # for i in range(num_selfmodeling_layer):
        #     self_modeling_blocks.append(
        #         Sparse1ConvBlock(dim, dim, norm_fn=norm_fn, indice_key=f'{indice_key}_selfmodeling'))
        # self.self_modeling_blocks = spconv.SparseSequential(*self_modeling_blocks)

        ### initialize deformable sampler ###
        self.deformable_sampler = DeformableSampler(**def_sampler_cfg)
        
        ### initialize QSSM ###
        self.qssm = QSSM(**qssm_cfg)

        ### initialize GSSM ###
        self.gssm = GSSM_v4(**gssm_cfg)

        ### initialize Query FFN ###
        self.query_ffn = QueryFFN(**query_ffn_cfg)
        self.curve_template = query_ffn_cfg['curve_template']
        self.hilbert_spatial_size = query_ffn_cfg['hilbert_spatial_size']
        
        ### initialize Fusion Module ###
        self.fusion_module = FusionModule(**fusion_module_cfg)

        # For recording memory usage
        self.memory_log_path = "info_log.csv"
        if not os.path.exists(self.memory_log_path):
            with open(self.memory_log_path, "w", newline="") as f:
                f.write("Time,  Stage,  Memory-Delta(MB),  Memory-Delta(GB),  NEVoxel-Delta,  Memory-Before(MB),  Memory-After(MB)\n")


    @staticmethod
    def _record_info(
            # self, 
            memory_log_path,
            stage_name, 
            memory_before=None, 
            feats_after=None,
            feats_before=None,
            newline=False,
            execute=True,
        ):
        if not execute:
            return None

        current_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

        memory_after = torch.cuda.memory_allocated() / (1024 ** 2)
        memory_before = memory_before if memory_before else memory_after
        delta_mb = memory_after - memory_before
        delta_gb = delta_mb / 1024
        # ne_voxel_before = feats_before.shape[0].item() if feats_before is not None else None
        ne_voxel_after = feats_after.shape[0] if feats_after is not None else None
        # delta_ne_voxel_num = ne_voxel_after - ne_voxel_before
        
        if stage_name is None:
            with open(memory_log_path, "a", newline="") as f:
                log_line = (
                    f"\n"
                )
                f.write(log_line)  
            return None

        if stage_name.startswith("Stage"):

            char_num = 30 if stage_name == "Stage_0_Initial" else 15
            banner = "=" * 15 + f" MEMORY TRACKING: {stage_name} " + "=" * char_num
            with open(memory_log_path, "a", newline="") as f:
                f.write(f"{current_time}, {banner}\n")
        
        
        with open(memory_log_path, "a", newline="") as f:
            log_line = (
                f"{current_time:<19}, "
                f"{stage_name:<20}, "
                f"{delta_mb:>10.2f} Mb, "
                f"{delta_gb:>8.4f} Gb, "
                # f"{(ne_voxel_before if ne_voxel_before else None):>10}, " 
                f"{ne_voxel_after:>10}, "
                f"{memory_before:>10.2f}, "
                f"{memory_after:>10.2f}, "
                f"\n"
            ) + ("\n" if newline else '')
            f.write(log_line)        
        
        return memory_after


    def forward(self,
            window_x : WindowSparseConvTensor,
            qssm_pos_embed,
            gssm_pos_embed,
            num_stage  # current stage index
            ):
        # debug
        debug = True
        
        current_memory = self._record_info(self.memory_log_path, f"Stage_{num_stage}_Initial", feats_after=window_x.sparse_tensor.features, execute=debug)
        # current_memory = self._record_info(f"Stage_{num_stage}_Initial", feats_after=window_x.unify_sparse_tensor.features)

        query_pos_embed, query_embed_res = self.query_ffn(query=window_x.window_pos)

        F_g = self.gssm(
            window_x.unify_sparse_tensor,
            gssm_pos_embed
        )
        window_x.update_with_sparse(F_g, activate=True) 

        current_memory = self._record_info(self.memory_log_path, "After GSSM", current_memory, feats_after=F_g.features, execute=debug)


        window_sample_x, window_enhanced_main_x, window_sample_pos_global = self.deformable_sampler(
            window_x, sparse_grid_sample=True)
        current_memory = self._record_info(self.memory_log_path, "After Def-Sampler", current_memory, feats_after=window_enhanced_main_x.sparse_tensor.features, execute=debug)
        # current_memory = self._record_info("After Def-Sampler", current_memory, feats_after=window_enhanced_main_x.unify_sparse_tensor.features)


        window_F_q, enhanced_query_pos_embed = self.qssm(
                window_sample_x, window_enhanced_main_x, window_sample_pos_global, 
                query_pos_embed, qssm_pos_embed)
        current_memory = self._record_info(self.memory_log_path, "After QSSM", current_memory, feats_after=window_F_q.sparse_tensor.features, execute=debug)
        # current_memory = self._record_info("After QSSM", current_memory, feats_after=window_F_q.unify_sparse_tensor.features)
        
        F_q_sp = window_F_q.merge_all_window(merge_type='mean') 
        current_memory = self._record_info(self.memory_log_path, "After Merge", current_memory, feats_after=window_F_q.sparse_tensor.features, execute=debug)
        # current_memory = self._record_info("After Merge", current_memory, feats_after=window_F_q.unify_sparse_tensor.features)

        updated_query_pos = self.query_ffn(query=window_x.window_pos, query_embed_res=query_embed_res, # TODO: The residual of `query_embed_res` is just possible
                                        hidden_query_updated=enhanced_query_pos_embed)
        
        # NOTE: update new query sequence to sparse_tensor (for future)
        window_F_q.window_pos = updated_query_pos
        window_F_q.query_sort(self.curve_template, self.hilbert_spatial_size, updated_query_pos)
        window_F_q.update_with_points()

        fm_output = self.fusion_module(F_g, F_q_sp)
        F_down, coords_down = fm_output[0], fm_output[1]
        current_memory = self._record_info(self.memory_log_path, "After Fusion", current_memory, feats_after=F_down, newline=False, execute=debug)
        
        return F_down, coords_down, window_F_q
    

class QueryDeformableMambaLayer_v1(nn.Module):
    '''
    GLSSM (Encoder-Decoder Architecture)
    GSSM -> GSSM -> QSSM -> GSSM -> GSSM (G G Q G G)
    '''
    def __init__(
            self,
            def_sampler_cfg: Dict,
            qssm_cfg: Dict,
            query_ffn_cfg: Dict,
            dim: int,
            window_overlap: int,
            gssm_kernel_size: List,
            gssm_stride: List,
            sub_num: List,
            curve_template: Dict,
            hilbert_spatial_size: Dict,
            norm_epsilon,
            down_resolution: List,
            rms_norm: bool,
            fused_add_norm: bool,
            residual_in_fp32: bool,
            ssm_lvl: List,
            num_stage: int,
            num_block: int,
            merge_threshold: List,
            module_list: List,
            device: str = None,
            dtype=None,
            **kwargs
        ):
        super().__init__()

        self.norm1d_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm2d_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.layernorm_fn = partial(nn.LayerNorm, eps=1e-5)

        self.sparse_shape_list = def_sampler_cfg['sparse_shape']
        self.window_shape_list = def_sampler_cfg['window_shape']
        self.merge_threshold_list = def_sampler_cfg['merge_threshold']
        self.module_list = module_list


        indice_key = f"layer_{num_stage}_{num_block}"

        block = SparseBasicBlock3D
        # SubSparseConv3d for first layout (1x, so without down-sampling)
        self.encoder = nn.ModuleList(
            [spconv.SparseSequential(
                *[block(dim, indice_key=f"{indice_key}_encoder") for _ in range(sub_num[0])])]
        )

        # down sample layer
        self.num_levels = len(gssm_stride)
        for idx in range(1, self.num_levels):
            cur_layers = [
                post_act_block(
                    in_channels=dim, out_channels=dim, kernel_size=gssm_kernel_size[idx], 
                    stride=gssm_stride[idx], padding=gssm_kernel_size[idx] // 2,
                    conv_type='spconv', indice_key=f'enc_spconv_{indice_key}_{idx}', norm_fn=self.layernorm_fn),
                *[block(dim, indice_key=f"{indice_key}_{idx}") for _ in range(sub_num[idx])]
            ]
            self.encoder.append(spconv.SparseSequential(*cur_layers))

        # up sample layer
        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(self.num_levels - 1, 0, -1):
            self.decoder.append(
                post_act_block(
                    in_channels=dim, out_channels=dim, kernel_size=gssm_kernel_size[idx], 
                    conv_type='inverseconv', indice_key=f'enc_spconv_{indice_key}_{idx}', norm_fn=self.layernorm_fn))
            self.decoder_norm.append(self.layernorm_fn(dim))

        # ssm_cfg = {}
        factory_kwargs = {'device': device, 'dtype':dtype}
        self.curve_template = curve_template
        self.hilbert_spatial_size = hilbert_spatial_size
        
        self.total_layers = (self.num_levels - 1) * 2 + 1
        self.block_list = nn.ModuleList()
        for idx, module in enumerate(module_list):
            sparse_shape, window_shape = self.sparse_shape_list[idx], self.window_shape_list[idx]
            if module == 'GSSM':
                self.block_list.append(
                    GSSM_v1(dim=dim, num_stage=num_stage, num_block=num_block, num_lvl=idx, gssm_lvl=ssm_lvl[idx], 
                         norm_epsilon=norm_epsilon, rms_norm=rms_norm, curve_template=curve_template, 
                         hilbert_spatial_size=hilbert_spatial_size, residual_in_fp32=residual_in_fp32, 
                         fused_add_norm=fused_add_norm, device=device, dtype=dtype,
                         window_shape=window_shape, window_overlap=window_overlap, sparse_shape=sparse_shape)
                )

            elif module == 'LSSM':
                def_sampler_cfg['window_shape'], def_sampler_cfg['sparse_shape'] = window_shape, sparse_shape
                qssm_cfg['window_shape'], qssm_cfg['sparse_shape'] = window_shape, sparse_shape
                self.block_list.append(
                    nn.Sequential(DeformableSampler(**def_sampler_cfg), QSSM(**qssm_cfg))
                )
                # def_sampler_cfg['window_shape'], def_sampler_cfg['sparse_shape'] = window_shape, sparse_shape
                # qssm_cfg['window_shape'], qssm_cfg['sparse_shape'] = window_shape, sparse_shape
                # self.block_list.append(
                #     DeformableSampler(**def_sampler_cfg))

        ### initialize Query FFN ###
        self.query_ffn = QueryFFN(**query_ffn_cfg)
        self.curve_template = query_ffn_cfg['curve_template']
        self.hilbert_spatial_size = query_ffn_cfg['hilbert_spatial_size']

        # For recording memory usage
        self.memory_log_path = "info_log.csv"
        
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
        ne_voxel_after = feats_after.shape[0] if feats_after is not None else None
        # delta_ne_voxel_num = ne_voxel_after - ne_voxel_before



        if stage_name.startswith("Stage"):

            char_num = 30 if stage_name == "Stage_0_Initial" else 15
            banner = "=" * 15 + f" MEMORY TRACKING: {stage_name} " + "=" * char_num
            with open(memory_log_path, "a", newline="") as f:
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}, {banner}\n")
        
        
        with open(memory_log_path, "a", newline="") as f:
            log_line = (
                f"{dt.strftime('%Y-%m-%d %H:%M:%S'):<19}, "
                f"{stage_name:<20}, "
                f"{delta_time:>10.3f} s"
                f"{delta_mb:>10.2f} Mb, "
                f"{delta_gb:>8.4f} Gb, "
                # f"{(ne_voxel_before if ne_voxel_before else None):>10}, " 
                f"{ne_voxel_after:>10}, "
                f"{memory_before:>10.2f}, "
                f"{memory_after:>10.2f}, "
                f"\n"
            ) + ("\n" if newline else '')
            f.write(log_line)        

        info_current = {}
        info_current['time'] = current_time
        info_current['memory'] = memory_after
        return info_current
        
        
        # return memory_after


    def forward(self,
            window_x : WindowSparseConvTensor,
            qssm_pos_embed=None,
            gssm_pos_embed=None,
            num_stage=None, # current stage index
            debug=False
            ):
        # debug = debug
        
        info_current = self._record_info(self.memory_log_path, f"Stage_{num_stage}_Initial", feats_after=window_x.sparse_tensor.features, execute=debug)

        idx = 0
        feats_list = []
        x = window_x.unify_sparse_tensor

        query_pos_embed, query_embed_res = self.query_ffn(query=window_x.window_pos)
        info_current = self._record_info(self.memory_log_path, "After QFFN", info_current, feats_after=window_x.unify_sparse_tensor.features, execute=debug)

        for conv in self.encoder:

            x = conv(x)
            info_current = self._record_info(self.memory_log_path, "x = conv(x)", info_current, feats_after=window_x.unify_sparse_tensor.features, execute=debug)
        
            if self.module_list[idx] == 'GSSM':
                x = self.block_list[idx](x, gssm_pos_embed, debug)
                info_current = self._record_info(self.memory_log_path, "GSSM op", info_current, feats_after=window_x.unify_sparse_tensor.features, execute=debug)
            elif self.module_list[idx] == 'LSSM':
                window_x.update_with_sparse(x, activate=True,
                        window_shape=self.window_shape_list[idx], 
                        merge_threshold=self.merge_threshold_list[idx], debug=debug)                
                
                info_current = self._record_info(self.memory_log_path, "window.update", info_current, feats_after=window_x.unify_sparse_tensor.features, execute=debug)
                
                deformable_sampler, qssm = self.block_list[idx][0], self.block_list[idx][1]
                window_sample_x, window_enhanced_main_x, window_sample_pos_global = deformable_sampler(
            window_x, sparse_grid_sample=True, debug=debug)
                info_current = self._record_info(self.memory_log_path, "Defsp op", info_current, feats_after=window_x.unify_sparse_tensor.features, execute=debug)
                
                window_F_q, enhanced_query_pos_embed = qssm(window_sample_x, window_enhanced_main_x, 
            window_sample_pos_global, query_pos_embed, qssm_pos_embed, debug=debug)
                x = window_x.unify_sparse_tensor.replace_feature(window_F_q.unify_sparse_tensor.features)
                info_current = self._record_info(self.memory_log_path, "QSSM op", info_current, feats_after=window_x.unify_sparse_tensor.features, execute=debug)

            feats_list.append(x)
            idx += 1
        

        x = feats_list[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats_list[:-1][::-1]):

            x = deconv(x)
            info_current = self._record_info(self.memory_log_path, "x = deconv(x)", info_current, feats_after=window_x.unify_sparse_tensor.features, execute=debug)
            if self.module_list[idx] == 'GSSM':
                x = self.block_list[idx](x, gssm_pos_embed, debug)
                info_current = self._record_info(self.memory_log_path, "GSSM op", info_current, feats_after=window_x.unify_sparse_tensor.features, execute=debug)

            x = x.replace_feature(norm(x.features + up_x.features))
            idx += 1


        updated_query_pos = self.query_ffn(query=window_x.window_pos, query_embed_res=query_embed_res, # TODO: The residual of `query_embed_res` is just possible
                                        hidden_query_updated=enhanced_query_pos_embed)
        info_current = self._record_info(self.memory_log_path, "After QFFN", info_current, feats_after=window_x.unify_sparse_tensor.features, execute=debug)

        window_F_q.window_pos = updated_query_pos
        # window_F_q.query_sort(self.curve_template, self.hilbert_spatial_size, updated_query_pos)
        # info_current = self._record_info(self.memory_log_path, "window.querysort", info_current, feats_after=window_F_q.unify_sparse_tensor.features, execute=debug)
        window_F_q.update_with_points(debug=debug)
        info_current = self._record_info(self.memory_log_path, "window.update", info_current, feats_after=window_F_q.unify_sparse_tensor.features, execute=debug)

        return x.features, x.indices, window_F_q, info_current



class QueryDeformableMambaLayer_v5(nn.Module):
    '''
    GLSSM (Encoder-Decoder Architecture)
    '''
    def __init__(
            self,
            dim: int,
            ffn_cfg: Dict,
            qssm_cfg: Dict,
            query_ffn_cfg: Dict,
            sparse_shape: List,
            gssm_kernel_size: List,
            gssm_stride: List,
            sub_num: List,
            norm_epsilon,
            rms_norm: bool,
            fused_add_norm: bool,
            residual_in_fp32: bool,
            force_layernorm: bool,
            total_queries: int,
            num_stage: int,
            num_block: int,
            revise_resolution: List,
            scan_axis_list: List,
            module_list: List,
            space_filing_curve: str,
            window_pos_embed: bool = False,
            curve_template: Dict = None,
            hilbert_spatial_size: Dict = None,
            device: str = None,
            dtype=None,
            **kwargs
        ):
        super().__init__()
        self.layernorm_fn = partial(nn.LayerNorm, eps=1e-5)
        self.norm1d_fn = self.layernorm_fn if force_layernorm == True else partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm2d_fn = self.layernorm_fn if force_layernorm == True else partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.total_queries = total_queries
        self.force_layernorm = force_layernorm

        self.sparse_shape_list = sparse_shape

        self.gssm_kernel_size = gssm_kernel_size[num_stage]
        self.gssm_stride = gssm_stride[num_stage]
        self.sub_num = sub_num[num_stage]
        self.module_list = module_list[num_stage]
        self.scan_axis_list = scan_axis_list[num_stage]
        self.revise_resolution = revise_resolution[num_stage]

        indice_key = f"layer_{num_stage}_{num_block}"

        self.curve_template = curve_template
        self.hilbert_spatial_size = hilbert_spatial_size
        
        ### for self-modeling ###
        # dim = self_modeling_cfg['dim']
        # indice_key = f"stage_{num_stage}_{num_block}"
        # self.self_modeling_blocks = []
        # self.self_modeling_blocks = nn.ModuleList([
        #     spconv.SparseSequential(
        #         *[Sparse1ConvBlock(
        #             dim, dim, norm_fn=self.norm1d_fn, indice_key=f'{indice_key}_{i}_{j}_selfmodeling', device=device
        #         ) for j in range(layer_num)]
        #     )
        #     for i, layer_num in enumerate(self_modeling_cfg['num_selfmodeling_layers'][num_stage][num_block])
        # ])

        self.ffn_blocks = nn.ModuleList()
        for i, layer_num in enumerate(ffn_cfg['num_ffn_layers'][num_stage][num_block]):
            if layer_num == 0:
                self.ffn_blocks.append(nn.Identity())
                continue
            for j in range(layer_num):
                self.ffn_blocks.append(SimpleFFN(norm_fn=self.layernorm_fn, **ffn_cfg))

        # For downsample and upsample conv
        block = SparseBasicBlock3D
        self.sample_conv = nn.ModuleList()
        self.sample_norm = nn.ModuleList()

        self.num_levels = len(self.module_list)
        for idx in range(0, self.num_levels):
            if idx < (self.num_levels + 1) // 2 and self.revise_resolution[idx] == True:
                cur_layers = [
                    post_act_block(
                        in_channels=dim, out_channels=dim, kernel_size=self.gssm_kernel_size[idx], 
                        stride=self.gssm_stride[idx], padding=self.gssm_kernel_size[idx] // 2,
                        conv_type='spconv', indice_key=f'sample_spconv_{indice_key}_{idx}', norm_fn=self.norm1d_fn),
                    *[block(dim, norm_fn=self.norm1d_fn, indice_key=f"{indice_key}_{idx}") for _ in range(self.sub_num[idx])]
                ]
                self.sample_conv.append(spconv.SparseSequential(*cur_layers))
            elif idx >= (self.num_levels + 1) // 2 and self.revise_resolution[idx] == True:
                    self.sample_norm.append(self.norm1d_fn(dim))
                    self.sample_conv.append(
                        post_act_block(
                            in_channels=dim, out_channels=dim, kernel_size=self.gssm_kernel_size[idx], 
                            conv_type='inverseconv', indice_key=f'sample_spconv_{indice_key}_{self.num_levels - idx}', norm_fn=self.norm1d_fn))


        self.total_layers = self.num_levels
        self.block_list = nn.ModuleList()
        for idx, module in enumerate(self.module_list):
            sparse_shape = self.sparse_shape_list[idx]
            if module == 'GSSM':
                self.block_list.append(
                    GSSM_v5(dim=dim, num_stage=num_stage, num_block=num_block, ssm_idx=idx, num_lvl=idx, sparse_shape=sparse_shape,
                         norm_epsilon=norm_epsilon, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32, force_layernorm=force_layernorm,
                         fused_add_norm=fused_add_norm, device=device, dtype=dtype, window_pos_embed=window_pos_embed,
                         space_filing_curve=space_filing_curve, curve_template=curve_template, hilbert_spatial_size=hilbert_spatial_size)
                )
            elif module == 'Bi_GSSM':
                self.block_list.append(
                    Bi_GSSM_v5(dim=dim, num_stage=num_stage, num_block=num_block, ssm_idx=idx, num_lvl=idx, sparse_shape=sparse_shape,
                         norm_epsilon=norm_epsilon, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32, force_layernorm=force_layernorm,
                         fused_add_norm=fused_add_norm, device=device, dtype=dtype, window_pos_embed=window_pos_embed,
                         space_filing_curve=space_filing_curve, curve_template=curve_template, hilbert_spatial_size=hilbert_spatial_size)
                )

            elif module == 'QSSM':
                qssm_cfg['sparse_shape'] = sparse_shape
                qssm_cfg['curve_template'] = curve_template
                qssm_cfg['hilbert_spatial_shape'] = hilbert_spatial_size
                self.block_list.append(
                    QSSM_v5(**qssm_cfg)
                )

        ### Query FFN ###
        if self.total_queries > 0:
            assert query_ffn_cfg['ffn_type'] in ['simple', 'complex']
            self.query_ffn_blocks = nn.ModuleList()
            query_ffn_block = QueryFFN_M if query_ffn_cfg['ffn_type'] == 'complex' else SimpleFFN
            for i, layer_num in enumerate(query_ffn_cfg['num_qffn_layers'][num_stage][num_block]):
                if layer_num == 0:
                    self.query_ffn_blocks.append(nn.Identity())
                    continue
                for j in range(layer_num):
                    self.query_ffn_blocks.append(query_ffn_block(norm_fn=self.layernorm_fn, **query_ffn_cfg))
                    # self.query_ffn_blocks.append(SimpleFFN(norm_fn=self.norm1d_fn, **query_ffn_cfg))

        # For recording memory usage
        self.memory_log_path = "info_log.csv"

        
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
        ne_voxel_num = feats_after.shape[0] if feats_after is not None else 0
        # ne_voxel_before = feats_before.shape[0].item() if feats_before is not None else None
        # ne_voxel_after = feats_after.shape[0] if feats_after is not None else None
        # delta_ne_voxel_num = ne_voxel_after - ne_voxel_before


        if stage_name.startswith("Stage"):

            char_num = 30 if stage_name == "Stage_0_Initial" else 15
            banner = "=" * 15 + f" MEMORY TRACKING: {stage_name} " + "=" * char_num
            with open(memory_log_path, "a", newline="") as f:
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}, {banner}\n")
        
        
        with open(memory_log_path, "a", newline="") as f:
            log_line = (
                f"{dt.strftime('%Y-%m-%d %H:%M:%S'):<19}, "
                f"{stage_name:<20}, "
                f"{delta_time:>10.3f} s"
                f"{delta_mb:>10.2f} Mb, "
                f"{ne_voxel_num:>10f} "
                f"{delta_gb:>8.4f} Gb, "
                f"{memory_before:>10.2f}, "
                f"{memory_after:>10.2f}, "
                f"\n"
            ) + ("\n" if newline else '')
            f.write(log_line)        

        info_current = {}
        info_current['time'] = current_time
        info_current['memory'] = memory_after
        return info_current
        
        
        # return memory_after


    def forward(self,
            feats,
            coords,
            hidden_queries,
            pos_embed=None,
            num_stage=None, # current stage index
            debug=False
            ):
        # debug = debug
        
        info_current = self._record_info(self.memory_log_path, f"Stage_{num_stage}_Initial", execute=debug)

        x = spconv.SparseConvTensor(
            features=feats,
            indices=coords,
            spatial_shape=self.sparse_shape_list[0],
            batch_size=coords[:, 0].max().item() + 1
        )

        if self.total_queries > 0:
            out_hidden_queries = hidden_queries.clone()
        else:
            out_hidden_queries = None

        conv_idx = 0
        feats_list = []

        for block_idx, block in enumerate(self.block_list):

            if self.revise_resolution[block_idx] == True:
                x  = self.sample_conv[conv_idx](x)
                conv_idx += 1
            if block_idx >= (self.num_levels + 1) // 2 and self.revise_resolution[block_idx] == True:
                up_x = feats_list[-1]
                feats_list.pop()
                x = x.replace_feature(self.sample_norm[block_idx - (self.num_levels + 1) // 2 - (1 - self.num_levels % 2)](x.features + up_x.features))

            info_current = self._record_info(self.memory_log_path, "conv&modeling", info_current, feats_after=x.features, execute=debug)

            if self.module_list[block_idx] == 'GSSM' or self.module_list[block_idx] == 'Bi_GSSM':
                x = block(x, pos_embed, self.scan_axis_list[block_idx], num_stage, debug)
                info_current = self._record_info(self.memory_log_path, "GSSM op", info_current, feats_after=x.features, execute=debug)
            elif self.module_list[block_idx] == 'QSSM':
                out_fq, out_hidden_queries = block(x, out_hidden_queries, pos_embed, self.scan_axis_list[block_idx], debug)
                x = x.replace_feature(out_fq)
                info_current = self._record_info(self.memory_log_path, "QSSM op", info_current, feats_after=x.features, execute=debug)
                if self.total_queries > 0:
                    out_hidden_queries = self.query_ffn_blocks[block_idx](out_hidden_queries)
                    info_current = self._record_info(self.memory_log_path, "FFN op", info_current, feats_after=x.features, execute=debug)

            x = x.replace_feature(self.ffn_blocks[block_idx](x.features))

            if block_idx < (self.num_levels // 2 - (1 - self.num_levels % 2)):
                feats_list.append(x)
            
        return x.features, x.indices, out_hidden_queries, info_current
