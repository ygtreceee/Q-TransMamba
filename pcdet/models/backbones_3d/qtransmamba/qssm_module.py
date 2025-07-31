import math
import copy
import os
import csv
from sympy import Identity
import torch
import numpy as np
import torch_scatter
import torch.nn as nn
import time
from datetime import datetime
from functools import partial
from collections import defaultdict
import torch.utils.checkpoint as cp
from torch.nn import functional as F 
from collections import namedtuple, defaultdict
from typing import List, Tuple, Dict, Any, Union, Optional, Iterable

from einops import rearrange
from mamba_ssm.models.mixer_seq_simple import create_block
import torch_scatter.scatter

from pcdet.utils.spconv_utils import replace_feature, spconv

from pcdet.models.backbones_3d.qdefmamba.basic_utils import get_hilbert_index_3d_mamba_lite, get_random_index_3d, \
        Sparse1ConvBlock, EfficientAttention1D, XFormersCrossAttention, sparse_add, SimpleFFN, \
        DepthwiseDownsample1d, DepthwiseUpsample1d, SequenceUpsampler, SequenceReducer
from pcdet.models.backbones_3d.qdefmamba.deformable_sampler_4d_module import DeformableSampler
from pcdet.models.backbones_3d.qdefmamba.window_sparseconv_tensor import WindowSparseConvTensor
# from pcdet.models.backbones_3d.qdefmamba.fusion_module import SimpleFFN

from morton_encoding import get_morton_index_3d



class QSSM(nn.Module):
    def __init__(self,
                 dim,
                 num_stage,
                 num_block,
                 num_query,
                 sparse_shape,
                 window_shape,
                 window_overlap,
                 mamba_cross_attn,
                 hidden_query_attn,
                 norm_epsilon, 
                 rms_norm,
                 curve_template,
                 hilbert_spatial_size,
                 ssm_cfg=None,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 device=None,
                 dtype=None,
                 **kwargs,
                ):
        super().__init__()

        self.dim = dim
        self.num_query = num_query
        self.sparse_shape = sparse_shape
        self.window_shape = window_shape
        self.window_overlap = window_overlap
        self.device = device
        self.dtype = dtype
        
        self.norm1d_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm2d_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.layernorm_fn = partial(nn.LayerNorm, eps=1e-5)

        self.mode = 'local_main'

        ssm_cfg = {}
        self.d_conv = 4 # NOTE: what is it
            
        factory_kwargs = {'device': device, 'dtype':dtype}

        mamba_encoder_i = create_block(
            d_model=dim,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=0,
            z_residual=True,
            **factory_kwargs,
        )

        mamba_encoder_j = create_block(
            d_model=dim,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=1, # TODO 06.13: 0 -> 1 尝试一下
            z_residual=False,
            # z_residual=True,
            **factory_kwargs,
        )

        self.mamba_encoder_list = nn.ModuleList([mamba_encoder_i, mamba_encoder_j])

        # NOTE: here we use LayerNorm for sequence following VoxelMamba
        self.ssm_norm_i = self.layernorm_fn(dim)
        self.ssm_norm_j = self.layernorm_fn(dim)

        # For Pre-modeling
        self.conv_i = Sparse1ConvBlock(dim, dim, norm_fn=self.layernorm_fn, indice_key="qssm_conv", activate='gelu')
        self.conv_j = Sparse1ConvBlock(dim, dim, norm_fn=self.layernorm_fn, indice_key="qssm_conv", activate='gelu')

        # For Cross Mamba
        # self.linear_v = nn.Linear(dim, dim)
        # self.s_j_norm = self.layernorm_fn(dim)
        self.curve_template = curve_template
        self.hilbert_spatial_size = hilbert_spatial_size
        
        self.cross_attention = XFormersCrossAttention(
            dim, attn_type=mamba_cross_attn, proj_type='conv1d', # 'conv1d' \ 'None'
            dropout=0.0, attn_cfg=None)
        self.res_norm_attn = self.layernorm_fn(dim)
        if hidden_query_attn == 'cross-attn':
            self.query_attn_module = XFormersCrossAttention(
                dim, attn_type=mamba_cross_attn, proj_type='conv1d', # 'conv1d' \ 'None'
                res_norm_fn=None, dropout=0.0, attn_cfg=None
            )
        elif hidden_query_attn == 'efficient-attn':
            self.query_attn_module = EfficientAttention1D(
                in_channels=dim,
                key_channels=dim,
                head_count=4,
                value_channels=dim
            )

        # downsampling for attn        
        # self.q_ds_1 = DepthwiseDownsample1d(channels=dim, kernel_size=3)
        # self.kv_ds_1 = DepthwiseDownsample1d(channels=dim, kernel_size=3)
        # self.q_us_1 = DepthwiseUpsample1d(channels=dim, kernel_size=3)
        # self.kv_ds_2 = DepthwiseDownsample1d(channels=dim, kernel_size=3)
        # self.q_ds_1 = SequenceReducer('mean')
        # self.kv_ds_1 = SequenceReducer('mean')
        # self.q_us_1 = SequenceUpsampler('linear')  # 'linear' \ 'nearest'
        # self.kv_ds_2 = SequenceReducer('mean')
        
        # For Feedback (Channel Gate)
        self.fb_norm = self.layernorm_fn(dim)
        self.linear_pool = nn.Linear(dim, dim) 
        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.activation = nn.ReLU() 

        # For recording memory usage
        self.memory_log_path = "info_log_qssm.csv"
        if not os.path.exists(self.memory_log_path):
            with open(self.memory_log_path, "w", newline="") as f:
                f.write("Time,  Stage,  Memory-Delta(MB),  Memory-Delta(GB),  Memory-Before(MB),  Memory-After(MB)\n")



    @staticmethod
    @torch.no_grad()
    def _record_info(
            # self, 
            memory_log_path,
            stage_name,
            info_before=None,
            # feats_after=None,
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
        # ne_voxel_after = feats_after.shape[0] if feats_after is not None else None
        # delta_ne_voxel_num = ne_voxel_after - ne_voxel_before

        if stage_name.startswith("Enter"):

            char_num = 30 if stage_name == "Stage_0_Initial" else 15
            banner = "=" * 15 + f" MEMORY TRACKING: {stage_name} " + "=" * char_num
            with open(memory_log_path, "a", newline="") as f:
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}, {banner}\n")
        
        if info_before is not None and 'attn_info' in info_before:
            attn_info = info_before['attn_info']
            with open(memory_log_path, "a", newline="") as f:
                log_line = (
                    f"{dt.strftime('%Y-%m-%d %H:%M:%S'):<19}, "
                    f"{stage_name:<20}, "
                    f"{delta_time:>10.3f} s"
                    f"{attn_info[0]:>6f}  "
                    f"{attn_info[1]:>6f}  "
                    f"{attn_info[2]:>6f}  "
                    f"{delta_mb:>10.2f} Mb, "
                    f"{delta_gb:>8.4f} Gb, "
                    f"{memory_before:>10.2f}, "
                    f"{memory_after:>10.2f}, "
                    f"\n"
                ) + ("\n" if newline else '')
                f.write(log_line) 
        else:
            with open(memory_log_path, "a", newline="") as f:
                log_line = (
                    f"{dt.strftime('%Y-%m-%d %H:%M:%S'):<19}, "
                    f"{stage_name:<20}, "
                    f"{delta_time:>10.3f} s"
                    f"{delta_mb:>10.2f} Mb, "
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

        

    def feedback(
            self, 
            attn_hidden_queries, 
            hidden_queries_batch,  # [window_length, C]
            # query_weights_batch,   # [window_length]
            residual=True
        ):
        '''
        attn_hidden_queries: [N, C]
        hidden_queries_batch: [N, C]
        query_weights_batch,   # [N]
        '''
        concatenated = torch.cat([attn_hidden_queries, hidden_queries_batch], dim=0) # [2N, C]
        concatenated = self.fb_norm(concatenated)  # NOTE: LayerNorm added here 
        pooled = torch.mean(concatenated, dim=0, keepdim=True)  # glocal_avg_pool -> [1, C] 
        l_pooled = self.linear_pool(pooled)
        channel_weight = self.activation(l_pooled)  # [1, C] NOTE: alternative activation is Sigmoid ([0, 1]), for stepping back

        weighted_enhanced = attn_hidden_queries * channel_weight # [N, C]
        out_hidden_queries = self.gamma * weighted_enhanced
        if residual == True:
            out_hidden_queries = out_hidden_queries + hidden_queries_batch  # [N, C]
        
        # out_hidden_queries = self.activation((residual_output))  # [N, C]

        # NOTE: 06.16 使用权重更新, 暂时直接将权重乘到 out_hidden_queries 上, 或许可以有别的操作
        # out_hidden_queries = out_hidden_queries * query_weights_batch[:, None]  # [N, C], 这里的 query_weights_batch 是 [N] 的权重向量
        
        return out_hidden_queries
    


    def gather_and_broadcast(
        self,
        valid_indices: torch.Tensor,  # [K] 主查询点索引
        attn_hidden_queries: torch.Tensor,  # [K, C] 主窗口的隐藏状态
        batch_hidden_queries: torch.Tensor,  # [total_queries, C] 当前批次的隐藏查询
        batch_query_index_map: torch.Tensor, # [total_queries] 查询索引映射
        batch_query_weight_map: torch.Tensor, # [total_queries] 查询权重映射
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        group_indices_mask = torch.isin(batch_query_index_map, valid_indices)
        group_indices = torch.where(group_indices_mask)[0]  # 所有属于这些组的原始查询点
        # 创建临时存储张量
        # attn_features = torch.zeros_like(batch_hidden_queries)  # [total_queries, C]
        # weight_mask = torch.zeros_like(attn_features[..., 0])  # 权重累加掩码
        if len(valid_indices) > 0 and len(group_indices) > 0:                
            # 步骤1: 创建映射张量
            max_idx = int(torch.max(valid_indices).item()) + 1
            idx_map = torch.full((max_idx,), -1, dtype=torch.long, device=batch_hidden_queries.device)
            idx_map[valid_indices] = torch.arange(len(valid_indices), device=batch_hidden_queries.device)
            
            # 步骤2: 获取子查询点对应的主查询点位置
            master_indices = batch_query_index_map[group_indices]  # 子查询点对应的主查询点索引
            valid_mask = (master_indices < max_idx) & (master_indices >= 0)
            
            if valid_mask.any():
                # 筛选有效的子查询点
                valid_sub_indices = group_indices[valid_mask]
                valid_master_indices = master_indices[valid_mask]
                
                # 获取主查询点在attn_hidden_queries中的位置
                master_positions = idx_map[valid_master_indices]
                
                # 确保索引有效
                valid_pos_mask = (master_positions >= 0)
                
                if valid_pos_mask.any():
                    # 应用最终筛选
                    final_sub_indices = valid_sub_indices[valid_pos_mask]
                    final_master_pos = master_positions[valid_pos_mask]
                    
                    # 获取主查询点特征
                    master_feats = attn_hidden_queries[final_master_pos]
                    
                    # 获取子查询点权重
                    weights = batch_query_weight_map[final_sub_indices]
                    
                    # 计算加权特征
                    weighted_feats = master_feats * weights.unsqueeze(-1)

                    unit_attn_query_features = weighted_feats
                    unit_window_query_features = batch_hidden_queries[group_indices]

        return unit_attn_query_features, unit_window_query_features, group_indices




    def create_sliding_sample_groups(self,
                                    sample_window: WindowSparseConvTensor,
                                    main_window: WindowSparseConvTensor,
                                    window_size: int,
                                    stride: int,
                                    mode: str = 'shared_main',
                                    delta_distance: int = 1):
        """
        创建基于实际有效窗口数的滑动窗口式子集采样生成器
        """
        sample_window_num = sample_window.window_num
        main_window_num = main_window.window_num
        batch_size = sample_window.origin_bs
        
        for b in range(batch_size):
            valid_sample_indices = [i for i in range(sample_window_num) if sample_window.valid_mask[b][i]]
            valid_sample_window_num = len(valid_sample_indices)
            valid_main_indices = [i for i in range(main_window_num) if main_window.valid_mask[b][i]]
            valid_main_window_num = len(valid_main_indices)
            assert valid_sample_window_num > 0, '理论有效网格不应为0个'

            
            # 滑动窗口遍历 - 只考虑有效窗口
            for start in range(0, valid_sample_window_num - window_size + 1, stride):
                end_idx = start + window_size if start + window_size <= valid_sample_window_num else valid_sample_window_num
                sample_indices = valid_sample_indices[start:end_idx]  # 当前组的有效窗口索引

                # ==================== 处理采样窗口组 ====================
                sample_point_ranges = [sample_window.window_points_local_range[b][idx] for idx in sample_indices]
                sample_point_indices = [i for start, end in sample_point_ranges for i in range(start, end)]
                sample_data = (
                    sample_indices,
                    sample_point_indices, # List ?
                    b,
                )
                    
                # ==================== 处理主窗口 ====================
                if mode == 'shared_main':
                    # 模式1: 共享完整main_x
                    main_data = (None, valid_main_indices)
                elif mode == 'local_main':
                    # 模式2: 局部化main_x子集
                    # 使用采样窗口组的中间索引
                    center_idx = sample_indices[len(sample_indices) // 2]
                    # 找到最接近的有效主窗口索引
                    closest_idx = min(valid_main_indices, key=lambda x: abs(x - center_idx))
                    center_index_in_main = valid_main_indices.index(closest_idx)
                    # 计算局部范围
                    local_start = max(0, center_index_in_main - delta_distance)
                    local_end = min(main_window_num, center_index_in_main + delta_distance + 1)
                    # 获取局部范围内的有效窗口索引
                    main_indices = valid_main_indices[local_start:local_end]
                    main_point_ranges = [main_window.window_points_local_range[b][idx] for idx in main_indices]

                    main_point_indices = [i for start, end in main_point_ranges for i in range(start, end)]
                    main_data = (
                        main_indices,
                        main_point_indices,
                        b,
                    )

                yield main_data, sample_data



    # def forward_single(self,
    #                 main_x: spconv.SparseConvTensor,
    #                 sample_x: spconv.SparseConvTensor,
    #                 main_indices: torch.Tensor = None,
    #                 sample_indices: torch.Tensor = None,
    #                 valid_mask: torch.Tensor = None,
    #                 query_index_map: torch.Tensor = None,
    #                 query_weight_map: torch.Tensor = None,
    #                 origin_bs: int = None,
    #                 hidden_queries: torch.Tensor = None,
    #                 pos_embed: nn.Module = None,
    #                 s_j_residual: bool = True,
    #                 current_batch: int=None,
    #                 info_current=None,
    #                 debug=False):
        
    #     # debug = False
    #     ori_main_coords = main_x.indices.clone()
    #     ori_sample_coords = sample_x.indices.clone()
    #     # ori_main_coords[:, 0] = ori_main_coords[:, 0] % origin_bs
    #     # ori_sample_coords[:, 0] = ori_sample_coords[:, 0] % origin_bs
    #     win_start_idx, win_end_idx = sample_indices[0], sample_indices[-1]
        
    #     mamba_layer_i = self.mamba_encoder_list[0]
    #     mamba_layer_j = self.mamba_encoder_list[1]
        

    #     coords_main = main_x.indices
    #     assert coords_main[:, 0].max() < origin_bs and coords_main[:, 0].min() >= 0
    #     assert coords_main[:, 1].max() < self.sparse_shape[0] and coords_main[:, 1].min() >= 0
    #     assert coords_main[:, 2].max() < self.sparse_shape[1] and coords_main[:, 2].min() >= 0
    #     assert coords_main[:, 3].max() < self.sparse_shape[2] and coords_main[:, 3].min() >= 0
    #     coords_sample = sample_x.indices
    #     assert coords_sample[:, 0].max() < origin_bs and coords_sample[:, 0].min() >= 0
    #     assert coords_sample[:, 1].max() < self.sparse_shape[0] and coords_sample[:, 1].min() >= 0
    #     assert coords_sample[:, 2].max() < self.sparse_shape[1] and coords_sample[:, 2].min() >= 0
    #     assert coords_sample[:, 3].max() < self.sparse_shape[2] and coords_sample[:, 3].min() >= 0
    #     main_x = self.conv_i(main_x)
    #     sample_x = self.conv_j(sample_x)

    #     main_feats = main_x.features
    #     sample_feats = sample_x.features
    #     main_coords = main_x.indices
    #     sample_coords = sample_x.indices

    #     info_current = self._record_info(self.memory_log_path, "qssm.2.2.1", info_current, execute=debug)

    #     assert sum(ori_main_coords[:, 0] == current_batch) == ori_main_coords.shape[0]
    #     assert sum(ori_sample_coords[:, 0] == current_batch) == ori_sample_coords.shape[0]


    #     updated_hidden_queries = hidden_queries.clone()
        
    #     feats_i = main_feats
    #     feats_j = sample_feats
    #     feats_i_m = main_feats[None] # [1, N, C]
    #     feats_j_m = sample_feats[None]

    #     # if feats_i_m.size(1) < self.d_conv:
    #     #     print(f"Warning: main window sequence length {feats_i_m.size(1)} is less than minimum required {self.d_conv}. Skipping Mamba processing.")
    #     #     out_feats_i_m = feats_i_m
    #     # else:
    #     out_feats_i_m = mamba_layer_i(feats_i_m, None)  # ([1, N, C], _)
    #     out_feats_i = out_feats_i_m[0].squeeze(0) # [N, C]

    #     # if feats_j_m.size(1) < self.d_conv:
    #     #     print(f"Warning: main window sequence length {feats_j_m.size(1)} is less than minimum required {self.d_conv}. Skipping Mamba processing.")
    #     #     out_feats_j_m = feats_j_m
    #     # else:
    #     out_feats_j_m = mamba_layer_j(feats_j_m, None)
    #     out_feats_j = out_feats_j_m[0].squeeze(0)  # [N, C]
        
    #     out_feats_i = self.ssm_norm_i(out_feats_i)
    #     out_feats_j = self.ssm_norm_j(out_feats_j)

    #     info_current = self._record_info(self.memory_log_path, "qssm.2.2.2", info_current, execute=debug)

    #     # out_feats_j_attn = self.q_ds_1(out_feats_j, reduction_factor=3)  # [L, C]
    #     # out_feats_i_attn = self.kv_ds_1(out_feats_i, reduction_factor=3)
    #     # out_feats_j_attn = self.q_ds_1(out_feats_j, k=3)  # [L, C]
    #     # out_feats_i_attn = self.kv_ds_1(out_feats_i, k=3)  # [L, C]

    #     # fused_feats_j = self.cross_attention(Q=out_feats_j.unsqueeze(0), K=out_feats_i.unsqueeze(0), V=out_feats_i.unsqueeze(0))

    #     # fused_feats_j = self.q_us_1(fused_feats_j_attn[0], scale_factor=3, original_length=feats_j.shape[0])  # [L, C]
    #     # fused_feats_j = self.q_us_1(fused_feats_j_attn[0], k=3, original_length=feats_j.shape[0])  # [L, C]

    #     # fused_feats_j = self.res_norm_attn(fused_feats_j + feats_j)

    #     fused_feats_j = out_feats_j + out_feats_i + feats_j
    #     if debug:
    #         info_current['attn_info'] = [out_feats_j.shape[0], out_feats_i.shape[0], out_feats_i.shape[0]]
    #     info_current = self._record_info(self.memory_log_path, "qssm.2.2.3", info_current, execute=debug)


    #     window_indices = torch.arange(win_start_idx, win_end_idx, device=hidden_queries.device)
    #     sub_valid_mask = valid_mask[current_batch][window_indices]
    #     valid_indices = window_indices[sub_valid_mask]
    #     query_seq = hidden_queries[current_batch, valid_indices]

    #     # fused_feats_j_attn = self.kv_ds_2(fused_feats_j, reduction_factor=3)  # [L, C]        
    #     # fused_feats_j_attn = self.kv_ds_2(fused_feats_j, k=3)
    #     # attn_hidden_queries = self.query_attn_module(
    #     #     Q=query_seq.unsqueeze(0),
    #     #     K=fused_feats_j,
    #     #     V=fused_feats_j,
    #     # ) # [1, L, C]    
    #     # attn_hidden_queries = attn_hidden_queries[0]
    #     attn_hidden_queries = query_seq + 0 * fused_feats_j.mean()
    #     # attn_hidden_queries = query_seq.clone()
    #     # updated_hidden_queries[current_batch][valid_indices] = attn_hidden_queries
    #     if debug:
    #         info_current['attn_info'] = [query_seq.shape[0], fused_feats_j.shape[1], fused_feats_j.shape[1]]
    #     info_current = self._record_info(self.memory_log_path, "qssm.2.2.4", info_current, execute=debug)


    #     # unit_attn_query_features, unit_window_query_features, group_indices = self.gather_and_broadcast(
    #     #     valid_indices=valid_indices,  # [K] 主查询点索引
    #     #     attn_hidden_queries=attn_hidden_queries,
    #     #     batch_hidden_queries=hidden_queries[current_batch],  # 当前批次的隐藏查询
    #     #     batch_query_index_map=query_index_map[current_batch],
    #     #     batch_query_weight_map=query_weight_map[current_batch],
    #     # )

    #     group_indices = valid_indices
    #     unit_attn_query_features = attn_hidden_queries
    #     unit_window_query_features = query_seq

    #     info_current = self._record_info(self.memory_log_path, "qssm.2.2.5", info_current, execute=debug)


    #     feedbacked_hidden_queries = self.feedback(
    #         unit_attn_query_features,  # [window_length, C]
    #         unit_window_query_features,  # [window_length, C]
    #     )

    #     info_current = self._record_info(self.memory_log_path, "qssm.2.2.6", info_current, execute=debug)


    #     updated_hidden_queries[current_batch][group_indices] = feedbacked_hidden_queries

    #     # enhanced_main_sp = spconv.SparseConvTensor(
    #     #     features=main_feats + out_feats_i,
    #     #     indices=main_coords,
    #     #     spatial_shape=main_x.spatial_shape,
    #     #     batch_size=main_x.batch_size
    #     # )
    #     # out_feats_3d_sj_sp = spconv.SparseConvTensor(
    #     #     features=out_feats_j,
    #     #     indices=sample_coords,
    #     #     spatial_shape=sample_x.spatial_shape,
    #     #     batch_size=sample_x.batch_size
    #     # )

    #     main_x = main_x.replace_feature(main_feats + out_feats_i + fused_feats_j)

    #     info_current = self._record_info(self.memory_log_path, "qssm.2.2.7", info_current, execute=debug)
        
    #     # F_q_sp = sparse_add(enhanced_main_sp, out_feats_3d_sj_sp)

    #     info_current = self._record_info(self.memory_log_path, "qssm.2.2.8", info_current, execute=debug)

    #     # return F_q, updated_hidden_queries
    #     # return F_q_sp, updated_hidden_queries, info_current
    #     return main_x, updated_hidden_queries, info_current


    
    
    # def forward(
    #     self,
    #     window_sample_x: WindowSparseConvTensor, 
    #     enhanced_main_x: WindowSparseConvTensor, 
    #     window_sample_pos_global: torch.Tensor,  # [N_total, 4]
    #     hidden_queries,  # [bs, total_queries, dim]
    #     pos_embed,
    #     s_j_residual=True,
    #     debug=False
    # ) -> Union[WindowSparseConvTensor, torch.Tensor]:
        
    #     # debug = False
    #     info_current = self._record_info(self.memory_log_path, "Enter qssm", execute=debug)

    #     output_features = enhanced_main_x.unify_sparse_tensor.features.clone()
    #     output_indices = enhanced_main_x.unify_sparse_tensor.indices.clone()
    #     enhanced_hidden_queries = hidden_queries.clone()
    #     device = output_features.device
    #     batch_size = enhanced_main_x.origin_bs

    #     agg_buffer = {
    #         'indices': output_indices.clone(),
    #         'features': output_features.clone(),
    #         'counts': torch.ones(output_features.shape[0], device=device, dtype=torch.long)
    #     }

    #     with torch.no_grad():

    #         sample_valid_count = torch.any(window_sample_x.valid_mask, dim=0).sum().item()
    #         main_valid_count = torch.any(enhanced_main_x.valid_mask, dim=0).sum().item()
    #         window_size = int(max(math.ceil(sample_valid_count * 0.10), 1)) # 1/10
    #         delta_distance = int(max(math.ceil(main_valid_count * 0.25), 1)) # 1/4
    #         # NOTE: print info
    #         # print(f'有效采样窗口: {sample_valid_count}/{window_sample_x.window_num}\n'
    #         # f'有效主窗口: {main_valid_count}/{enhanced_main_x.window_num}\n'
    #         # f'window_size={window_size}, delta_distance={delta_distance}\n')

    #         ori_main_coords = enhanced_main_x.sparse_tensor.indices.clone()
    #         ori_main_coords[:, 0] = ori_main_coords[:, 0] % enhanced_main_x.origin_bs
    #         ori_sample_coords = window_sample_pos_global.clone()
    #         ori_sample_coords[:, 1:] = torch.round(ori_sample_coords[:, 1:]) # TODO: 06.29 采用四舍五入法将采样坐标归纳到就近整数体素坐标, 需进一步研究可行性
    #         # window_sample_x.sparse_tensor.indices = ori_sample_coords #### NOTE: !!!
    #         ori_sample_coords[:, 0] = ori_sample_coords[:, 0] % window_sample_x.origin_bs

    #         sliding_groups = self.create_sliding_sample_groups(
    #             sample_window=window_sample_x,
    #             main_window=enhanced_main_x,
    #             window_size=window_size,
    #             delta_distance=delta_distance,
    #             stride=window_size, # 不重叠的滑动
    #             mode=self.mode,
    #         )
            
    #         info_current = self._record_info(self.memory_log_path, "qssm.0", info_current, execute=debug)
            

        
    #     clvl_cruve_template_ij = self.curve_template['curve_template_rank9']
    #     clvl_hilbert_spatial_size_ij = self.hilbert_spatial_size['curve_template_rank9']

    #     main_points_hilbert_indices = get_hilbert_index_3d_mamba_lite(
    #         clvl_cruve_template_ij, 
    #         ori_main_coords, 
    #         batch_size=enhanced_main_x.origin_bs,
    #         z_dim=self.sparse_shape[0],
    #         hilbert_spatial_size=clvl_hilbert_spatial_size_ij,
    #         shift=(0, 0, 0),
    #     )
        
    #     # sample_points_hilbert_indices = get_hilbert_index_3d_mamba_lite(
    #     #     clvl_cruve_template_ij,
    #     #     ori_sample_coords,
    #     #     batch_size=window_sample_x.origin_bs,
    #     #     z_dim=self.sparse_shape[0],
    #     #     hilbert_spatial_size=clvl_hilbert_spatial_size_ij,
    #     #     shift=(0, 0, 0),
    #     # )
    #     sample_points_hilbert_indices = main_points_hilbert_indices

    #     window_main_coords_m = {}
    #     window_main_feats_m = {}
    #     window_sample_coords_m = {}
    #     window_sample_feats_m = {}

    #     info_current = self._record_info(self.memory_log_path, "qssm.1", info_current, execute=debug)

    #     for b in range(batch_size):
    #         batch_mask = ori_main_coords[:, 0] == b
    #         ## TODO: is it right ? 
    #         window_main_coords_m[b] = ori_main_coords[batch_mask][main_points_hilbert_indices['inds_curt_to_next'][b]]
    #         window_main_feats_m[b] = enhanced_main_x.sparse_tensor.features[batch_mask][main_points_hilbert_indices['inds_curt_to_next'][b]]

    #         batch_mask = ori_sample_coords[:, 0] == b
    #         ### NOTE: !!!
    #         # window_sample_coords_m[batch_mask] = ori_sample_coords[batch_mask][sample_points_hilbert_indices['inds_curt_to_next'][b]]
    #         # window_sample_coords_m[batch_mask] = window_main_coords_m[batch_mask][sample_points_hilbert_indices['inds_curt_to_next'][b]]
    #         window_sample_feats_m[b] = window_sample_x.sparse_tensor.features[batch_mask][sample_points_hilbert_indices['inds_curt_to_next'][b]]
    #         window_sample_coords_m[b] = window_main_coords_m[b]
        
    #     info_current = self._record_info(self.memory_log_path, "qssm.2", info_current, execute=debug)
                    

    #     for group_idx, (main_data, sample_data) in enumerate(sliding_groups): # 48 groups(6, 15)

    #         info_current = self._record_info(self.memory_log_path, "qssm.2.1", info_current, execute=debug)

    #         sample_group_indices, sample_point_indices, current_batch = sample_data
    #         sample_point_indices_m = sample_points_hilbert_indices['inds_next_to_curt'][current_batch][sample_point_indices]
    #         sample_feats_m = window_sample_feats_m[current_batch][sample_point_indices_m]
    #         sample_coords_m = window_sample_coords_m[current_batch][sample_point_indices_m]

    #         sample_x = spconv.SparseConvTensor(
    #             features=sample_feats_m,
    #             indices=sample_coords_m.int(),
    #             spatial_shape=window_sample_x.sparse_tensor.spatial_shape,
    #             batch_size=window_sample_x.origin_bs
    #         )

    #         if self.mode == 'shared_main':
    #             main_x = enhanced_main_x.sparse_tensor
    #             main_indices = torch.arange(enhanced_main_x.total_points, device=device)
    #         else:  # local_main
    #             main_coords = enhanced_main_x.sparse_tensor.indices
    #             main_group_indices, main_point_indices, current_batch = main_data
    #             main_point_indices_m = main_points_hilbert_indices['inds_next_to_curt'][current_batch][main_point_indices]
    #             main_feats_m = window_main_feats_m[current_batch][main_point_indices_m]
    #             main_coords_m = window_main_coords_m[current_batch][main_point_indices_m]
    #             main_x = spconv.SparseConvTensor(
    #                 features=main_feats_m,
    #                 indices=main_coords_m.int(),
    #                 spatial_shape=enhanced_main_x.sparse_tensor.spatial_shape,
    #                 batch_size=enhanced_main_x.origin_bs
    #             )

    #         info_current = self._record_info(self.memory_log_path, "qssm.2.2", info_current, execute=debug)

    #         F_q_sp, enhanced_hidden_queries, info_current = self.forward_single(
    #             main_x=main_x,
    #             sample_x=sample_x,
    #             main_indices=main_group_indices,
    #             sample_indices=sample_group_indices,
    #             # main_index_info=main_index_info,
    #             # sample_index_info=sample_index_info,
    #             valid_mask=window_sample_x.valid_mask,
    #             query_index_map=window_sample_x.query_index_map,
    #             query_weight_map=window_sample_x.query_weight_map,
    #             origin_bs=window_sample_x.origin_bs,
    #             hidden_queries=enhanced_hidden_queries,
    #             pos_embed=pos_embed,
    #             s_j_residual=s_j_residual,
    #             current_batch=current_batch,
    #             info_current=info_current,
    #             debug=debug
    #         )

    #         update_indices = F_q_sp.indices
    #         update_features = F_q_sp.features
            
    #         # ================ 增量聚合 ================
    #         with torch.no_grad():
    #             # 1. 对当前组内特征进行预聚合（避免组内重复点）
    #             current_unique_indices, inverse, counts = torch.unique(
    #                 update_indices,
    #                 return_inverse=True,
    #                 return_counts=True,
    #                 dim=0
    #             )
                
    #             # 计算组内平均特征
    #             current_unique_features = torch.zeros(
    #                 len(current_unique_indices), 
    #                 update_features.size(-1),
    #                 dtype=update_features.dtype,
    #                 device=device
    #             )
    #             current_unique_features.scatter_add_(0, inverse.view(-1, 1).expand(-1, update_features.size(-1)), update_features)
    #             current_unique_features /= counts.float().unsqueeze(-1)
                
    #             # 2. 合并到全局缓冲区
    #             combined_indices = torch.cat([
    #                 agg_buffer['indices'], 
    #                 current_unique_indices
    #             ])
                
    #             combined_features = torch.cat([
    #                 agg_buffer['features'], 
    #                 current_unique_features
    #             ])
                
    #             combined_counts = torch.cat([
    #                 agg_buffer['counts'], 
    #                 counts
    #             ])
                
    #             # 3. 全局聚合（高效处理重复点）
    #             agg_unique_indices, inverse, agg_counts = torch.unique(
    #                 combined_indices,
    #                 return_inverse=True,
    #                 return_counts=True,
    #                 dim=0
    #             )
                
    #             # 计算全局平均特征
    #             new_features = torch.zeros(
    #                 len(agg_unique_indices), 
    #                 combined_features.size(-1),
    #                 dtype=combined_features.dtype,
    #                 device=device
    #             )
    #             new_features.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, combined_features.size(-1)), combined_features)
    #             new_features /= agg_counts.float().unsqueeze(-1)
                
    #             # 4. 更新缓冲区
    #             agg_buffer['indices'] = agg_unique_indices
    #             agg_buffer['features'] = new_features
    #             agg_buffer['counts'] = agg_counts
            
    #         info_current = self._record_info(self.memory_log_path, "qssm.2.3", info_current, execute=debug)


    #     info_current = self._record_info(self.memory_log_path, "qssm.3", info_current, execute=debug)

    #     # ================ 直接使用聚合结果 ================
    #     output_features = agg_buffer['features'].clone()
        
    #     window_F_q = enhanced_main_x.clone()
    #     window_F_q.unify_sparse_tensor.replace_feature(output_features)
        
    #     info_current = self._record_info(self.memory_log_path, "qssm.4", info_current, execute=debug)
    #     return window_F_q, enhanced_hidden_queries




    def forward_single(self,
                    main_x: spconv.SparseConvTensor,
                    sample_x: spconv.SparseConvTensor,
                    main_indices: torch.Tensor = None,
                    sample_indices: torch.Tensor = None,
                    valid_mask: torch.Tensor = None,
                    query_index_map: torch.Tensor = None,
                    query_weight_map: torch.Tensor = None,
                    origin_bs: int = None,
                    hidden_queries: torch.Tensor = None,
                    pos_embed: nn.Module = None,
                    s_j_residual: bool = True,
                    current_batch: int=None,
                    info_current=None,
                    debug=False):
        
        # debug = False
        ori_main_coords = main_x.indices.clone()
        ori_sample_coords = sample_x.indices.clone()
        # ori_main_coords[:, 0] = ori_main_coords[:, 0] % origin_bs
        # ori_sample_coords[:, 0] = ori_sample_coords[:, 0] % origin_bs
        win_start_idx, win_end_idx = sample_indices[0], sample_indices[-1]
        
        mamba_layer_i = self.mamba_encoder_list[0]
        mamba_layer_j = self.mamba_encoder_list[1]

        # coords_main = main_x.indices
        # assert coords_main[:, 0].max() < origin_bs and coords_main[:, 0].min() >= 0
        # assert coords_main[:, 1].max() < self.sparse_shape[0] and coords_main[:, 1].min() >= 0
        # assert coords_main[:, 2].max() < self.sparse_shape[1] and coords_main[:, 2].min() >= 0
        # assert coords_main[:, 3].max() < self.sparse_shape[2] and coords_main[:, 3].min() >= 0
        # coords_sample = sample_x.indices
        # assert coords_sample[:, 0].max() < origin_bs and coords_sample[:, 0].min() >= 0
        # assert coords_sample[:, 1].max() < self.sparse_shape[0] and coords_sample[:, 1].min() >= 0
        # assert coords_sample[:, 2].max() < self.sparse_shape[1] and coords_sample[:, 2].min() >= 0
        # assert coords_sample[:, 3].max() < self.sparse_shape[2] and coords_sample[:, 3].min() >= 0
        
        main_x = self.conv_i(main_x)
        sample_x = self.conv_j(sample_x)

        main_feats = main_x.features
        sample_feats = sample_x.features
        main_coords = main_x.indices
        sample_coords = sample_x.indices

        info_current = self._record_info(self.memory_log_path, "qssm.2.2.1", info_current, execute=debug)

        assert sum(ori_main_coords[:, 0] == current_batch) == ori_main_coords.shape[0]
        assert sum(ori_sample_coords[:, 0] == current_batch) == ori_sample_coords.shape[0]


        updated_hidden_queries = hidden_queries.clone()
        
        feats_i = main_feats
        feats_j = sample_feats
        feats_i_m = main_feats[None] # [1, N, C]
        feats_j_m = sample_feats[None]

        # if feats_i_m.size(1) < self.d_conv:
        #     print(f"Warning: main window sequence length {feats_i_m.size(1)} is less than minimum required {self.d_conv}. Skipping Mamba processing.")
        #     out_feats_i_m = feats_i_m
        # else:
        out_feats_i_m = mamba_layer_i(feats_i_m, None)  # ([1, N, C], _)
        out_feats_i = out_feats_i_m[0].squeeze(0) # [N, C]

        # if feats_j_m.size(1) < self.d_conv:
        #     print(f"Warning: main window sequence length {feats_j_m.size(1)} is less than minimum required {self.d_conv}. Skipping Mamba processing.")
        #     out_feats_j_m = feats_j_m
        # else:
        out_feats_j_m = mamba_layer_j(feats_j_m, None)
        out_feats_j = out_feats_j_m[0].squeeze(0)  # [N, C]
        
        out_feats_i = self.ssm_norm_i(out_feats_i)
        out_feats_j = self.ssm_norm_j(out_feats_j)

        info_current = self._record_info(self.memory_log_path, "qssm.2.2.2", info_current, execute=debug)

        fused_feats_j = self.cross_attention(Q=out_feats_j.unsqueeze(0), K=out_feats_i.unsqueeze(0), V=out_feats_i.unsqueeze(0))
        # fused_feats_j = out_feats_j + out_feats_i + feats_j

        fused_feats_j = self.res_norm_attn(fused_feats_j + feats_j) # NOTE 07.03: 加回该行

        if debug:
            info_current['attn_info'] = [out_feats_j.shape[0], out_feats_i.shape[0], out_feats_i.shape[0]]
        info_current = self._record_info(self.memory_log_path, "qssm.2.2.3", info_current, execute=debug)

        batch_valid_mask = valid_mask[current_batch]
        query_seq = hidden_queries[current_batch, batch_valid_mask]

        attn_hidden_queries = self.query_attn_module(
            Q=query_seq.unsqueeze(0),
            K=fused_feats_j,
            V=fused_feats_j,
        ) # [1, L, C]    
        attn_hidden_queries = attn_hidden_queries[0]
        # attn_hidden_queries = query_seq + 0 * fused_feats_j.mean()
        # attn_hidden_queries = query_seq.clone()
        # updated_hidden_queries[current_batch][valid_indices] = attn_hidden_queries
        if debug:
            info_current['attn_info'] = [query_seq.shape[0], fused_feats_j.shape[1], fused_feats_j.shape[1]]

        unit_attn_query_features = attn_hidden_queries
        unit_window_query_features = query_seq

        info_current = self._record_info(self.memory_log_path, "qssm.2.2.4", info_current, execute=debug)

        feedbacked_hidden_queries = self.feedback(
            unit_attn_query_features,  # [window_length, C]
            unit_window_query_features,  # [window_length, C]
        )

        info_current = self._record_info(self.memory_log_path, "qssm.2.2.5", info_current, execute=debug)

        # updated_hidden_queries[current_batch, batch_valid_mask] = feedbacked_hidden_queries
        group_indices = batch_valid_mask.nonzero(as_tuple=False)[:, 0]
        indices = [
            torch.full_like(group_indices, current_batch),
            group_indices                                
        ]
        updated_hidden_queries.index_put_(
            indices, 
            feedbacked_hidden_queries, 
            accumulate=False
        )
        replace_feature(main_x, main_feats + out_feats_i + fused_feats_j.squeeze(0))

        info_current = self._record_info(self.memory_log_path, "qssm.2.2.6", info_current, execute=debug)

        # return F_q_sp, updated_hidden_queries, info_current
        return main_x, updated_hidden_queries, info_current


  
    
    
    def forward(
        self,
        window_sample_x: WindowSparseConvTensor, 
        enhanced_main_x: WindowSparseConvTensor, 
        window_sample_pos_global: torch.Tensor,  # [N_total, 4]
        hidden_queries,  # [bs, total_queries, dim]
        pos_embed,
        s_j_residual=True,
        debug=False
    ) -> Union[WindowSparseConvTensor, torch.Tensor]:
        
        # debug = False
        info_current = self._record_info(self.memory_log_path, "Enter qssm", execute=debug)

        enhanced_hidden_queries = hidden_queries.clone()
        batch_size = enhanced_main_x.origin_bs

        with torch.no_grad():
            ori_main_coords = enhanced_main_x.sparse_tensor.indices.clone()
            ori_main_coords[:, 0] = ori_main_coords[:, 0] % enhanced_main_x.origin_bs
            ori_sample_coords = ori_main_coords.clone()
            # ori_sample_coords = window_sample_pos_global.clone()
            # ori_sample_coords[:, 1:] = torch.round(ori_sample_coords[:, 1:]) # TODO: 06.29 采用四舍五入法将采样坐标归纳到就近整数体素坐标, 需进一步研究可行性
            # ori_sample_coords[:, 0] = ori_sample_coords[:, 0] % window_sample_x.origin_bs
            info_current = self._record_info(self.memory_log_path, "qssm.0", info_current, execute=debug)

            # clvl_cruve_template_ij = self.curve_template['curve_template_rank9']
            # clvl_hilbert_spatial_size_ij = self.hilbert_spatial_size['curve_template_rank9']

            # main_points_hilbert_indices = get_hilbert_index_3d_mamba_lite(
            #     clvl_cruve_template_ij, 
            #     ori_main_coords, 
            #     batch_size=enhanced_main_x.origin_bs,
            #     z_dim=self.sparse_shape[0],
            #     hilbert_spatial_size=clvl_hilbert_spatial_size_ij,
            #     shift=(0, 0, 0),
            # )
            
            # main_points_hilbert_indices = get_morton_index_3d(
            #     coors=ori_main_coords, 
            #     batch_size=enhanced_main_x.origin_bs,
            #     spatial_size=self.sparse_shape, 
            #     primary_axis='x'
            # )

            main_points_hilbert_indices = get_morton_index_3d(
                ori_main_coords, 
                enhanced_main_x.origin_bs,
                self.sparse_shape, 
                primary_axis='x'
            )

            # sample_points_hilbert_indices = get_hilbert_index_3d_mamba_lite(
            #     clvl_cruve_template_ij,
            #     ori_sample_coords,
            #     batch_size=window_sample_x.origin_bs,
            #     z_dim=self.sparse_shape[0],
            #     hilbert_spatial_size=clvl_hilbert_spatial_size_ij,
            #     shift=(0, 0, 0),
            # )

            sample_points_hilbert_indices = main_points_hilbert_indices

        window_main_coords_m = {}
        window_main_feats_m = {}
        window_sample_coords_m = {}
        window_sample_feats_m = {}

        info_current = self._record_info(self.memory_log_path, "qssm.1", info_current, execute=debug)
        
        for b in range(batch_size):
            batch_mask = ori_main_coords[:, 0] == b
            window_main_coords_m[b] = ori_main_coords[batch_mask][main_points_hilbert_indices['inds_curt_to_next'][b]]
            window_main_feats_m[b] = enhanced_main_x.sparse_tensor.features[batch_mask][main_points_hilbert_indices['inds_curt_to_next'][b]]

            batch_mask = ori_sample_coords[:, 0] == b
            window_sample_coords_m[b] = window_main_coords_m[b]
            window_sample_feats_m[b] = window_sample_x.sparse_tensor.features[batch_mask][sample_points_hilbert_indices['inds_curt_to_next'][b]]
        
        info_current = self._record_info(self.memory_log_path, "qssm.2", info_current, execute=debug)

        out_features = enhanced_main_x.sparse_tensor.features # 07.03: 原地操作代替克隆防止显存问题
        # out_features = enhanced_main_x.sparse_tensor.features.clone()
                    
        for current_batch in range(batch_size):

            with torch.no_grad(): # 10~12ms -> 8~10ms
                batch_mask = ori_main_coords[:, 0] == current_batch
                
                sample_point_indices_m = sample_points_hilbert_indices['inds_next_to_curt'][current_batch]
                sample_feats_m = window_sample_feats_m[current_batch][sample_point_indices_m]
                sample_coords_m = window_sample_coords_m[current_batch][sample_point_indices_m]

                main_point_indices_m = main_points_hilbert_indices['inds_next_to_curt'][current_batch]
                main_coords_m = window_main_coords_m[current_batch][main_point_indices_m]
            main_feats_m = window_main_feats_m[current_batch][main_point_indices_m]

            sample_x = spconv.SparseConvTensor(
                features=sample_feats_m,
                indices=sample_coords_m.int(),
                spatial_shape=window_sample_x.sparse_tensor.spatial_shape,
                batch_size=window_sample_x.origin_bs
            )
            main_x = spconv.SparseConvTensor(
                features=main_feats_m,
                indices=main_coords_m.int(),
                spatial_shape=enhanced_main_x.sparse_tensor.spatial_shape,
                batch_size=enhanced_main_x.origin_bs
            )

            F_q_sp, enhanced_hidden_queries, info_current = self.forward_single(
                main_x=main_x,
                sample_x=sample_x,
                sample_indices=[0, enhanced_main_x.window_num - 1],
                valid_mask=window_sample_x.valid_mask,
                query_index_map=window_sample_x.query_index_map,
                query_weight_map=window_sample_x.query_weight_map,
                origin_bs=window_sample_x.origin_bs,
                hidden_queries=enhanced_hidden_queries,
                pos_embed=pos_embed,
                s_j_residual=s_j_residual,
                current_batch=current_batch,
                info_current=info_current,
                debug=debug
            )
            
            # out_features[batch_mask] = F_q_sp.features
            with torch.no_grad(): # 07.03: 直接明确需要的索引而不是每个索引都判断 1ms -> 0ms
                out_features[batch_mask.nonzero(as_tuple=True)] = F_q_sp.features
            
        info_current = self._record_info(self.memory_log_path, "qssm.3", info_current, execute=debug)

        window_F_q = enhanced_main_x # 既然后续没有enhanced_main_x的原地修改, 无需再克隆大型变量
        # window_F_q = enhanced_main_x.clone()

        # window_F_q.unify_sparse_tensor = window_F_q.unify_sparse_tensor.replace_feature(out_features)
        window_F_q.sparse_tensor = window_F_q.sparse_tensor.replace_feature(out_features)
        window_F_q.merge_all_window(merge_type='mean')
        
        info_current = self._record_info(self.memory_log_path, "qssm.4", info_current, execute=debug)
        return window_F_q, enhanced_hidden_queries



class MultiScaleModeling(nn.Module):
    def __init__(
            self,
            dim, 
            groups=4,
            residual=True, # 默认为 post-norm
            residual_in_fp32=True,
            ffn_out=True, # TODO
        ):
        super().__init__()
        assert groups == 4, 'Now just support for 4 groups channel split.'
        assert dim % groups == 0
        self.dim = dim
        self.groups = groups
        self.residual = residual 
        self.residual_in_fp32 = residual_in_fp32
        
        hidden_dim = dim // groups
        self.norm_out = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.layers = nn.ModuleList([
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    hidden_dim, hidden_dim, 
                    kernel_size=(1,3,3), stride=1, padding=(1,1,1), dilation=1,
                    bias=False, indice_key='subm_1'
                ),
                # nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    hidden_dim, hidden_dim, 
                    kernel_size=(1,5,5), stride=2, padding=(1,2,2), dilation=1,
                    bias=False, indice_key='subm_2'
                ),
                # nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    hidden_dim, hidden_dim,
                    kernel_size=(3,3,3), stride=1, padding=(1,1,1), dilation=1,
                    bias=False, indice_key='subm_3'
                ),
                # nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    hidden_dim, hidden_dim,
                    kernel_size=(3,3,3), stride=1, padding=(1,3,3), dilation=(1,3,3),
                    bias=False, indice_key='subm_4'
                ),
                # nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        ])

    def forward(self, x: spconv.SparseConvTensor):
        features = x.features  # [num_voxels, dim]
        indices = x.indices    # [num_voxels, 4] (batch_idx, z, y, x)
        spatial_shape = x.spatial_shape
        batch_size = x.batch_size
        ori_dtype = features.dtype

        if self.residual == True:
            res_feats = features.clone().to(torch.float32) if self.residual_in_fp32 == True else features.clone()

        group_size = self.dim // self.groups
        features_split = torch.split(features, group_size, dim=1)
        processed_features = []
        for i in range(self.groups):
            processed_features.append(
                self.layers[i](
                    spconv.SparseConvTensor(
                        features=features_split[i],
                        indices=indices,
                        spatial_shape=spatial_shape,
                        batch_size=batch_size
                    )
                ).features
            )
        out_feats = torch.cat(processed_features, dim=1).to(torch.float32) if self.residual_in_fp32 == True else torch.cat(processed_features, dim=1)
        
        if self.residual == True:
            out_feats = self.norm_out(out_feats + res_feats)
        else:
            out_feats = self.norm_out(out_feats)
        
        return spconv.SparseConvTensor(
            features=out_feats.to(ori_dtype),
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )


# NOTE 自建模模块中先统一进行多尺度建模（分轴、分通道）
class HighFreqMultiSourceExtractor(nn.Module):
    def __init__(
            self,
            dim,
            groups=4,
            topk_r=0.1,    
        ):
        super().__init__()
        self.dim = dim
        self.groups = groups
        self.topk_r = topk_r

        # For DEBUG # 注册缓冲区用于统计 topkr 长度
        # self.print_interval = 100
        # self.register_buffer("topkr_sum", torch.tensor(0.0, requires_grad=False))
        # self.register_buffer("topkr_count", torch.tensor(0, requires_grad=False))
        # self.register_buffer("topkr_avg", torch.tensor(0.0, requires_grad=False))
    
    # @torch.no_grad()
    # def _update_debug_statistics(self, current_topkr, debug=False):
    #     if not debug:
    #         return
    #     self.topkr_sum += current_topkr
    #     self.topkr_count += 1
    #     self.topkr_avg = self.topkr_sum / self.topkr_count
    #     if self.topkr_count % self.print_interval == 0: 
    #         print(f"Step {self.topkr_count}: actual topkr avg = {self.topkr_avg.item():.2f}%")
    
    def forward(self, feats, coords, debug=False):
        device = feats.device
        batch_size = coords[:, 0].max().int().item() + 1
        
        with torch.no_grad():
            # 存储所有批次的高频信息特征和索引掩码
            all_sample_feats = []
            all_sample_coords = []
            all_sample_masks = []
            
            # TODO 07.16 for bs problem: 此处是否也会引发 bs 问题? 应该不会导致.
            for b in range(batch_size):
                # 获取当前批次的点
                batch_mask = coords[:, 0] == b
                batch_feats = feats[batch_mask]  # [N_b, C]
                batch_coords = coords[batch_mask] # [N_b, 4]
                # batch_indices = torch.where(batch_mask)[0]  # 当前批次在全局序列中的索引
                
                # 将特征沿通道维度分为 groups 组
                group_feats = batch_feats.chunk(self.groups, dim=1)  # groups * [N_b, C//groups]
                
                # 存储每组的高频信息索引
                group_indices = []
                
                for group_idx in range(self.groups):
                    # 计算当前组的平均特征值作为显著性分数
                    group_scores = group_feats[group_idx].mean(dim=1)  # [N_b]
                    
                    # 根据 topk_r 选择高频信息点
                    k = max(1, int(len(group_scores) * self.topk_r))
                    _, top_indices = torch.topk(group_scores, k)
                    
                    group_indices.append(top_indices)
                
                all_group_indices = torch.cat(group_indices) # 合并所有组的高频信息索引
                unique_indices = torch.unique(all_group_indices) # 获取唯一索引（去除重复）
                
                # 提取当前批次的高频信息特征
                sample_feats_b = batch_feats[unique_indices]  # [N'_b, C]
                sample_coords_b = batch_coords[unique_indices]  # [N'_b, 4]
                
                all_sample_feats.append(sample_feats_b)
                all_sample_coords.append(sample_coords_b)
                all_sample_masks.append(unique_indices)

                # # if debug: # TODO 可以在tb监控变化
                # #   print(f"Actual topkr = {100.0 * unique_indices.size(0) / batch_coords.size(0):.2f}% at {b} batch")
                # self._update_debug_statistics(100.0 * unique_indices.size(0) / batch_coords.size(0), debug=True)
            
            # 拼接所有批次的高频信息特征和索引掩码
            sample_feats = torch.cat(all_sample_feats, dim=0)  # [N', C]
            sample_coords = torch.cat(all_sample_coords, dim=0)  # [N', 4]
            # all_sample_masks = torch.cat(all_sample_masks, dim=0)  # [N']

            # if sample_feats.shape[0] > 30000:
                # print(f"Large size: {sample_feats.shape[0]}(origin: {feats.shape[0]}) with topkr = {100.0 * unique_indices.size(0) / batch_coords.size(0):.2f}%")
        
        return sample_feats, sample_coords, all_sample_masks


class QSSM_v5(nn.Module):
    def __init__(self,
                 dim,
                 num_stage,
                 num_block,
                 sparse_shape,
                 offset_layer_num,
                 total_queries,
                 query_attn_type,
                 mix_type, # NOTE [上分支, 下分支, 融合分支]
                 norm_epsilon, 
                 rms_norm,
                 topk_r,
                 space_filing_curve,
                 window_pos_embed: bool = False,
                 curve_template=None, 
                 hilbert_spatial_size=None,
                 multiscale_modeling=True,
                 groups=4, # 1 组相当于正常的 topk
                 gated_feedback=True,
                 hybrid_gate=False,
                 z_residual=None, # NOTE [上分支, 下分支, 融合分支]
                 skip_connect=None, # NOTE [上分支, 下分支]
                 fusion_simulate_z_res=False,
                 ssm_cfg=None,
                 attn_cfg=None,
                 residual_in_fp32=True,
                 query_in_fp32=True,
                 query_attn_dropout=0.1,
                 fused_add_norm=True,
                 extra_post_norm=False, # NOTE: 模块输出与残差连接融合后额外接入一个 Norm(相当于 post-norm) NOTE 且只有在 Fj 也经过 CrossAttn 更新才需要启用
                #  force_layernorm: bool,
                 device=None,
                 dtype=None,
                 **kwargs,
                ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype':dtype}

        self.dim = dim
        # self.num_query = num_query
        self.sparse_shape = sparse_shape
        self.device = device
        self.dtype = dtype
        self.extra_post_norm = extra_post_norm
        self.residual_in_fp32 = residual_in_fp32
        self.query_in_fp32 = query_in_fp32
        self.mix_type = mix_type
        self.gated_feedback = gated_feedback
        self.hybrid_gate = hybrid_gate
        self.total_queries = total_queries

        self.norm1d_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm2d_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.layernorm_fn = partial(nn.LayerNorm, eps=1e-5)
        
        self.multiscale_modeling = multiscale_modeling
        if multiscale_modeling == True:
            self.pre_modeling = MultiScaleModeling(dim, groups=groups)

        self.window_pos_embed = window_pos_embed
        self.space_filing_curve = space_filing_curve
        self.curve_template=curve_template
        self.hilbert_spatial_size=hilbert_spatial_size

        self.topk_r = topk_r
        # self.defsampler = DeformableSampler(
        #     dim=dim, 
        #     num_stage=num_stage, 
        #     num_block=num_block, 
        #     sparse_shape=sparse_shape,
        #     offset_layer_num=offset_layer_num,
        # )
        
        self.extractor = HighFreqMultiSourceExtractor(
            dim=dim,
            groups=groups,  # 分为4组
            topk_r=topk_r,
        )
        
        if z_residual is None:
            z_residual = [True, True, False]
        if skip_connect is None:
            skip_connect = [True, True, False]
            # skip_connect = [True, True]
        self.skip_connect = skip_connect

        self.fusion_simulate_z_res = (fusion_simulate_z_res and (z_residual[0] == False) and (z_residual[1] == False))       
        if self.fusion_simulate_z_res == True:
            self.fusion_post_act = nn.SiLU(inplace=True)

        if ssm_cfg is None:
            ssm_cfg = {}
        self.d_conv = 4
        mamba_encoder_i = create_block(
            d_model=dim,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=0,
            z_residual=z_residual[0],
            **factory_kwargs,
        )
        mamba_encoder_j = create_block(
            d_model=dim,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=1,
            z_residual=z_residual[1], # NOTE: 是否需要？
            # z_residual=True,
            **factory_kwargs,
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.Softplus(),
        )
        mamba_encoder_fusion = create_block(
            d_model=dim,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=2,
            z_residual=z_residual[2], # NOTE: 是否需要？
            # z_residual=True,
            **factory_kwargs,
        )
        self.mamba_encoder_list = nn.ModuleList([mamba_encoder_i, mamba_encoder_j, mamba_encoder_fusion])

        # self.skip_scale = nn.Parameter(torch.ones(1))
        # self.skip_gate = nn.Sigmoid()

        # NOTE: here we use LayerNorm for sequence following VoxelMamba
        self.ssm_norm_i = self.layernorm_fn(dim)
        self.ssm_norm_j = self.layernorm_fn(dim)
        self.ssm_norm_fusion = self.layernorm_fn(dim)

        # self.ssm_norm_out_j = self.layernorm_fn(dim)
        # self.norm_out = self.layernorm_fn(dim)
        self.ssm_norm_out_j = self.norm1d_fn(dim)
        self.norm_out = self.norm1d_fn(dim)
        assert not self.extra_post_norm
        # self.post_norm_out = self.layernorm_fn(dim) if self.extra_post_norm == True else nn.Identity()

        # For Pre-modeling
        # self.conv_i = Sparse1ConvBlock(dim, dim, norm_fn=self.layernorm_fn, indice_key="qssm_conv", activate='gelu')
        # self.conv_j = Sparse1ConvBlock(dim, dim, norm_fn=self.layernorm_fn, indice_key="qssm_conv", activate='gelu')

        # For Cross Mamba
        # self.curve_template = curve_template
        # self.hilbert_spatial_size = hilbert_spatial_size
        
        if self.total_queries > 0:
            self.query_attn_module = XFormersCrossAttention(
                dim, attn_type=query_attn_type, 
                proj_type='linear', # 'linear'(1~2ms) \ 'conv1d'(6~8ms) \ 
                res_norm_fn=None if gated_feedback == True else self.layernorm_fn,
                residual=False if gated_feedback == True else True, # NOTE: 有 feedback 充当残差设计
                residual_in_fp32=residual_in_fp32,
                dropout=query_attn_dropout if self.training else 0.0,
                attn_cfg=attn_cfg,
                keyname='query update attn'
            )
            
            # For Feedback
            if gated_feedback == True:
                if hybrid_gate == True:
                    self.attn_gate = nn.Linear(dim * 2, 1, dtype=torch.float32, device=self.device)
                else:
                    self.attn_gate = nn.Linear(dim, 1, dtype=torch.float32, device=self.device)
                self.attn_norm = self.norm1d_fn(dim) # nn.Identity()
                # self.attn_norm = self.layernorm_fn(dim) # nn.Identity()

        # For recording memory usage
        self.memory_log_path = "info_log_qssm.csv"
        if not os.path.exists(self.memory_log_path):
            with open(self.memory_log_path, "w", newline="") as f:
                f.write("Time,  Stage,  Memory-Delta(MB),  Memory-Delta(GB),  Memory-Before(MB),  Memory-After(MB)\n")

        # For DEBUG # 注册缓冲区用于统计 out_feats_j 长度
        # self.print_interval = 100
        # self.register_buffer("out_feats_j_length_sum", torch.tensor(0.0, requires_grad=False))
        # self.register_buffer("out_feats_j_count", torch.tensor(0, requires_grad=False))
        # self.register_buffer("out_feats_j_avg", torch.tensor(0.0, requires_grad=False))

    @staticmethod
    @torch.no_grad()
    def _record_info(
            # self, 
            memory_log_path,
            stage_name,
            info_before=None,
            # feats_after=None,
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
        # ne_voxel_after = feats_after.shape[0] if feats_after is not None else None
        # delta_ne_voxel_num = ne_voxel_after - ne_voxel_before

        if stage_name.startswith("Enter"):

            char_num = 30 if stage_name == "Stage_0_Initial" else 15
            banner = "=" * 15 + f" MEMORY TRACKING: {stage_name} " + "=" * char_num
            with open(memory_log_path, "a", newline="") as f:
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}, {banner}\n")
        
        if info_before is not None and 'attn_info' in info_before:
            attn_info = info_before['attn_info']
            with open(memory_log_path, "a", newline="") as f:
                log_line = (
                    f"{dt.strftime('%Y-%m-%d %H:%M:%S'):<19}, "
                    f"{stage_name:<20}, "
                    f"{delta_time:>10.3f} s"
                    f"{attn_info[0]:>6f}  "
                    f"{attn_info[1]:>6f}  "
                    f"{attn_info[2]:>6f}  "
                    f"{delta_mb:>10.2f} Mb, "
                    f"{delta_gb:>8.4f} Gb, "
                    f"{memory_before:>10.2f}, "
                    f"{memory_after:>10.2f}, "
                    f"\n"
                ) + ("\n" if newline else '')
                f.write(log_line) 
        else:
            with open(memory_log_path, "a", newline="") as f:
                log_line = (
                    f"{dt.strftime('%Y-%m-%d %H:%M:%S'):<19}, "
                    f"{stage_name:<20}, "
                    f"{delta_time:>10.3f} s"
                    f"{delta_mb:>10.2f} Mb, "
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

    # @torch.no_grad()
    # def _update_debug_statistics(self, current_length, debug=False):
    #     if not debug:
    #         return
    #     # 更新统计量
    #     self.out_feats_j_length_sum += current_length
    #     self.out_feats_j_count += 1
    #     self.out_feats_j_avg = self.out_feats_j_length_sum / self.out_feats_j_count
        
    #     # 按间隔打印
    #     if self.out_feats_j_count % self.print_interval == 0: 
    #         print(f"Step {self.out_feats_j_count}: out_feats_j avg length = {self.out_feats_j_avg.item():.2f}")
    
    # TODO: simple \ classic version(可学习转译矩阵)
    def feedback(
            self, 
            hidden_queries, # [bs\None, num_queries, C]
            attn_hidden_queries, # [bs\None, num_queries, C]
            # residual=True,
        ):
        ori_dtype = hidden_queries.dtype
        if ori_dtype in (torch.float16, torch.bfloat16) and self.query_in_fp32 == True: # NOTE: 可强制全精度 feedback
            if self.hybrid_gate == True:
                g = torch.sigmoid(self.attn_gate(torch.cat([hidden_queries, attn_hidden_queries], dim=1).to(dtype=torch.float32))) # [N, 1]
            else:
                g = torch.sigmoid(self.attn_gate(hidden_queries.to(dtype=torch.float32))) # [N, 1]
            # g = g.clamp(min=0.01, max=0.99)
            out_hidden_queries = g * attn_hidden_queries.to(dtype=torch.float32) + (1 - g) * hidden_queries.to(dtype=torch.float32)
            return out_hidden_queries.to(dtype=ori_dtype)
            # return self.attn_norm(out_hidden_queries).to(dtype=ori_dtype)
        else:
            if self.hybrid_gate == True:
                g = torch.sigmoid(self.attn_gate(torch.cat([hidden_queries, attn_hidden_queries], dim=1))) # [N, 1]
            else:
                g = torch.sigmoid(self.attn_gate(hidden_queries)) # [N, 1]
            # g = g.clamp(min=0.01, max=0.99)
            out_hidden_queries = g * attn_hidden_queries + (1 - g) * hidden_queries
            # print(f"g={g.mean().item()}")
            return out_hidden_queries
            # return self.attn_norm(out_hidden_queries)
    
    
    def safe_fusion(self, feats_i, feats_j, sample_mask=None, save_first=True, fusion_type='mul'):
        """
        安全实现：fusion_feats[sample_mask] = feats_i[sample_mask] * feats_j
        保持原始逻辑不变，避免梯度断裂
        参数:
            feats_i: [N, C] 主特征
            feats_j: [M, C] 采样特征 (M = len(sample_mask))
            sample_mask: [M] 整数索引，标记采样位置
        返回:
            fusion_feats: [N, C] 融合后的特征
        """
        if sample_mask is not None:
            if save_first == True:
                fusion_feats = feats_i.clone() # 创建全量特征张量, 输出结果形状为 [N, C]
            else:
                # fusion_feats = torch.zeros_like(feats_j, device=feats_j.device, dtype=feats_j.dtype)
                fusion_feats = None # 输出结果形状为 [M, C]

            if fusion_type == 'mul':
                product = feats_i[sample_mask] * feats_j
            elif fusion_type == 'sum':
                product = feats_i[sample_mask] + feats_j
            else:
                raise NotImplementedError

            if save_first == True:
                expanded_indices = sample_mask.unsqueeze(1).expand(-1, feats_i.size(1)) # 创建扩展索引 [M, C]
                fusion_feats = fusion_feats.scatter( # 使用 scatter 安全更新
                    dim=0,
                    index=expanded_indices,
                    src=product
                )
            else:
                fusion_feats = product
        else:
            if fusion_type == 'mul':
                fusion_feats = feats_i * feats_j
            elif fusion_type == 'sum':
                fusion_feats = feats_i + feats_j
            else:
                raise NotImplementedError
        return fusion_feats

    def forward1(
            self,
            main_feats: torch.Tensor,
            sample_feats: torch.Tensor,
            debug=False):
        
        mamba_layer_i, mamba_layer_j = self.mamba_encoder_list[:2]

        feats_i_m = mamba_layer_i(main_feats[None].flip(1), None)  # ([1, N, C], _)
        feats_i = feats_i_m[0].squeeze(0).flip(0) # [N, C]
        
        feats_j_m = mamba_layer_j(sample_feats[None].flip(1), None)
        feats_j = feats_j_m[0].squeeze(0).flip(0)  # [N, C]

        feats_i = self.ssm_norm_i(feats_i)
        feats_j = self.ssm_norm_j(feats_j)
        return feats_i, feats_j
        
    def forward2(
            self,
            feats_i: torch.Tensor,
            feats_j: torch.Tensor,
            sample_mask: torch.Tensor = None,
            # main_scan_index_rev: torch.Tensor = None,
            sample_scan_index_rev: torch.Tensor = None,
            fusion_scan_index: torch.Tensor = None,
            fusion_scan_index_rev: torch.Tensor = None,
            hidden_queries: torch.Tensor = None,
            info_current=None,
            debug=False):
        
        mamba_layer_fusion = self.mamba_encoder_list[2]

        if self.skip_connect[1] == True:
            res_feats_j = feats_j.clone()

        fusion_feats = self.safe_fusion(
            feats_i, feats_j, sample_mask,
            save_first=True,
            fusion_type=self.mix_type[2]) # fusion_feats[sample_mask] = feats_i[sample_mask] * feats_j
        if fusion_scan_index is not None:
            fusion_feats = fusion_feats[fusion_scan_index]

        fusion_feats = rearrange(fusion_feats, "s d -> d s")
        fusion_feats = self.conv_fusion(fusion_feats)
        fusion_feats = rearrange(fusion_feats, "d s -> s d")

        feats_fusion_m = mamba_layer_fusion(fusion_feats[None].flip(1), None)[0].squeeze(0)
        feats_fusion = self.ssm_norm_fusion(feats_fusion_m.flip(0))
        
        if fusion_scan_index is not None: # 表示 fusion 曾转换过方向, 扫描完后也转回原序列方向
            feats_fusion = feats_fusion[fusion_scan_index_rev]
        info_current = self._record_info(self.memory_log_path, "qssm.3.1", info_current, execute=debug)

        if self.fusion_simulate_z_res == True:
            feats_fusion = self.fusion_post_act(feats_fusion)

        out_feats_i = self.safe_fusion(
            feats_i, feats_fusion, None,
            save_first=True,
            fusion_type=self.mix_type[0])
        # out_feats_i = feats_i * feats_fusion

        # NOTE 注意这里需要保留的只是 feats_j 部分的特征 [M, C] -> save_first=False
        out_feats_j = self.safe_fusion(
            feats_fusion, feats_j, sample_mask,
            save_first=False, # out_feats_j = feats_j * feats_fusion[sample_mask]
            fusion_type=self.mix_type[1])

        if fusion_scan_index is None: # None 表示都处于某一方向顺序
            if self.skip_connect[1] == True:
                out_feats_j = out_feats_j + res_feats_j # 'x' \ 'y'
            out_feats_j = out_feats_j[sample_scan_index_rev] # origin
        else: # 否则都已经是原方向
            if self.skip_connect[1] == True:
                out_feats_j = out_feats_j + res_feats_j # origin

        out_feats_j = self.ssm_norm_out_j(out_feats_j)

        # outout_feats = out_feats_i + res_main_feats * self.skip_gate(self.skip_scale)
        info_current = self._record_info(self.memory_log_path, "qssm.3.2", info_current, execute=debug)

        if self.total_queries > 0:
            # self._update_debug_statistics(current_length=out_feats_j.shape[0], debug=True)
            kv = out_feats_j.unsqueeze(0)
            attn_hidden_queries = self.query_attn_module( # NOTE: 是否需要强制全精度 CrossAttn?
                Q=hidden_queries.unsqueeze(0),
                K=kv,
                V=kv.clone(),
            ).squeeze(0)
            # attn_hidden_queries = hidden_queries + out_feats_j.mean() # JUST FOR TEST
            if debug == True:
                info_current['attn_info'] = [hidden_queries.shape[0], out_feats_j.shape[0], attn_hidden_queries.shape[0]]

            info_current = self._record_info(self.memory_log_path, "qssm.3.3", info_current, execute=debug)

            # NOTE: 有 feedback 则 query_attn_module 无需残差即 residual=False
            if self.gated_feedback == True:
                updated_hidden_queries = self.feedback(
                    hidden_queries=hidden_queries,
                    attn_hidden_queries=attn_hidden_queries,
                    # residual=True,
                )
            else:
                updated_hidden_queries = attn_hidden_queries # 若 gated_feedback 为 False, 则已在 Cross-Attn 中加入标准残差
        else:
            updated_hidden_queries = None

            info_current = self._record_info(self.memory_log_path, "qssm.3.4", info_current, execute=debug)

        if self.skip_connect[2] == True:
            return out_feats_i, out_feats_j, updated_hidden_queries, info_current
        else:
            return out_feats_i, None, updated_hidden_queries, info_current


    def forward(
        self,
        x: spconv.SparseConvTensor,
        hidden_queries,  # [bs, total_queries, dim]
        pos_embed,
        scan_primary_axis_list,
        num_stage=0, # NOTE: without input num_stage temporary (default is 0 now)
        debug=False,
    ) -> Tuple[torch.Tensor]:
        
        # debug = False
        info_current = self._record_info(self.memory_log_path, "Enter qssm", execute=debug)

        scan_primary_axis_main, scan_primary_axis_sample, scan_primary_axis_fusion = scan_primary_axis_list
        assert scan_primary_axis_main == scan_primary_axis_sample

        # TODO: 每个模块(QSSM\GSSM)包括 selfmodeling 和 ffn 的首尾组合(注意残差、归一化位置和耗用)
        if self.multiscale_modeling == True:
            x = self.pre_modeling(x)

        feats = x.features
        coords = x.indices
        dtype = feats.dtype
        batch_size = coords[:, 0].max().int().item() + 1

        if self.skip_connect[0] == True:
            res_main_feats = feats.clone()
        
        info_current = self._record_info(self.memory_log_path, "qssm.modeling", info_current, execute=debug)

        with torch.no_grad():
            info_current = self._record_info(self.memory_log_path, "qssm.0", info_current, execute=debug)
            # clvl_cruve_template_ij = self.curve_template['curve_template_rank9']
            # clvl_hilbert_spatial_size_ij = self.hilbert_spatial_size['curve_template_rank9']
            # main_points_hilbert_indices = get_hilbert_index_3d_mamba_lite(
            #     clvl_cruve_template_ij, 
            #     ori_main_coords, 
            #     batch_size=enhanced_main_x.origin_bs,
            #     z_dim=self.sparse_shape[0],
            #     hilbert_spatial_size=clvl_hilbert_spatial_size_ij,
            #     shift=(0, 0, 0),
            # )

            if self.space_filing_curve == 'z_order' or self.space_filing_curve == 'hierarchical_z_order':
                main_morton_indices = get_morton_index_3d(
                    coords, 
                    batch_size,
                    self.sparse_shape, 
                    primary_axis=scan_primary_axis_main
                )
                if scan_primary_axis_main != scan_primary_axis_fusion:
                    fusion_morton_indices = get_morton_index_3d( # NOTE: 注意这里前提是 fusion 仍为 main 的体素序列(即长度不变)
                        coords, 
                        batch_size,
                        self.sparse_shape, 
                        primary_axis=scan_primary_axis_fusion
                    )
                else:
                    fusion_morton_indices = None
                    # fusion_morton_indices = main_morton_indices
            elif self.space_filing_curve == 'hilbert':
                main_morton_indices = get_hilbert_index_3d_mamba_lite(
                    self.curve_template['curve_template_rank9'], 
                    coords, batch_size, self.sparse_shape[0],
                    self.hilbert_spatial_size['curve_template_rank9'], 
                    shift=(num_stage, num_stage, num_stage))
                fusion_morton_indices = None
                # fusion_morton_indices = main_morton_indices
            elif self.space_filing_curve == 'random':
                main_morton_indices = get_random_index_3d(
                    coords, batch_size, seed=42
                )
                fusion_morton_indices = None
                # fusion_morton_indices = main_morton_indices

            

        info_current = self._record_info(self.memory_log_path, "qssm.1", info_current, execute=debug)
        
        # sample_feats = self.defsampler(feats, coords, query_embed=hidden_queries, debug=debug)
        sample_feats, sample_coords, all_sample_masks = self.extractor(feats, coords, debug=debug)
        # sample_feats = feats.clone()
        # sample_coords = coords.clone()
        # sample_coords_mask = torch.ones(feats.size(0), dtype=torch.long).to(feats.device)

        info_current = self._record_info(self.memory_log_path, "qssm.sampler", info_current, execute=debug)

        if self.window_pos_embed == True:
            window_pos_embed = self.window_pos_embeding(coords, x.spatial_shape, pos_embed)
            feats = feats + window_pos_embed
            window_sample_pos_embed = self.window_pos_embeding(sample_coords, x.spatial_shape, pos_embed)
            sample_feats = sample_feats + window_sample_pos_embed

        info_current = self._record_info(self.memory_log_path, "qssm.2.1", info_current, execute=debug)

        with torch.no_grad():
            if self.space_filing_curve == 'z_order' or self.space_filing_curve == 'hierarchical_z_order':
                sample_morton_indices = get_morton_index_3d(
                    sample_coords, 
                    batch_size,
                    self.sparse_shape, 
                    primary_axis=scan_primary_axis_sample
                )
            elif self.space_filing_curve == 'hilbert':
                sample_morton_indices = get_hilbert_index_3d_mamba_lite(
                    self.curve_template['curve_template_rank9'], 
                    sample_coords, batch_size, self.sparse_shape[0],
                    self.hilbert_spatial_size['curve_template_rank9'], 
                    shift=(num_stage, num_stage, num_stage))
            elif self.space_filing_curve == 'random':
                sample_morton_indices = get_random_index_3d(
                    sample_coords, batch_size, seed=42
                )

        # coords_m = {}
        feats_m = {}
        sample_feats_m = {}
        all_sample_masks_m = {}

        info_current = self._record_info(self.memory_log_path, "qssm.2.2", info_current, execute=debug)
        
        with torch.no_grad():
            for b in range(batch_size):
                batch_mask = coords[:, 0] == b
                sample_batch_mask = sample_coords[:, 0] == b
                # coords_m[b] = coords[batch_mask][main_morton_indices['inds_curt_to_next'][b]]
                feats_m[b] = feats[batch_mask][main_morton_indices['inds_curt_to_next'][b]]
                sample_feats_m[b] = sample_feats[sample_batch_mask][sample_morton_indices['inds_curt_to_next'][b]]
                if fusion_morton_indices is None: # 仅当三者方向一致时需要转换 mask, 使其也为该方向; 否则按原序列顺序即可(因为 main\sample 则也会被转为原序列顺序)
                    all_sample_masks_m[b] = main_morton_indices['inds_next_to_curt'][b][all_sample_masks[b]]

        info_current = self._record_info(self.memory_log_path, "qssm.2.3", info_current, execute=debug)

        out_features = torch.zeros_like(feats)
        # out_features = feats.clone() # TODO: 注意这里的输出特征的基础特征到底需要什么, zeros? feats?

        enhanced_main_feats_m_list = []
        enhanced_sample_feats_m_list = []
        for b in range(batch_size):
            batch_feats_m = feats_m[b]
            batch_sample_feats_m = sample_feats_m[b]

            enhanced_main_feats_m, enhanced_sample_feats_m = self.forward1(
                main_feats=batch_feats_m,
                sample_feats=batch_sample_feats_m,
                debug=debug,
            )

            enhanced_main_feats_m_list.append(enhanced_main_feats_m)
            enhanced_sample_feats_m_list.append(enhanced_sample_feats_m)

        info_current = self._record_info(self.memory_log_path, "qssm.3", info_current, execute=debug)
            
        out_hidden_queries_list = []
        for b in range(batch_size):
            with torch.no_grad():
                batch_mask = coords[:, 0] == b
                # sample_batch_mask = sample_coords[:, 0] == b

                batch_point_indices_m = main_morton_indices['inds_next_to_curt'][b]
                batch_sample_point_indices_m = sample_morton_indices['inds_next_to_curt'][b]
                batch_fusion_point_indices_m = fusion_morton_indices['inds_curt_to_next'][b] if fusion_morton_indices is not None else None
                batch_fusion_point_indices_rev_m = fusion_morton_indices['inds_next_to_curt'][b] if fusion_morton_indices is not None else None
                # batch_coords_m = coords_m[b][batch_point_indices_m]

            if fusion_morton_indices is None: # 三者扫描方向相同, 保留原方向
                enhanced_main_feats = enhanced_main_feats_m_list[b]
                enhanced_sample_feats = enhanced_sample_feats_m_list[b]
                batch_sample_mask = all_sample_masks_m[b]
            else: # fusion 与 main\sample 方向不一致, 先转换 main\sample 回原序列顺序
                enhanced_main_feats = enhanced_main_feats_m_list[b][batch_point_indices_m]
                enhanced_sample_feats = enhanced_sample_feats_m_list[b][batch_sample_point_indices_m]
                batch_sample_mask = all_sample_masks[b]

            updated_main_feats, updated_sample_feats, updated_hidden_queries, info_current = self.forward2(
                feats_i=enhanced_main_feats,
                feats_j=enhanced_sample_feats,
                sample_mask=batch_sample_mask, # NOTE: 视 fusion 是否转换方向，若不是则需要存储的是转换为 main\sample 方向的索引
                fusion_scan_index=batch_fusion_point_indices_m,
                # main_scan_index_rev=batch_point_indices_m,
                sample_scan_index_rev=batch_sample_point_indices_m,
                fusion_scan_index_rev=batch_fusion_point_indices_rev_m,
                hidden_queries=hidden_queries[b] if self.total_queries > 0 else None,
                info_current=info_current,
                debug=debug,
            )

            if fusion_morton_indices is None: # 三者方向相同, 扫描完后才转回原序列顺序
                updated_main_feats = updated_main_feats[batch_point_indices_m]
                if updated_sample_feats is not None:
                    updated_sample_feats = updated_sample_feats[batch_sample_point_indices_m] # 若需要
            else:
                pass

            # with torch.no_grad():
            # indices = batch_mask.nonzero(as_tuple=True)[0]
            # out_features = out_features.index_put( # 使用 index_put 安全赋值, origin: out_features[batch_mask] = updated_main_feats
            #     indices=[indices],
            #     values=updated_main_feats,
            #     accumulate=False
            # )
            # NOTE: TODO: 为什么 out_features 和 feats 是 fp16, 经过 forward1 的两个特征都是 fp32, 导致这里 updated_main_feats 也是 fp32。类型不匹配
            # 这里暂时直接转换类型
            out_features[batch_mask] = self.safe_fusion(
                feats_i=updated_main_feats, feats_j=updated_sample_feats,
                sample_mask=batch_sample_mask,
                save_first=True,
                fusion_type='sum',
            ) if self.skip_connect[2] == True else updated_main_feats.to(dtype=dtype)
            # out_features[batch_mask] = updated_main_feats

            if updated_hidden_queries is not None:
                out_hidden_queries_list.append(updated_hidden_queries)

        if self.total_queries > 0:
            out_hidden_queries = torch.stack(out_hidden_queries_list, dim=0)
            out_hidden_queries = self.attn_norm(out_hidden_queries.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        else:
            out_hidden_queries = None

        # TODO NOTE: 若 Fj 经过 cross-transformer 更新后得到 enhanced_sample_feats, 则需要 norm_out(out_features + scale(enhanced_sample_feats))
        if self.skip_connect[0] == True:
            norm_out_features = self.norm_out(out_features) + res_main_feats
        else:
            norm_out_features = self.norm_out(out_features)

        info_current = self._record_info(self.memory_log_path, "qssm.4", info_current, execute=debug)
        return norm_out_features, out_hidden_queries


    def window_pos_embeding(
            self, 
            coords: torch.Tensor,
            spatial_shape: List,
            pos_embed,
        ):
    
        window_pos_coords = torch.zeros([coords.shape[0], 9], device=coords.device, dtype=torch.float32)
        window_pos_coords[:, 0] = coords[:, 1] / spatial_shape[0]
        window_pos_coords[:, 1:3] = (coords[:, 2:] // 12) / (spatial_shape[1]//12 + 1)
        window_pos_coords[:, 3:5] = (coords[:, 2:] % 12) / 12.0
        window_pos_coords[:, 5:7] = ((coords[:, 2:] + 6) // 12) / (spatial_shape[1]//12 + 1)
        window_pos_coords[:, 7:9] = ((coords[:, 2:] + 6) % 12) / 12.0
        window_pos_embed= pos_embed(window_pos_coords.float())

        return window_pos_embed
