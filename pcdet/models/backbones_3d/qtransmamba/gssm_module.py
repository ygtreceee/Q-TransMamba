import os
import math
import copy
import time
import csv
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

from pcdet.models.backbones_3d.qdefmamba.basic_utils import get_hilbert_index_3d_mamba_lite, post_act_block, get_random_index_3d
        
from pcdet.models.backbones_3d.qdefmamba.basic_utils import Sparse1ConvBlock, SparseBasicBlock3D

from morton_encoding import get_morton_index_3d


class GSSM_v2_v3(nn.Module):
    def __init__(self,
                 dim,
                 num_stage,
                 num_block,
                 gssm_kernel_size, # [3, 3]
                 gssm_stride,      # [1, 2]
                 sub_num,          # [2, 1]
                 gssm_lvl,   # ['...9', '...8', '...9']
                 down_resolution,
                 sparse_shape,
                 norm_epsilon, 
                 rms_norm,
                 curve_template,
                 hilbert_spatial_size,
                 window_shape=None,
                 window_overlap=None,
                 ssm_cfg=None,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.gssm_lvl = gssm_lvl
        self.down_resolution = down_resolution

        self.norm1d_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm2d_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.layernorm_fn = partial(nn.LayerNorm, eps=1e-5)

        indice_key = f"gssm_{num_stage}_{num_block}"

        ## for encoder-decoder architecture ##
        block = SparseBasicBlock3D
        # self.gssm_kernel_size = gssm_kernel_size
        # self.gssm_stride = gssm_stride
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
                    in_channels=self.dim, out_channels=self.dim, kernel_size=gssm_kernel_size[idx], 
                    stride=gssm_stride[idx], padding=gssm_kernel_size[idx] // 2,
                    conv_type='spconv', indice_key=f'spconv_{indice_key}_{idx}', norm_fn=self.layernorm_fn),
                *[block(dim, indice_key=f"{indice_key}_{idx}") for _ in range(sub_num[idx])]
            ]
            self.encoder.append(spconv.SparseSequential(*cur_layers))

        # up sample layer
        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(self.num_levels - 1, 0, -1):
            self.decoder.append(
                post_act_block(
                    in_channels=self.dim, out_channels=self.dim, kernel_size=gssm_kernel_size[idx], 
                    conv_type='inverseconv', indice_key=f'spconv_{indice_key}_{idx}', norm_fn=self.layernorm_fn))
            self.decoder_norm.append(self.layernorm_fn(dim))
 

        self.sparse_shape = sparse_shape
        self.window_shape = window_shape
        self.window_overlap = window_overlap

        # ssm_cfg = {}
        factory_kwargs = {'device': device, 'dtype':dtype}
        self.curve_template = curve_template
        self.hilbert_spatial_size = hilbert_spatial_size

        self.total_layers = (self.num_levels - 1) * 2 + 1
        self.mamba_encoder_list = nn.ModuleList()
        self.ssm_norm_list = nn.ModuleList()
        for idx in range(self.total_layers):
            self.mamba_encoder_list.append(
                create_block(
                    d_model=dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=idx,
                    z_residual=True,
                    **factory_kwargs,)
            )
            self.ssm_norm_list.append(self.layernorm_fn(dim))



    def forward(self,
                sparse_x,
                pos_embed):

        # x = spconv.SparseConvTensor(
        #     features=voxel_features,
        #     indices=voxel_coords.int(),
        #     spatial_shape=curt_spatial_shape,
        #     batch_size=batch_size
        # )

        x = sparse_x

        # NOTE: 是否需要设置反向扫描
        feats_list = []
        idx = 0
        for conv in self.encoder:
            x = conv(x)
            x = self.single_glocal_scan_forward(x=x, num_stage=idx, pos_embed=pos_embed)
            feats_list.append(x)
            idx += 1

        x = feats_list[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats_list[:-1][::-1]):
            x = deconv(x)
            x = self.single_glocal_scan_forward(x=x, num_stage=idx, pos_embed=pos_embed)
            x = replace_feature(x, norm(x.features + up_x.features))
            idx += 1

        return x


    def single_glocal_scan_forward(
            self,
            x,
            num_stage,
            pos_embed,
        ):
        
        mamba_layer = self.mamba_encoder_list[num_stage]

        feats = x.features
        coords = x.indices
        curt_spatial_shape = x.spatial_shape
        batch_size = x.batch_size

        # Pos Embedding 
        window_pos = self.global_window_position_embedding(
            coords, 
            curt_spatial_shape, 
            window_size=self.window_shape, 
            overlap=self.window_overlap
        )  # [N, 5]

        window_pos_embed = pos_embed(window_pos) # [N, 256]

        feats = feats + window_pos_embed # [N, 256]

        out_feats_3d = torch.zeros_like(feats)

        curve_template_rank = self.gssm_lvl[num_stage] # '...8 / ...9'
        clvl_cruve_template = self.curve_template[curve_template_rank]
        clvl_hilbert_spatial_size = self.hilbert_spatial_size[curve_template_rank]
        index_info = get_hilbert_index_3d_mamba_lite(clvl_cruve_template, coords, batch_size, self.sparse_shape[0], \
                                                        clvl_hilbert_spatial_size, shift=(num_stage, num_stage, num_stage))
        inds_curt_to_next = index_info['inds_curt_to_next'] # 'dict' 
        inds_next_to_curt = index_info['inds_next_to_curt']

        for i in range(batch_size):

            b_mask_m = coords[:, 0] == i
            feats_b = feats[b_mask_m]
            feats_b_m = feats_b[inds_curt_to_next[i]][None] # feats_i_m: [1, N, dim]
            out_feats_b_m = mamba_layer(feats_b_m, None) # out_feat_bi_m[0]: [1, N, dim])
            out_feats_b = out_feats_b_m[0].squeeze(0)[inds_next_to_curt[i]] # [N, dim]
            out_feats_3d[b_mask_m] = out_feats_b

        out_feats_3d = self.ssm_norm_list[num_stage](out_feats_3d)
        x = replace_feature(x, out_feats_3d)

        return x


    def global_window_position_embedding(self, coords, spatial_shape, window_size, overlap):
        """
        coords: Tensor of shape [N, 4] containing (batch_idx, z, y, x) coordinates
        spatial_shape: Tuple (Z, Y, X) of spatial dimensions
        window_size: Size of the sliding window (Y, X)
        overlap: Overlap size between consecutive windows
        """
        step = window_size[0] - overlap  # Step size for sliding window
        Z, Y, X = spatial_shape
        N = coords.shape[0]
        
        pos_embed = torch.zeros((N, 5), device=coords.device, dtype=torch.float32)
        
        pos_embed[:, 0] = coords[:, 1] / Z
        
        y_coords = coords[:, 2]  # [N]
        x_coords = coords[:, 3]  # [N]
        
        # 计算各维度的窗口数量
        window_num_y = int(torch.ceil(torch.tensor((Y - overlap) / step)))
        window_num_x = int(torch.ceil(torch.tensor((X - overlap) / step)))
        
        # 2. 预计算所有窗口起始位置
        window_starts_y = torch.arange(0, Y - overlap, step, device=coords.device)[:window_num_y]
        window_starts_x = torch.arange(0, X - overlap, step, device=coords.device)[:window_num_x]
        
        # 3. 向量化计算点与窗口的包含关系 - Y方向
        # 创建广播矩阵 [N, window_num_y]
        y_coords_exp = y_coords.unsqueeze(1)  # [N, 1]
        win_starts_y_exp = window_starts_y.unsqueeze(0)  # [1, window_num_y]
        
        # 计算哪些窗口包含该点 (判断Y方向)
        y_in_window = (y_coords_exp >= win_starts_y_exp) & \
                    (y_coords_exp < win_starts_y_exp + window_size[0])
        
        # 计算Y方向局部坐标并归一化
        local_y = (y_coords_exp - win_starts_y_exp) * y_in_window
        norm_local_y = local_y / window_size[0]  # [N, window_num_y]
        
        # 窗口索引归一化
        win_y_idx = torch.arange(window_num_y, device=coords.device).float()[None, :] / window_num_y
        
        # 4. 向量化计算点与窗口的包含关系 - X方向
        # 创建广播矩阵 [N, window_num_x]
        x_coords_exp = x_coords.unsqueeze(1)  # [N, 1]
        win_starts_x_exp = window_starts_x.unsqueeze(0)  # [1, window_num_x]
        
        # 计算哪些窗口包含该点 (判断X方向)
        x_in_window = (x_coords_exp >= win_starts_x_exp) & \
                    (x_coords_exp < win_starts_x_exp + window_size[1])
        
        # 计算X方向局部坐标并归一化
        local_x = (x_coords_exp - win_starts_x_exp) * x_in_window
        norm_local_x = local_x / window_size[1]  # [N, window_num_x]
        
        # 窗口索引归一化
        win_x_idx = torch.arange(window_num_x, device=coords.device).float()[None, :] / window_num_x
        
        # 5. 计算每个点归属的窗口数量(用于平均化)
        # [N, window_num_y] -> [N] 每个点在Y方向归属的窗口数
        y_window_count = y_in_window.sum(dim=1).float()
        # [N, window_num_x] -> [N] 每个点在X方向归属的窗口数
        x_window_count = x_in_window.sum(dim=1).float()
        
        # 6. 计算窗口贡献值 (避免直接累加导致的数值膨胀)
        # NOTE: 交叉加权
        # 假设一个点被3个Y方向窗口和2个X方向窗口覆盖，那么该点总共落在3 * 2=6个窗口内。
        # 我们希望聚合这些6个窗口的信息。但是，由于窗口索引和局部位置是分别计算Y和X方向，因此我们采用交叉权重：
        # 计算Y方向窗口索引的聚合时，每个Y方向窗口出现的次数为（该点在X方向覆盖的窗口数），即2次。这样相当于把每个Y方向窗口复制了2次（对应2个X方向窗口），然后求和。
        # Y方向贡献部分
        y_contrib = win_y_idx * x_window_count[:, None]  # 交叉加权
        y_contrib_sum = torch.einsum('nd,nd->n', y_in_window.float(), y_contrib)
        
        # X方向贡献部分
        x_contrib = win_x_idx * y_window_count[:, None]  # 交叉加权
        x_contrib_sum = torch.einsum('nd,nd->n', x_in_window.float(), x_contrib)
        
        # 7. 计算局部位置贡献值
        # Y局部位置
        local_y_contrib = norm_local_y * x_window_count[:, None]
        local_y_sum = torch.einsum('nd,nd->n', y_in_window.float(), local_y_contrib)
        
        # X局部位置
        local_x_contrib = norm_local_x * y_window_count[:, None]
        local_x_sum = torch.einsum('nd,nd->n', x_in_window.float(), local_x_contrib)
        
        # 8. 合并到最终位置嵌入
        pos_embed[:, 1] = x_contrib_sum  # X方向窗口索引加权和
        pos_embed[:, 2] = y_contrib_sum  # Y方向窗口索引加权和
        pos_embed[:, 3] = local_x_sum    # X局部位置加权和
        pos_embed[:, 4] = local_y_sum    # Y局部位置加权和
        
        return pos_embed
    



class GSSM_v1(nn.Module):
    def __init__(self,
                 dim,
                 num_stage,
                 num_block,
                 num_lvl,
                 gssm_lvl,   # ['...9', '...8', '...9']
                 sparse_shape,
                 window_shape,
                 window_overlap,
                 norm_epsilon, 
                 rms_norm,
                 curve_template,
                 hilbert_spatial_size,
                 ssm_cfg=None,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()

        # self.dim = dim
        self.gssm_lvl = gssm_lvl
        # self.num_lvl = num_lvl

        # self.norm1d_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # self.norm2d_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.layernorm_fn = partial(nn.LayerNorm, eps=1e-5)

        # indice_key = f"gssm_{num_stage}_{num_block}"
         
        self.sparse_shape = sparse_shape
        self.window_shape = window_shape
        self.window_overlap = window_overlap

        # ssm_cfg = {}
        factory_kwargs = {'device': device, 'dtype':dtype}
        self.curve_template = curve_template
        self.hilbert_spatial_size = hilbert_spatial_size

        self.mamba = create_block(
                    d_model=dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=num_lvl,
                    z_residual=True,
                    **factory_kwargs,)
        self.ssm_norm = self.layernorm_fn(dim)


    def forward(self,
                sparse_x,
                pos_embed,
                debug=False):

        x = self.single_glocal_scan_forward(x=sparse_x, pos_embed=pos_embed)

        return x


    def single_glocal_scan_forward(
            self,
            x,
            pos_embed,
        ):
        

        feats = x.features.clone()
        coords = x.indices.clone()
        curt_spatial_shape = x.spatial_shape
        batch_size = x.batch_size

        # Pos Embedding 
        window_pos = self.global_window_position_embedding(
            coords, 
            curt_spatial_shape, 
            window_size=self.window_shape, 
            overlap=self.window_overlap
        )  # [N, 5]

        window_pos_embed = pos_embed(window_pos) # [N, 256]

        feats = feats + window_pos_embed # [N, 256]

        # NOTE: 06.24 
        # out_feats_3d = torch.zeros_like(feats)
        # out_feats_3d = feats.clone()
        out_feats_list = []

        mamba_layer = self.mamba
        curve_template_rank = self.gssm_lvl # '...8 / ...9'
        clvl_cruve_template = self.curve_template[curve_template_rank]
        clvl_hilbert_spatial_size = self.hilbert_spatial_size[curve_template_rank]
        index_info = get_hilbert_index_3d_mamba_lite(clvl_cruve_template, coords, batch_size, self.sparse_shape[0], \
                                                        clvl_hilbert_spatial_size, shift=(0, 0, 0)) # TODO: shift lead to cuda error ?
        inds_curt_to_next = index_info['inds_curt_to_next'] # 'dict' 
        inds_next_to_curt = index_info['inds_next_to_curt']

        for i in range(batch_size):

            b_mask_m = coords[:, 0] == i
            feats_b = feats[b_mask_m]
            feats_b_m = feats_b[inds_curt_to_next[i]][None] # feats_i_m: [1, N, dim]
            out_feats_b_m = mamba_layer(feats_b_m, None) # out_feat_bi_m[0]: [1, N, dim])
            out_feats_b = out_feats_b_m[0].squeeze(0)[inds_next_to_curt[i]] # [N, dim]
            # out_feats_3d[b_mask_m] = out_feats_b
            out_feats_list.append(out_feats_b)

        out_feats_3d = torch.cat(out_feats_list, dim=0)
        out_feats_3d = self.ssm_norm(out_feats_3d)
        x = replace_feature(x, out_feats_3d)

        return x


    def global_window_position_embedding(self, coords, spatial_shape, window_size, overlap):
        """
        coords: Tensor of shape [N, 4] containing (batch_idx, z, y, x) coordinates
        spatial_shape: Tuple (Z, Y, X) of spatial dimensions
        window_size: Size of the sliding window (Y, X)
        overlap: Overlap size between consecutive windows
        """
        step = window_size[0] - overlap  # Step size for sliding window
        Z, Y, X = spatial_shape
        N = coords.shape[0]
        
        pos_embed = torch.zeros((N, 5), device=coords.device, dtype=torch.float32)
        
        pos_embed[:, 0] = coords[:, 1] / Z
        
        y_coords = coords[:, 2]  # [N]
        x_coords = coords[:, 3]  # [N]
        
        # 计算各维度的窗口数量
        window_num_y = int(torch.ceil(torch.tensor((Y - overlap) / step)))
        window_num_x = int(torch.ceil(torch.tensor((X - overlap) / step)))
        
        # 2. 预计算所有窗口起始位置
        window_starts_y = torch.arange(0, Y - overlap, step, device=coords.device)[:window_num_y]
        window_starts_x = torch.arange(0, X - overlap, step, device=coords.device)[:window_num_x]
        
        # 3. 向量化计算点与窗口的包含关系 - Y方向
        # 创建广播矩阵 [N, window_num_y]
        y_coords_exp = y_coords.unsqueeze(1)  # [N, 1]
        win_starts_y_exp = window_starts_y.unsqueeze(0)  # [1, window_num_y]
        
        # 计算哪些窗口包含该点 (判断Y方向)
        y_in_window = (y_coords_exp >= win_starts_y_exp) & \
                    (y_coords_exp < win_starts_y_exp + window_size[0])
        
        # 计算Y方向局部坐标并归一化
        local_y = (y_coords_exp - win_starts_y_exp) * y_in_window
        norm_local_y = local_y / window_size[0]  # [N, window_num_y]
        
        # 窗口索引归一化
        win_y_idx = torch.arange(window_num_y, device=coords.device).float()[None, :] / window_num_y
        
        # 4. 向量化计算点与窗口的包含关系 - X方向
        # 创建广播矩阵 [N, window_num_x]
        x_coords_exp = x_coords.unsqueeze(1)  # [N, 1]
        win_starts_x_exp = window_starts_x.unsqueeze(0)  # [1, window_num_x]
        
        # 计算哪些窗口包含该点 (判断X方向)
        x_in_window = (x_coords_exp >= win_starts_x_exp) & \
                    (x_coords_exp < win_starts_x_exp + window_size[1])
        
        # 计算X方向局部坐标并归一化
        local_x = (x_coords_exp - win_starts_x_exp) * x_in_window
        norm_local_x = local_x / window_size[1]  # [N, window_num_x]
        
        # 窗口索引归一化
        win_x_idx = torch.arange(window_num_x, device=coords.device).float()[None, :] / window_num_x
        
        # 5. 计算每个点归属的窗口数量(用于平均化)
        # [N, window_num_y] -> [N] 每个点在Y方向归属的窗口数
        y_window_count = y_in_window.sum(dim=1).float()
        # [N, window_num_x] -> [N] 每个点在X方向归属的窗口数
        x_window_count = x_in_window.sum(dim=1).float()
        
        # 6. 计算窗口贡献值 (避免直接累加导致的数值膨胀)
        # NOTE: 交叉加权
        # 假设一个点被3个Y方向窗口和2个X方向窗口覆盖，那么该点总共落在3 * 2=6个窗口内。
        # 我们希望聚合这些6个窗口的信息。但是，由于窗口索引和局部位置是分别计算Y和X方向，因此我们采用交叉权重：
        # 计算Y方向窗口索引的聚合时，每个Y方向窗口出现的次数为（该点在X方向覆盖的窗口数），即2次。这样相当于把每个Y方向窗口复制了2次（对应2个X方向窗口），然后求和。
        # Y方向贡献部分
        y_contrib = win_y_idx * x_window_count[:, None]  # 交叉加权
        y_contrib_sum = torch.einsum('nd,nd->n', y_in_window.float(), y_contrib)
        
        # X方向贡献部分
        x_contrib = win_x_idx * y_window_count[:, None]  # 交叉加权
        x_contrib_sum = torch.einsum('nd,nd->n', x_in_window.float(), x_contrib)
        
        # 7. 计算局部位置贡献值
        # Y局部位置
        local_y_contrib = norm_local_y * x_window_count[:, None]
        local_y_sum = torch.einsum('nd,nd->n', y_in_window.float(), local_y_contrib)
        
        # X局部位置
        local_x_contrib = norm_local_x * y_window_count[:, None]
        local_x_sum = torch.einsum('nd,nd->n', x_in_window.float(), local_x_contrib)
        
        # 8. 合并到最终位置嵌入
        pos_embed[:, 1] = x_contrib_sum  # X方向窗口索引加权和
        pos_embed[:, 2] = y_contrib_sum  # Y方向窗口索引加权和
        pos_embed[:, 3] = local_x_sum    # X局部位置加权和
        pos_embed[:, 4] = local_y_sum    # Y局部位置加权和
        
        return pos_embed 
    



class GSSM_v4(nn.Module):
    def __init__(self,
                 dim,
                 num_stage,
                 num_block,
                 gssm_kernel_size, # [3, 3]
                 gssm_stride,      # [1, 2]
                 sub_num,          # [2, 1]
                 gssm_lvl,   # ['...9', '...8', '...9']
                 down_resolution,
                 sparse_shape,
                 norm_epsilon, 
                 rms_norm,
                #  curve_template,
                #  hilbert_spatial_size,
                 ssm_cfg=None,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.gssm_lvl = gssm_lvl
        self.down_resolution = down_resolution

        self.norm1d_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm2d_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.layernorm_fn = partial(nn.LayerNorm, eps=1e-5)

        indice_key = f"gssm_{num_stage}_{num_block}"

        ## for encoder-decoder architecture ##
        block = SparseBasicBlock3D
        # self.gssm_kernel_size = gssm_kernel_size
        # self.gssm_stride = gssm_stride
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
                    in_channels=self.dim, out_channels=self.dim, kernel_size=gssm_kernel_size[idx], 
                    stride=gssm_stride[idx], padding=gssm_kernel_size[idx] // 2,
                    conv_type='spconv', indice_key=f'spconv_{indice_key}_{idx}', norm_fn=self.layernorm_fn),
                *[block(dim, indice_key=f"{indice_key}_{idx}") for _ in range(sub_num[idx])]
            ]
            self.encoder.append(spconv.SparseSequential(*cur_layers))

        # up sample layer
        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(self.num_levels - 1, 0, -1):
            self.decoder.append(
                post_act_block(
                    in_channels=self.dim, out_channels=self.dim, kernel_size=gssm_kernel_size[idx], 
                    conv_type='inverseconv', indice_key=f'spconv_{indice_key}_{idx}', norm_fn=self.layernorm_fn))
            self.decoder_norm.append(self.layernorm_fn(dim))
 

        self.sparse_shape = sparse_shape

        # ssm_cfg = {}
        factory_kwargs = {'device': device, 'dtype':dtype}
        # self.curve_template = curve_template
        # self.hilbert_spatial_size = hilbert_spatial_size

        self.total_layers = (self.num_levels - 1) * 2 + 1
        self.mamba_encoder_list = nn.ModuleList()
        self.ssm_norm_list = nn.ModuleList()
        for idx in range(self.total_layers):
            self.mamba_encoder_list.append(
                create_block(
                    d_model=dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=idx,
                    z_residual=True,
                    **factory_kwargs,)
            )
            self.ssm_norm_list.append(self.layernorm_fn(dim))



    def forward(self,
                x,
                pos_embed,
                debug=False):

        # NOTE: 是否需要设置反向扫描
        feats_list = []
        idx = 0
        for conv in self.encoder:
            x = conv(x)
            x = self.single_glocal_scan_forward(x=x, num_stage=idx, pos_embed=pos_embed)
            feats_list.append(x)
            idx += 1

        x = feats_list[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats_list[:-1][::-1]):
            x = deconv(x)
            x = self.single_glocal_scan_forward(x=x, num_stage=idx, pos_embed=pos_embed)
            x = replace_feature(x, norm(x.features + up_x.features))
            idx += 1

        return x


    def single_glocal_scan_forward(
            self,
            x,
            num_stage,
            pos_embed,
        ):
        
        mamba_layer = self.mamba_encoder_list[num_stage]

        feats = x.features
        coords = x.indices
        curt_spatial_shape = x.spatial_shape
        batch_size = x.batch_size

        # Pos Embedding 
        window_pos_embed = pos_embed(x.indices[:, 1:].float())

        feats = feats + window_pos_embed # [N, 256]

        out_feats_3d = torch.zeros_like(feats)

        # curve_template_rank = self.gssm_lvl[num_stage] # '...8 / ...9'
        # clvl_cruve_template = self.curve_template[curve_template_rank]
        # clvl_hilbert_spatial_size = self.hilbert_spatial_size[curve_template_rank]
        # index_info = get_hilbert_index_3d_mamba_lite(clvl_cruve_template, coords, batch_size, self.sparse_shape[0], \
        #                                                 clvl_hilbert_spatial_size, shift=(num_stage, num_stage, num_stage))
        index_info = get_morton_index_3d(coords, batch_size, self.sparse_shape, \
                                        shift=(num_stage, num_stage, num_stage), primary_axis='x')
        inds_curt_to_next = index_info['inds_curt_to_next'] # 'dict' 
        inds_next_to_curt = index_info['inds_next_to_curt']

        for i in range(batch_size):

            b_mask_m = coords[:, 0] == i
            feats_b = feats[b_mask_m]
            feats_b_m = feats_b[inds_curt_to_next[i]][None] # feats_i_m: [1, N, dim]
            out_feats_b_m = mamba_layer(feats_b_m, None) # out_feat_bi_m[0]: [1, N, dim])
            out_feats_b = out_feats_b_m[0].squeeze(0)[inds_next_to_curt[i]] # [N, dim]
            out_feats_3d[b_mask_m] = out_feats_b

        out_feats_3d = self.ssm_norm_list[num_stage](out_feats_3d)
        x = replace_feature(x, out_feats_3d)

        return x





class GSSM_v5(nn.Module):
    def __init__(self,
                 dim,
                 num_stage,
                 num_block,
                 ssm_idx,
                 num_lvl,
                 sparse_shape,
                 norm_epsilon, 
                 rms_norm,
                 force_layernorm,
                 space_filing_curve,
                 ssm_cfg=None,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 window_pos_embed=False,
                 curve_template=None, 
                 hilbert_spatial_size=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()

        # self.dim = dim
        # self.norm2d_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.layernorm_fn = partial(nn.LayerNorm, eps=1e-5)
        self.norm1d_fn = self.layernorm_fn if force_layernorm == True else partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = sparse_shape

        self.window_pos_embed = window_pos_embed
        self.space_filing_curve = space_filing_curve
        self.curve_template=curve_template
        self.hilbert_spatial_size=hilbert_spatial_size

        indice_key = f"stage_{num_stage}_{num_block}_{ssm_idx}"
        num_selfmodeling_layer = 1
        self.self_modeling_blocks = spconv.SparseSequential(
            *[Sparse1ConvBlock(
                dim, dim, norm_fn=self.norm1d_fn, indice_key=f'{indice_key}_selfmodeling_{i}', device=device
            ) for i in range(num_selfmodeling_layer)]
        )
        # num_selfmodeling_layer = 2
        # self_modeling_blocks = []
        # for i in range(num_selfmodeling_layer):
        #     self_modeling_blocks.append(
        #         Sparse1ConvBlock(dim, dim, norm_fn=self.norm1d_fn, indice_key=f'{indice_key}_selfmodeling_{i}'))
        # self.self_modeling_blocks = spconv.SparseSequential(*self_modeling_blocks)

        factory_kwargs = {'device': device, 'dtype':dtype}
        self.mamba_forward = create_block(
                    d_model=dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=num_lvl,
                    z_residual=True,
                    **factory_kwargs,)
        self.ssm_forward_norm = self.layernorm_fn(dim)


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

 


    def forward(self,
                x,
                pos_embed,
                scan_primary_axis='x',
                num_stage=0,
                debug=False):

        x = self.self_modeling_blocks(x)

        x = self.single_glocal_scan_forward(
            x=x, 
            pos_embed=pos_embed, 
            scan_primary_axis=scan_primary_axis,
            num_stage=num_stage,
            debug=debug
        )

        return x


    def single_glocal_scan_forward(
            self,
            x,
            pos_embed,
            scan_primary_axis='x',
            num_stage=0,
            flip=False,
            debug=False
        ):
        
        feats = x.features.clone()
        coords = x.indices.clone()
        curt_spatial_shape = x.spatial_shape
        batch_size = x.batch_size

        mamba_layer = self.mamba_forward
        mamba_norm = self.ssm_forward_norm

        # Pos Embedding
        if self.window_pos_embed == True:
            pos_coords = torch.zeros([coords.shape[0], 9], device=coords.device, dtype=torch.float32)
            pos_coords[:, 0] = coords[:, 1] / curt_spatial_shape[0]
            pos_coords[:, 1:3] = (coords[:, 2:] // 12) / (curt_spatial_shape[1]//12 + 1)
            pos_coords[:, 3:5] = (coords[:, 2:] % 12) / 12.0
            pos_coords[:, 5:7] = ((coords[:, 2:] + 6) // 12) / (curt_spatial_shape[1]//12 + 1)
            pos_coords[:, 7:9] = ((coords[:, 2:] + 6) % 12) / 12.0
            pos_embedding = pos_embed(pos_coords.float())
            feats = feats + pos_embedding # [N, 256]
        elif self.window_pos_embed == False:
            pos_embedding = pos_embed(x.indices[:, 1:].float())
            feats = feats + pos_embedding # [N, 256]

        if self.space_filing_curve == 'z_order' or self.space_filing_curve == 'hierarchical_z_order':
            index_info = get_morton_index_3d(
                coords, batch_size, self.sparse_shape,
                shift=(num_stage, num_stage, num_stage), primary_axis=scan_primary_axis
            )
        elif self.space_filing_curve == 'hilbert':
            index_info = get_hilbert_index_3d_mamba_lite(
                self.curve_template['curve_template_rank9'], 
                coords, batch_size, self.sparse_shape[0],
                self.hilbert_spatial_size['curve_template_rank9'],
                shift=(num_stage, num_stage, num_stage)
            )
        elif self.space_filing_curve == 'random':
            index_info = get_random_index_3d(
                coords, batch_size, seed=42
            )
        inds_curt_to_next = index_info['inds_curt_to_next'] # 'dict' 
        inds_next_to_curt = index_info['inds_next_to_curt']

        # out_feats_3d = feats.clone()

        # TODO 07.16 for bs problem: 应当用 zero_likes 接收最终特征, 而不是列表接收再 cat
        # out_feats_list = []
        # for i in range(batch_size):
        #     b_mask_m = (coords[:, 0] == i)
        #     feats_b = feats[b_mask_m]
        #     feats_b_m = feats_b[inds_curt_to_next[i]][None] # feats_i_m: [1, N, dim]
        #     out_feats_b_m = mamba_layer(feats_b_m, None) # out_feat_bi_m[0]: [1, N, dim])
        #     out_feats_b = out_feats_b_m[0].squeeze(0)[inds_next_to_curt[i]] # [N, dim]
        #     # out_feats_3d[b_mask_m] = out_feats_b
        #     out_feats_list.append(out_feats_b)
        # out_feats_3d = torch.cat(out_feats_list, dim=0)
        out_feats_3d = torch.zeros_like(feats)
        for i in range(batch_size):
            b_mask_m = (coords[:, 0] == i)
            feats_b = feats[b_mask_m]
            feats_b_m = feats_b[inds_curt_to_next[i]][None] # feats_i_m: [1, N, dim]
            if flip == True:
                feats_back = feats_b_m.flip(1)
                out_feats_b_m = mamba_layer(feats_back, None) # out_feat_bi_m[0]: [1, N, dim])
                out_feats_3d[b_mask_m] = out_feats_b_m[0].squeeze(0).flip(0)[inds_next_to_curt[i]].to(feats.dtype) # [N, dim]
            elif flip == False:
                out_feats_b_m = mamba_layer(feats_b_m, None) # out_feat_bi_m[0]: [1, N, dim])
                out_feats_3d[b_mask_m] = out_feats_b_m[0].squeeze(0)[inds_next_to_curt[i]].to(feats.dtype) # [N, dim]

        out_feats_3d = mamba_norm(out_feats_3d)
        x = replace_feature(x, out_feats_3d)

        return x





class Bi_GSSM_v5(nn.Module):
    def __init__(self,
                 dim,
                 num_stage,
                 num_block,
                 ssm_idx,
                 num_lvl,
                 sparse_shape,
                 norm_epsilon, 
                 rms_norm,
                 force_layernorm,
                 space_filing_curve,
                 ssm_cfg=None,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 window_pos_embed=False,
                 curve_template=None, 
                 hilbert_spatial_size=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()

        # self.dim = dim
        # self.norm2d_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.layernorm_fn = partial(nn.LayerNorm, eps=1e-5)
        self.norm1d_fn = self.layernorm_fn if force_layernorm == True else partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = sparse_shape

        self.window_pos_embed = window_pos_embed
        self.space_filing_curve = space_filing_curve
        self.curve_template=curve_template
        self.hilbert_spatial_size=hilbert_spatial_size

        indice_key = f"stage_{num_stage}_{num_block}_{ssm_idx}"
        num_selfmodeling_layer = 2
        self.self_modeling_blocks = spconv.SparseSequential(
            *[Sparse1ConvBlock(
                dim, dim, norm_fn=self.norm1d_fn, indice_key=f'{indice_key}_selfmodeling_{i}', device=device
            ) for i in range(num_selfmodeling_layer)]
        )
        # num_selfmodeling_layer = 2
        # self_modeling_blocks = []
        # for i in range(num_selfmodeling_layer):
        #     self_modeling_blocks.append(
        #         Sparse1ConvBlock(dim, dim, norm_fn=self.norm1d_fn, indice_key=f'{indice_key}_selfmodeling_{i}'))
        # self.self_modeling_blocks = spconv.SparseSequential(*self_modeling_blocks)

        factory_kwargs = {'device': device, 'dtype':dtype}
        self.mamba_forward = create_block(
                    d_model=dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=num_lvl,
                    z_residual=True,
                    **factory_kwargs,)
        self.mamba_backward = create_block(
                    d_model=dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=num_lvl,
                    z_residual=True,
                    **factory_kwargs,)
        self.ssm_forward_norm = self.layernorm_fn(dim)
        self.ssm_backward_norm = self.layernorm_fn(dim)
        self.fusion_norm = self.layernorm_fn(dim)


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

 


    def forward(self,
                x,
                pos_embed,
                scan_primary_axis='x',
                num_stage=0,
                debug=False):

        x_list = []
        for conv in self.self_modeling_blocks:
            x = conv(x)
            x_list.append(x)

        x_forward = self.single_glocal_scan_forward(
            x=x_list[0], 
            pos_embed=pos_embed, 
            scan_primary_axis=scan_primary_axis,
            num_stage=num_stage,
            flip=False,
            debug=debug
        )

        x_backward = self.single_glocal_scan_forward(
            x=x_list[1], 
            pos_embed=pos_embed, 
            scan_primary_axis=scan_primary_axis,
            num_stage=num_stage,
            flip=True,
            debug=debug
        )

        out_feats = self.fusion_norm(x_forward.features + x_backward.features + x_list[0].features)
        x = x.replace_feature(out_feats)

        return x


    def single_glocal_scan_forward(
            self,
            x,
            # feats,
            # coords,
            # curt_spatial_shape,
            # batch_size,
            pos_embed,
            scan_primary_axis='x',
            num_stage=0,
            flip=False,
            debug=False
        ):
        
        feats = x.features.clone()
        coords = x.indices.clone()
        curt_spatial_shape = x.spatial_shape
        batch_size = x.batch_size

        if flip == False:
            mamba_layer = self.mamba_forward
            mamba_norm = self.ssm_forward_norm
        elif flip == True:
            mamba_layer = self.mamba_backward
            mamba_norm = self.ssm_backward_norm


        # Pos Embedding
        if self.window_pos_embed == True:
            pos_coords = torch.zeros([coords.shape[0], 9], device=coords.device, dtype=torch.float32)
            pos_coords[:, 0] = coords[:, 1] / curt_spatial_shape[0]
            pos_coords[:, 1:3] = (coords[:, 2:] // 12) / (curt_spatial_shape[1]//12 + 1)
            pos_coords[:, 3:5] = (coords[:, 2:] % 12) / 12.0
            pos_coords[:, 5:7] = ((coords[:, 2:] + 6) // 12) / (curt_spatial_shape[1]//12 + 1)
            pos_coords[:, 7:9] = ((coords[:, 2:] + 6) % 12) / 12.0
            pos_embedding = pos_embed(pos_coords.float())
            feats = feats + pos_embedding # [N, 256]
        elif self.window_pos_embed == False:
            pos_embedding = pos_embed(x.indices[:, 1:].float())
            feats = feats + pos_embedding # [N, 256]

        if self.space_filing_curve == 'z_order' or self.space_filing_curve == 'hierarchical_z_order':
            index_info = get_morton_index_3d(
                coords, batch_size, self.sparse_shape,
                shift=(num_stage, num_stage, num_stage), primary_axis=scan_primary_axis
            )
        elif self.space_filing_curve == 'hilbert':
            index_info = get_hilbert_index_3d_mamba_lite(
                self.curve_template['curve_template_rank9'], 
                coords, batch_size, self.sparse_shape[0],
                self.hilbert_spatial_size['curve_template_rank9'],
                shift=(num_stage, num_stage, num_stage)
            )
        elif self.space_filing_curve == 'random':
            index_info = get_random_index_3d(
                coords, batch_size, seed=42
            )
        inds_curt_to_next = index_info['inds_curt_to_next'] # 'dict' 
        inds_next_to_curt = index_info['inds_next_to_curt']

        # out_feats_3d = feats.clone()

        # TODO 07.16 for bs problem: 应当用 zero_likes 接收最终特征, 而不是列表接收再 cat
        # out_feats_list = []
        # for i in range(batch_size):
        #     b_mask_m = (coords[:, 0] == i)
        #     feats_b = feats[b_mask_m]
        #     feats_b_m = feats_b[inds_curt_to_next[i]][None] # feats_i_m: [1, N, dim]
        #     out_feats_b_m = mamba_layer(feats_b_m, None) # out_feat_bi_m[0]: [1, N, dim])
        #     out_feats_b = out_feats_b_m[0].squeeze(0)[inds_next_to_curt[i]] # [N, dim]
        #     # out_feats_3d[b_mask_m] = out_feats_b
        #     out_feats_list.append(out_feats_b)
        # out_feats_3d = torch.cat(out_feats_list, dim=0)
        out_feats_3d = torch.zeros_like(feats)
        for i in range(batch_size):
            b_mask_m = (coords[:, 0] == i)
            feats_b = feats[b_mask_m]
            feats_b_m = feats_b[inds_curt_to_next[i]][None] # feats_i_m: [1, N, dim]
            if flip == True:
                feats_back = feats_b_m.flip(1)
                out_feats_b_m = mamba_layer(feats_back, None) # out_feat_bi_m[0]: [1, N, dim])
                out_feats_3d[b_mask_m] = out_feats_b_m[0].squeeze(0).flip(0)[inds_next_to_curt[i]].to(feats.dtype) # [N, dim]
            elif flip == False:
                out_feats_b_m = mamba_layer(feats_b_m, None) # out_feat_bi_m[0]: [1, N, dim])
                out_feats_3d[b_mask_m] = out_feats_b_m[0].squeeze(0)[inds_next_to_curt[i]].to(feats.dtype) # [N, dim]


        out_feats_3d = mamba_norm(out_feats_3d)
        x = replace_feature(x, out_feats_3d)

        return x


