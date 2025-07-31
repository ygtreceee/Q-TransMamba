import math
import copy
import csv
import numpy as np
from math import sqrt
from functools import partial
import scipy as sp
import torch
import torch_scatter
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn import functional as F 
from torch.nn.modules.utils import _pair, _triple
from torch.nn.utils.parametrize import register_parametrization
from collections import namedtuple, defaultdict
from typing import List, Tuple, Dict, Any, Union, Optional, Iterable
from einops import rearrange

from pcdet.utils.spconv_utils import replace_feature, spconv

norm1d_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
norm2d_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
layernorm_fn = partial(nn.LayerNorm, eps=1e-5)


############################### Activation Functions ################################


class NeLU(nn.Module):
    def __init__(self, alpha=0.2):
        """
        Initialize NeLU activation function
        Args:
            alpha: Hyperparameter for negative region (default=0.2)
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass of NeLU activation function
        Formula:
          x > 0  : f(x) = x
          x <= 0 : f(x) = -alpha / (1 + x^2)
        """
        return torch.where(x > 0, x, -self.alpha / (1 + torch.pow(x, 2)))
    

class TeLU(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.tanh( torch.exp(input) )
    

class Logish(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.log( 1 + torch.sigmoid(input) )
    

class Smish(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.tanh( torch.log( 1 + torch.sigmoid(input) ) )


####################################################################################

class SimpleFFN(nn.Module):
    '''
        Post-Norm
        (but Pre-Norm in UniMamba)
    '''
    def __init__(
            self,
            dim,
            ffn_rate=4,
            norm_fn=nn.LayerNorm,
            residual=True, # or False
            residual_in_fp32=False,
            pre_norm=False,
            act_type='gelu',
            device=None,
            **kwargs
        ):
        super().__init__()
        assert act_type in ['gelu', 'relu']
        self.dim = dim
        self.residual = residual if residual is not None else False
        self.residual_in_fp32 = residual_in_fp32
        self.pre_norm = pre_norm
        self.hidden_dim = int(ffn_rate * dim)
        self.fc1 = nn.Linear(dim, self.hidden_dim, device=device)
        self.act_func = nn.ReLU() if act_type == 'relu' else nn.GELU()
        # self.act_fn = nn.ReLU() if act_type == 'relu' else nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, dim, device=device)
        self.norm_out = norm_fn(dim, device=device) if norm_fn is not None else nn.Identity()

    def forward(self, x):
        ori_dtype = x.dtype
        original_shape = x.shape

        if x.dim() == 3: # 合并前两个维度 (batch_size * N, channels)
            x = x.view(-1, x.size(-1))

        residual = x
        x = self.fc2(self.act_func(self.fc1(x)))
        # x = self.fc2(self.act_fn(self.fc1(x)))
        if self.residual == True:
            if self.pre_norm == True:
                if ori_dtype in [torch.float16, torch.bfloat16] and self.residual_in_fp32 == True:
                    return (self.norm_out(x).to(dtype=torch.float32) + residual.to(dtype=torch.float32)).to(dtype=ori_dtype)
                return self.norm_out(x) + residual
            if ori_dtype in [torch.float16, torch.bfloat16] and self.residual_in_fp32 == True:
                x = x.to(dtype=torch.float32) + residual.to(dtype=torch.float32)
            else:
                x = x + residual

        x = self.norm_out(x).to(dtype=ori_dtype)

        if len(original_shape) == 3: # 恢复原始形状
            x = x.view(original_shape[0], original_shape[1], -1)
        return x


class QueryFFN_M(nn.Module):
    '''
        Pre-Norm
    '''
    def __init__(
            self,
            dim,
            ffn_rate=4,
            norm_fn=nn.BatchNorm1d,
            dropout=0.1,
            act_type='relu',
            residual=True, # or False
            residual_in_fp32=False,
            device=None,
            dtype=None,
            **kwargs
        ):
        super().__init__()
        assert ffn_rate in [2, 4]
        assert act_type in ['relu', 'silu', 'gelu']

        self.dim = dim
        self.residual = residual if residual is not None else False
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dim = int(ffn_rate * dim)
        self.immediate_hidden_dim = self.hidden_dim if ffn_rate == 2 else self.hidden_dim // 2
        
        dropout_layer = nn.Dropout(dropout) if 0 < dropout < 1 and self.training else nn.Identity()
        self.layers = nn.Sequential(
            nn.Linear(dim, self.hidden_dim, device=device),
            norm_fn(self.hidden_dim, device=device),
            nn.SiLU() if act_type == 'silu' else (nn.GELU() if act_type == 'gelu' else nn.ReLU()),
            dropout_layer,

            nn.Linear(self.hidden_dim, self.immediate_hidden_dim, device=device),
            norm_fn(self.immediate_hidden_dim, device=device),
            nn.SiLU() if act_type == 'silu' else (nn.GELU() if act_type == 'gelu' else nn.ReLU()),
            
            nn.Linear(self.immediate_hidden_dim, dim, device=device),
            norm_fn(dim, device=device),
        )
        self.post_act = nn.SiLU() if act_type == 'silu' else (nn.GELU() if act_type == 'gelu' else nn.ReLU())
        self.layers[0].indice_key = "fc1"
        self.layers[2].indice_key = "act"
        self.layers[4].indice_key = "fc2"
        self.layers[7].indice_key = "out_proj"

    def forward(self, x):
        ori_dtype = x.dtype
        original_shape = x.shape
        if x.dim() == 3: # 合并前两个维度 (batch_size * N, channels)
            x = x.view(-1, x.size(-1))

        residual = x
        x = self.layers(x)
        if self.residual == True:
            if ori_dtype in [torch.float16, torch.bfloat16] and self.residual_in_fp32 == True:
                x = x.to(dtype=torch.float32) + residual.to(dtype=torch.float32)
            else:
                x = x + residual
        x = self.post_act(x).to(dtype=ori_dtype)

        if len(original_shape) == 3: # 恢复原始形状
            x = x.view(original_shape[0], original_shape[1], -1)
        return x


class ResMLP(nn.Module):
    def __init__(
            self, 
            dim,
            norm1d_fn,
            device=None,
            dtype=None,
        ):
        super().__init__()

        self.dim = dim
        self.hidden_state = 2 * dim

        self.fc = nn.Sequential(
            nn.Linear(dim, self.hidden_state),
            nn.LayerNorm(self.hidden_state),
            nn.ReLU(),
            nn.Linear(self.hidden_state, dim),
            nn.LayerNorm(dim),
        )

        self.post_act = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = x + residual
        return self.post_act(x)


class AnchorFormer(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.scale = channels ** -0.5  # 1/√d

    def forward(self, query, content, residual=None):
        """
        query: [M, C] (as Q)
        content: [N, C] (as K,V)
        """
        # 1. 计算内容特征与anchor的相似度
        attn_logits = torch.matmul(content, query.t()) * self.scale  # [N, m]
        A = torch.softmax(attn_logits, dim=-1)
        
        # 2. 构建转移矩阵Δ
        delta = torch.diag_embed(A.sum(dim=0))  # [m, m]
        
        # 3. 马尔可夫过程计算全局注意力
        # 重排计算顺序避免O(N^2)
        M1 = torch.matmul(A.t(), content)  # [m, C]
        M2 = torch.linalg.solve(delta, M1)  # Δ^{-1}·M1
        global_attn = torch.matmul(A, M2)   # A·M2
        
        updated_query = global_attn
        # 4. 用全局注意力更新query
        if residual is not None:
            updated_query = query + residual
        return updated_query
    


class MLPDecoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=3, dropout_r=0.0, sigmoid=True):
        super().__init__()
        dim = in_channels
        dropout_layer = nn.Dropout(dropout_r) if 0 < dropout_r < 1 else nn.Identity()
        self.layers = nn.Sequential(
            # 第1层：特征扩展与非线性激活
            nn.Linear(dim, dim * 2),  # 扩展维度以增强表达能力
            nn.LayerNorm(dim * 2),          # 层归一化稳定训练
            nn.GELU(),                  # 平滑激活函数（优于ReLU）
            dropout_layer,            # 轻微正则化防过拟合
            
            # 第2层：特征压缩
            nn.Linear(dim * 2, dim),        # 逐步压缩维度
            nn.LayerNorm(dim),
            nn.GELU(),
            
            # 第3层：输出适配
            nn.Linear(dim, dim // 2),         # 进一步压缩至低维空间
            nn.GELU(),
            
            # 输出层：无激活函数
            nn.Linear(dim // 2, out_channels)   # 直接输出位置坐标
        )
        self.layers[4].indice_key = "fc2"
        self.layers[7].indice_key = "out_proj"
        self.modulation = nn.Sigmoid() if sigmoid else nn.Identity()
        
    def forward(self, x):
        return self.modulation(self.layers(x))



class PointEmbedding(nn.Module):
    def __init__(self, point_dim, out_dim):
        super().__init__() 
        self.stem = nn.Conv1d(point_dim, out_dim // 2, kernel_size=1)
        self.norm1 = norm1d_fn(out_dim // 2)
        self.act_fn = nn.ReLU()
        self.fc2 = nn.Linear(out_dim // 2, out_dim)
        self.norm2 = norm1d_fn(out_dim)
        self.act2 = nn.ReLU()
    def forward(self, x):
        # input  : [batch, L, point_dim]
        # output : [batch, L, out_dim]
        x = self.act_fn(self.norm1(self.stem(x.transpose(1, 2).contiguous())))
        x = self.fc2(x.transpose(1, 2).contiguous())
        x = self.act2(self.norm2(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous())
        return x



class XFormersCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_type: str = 'vanilla',
        proj_type: str = 'linear',
        res_norm_fn: Optional[nn.Module] = None,
        residual: bool = False,
        residual_in_fp32: bool = False,
        pre_norm: bool = False,
        dropout: float = 0.0,
        attn_cfg: Optional[Dict] = None,
        keyname: str = None,
    ):
        """
        重构的高效跨注意力模块，支持不同长度的Q和K
        
        新增:
            - 'flash': 最快的高效注意力实现 (需安装flash_attn库)
        
        参数:
            dim: 特征维度
            attn_type: 注意力类型 
                - 'vanilla': xFormers内存优化注意力 (高效)
                - 'flash': Flash Attention (当前最快实现，强烈推荐)
                - 'base': 基础实现 (稳定)
                - 'linear': 线性注意力 (近似)
                - 'low_rank': 低秩注意力 (近似)
                - 'window': 局部窗口注意力
                - 'dynamic': 动态稀疏注意力
            proj_type: 投影类型
            res_norm_fn: 残差连接规范化函数
            dropout: 注意力dropout率
            attn_cfg: 类型相关配置字典
        """
        super().__init__()
        self.dim = dim
        self.attn_type = attn_type
        self.proj_type = proj_type
        self.dropout = dropout
        self.residual = residual
        self.residual_in_fp32 = residual_in_fp32
        self.pre_norm = pre_norm
        self.keyname = keyname
        self.attn_cfg = attn_cfg or {}
        
        if self.proj_type == 'linear':
            self.q_proj = nn.Linear(dim, dim)    
            self.k_proj = nn.Linear(dim, dim)    
            self.v_proj = nn.Linear(dim, dim)    
        elif self.proj_type == 'conv1d':
            self.q_proj = nn.Conv1d(dim, dim, 1)
            self.k_proj = nn.Conv1d(dim, dim, 1)
            self.v_proj = nn.Conv1d(dim, dim, 1)
        else:
            self.q_proj = nn.Identity()
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()

        # 设置默认配置
        defaults = {
            'num_heads': 4, # base 方法头数

            'window_size': 32,
            'topk': 0.1, # 小数为动态比例，整数为预设固定个数
            'topk_q': 0.3,
            'topk_k': 0.2,
            'low_rank': max(4, dim // 4), # dim // 8
            'phi': 'relu',

            'chunk_size_q': -1, # 512 \ 1024 \ 2048 \ 4096
            'chunk_size_k': -1, # 4096

            # Flash Attention特定配置
            'flash_causal': False,
            'flash_num_heads': 4,  # 默认为单头注意力 1 \ 4
            'flash_backend': 'triton',
            'flash_window_size': (-1, -1),  # 滑动窗口大小分布表示左右延展每个q_token对K可见长度 (-1,-1) \ (256,256) \ (128, 128)
            'flash_softcap': 0.0,             # 平滑截断系数 0.0 \ 0.2
            'flash_use_alibi': False,           # 启用ALiBi位置偏置(搭配多头使用) False \ True TODO: FP16 报错
            'flash_deterministic': False,      # 训练时禁用确定性计算 False
            # 'return_softmax': True # return out if not return_softmax else (out, softmax_lse, S_dmask) 给 True 即可拿到
        }
        for k, v in defaults.items():
            if k not in self.attn_cfg:
                self.attn_cfg[k] = v
        
        # 支持的类型列表
        valid_types = ['vanilla', 'flash', 'base', 'linear', 'low_rank', 'window', 'dynamic']
        assert attn_type in valid_types, f"无效的注意力类型: {attn_type}，可选: {valid_types}"
        
        # 残差连接规范化
        self.res_norm_fn = res_norm_fn(self.dim) if res_norm_fn is not None and residual else nn.Identity()
        self.scale = 1.0 / sqrt(dim)  # 更精确的缩放系数
        
        # 特定类型初始化
        if attn_type == 'linear':
            # 线性注意力核函数
            phi_dict = {
                'relu': nn.ReLU(),
                'elu': nn.ELU(),
                'softplus': nn.Softplus(),
                'identity': nn.Identity()
            }
            self.phi = phi_dict.get(self.attn_cfg['phi'], phi_dict['relu'])
            
        elif attn_type == 'low_rank':
            # 低秩分解矩阵
            rank = min(self.attn_cfg['low_rank'], dim)
            self.U = nn.Linear(dim, rank, bias=False)
            self.V = nn.Linear(dim, rank, bias=False)
        
        # 对于高效注意力类型，检查依赖库可用性
        if attn_type == 'vanilla':
            self._check_xformers_available()
        elif attn_type == 'flash':
            self._setup_flash_attention()
        
        self.autocast_enabled = True

    def _check_xformers_available(self):
        """验证xFormers可用性"""
        try:
            import xformers.ops as xops # NOTE: pip install xformers==0.0.22.post7 (for torch=2.1.0, cuda=11.8)
            self.xops = xops
        except ImportError:
            raise ImportError("未安装xFormers库。若要使用'vanilla'类型，请先安装: pip install xformers")

    def _setup_flash_attention(self):
        """设置Flash Attention需要的配置"""
        try:
            import flash_attn
            from importlib.metadata import version
            
            # 获取Flash Attention版本
            flash_version = version('flash_attn') # NOTE: MAX_JOBS=64 pip install -v flash-attn==2.6.1 --no-build-isolation
            # print(f"检测到Flash Attention版本: {flash_version}")
            
            # 根据版本选择不同的导入方式
            if flash_version >= '2.0.0':
                from flash_attn.flash_attn_interface import flash_attn_func
                self.flash_attn = flash_attn_func
                self.flash_version = 2
            else:
                from flash_attn.flash_attn_triton import flash_attn_func
                self.flash_attn = flash_attn_func
                self.flash_version = 1
            
            # 验证头数配置
            self.num_heads = self.attn_cfg['flash_num_heads']
            assert self.dim % self.num_heads == 0, (
                f"特征维度({self.dim})必须能被头数({self.num_heads})整除"
            )
            self.head_dim = self.dim // self.num_heads
            
        except ImportError as e:
            raise ImportError(
                "未安装Flash Attention库。若要使用'flash'类型，请先安装:\n"
                "标准版: pip install flash-attn\n"
                "Triton后端: pip install flash-attn-triton"
            ) from e

    def _reshape_for_flash(self, tensor: torch.Tensor) -> torch.Tensor:
        """为Flash Attention重塑张量形状"""
        B, L, _ = tensor.shape
        # 重塑为 [batch_size, seqlen, num_heads, head_dim]
        return tensor.view(B, L, self.num_heads, self.head_dim).contiguous()

    # def _process_flash_chunk(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    #     """处理单个分块的Flash Attention核心逻辑"""
    #     # 保存原始数据类型
    #     orig_dtype = Q.dtype
        
    #     # 自动转换到半精度
    #     if orig_dtype not in (torch.float16, torch.bfloat16):
    #         target_dtype = torch.float16 if Q.device.type == 'cuda' else torch.bfloat16
    #         Q, K, V = Q.to(target_dtype), K.to(target_dtype), V.to(target_dtype)
        
    #     # 确保连续内存并重塑形状
    #     Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    #     Q_reshaped = self._reshape_for_flash(Q)
    #     K_reshaped = self._reshape_for_flash(K)
    #     V_reshaped = self._reshape_for_flash(V)
        
    #     # 准备Flash参数
    #     dropout_p = self.dropout if self.training else 0.0
    #     causal = self.attn_cfg['flash_causal']
        
    #     # 根据Flash版本调用不同参数顺序
    #     try:
    #         if self.flash_version >= 2:
    #             attn_output = self.flash_attn(
    #                 Q_reshaped, K_reshaped, V_reshaped,
    #                 dropout_p=dropout_p,
    #                 softmax_scale=self.scale,
    #                 causal=causal,
                    
    #             )
    #         else:
    #             # Flash v1使用位置参数
    #             attn_output = self.flash_attn(
    #                 Q_reshaped, K_reshaped, V_reshaped,
    #                 dropout_p,
    #                 self.scale,
    #                 causal
    #             )
    #     except TypeError:
    #         # 回退到最简调用
    #         attn_output = self.flash_attn(Q_reshaped, K_reshaped, V_reshaped)
        
    #     # 恢复形状和数据类型
    #     B, L_q, _, _ = Q_reshaped.shape
    #     output = attn_output.reshape(B, L_q, self.dim)
    #     return output.to(orig_dtype) if orig_dtype != output.dtype else output

    # def forward_flash_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    #     """优化的Flash Attention实现，支持chunk_size并提高计算效率"""
    #     assert K.size(1) == V.size(1), "V序列长度必须与K相同"
        
    #     # 保存原始数据类型和形状
    #     # orig_dtype = Q.dtype
    #     B, L_q, C = Q.shape
    #     L_k = K.size(1)
        
    #     # 获取chunk_size配置
    #     chunk_size_q = self.attn_cfg['chunk_size_q']
    #     chunk_size_k = self.attn_cfg['chunk_size_k']
        
    #     # 确定是否启用分块策略
    #     use_chunking = (chunk_size_q > 0 and L_q > chunk_size_q) or (chunk_size_k > 0 and L_k > chunk_size_k)
        
    #     # 分块处理策略
    #     if use_chunking:
    #         # 实际分块大小（确保不超边界）
    #         actual_chunk_size_q = min(chunk_size_q, L_q) if chunk_size_q > 0 else L_q
    #         actual_chunk_size_k = min(chunk_size_k, L_k) if chunk_size_k > 0 else L_k
            
    #         # 收集分块结果
    #         outputs = []
    #         for i in range(0, L_q, actual_chunk_size_q):
    #             q_start, q_end = i, min(i + actual_chunk_size_q, L_q)
    #             Q_chunk = Q[:, q_start:q_end, :]
                
    #             # 选择相关的K/V块
    #             k_start = max(0, i - actual_chunk_size_k // 2)
    #             k_end = min(L_k, i + actual_chunk_size_k)
    #             K_chunk = K[:, k_start:k_end, :]
    #             V_chunk = V[:, k_start:k_end, :]
                
    #             # 处理单个分块
    #             chunk_output = self._process_flash_chunk(Q_chunk, K_chunk, V_chunk)
    #             outputs.append(chunk_output)
                
    #         # 合并所有输出
    #         return torch.cat(outputs, dim=1)
    #     else:
    #         # 不分块，处理整个序列
    #         return self._process_flash_chunk(Q, K, V)

    def _process_flash_chunk(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """处理单个分块的Flash Attention核心逻辑"""
        # 保存原始数据类型
        orig_dtype = Q.dtype
        self.device = Q.device
        
        # 自动转换到半精度
        if orig_dtype not in (torch.float16, torch.bfloat16):
            target_dtype = torch.float16 if Q.device.type == 'cuda' else torch.bfloat16
            Q, K, V = Q.to(target_dtype), K.to(target_dtype), V.to(target_dtype)
        else:
            target_dtype = orig_dtype
        self.dtype = target_dtype

        # 确保连续内存并重塑形状
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        Q_reshaped = self._reshape_for_flash(Q)
        K_reshaped = self._reshape_for_flash(K)
        V_reshaped = self._reshape_for_flash(V)
        
        # 准备Flash参数（使用2.7.0版本的高级参数）
        params = {
            'dropout_p': self.dropout if self.training else 0.0,
            'softmax_scale': self.scale,
            'causal': self.attn_cfg['flash_causal'],
            'window_size': tuple(self.attn_cfg.get('flash_window_size', (-1, -1))),
            'softcap': self.attn_cfg.get('flash_softcap', 0.0),
            'alibi_slopes': self._generate_alibi_slopes(),
            'deterministic': self.attn_cfg.get('flash_deterministic', False),
            'return_attn_probs': False
        }
        
        # 调用Flash Attention
        try:
            # print(f"Q={Q_reshaped.dtype}, K={K_reshaped.dtype}, V={V_reshaped.dtype}")
            # print(f"Q={Q_reshaped.shape}, K={K_reshaped.shape}, V={V_reshaped.shape}")
            attn_output = self.flash_attn(
                Q_reshaped, K_reshaped, V_reshaped,
                **params
            )
            # print(f"attn={attn_output.dtype}")
        except TypeError as e:
            print(f"Flash高级调用失败: {e}, 尝试最简模式")
            # 回退到最简调用
            attn_output = self.flash_attn(Q_reshaped, K_reshaped, V_reshaped)
        
        # 恢复形状和数据类型
        B, L_q, _, _ = Q_reshaped.shape
        output = attn_output.reshape(B, L_q, self.dim)
        return output.to(orig_dtype) if orig_dtype != output.dtype else output

    def _generate_alibi_slopes(self) -> Optional[torch.Tensor]:
        """生成ALiBi位置偏置斜率"""
        if not self.attn_cfg.get('flash_use_alibi', False):
            return None
        # print(self.attn_cfg.get('flash_use_alibi', False))

        # 为每个注意力头生成唯一的斜率
        num_heads = self.attn_cfg['flash_num_heads']
        # assert num_heads > 0
        slopes = torch.tensor([1 / (2 ** (i / num_heads)) for i in range(1, num_heads + 1)], device=self.device)
        return slopes

    def forward_flash_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """优化的Flash Attention实现，支持高级参数和chunk_size分块策略"""
        assert K.size(1) == V.size(1), "V序列长度必须与K相同"
        
        # 保存原始数据类型和形状
        B, L_q, C = Q.shape
        L_k = K.size(1)
        
        # 获取chunk_size配置
        chunk_size_q = self.attn_cfg['chunk_size_q']
        chunk_size_k = self.attn_cfg['chunk_size_k']
        
        # 确定是否启用分块策略
        use_chunking = (chunk_size_q > 0 and L_q > chunk_size_q) or (chunk_size_k > 0 and L_k > chunk_size_k)
        
        # 分块处理策略
        if use_chunking:
            # 实际分块大小（确保不超边界）
            actual_chunk_size_q = min(chunk_size_q, L_q) if chunk_size_q > 0 else L_q
            actual_chunk_size_k = min(chunk_size_k, L_k) if chunk_size_k > 0 else L_k
            
            # 收集分块结果
            outputs = []
            for i in range(0, L_q, actual_chunk_size_q):
                q_start, q_end = i, min(i + actual_chunk_size_q, L_q)
                Q_chunk = Q[:, q_start:q_end, :]
                
                # 选择相关的K/V块（带上下文重叠）
                k_start = max(0, i - actual_chunk_size_k // 2)
                k_end = min(L_k, i + actual_chunk_size_k)
                K_chunk = K[:, k_start:k_end, :]
                V_chunk = V[:, k_start:k_end, :]
                
                # 处理单个分块
                chunk_output = self._process_flash_chunk(Q_chunk, K_chunk, V_chunk)
                outputs.append(chunk_output)
                
            # 合并所有输出
            return torch.cat(outputs, dim=1)
        else:
            # 不分块，处理整个序列
            return self._process_flash_chunk(Q, K, V)

    def forward_vanilla_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """使用xFormers内存优化注意力处理不同长度的Q和K"""
        # 确保V的序列长度与K一致
        assert K.size(1) == V.size(1), (
            f"V的序列长度必须与K相同! "
            f"K长度: {K.size(1)}, V长度: {V.size(1)}"
        )
        with torch.cuda.amp.autocast(enabled=self.autocast_enabled):
            Q = Q.contiguous()
            K = K.contiguous()
            V = V.contiguous()
            return self.xops.memory_efficient_attention(
                Q, K, V,
                scale=self.scale,
                p=self.dropout if self.training else 0.0,
                attn_bias=self.attn_cfg.get('attn_bias', None)
            )

    # 原始无多头实现
    # def forward_base_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    #     """基础注意力实现，高效处理不同长度的Q和K"""
    #     # 确保V的序列长度与K一致
    #     assert K.size(1) == V.size(1), (
    #         f"V的序列长度必须与K相同! "
    #         f"K长度: {K.size(1)}, V长度: {V.size(1)}"
    #     )
        
    #     # 计算注意力分数
    #     attn_logits = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * self.scale
        
    #     # 应用softmax
    #     attn_weights = F.softmax(attn_logits, dim=-1)
        
    #     # 应用dropout
    #     if self.dropout > 0.0 and self.training:
    #         attn_weights = F.dropout(attn_weights, p=self.dropout)
        
    #     # 加权求和值
    #     return torch.matmul(attn_weights, V)

    def forward_base_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """基础注意力实现，支持多头注意力"""
        assert K.size(1) == V.size(1), (
            f"V的序列长度必须与K相同! "
            f"K长度: {K.size(1)}, V长度: {V.size(1)}"
        )
        
        num_heads = self.attn_cfg.get('num_heads', 1) # 获取多头配置
        head_dim = self.dim // num_heads
        assert self.dim % num_heads == 0, (
            f"特征维度({self.dim})必须能被头数({num_heads})整除"
        )
        
        # 重塑输入为多头格式 [B, L, num_heads, head_dim]
        Q = Q.view(Q.size(0), Q.size(1), num_heads, head_dim)
        K = K.view(K.size(0), K.size(1), num_heads, head_dim)
        V = V.view(V.size(0), V.size(1), num_heads, head_dim)
        
        # 转置以准备矩阵乘法 [B, num_heads, L, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算注意力分数 [B, num_heads, L_q, L_k]
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用softmax
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # 应用dropout
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # 加权求和值 [B, num_heads, L_q, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
        # 转置回原始形状 [B, L_q, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        # 合并多头 [B, L_q, dim]
        return attn_output.contiguous().view(attn_output.size(0), attn_output.size(1), -1)
    
    def forward_linear_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """带分块的线性注意力实现"""
        assert K.size(1) == V.size(1), "K和V序列长度必须相同"
        
        B, Lq, C = Q.shape
        Lk = K.size(1)
        
        # 获取配置参数
        chunk_size_q = self.attn_cfg['chunk_size_q']
        chunk_size_k = self.attn_cfg['chunk_size_k']
        
        # 确保分块大小有效
        chunk_size_q = min(chunk_size_q, Lq) if chunk_size_q > 0 else Lq
        chunk_size_k = min(chunk_size_k, Lk) if chunk_size_k > 0 else Lk
        
        # 初始化KV聚合张量
        kv = torch.zeros(B, C, V.size(-1), device=Q.device, dtype=Q.dtype)
        
        # ================================
        # 阶段1: KV聚合 (K和V分块处理)
        # ================================
        if chunk_size_k >= Lk:  # 无需分块
            K_chunk = self.phi(K)
            kv += torch.einsum('bkc,bkv->bcv', K_chunk, V)
        else:  # KV分块处理
            for i in range(0, Lk, chunk_size_k):
                # 计算当前分块的起始和结束位置
                k_start, k_end = i, min(i + chunk_size_k, Lk)
                
                K_chunk = K[:, k_start:k_end, :]
                V_chunk = V[:, k_start:k_end, :]
                
                # 特征变换和KV聚合
                K_chunk = self.phi(K_chunk)
                kv += torch.einsum('bkc,bkv->bcv', K_chunk, V_chunk)
        
        # ================================
        # 阶段2: 计算输出 (Q分块处理)
        # ================================
        outputs = []
        
        if chunk_size_q >= Lq:  # 无需分块
            Q_chunk = self.phi(Q)
            output_chunk = torch.einsum('bqc,bcv->bqv', Q_chunk, kv)
            outputs.append(output_chunk * self.scale)
        else:  # Q分块处理
            for j in range(0, Lq, chunk_size_q):
                # 计算当前分块的起始和结束位置
                q_start, q_end = j, min(j + chunk_size_q, Lq)
                
                Q_chunk = Q[:, q_start:q_end, :]
                
                # 特征变换和输出计算
                Q_chunk = self.phi(Q_chunk)
                output_chunk = torch.einsum('bqc,bcv->bqv', Q_chunk, kv)
                outputs.append(output_chunk * self.scale)
        
        # 合并所有输出分块
        return torch.cat(outputs, dim=1)

    def forward_low_rank_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """带分块的低秩注意力实现"""
        assert K.size(1) == V.size(1), "K和V序列长度必须相同"
        
        B, Lq, C = Q.shape
        Lk = K.size(1)
        
        # 获取配置参数
        chunk_size_q = self.attn_cfg['chunk_size_q']
        chunk_size_k = self.attn_cfg['chunk_size_k']
        
        # 确保分块大小有效
        chunk_size_q = min(chunk_size_q, Lq) if chunk_size_q > 0 else Lq
        chunk_size_k = min(chunk_size_k, Lk) if chunk_size_k > 0 else Lk
        
        # 初始化KV聚合张量
        kv = torch.zeros(B, self.U.out_features, V.size(-1), 
                         device=Q.device, dtype=Q.dtype)
        
        # ================================
        # 阶段1: KV聚合 (K和V分块处理)
        # ================================
        if chunk_size_k >= Lk:  # 无需分块
            UK_chunk = self.U(K)
            kv += torch.einsum('bkc,bkv->bcv', UK_chunk, V)
        else:  # KV分块处理
            for i in range(0, Lk, chunk_size_k):
                # 计算当前分块的起始和结束位置
                k_start, k_end = i, min(i + chunk_size_k, Lk)
                
                K_chunk = K[:, k_start:k_end, :]
                V_chunk = V[:, k_start:k_end, :]
                
                # 低秩投影和KV聚合
                UK_chunk = self.U(K_chunk)
                kv += torch.einsum('bkc,bkv->bcv', UK_chunk, V_chunk)
        
        # ================================
        # 阶段2: 计算输出 (Q分块处理)
        # ================================
        outputs = []
        
        if chunk_size_q >= Lq:  # 无需分块
            UQ_chunk = self.U(Q)
            output_chunk = torch.einsum('bqc,bcv->bqv', UQ_chunk, kv)
            outputs.append(output_chunk * self.scale)
        else:  # Q分块处理
            for j in range(0, Lq, chunk_size_q):
                # 计算当前分块的起始和结束位置
                q_start, q_end = j, min(j + chunk_size_q, Lq)
                
                Q_chunk = Q[:, q_start:q_end, :]
                
                # 低秩投影和输出计算
                UQ_chunk = self.U(Q_chunk)
                output_chunk = torch.einsum('bqc,bcv->bqv', UQ_chunk, kv)
                outputs.append(output_chunk * self.scale)
        
        # 合并所有输出分块
        return torch.cat(outputs, dim=1)    
    
    def forward_window_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """局部窗口注意力处理不同长度的Q和K"""
        # 确保V的序列长度与K一致
        assert K.size(1) == V.size(1), (
            f"V的序列长度必须与K相同! "
            f"K长度: {K.size(1)}, V长度: {V.size(1)}"
        )
        
        B, Lq, C = Q.shape
        Lk = K.size(1)
        w = self.attn_cfg['window_size']
        
        # 对Q和K进行填充以适应窗口
        pad_len_q = (w - Lq % w) % w
        pad_len_k = (w - Lk % w) % w
        
        Q_pad = F.pad(Q, (0, 0, 0, pad_len_q))
        K_pad = F.pad(K, (0, 0, 0, pad_len_k))
        V_pad = F.pad(V, (0, 0, 0, pad_len_k))
        
        # 窗口分割和注意力计算
        Q_win = rearrange(Q_pad, 'b (w n) c -> b w n c', w=(Lq + pad_len_q) // w)
        K_win = rearrange(K_pad, 'b (w n) c -> b w n c', w=(Lk + pad_len_k) // w)
        V_win = rearrange(V_pad, 'b (w n) c -> b w n c', w=(Lk + pad_len_k) // w)
        
        # 计算注意力
        attn_logits = torch.einsum('bwik,bwjk->bwij', Q_win, K_win) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_output = torch.einsum('bwij,bwjk->bwik', attn_weights, V_win)
        
        # 恢复原始形状
        attn_output = rearrange(attn_output, 'b w n c -> b (w n) c')
        return attn_output[:, :Lq, :]

    def forward_dynamic_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """动态稀疏注意力处理不同长度的Q和K"""
        # 确保V的序列长度与K一致
        assert K.size(1) == V.size(1), (
            f"V的序列长度必须与K相同! "
            f"K长度: {K.size(1)}, V长度: {V.size(1)}"
        )
        
        B, Lq, C = Q.shape
        Lk = K.size(1)
        # topk = min(self.attn_cfg['topk'], Lk)
        topk_val = self.attn_cfg['topk']
        if 0 < topk_val < 1:
            topk = int(topk_val * Lk)
        else:
            topk = int(min(topk_val, Lk))
        
        # 计算稀疏注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * self.scale
        
        # 选择top-k键值对
        topk_scores, topk_indices = torch.topk(scores, topk, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)
        
        # 收集对应的值向量
        batch_idx = torch.arange(B, device=Q.device)[:, None, None]
        topk_V = V[batch_idx, topk_indices, :]
        
        # 加权求和
        return torch.einsum('bqk,bqkc->bqc', topk_weights, topk_V)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            Q: 查询张量 [B, L_q, C] (序列长度L_q)
            K: 键张量 [B, L_k, C] (序列长度L_k)
            V: 值张量 [B, L_v, C] (必须L_v == L_k)
            residual: 残差连接张量 [B, L_q, C]
            
        返回:
            注意力输出张量 [B, L_q, C]
        """
        # 验证输入维度
        assert Q.dim() == 3, f"Q应为3D张量，实际为{Q.dim()}D"
        assert K.dim() == 3, f"K应为3D张量，实际为{K.dim()}D"
        assert V.dim() == 3, f"V应为3D张量，实际为{V.dim()}D"
        assert Q.size(0) == K.size(0) == V.size(0), "批次大小不一致"
        assert Q.size(2) == K.size(2) == V.size(2), "特征维度不一致"
        assert K.size(1) == V.size(1), "K和V序列长度必须相同"
        # assert (self.residual == True and residual is not None) or (self.residual == False and residual is None)
        if residual is None and self.residual == True:
            residual = Q.clone()
            if self.attn_type == 'flash':
                residual = residual.to(dtype=torch.float16 if Q.device.type == 'cuda' else torch.bfloat16)

        if self.proj_type == 'conv1d':
            Q = self.q_proj(Q.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            K = self.k_proj(K.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            V = self.v_proj(V.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        else:
            Q = self.q_proj(Q)
            K = self.k_proj(K)
            V = self.v_proj(V)

        # 根据注意力类型选择实现
        if self.attn_type == 'vanilla':
            attn_output = self.forward_vanilla_attention(Q, K, V)
        elif self.attn_type == 'flash':  # 新增Flash Attention分支
            attn_output = self.forward_flash_attention(Q, K, V)
        elif self.attn_type == 'base':
            attn_output = self.forward_base_attention(Q, K, V)
        elif self.attn_type == 'linear':
            attn_output = self.forward_linear_attention(Q, K, V)
        elif self.attn_type == 'low_rank':
            attn_output = self.forward_low_rank_attention(Q, K, V)
        elif self.attn_type == 'window':
            attn_output = self.forward_window_attention(Q, K, V)
        elif self.attn_type == 'dynamic':
            attn_output = self.forward_dynamic_attention(Q, K, V)
        

        # # ===== 添加测试代码开始 =====
        # # 创建与 attn_output 相同形状的布尔掩码
        # nan_mask_test = torch.zeros_like(attn_output, dtype=torch.bool)
        # if nan_mask_test.numel() > 0:  # 确保张量不为空
        #     nan_mask_test[0, 0, 0] = True  # 设置第一个位置为 True
            
        # # 生成包含 NaN 的张量
        # nan_tensor = torch.tensor([float('nan')], device=attn_output.device)
        # attn_output = torch.where(
        #     nan_mask_test,
        #     nan_tensor.expand_as(attn_output),  # 将单一 NaN 扩展到相同形状
        #     attn_output
        # )
        # if residual is not None:
        #     nan_tensor_2 = torch.tensor([float('nan')], device=residual.device)
        #     residual = torch.where(
        #         nan_mask_test,
        #         nan_tensor_2.expand_as(residual),  # 将单一 NaN 扩展到相同形状
        #         residual
        #     )
        # print("TEST: Injected NaN into attn_output for debugging")
        # # ===== 添加测试代码结束 =====

        # 确保所有值都有效
        nan_mask1 = torch.isnan(attn_output) # | torch.isinf(residual)
        if self.training and nan_mask1.any():
            print(f"Warning: attn_output got NaN values at {self.keyname}!") # 替换无效值为0并添加微小噪声
            attn_output = torch.nan_to_num(attn_output, nan=0.0)
            noise = torch.randn_like(attn_output, device=attn_output.device, dtype=attn_output.dtype) * 1e-8
            attn_output = attn_output + noise
            
        # 残差连接
        if residual is not None:
            # 确保残差维度匹配
            if residual.size() != Q.size():
                assert residual.size() == Q.size(), (
                    f"残差形状{residual.shape}必须与查询形状{Q.shape}匹配"
                )

            # 确保所有值都有效
            nan_mask2 = torch.isnan(residual) # | torch.isinf(residual)
            if nan_mask2.any():
                print(f"Warning: residual got NaN values at {self.keyname}!") # 替换无效值为0并添加微小噪声
                residual = torch.nan_to_num(residual, nan=0.0)
                noise = torch.randn_like(residual, device=residual.device, dtype=residual.dtype) * 1e-8
                residual = residual + noise

            ori_dtype = attn_output.dtype
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
                attn_output = attn_output.to(torch.float32)

            if self.res_norm_fn:
                if self.pre_norm == True:
                    output = self.res_norm_fn(attn_output) + residual
                else:
                    output = self.res_norm_fn(residual + attn_output)
                if ori_dtype in (torch.float16, torch.bfloat16):
                    return output.to(dtype=ori_dtype)
                return output
            return residual + attn_output
        
        return attn_output




class CrossTransformer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_layers: int = 1,
            attn_type: str = 'flash',
            proj_type: str = 'linear',
            res_norm_fn: Optional[nn.Module] = None,
            residual: bool = True,
            residual_in_fp32: bool = False,
            pre_norm: bool = False,
            dropout: float = 0.0,
            attn_cfg: Optional[Dict] = None, # 针对所选 attn_type 的独立配置
            ffn_cfg: Optional[Dict] = None,
            norm_fn=nn.LayerNorm,
            device=None,
            **kwargs,
        ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.device = device
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            cross_attn = XFormersCrossAttention(
                dim=dim, attn_type=attn_type, proj_type=proj_type,
                res_norm_fn=res_norm_fn, residual=residual, residual_in_fp32=residual_in_fp32, pre_norm=pre_norm,
                dropout=dropout if self.training else 0.0,
                attn_cfg=attn_cfg,
                keyname='Cross Transformer',
            )
            ffn = SimpleFFN(
                dim=dim, device=device, pre_norm=pre_norm, residual=residual,
                norm_fn=norm_fn,
                **ffn_cfg if ffn_cfg else {}
            )
            self.layers.append(nn.ModuleDict({
                'cross_attn': cross_attn,
                'ffn': ffn
            }))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        Q = Q.to(self.device)
        K = K.to(self.device)
        V = V.to(self.device)
        for layer in self.layers:
            Q = layer['cross_attn'](Q, K, V)
            Q = layer['ffn'](Q)
        return Q




class CustomSubMConv3d(spconv.SubMConv3d):
    """
    支持自定义卷积核形状的稀疏子流形卷积
    Tips: 曼哈顿卷积不再使用该类, 换做使用 class ManhattanSubMConv3d
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_template: 卷积核形状模板函数
            func(k: int) -> List[Tuple[int, int, int]]
            返回偏移量列表[(dz, dy, dx), ...]
        kernel_size: 模板函数参数
        **kwargs: SubMConv3d的其他参数
    """
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_template,  # 模板函数
        kernel_size: int = 1,
        **kwargs
    ):
        # 生成自定义卷积核偏移
        offsets = kernel_template(kernel_size)
        offsets.sort(key=lambda x: (abs(x[0]), abs(x[1]), abs(x[2])))
        
        # 创建标准卷积核尺寸（覆盖所有偏移）
        max_dims = np.array([[abs(dz), abs(dy), abs(dx)] for dz, dy, dx in offsets]).max(axis=0)
        kernel_dims = max_dims * 2 + 1
        
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=tuple(kernel_dims.tolist()),
            **kwargs
        )
        
        # 存储自定义偏移映射
        self.offset_map = self._create_offset_map(offsets)
        
        # 创建权重掩码而不是特征掩码
        with torch.no_grad():
            weight_mask = torch.zeros_like(self.weight)  # [out_channels, in_channels, D, H, W]
            for dz, dy, dx in offsets:
                kz, ky, kx = self.offset_map[(dz, dy, dx)]
                weight_mask[:, :, kz, ky, kx] = 1.0
            self.weight.mul_(weight_mask)  # 初始应用掩码
            
        # 存储掩码用于后续权重重置
        self.register_buffer('weight_mask', weight_mask)
        
        # 注册钩子以在每次梯度更新后重置权重
        self._register_post_accumulate_grad_hook()
        
        # self.register_buffer('custom_mask', self._create_mask(offsets))


    def _create_offset_map(self, offsets: List[Tuple[int, int, int]]) -> Dict:
        """创建偏移到卷积核位置的映射"""
        center = np.array(self.kernel_size) // 2
        offset_map = {}
        for dz, dy, dx in offsets:
            # 计算在标准卷积核中的位置
            pos = center + np.array([dz, dy, dx])
            key = (dz, dy, dx)
            offset_map[key] = tuple(pos.astype(int).tolist())
            
        return offset_map

    def _create_mask(self, offsets: List[Tuple[int, int, int]]) -> torch.Tensor:
        """创建自定义卷积核的二进制掩码"""
        mask = torch.zeros((
            1, 1, 
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2]
        ), dtype=torch.float32)
        
        for dz, dy, dx in offsets:
            kz, ky, kx = self.offset_map[(dz, dy, dx)]
            mask[0, 0, kz, ky, kx] = 1.0
            
        return mask

    def _register_post_accumulate_grad_hook(self):
        """注册一个在梯度累积后重置权重的钩子"""
        def _post_accumulate_hook(param):
            """在每个权重更新周期后重置权重"""
            if param.grad is not None:
                with torch.no_grad():
                    self.weight.mul_(self.weight_mask)
                    self.weight[self.weight_mask == 0] = 0
        
        # 在权重参数上注册钩子
        self.weight.register_post_accumulate_grad_hook(_post_accumulate_hook)

    def forward(self, input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return super().forward(input)
        
    

# NOTE: 实际上有些部分确实需要修改, 特别是权重限制步骤
# 该版本模型似乎未使用上曼哈顿掩码
# class ManhattanSubMConv3d(spconv.SubMConv3d):
#     """
#     增强版曼哈顿稀疏子流形卷积，支持：
#     1. 向量化曼哈顿核生成 2. 可控的非曼哈顿位置权重 3. 自动梯度约束
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         manhattan_kernel_template,  # 曼哈顿模板函数
#         manhattan_distance: Union[int, Tuple[int, int, int]],  # 曼哈顿距离阈值
#         kernel_size: Union[int, Tuple[int, int, int]],  # 卷积核尺寸
#         include_non_manhattan: bool = False,
#         non_manhattan_weight: float = 0.1,
#         non_manhattan_weight_max_scale: float = 1.2,
#         scale_decay: Tuple[float, float] = (0.1, 1.0),
#         **kwargs
#     ):
#         # 保存参数
#         self.manhattan_distance = manhattan_distance
#         self.min_decay, self.max_decay = scale_decay
#         assert self.min_decay <= self.max_decay, "scale_decay min必须小于等于max"
#         self.non_manhattan_weight_max_scale = non_manhattan_weight_max_scale
        
#         # 规范化参数格式
#         if isinstance(manhattan_distance, int):
#             manhattan_distance = (manhattan_distance, manhattan_distance, manhattan_distance)
        
#         if isinstance(kernel_size, int):
#             kernel_size = (kernel_size, kernel_size, kernel_size)
        
#         # 验证参数有效性
#         for i in range(3):
#             assert kernel_size[i] >= 2 * manhattan_distance[i] + 1, (
#                 f"维度 {['Z', 'Y', 'X'][i]} 的kernel_size({kernel_size[i]}) "
#                 f"必须 ≥ 2*manhattan_distance({manhattan_distance[i]}) + 1 = "
#                 f"{2 * manhattan_distance[i] + 1}"
#             )
        
#         # 生成偏移并排序（按距离和轴向）
#         offsets = manhattan_kernel_template(manhattan_distance)
#         offsets.sort(key=lambda x: (abs(x[0])+abs(x[1])+abs(x[2]), abs(x[0]), abs(x[1]), abs(x[2])))
        
#         # 调用父类构造函数
#         super().__init__(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             **kwargs
#         )
        
#         # 核心优化：创建空间掩码（不是权重掩码）
#         self.manhattan_spatial_mask = self._create_spatial_mask(offsets)
#         self.include_non_manhattan = include_non_manhattan
#         self.non_manhattan_weight = non_manhattan_weight
        
#         # 创建空间距离张量（用于非曼哈顿位置）
#         self.register_buffer("manhattan_dist", self._create_distance_tensor())
        
#         # 注册梯度钩子（基于原始版本）
#         self._register_post_accumulate_grad_hook()
    
#     def _create_spatial_mask(self, offsets):
#         """创建空间维度的曼哈顿位置掩码"""
#         kz, ky, kx = self.kernel_size
#         center_z, center_y, center_x = kz//2, ky//2, kx//2
#         spatial_mask = torch.zeros(self.kernel_size, device=self.weight.device)
        
#         for dz, dy, dx in offsets:
#             kz_pos = center_z + dz
#             ky_pos = center_y + dy
#             kx_pos = center_x + dx
#             if (0 <= kz_pos < kz and 
#                 0 <= ky_pos < ky and 
#                 0 <= kx_pos < kx):
#                 spatial_mask[kz_pos, ky_pos, kx_pos] = 1.0
        
#         return spatial_mask
    
#     def _create_distance_tensor(self):
#         """创建曼哈顿距离张量"""
#         kz, ky, kx = self.kernel_size
#         center = torch.tensor([kz//2, ky//2, kx//2], device=self.weight.device)
        
#         # 创建三维网格
#         grid_z, grid_y, grid_x = torch.meshgrid(
#             torch.arange(kz, device=self.weight.device),
#             torch.arange(ky, device=self.weight.device),
#             torch.arange(kx, device=self.weight.device),
#             indexing='ij'
#         )
        
#         # 计算相对位置
#         rel_pos = torch.stack([grid_z, grid_y, grid_x], dim=-1) - center
#         return rel_pos.abs().sum(dim=-1)  # 曼哈顿距离
    
#     def _register_post_accumulate_grad_hook(self):
#         """自动梯度约束机制（基于原始版本）"""
#         def _post_accumulate_grad_hook(param):
#             with torch.no_grad():
#                 kz, ky, kx = self.kernel_size
#                 weight = self.weight
                
#                 # 重塑权重为空间视图 [out, in, kz, ky, kx]
#                 weight_view = weight.view(
#                     self.out_channels, 
#                     self.in_channels, 
#                     kz, ky, kx
#                 )
                
#                 # 1. 处理曼哈顿位置
#                 manhattan_mask = self.manhattan_spatial_mask.bool()
#                 # 正确扩展掩码维度 [1, 1, kz, ky, kx] -> [out, in, kz, ky, kx]
#                 manhattan_mask_expanded = manhattan_mask.unsqueeze(0).unsqueeze(0).expand_as(weight_view)
                
#                 # 2. 处理非曼哈顿位置
#                 if self.include_non_manhattan:
#                     non_manhattan_mask = ~manhattan_mask_expanded
                    
#                     # 应用衰减因子
#                     decay = torch.clamp(
#                         1.0 / (self.manhattan_dist + 1), # TODO: 是否有更合理的算法
#                         min=self.min_decay, 
#                         max=self.max_decay
#                     )
                    
#                     # 正确扩展加权值 [kz, ky, kx] -> [out, in, kz, ky, kx]
#                     weighted_value = self.non_manhattan_weight * decay
#                     weighted_value_expanded = weighted_value.unsqueeze(0).unsqueeze(0).expand_as(weight_view)
                    
#                     # 应用权重限制（仅约束不归零）
#                     constrained_weights = torch.min(
#                         weight_view[non_manhattan_mask],
#                         weighted_value_expanded[non_manhattan_mask]
#                     )
#                     weight_view[non_manhattan_mask] = constrained_weights
#                 else:
#                     # 未启用非曼哈顿位置时归零
#                     non_manhattan_mask = ~manhattan_mask_expanded
#                     weight_view[non_manhattan_mask] = 0
                
#                 # 将更新后的权重复制回原始权重
#                 self.weight.data.copy_(weight_view.view_as(self.weight))
                
#         # 在权重参数上注册钩子（每次梯度累积后调用）
#         self.weight.register_post_accumulate_grad_hook(_post_accumulate_grad_hook)

#     def forward(self, input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
#         # 添加梯度监控（调试用）
#         if torch.is_grad_enabled():
#             total_norm = 0
#             for p in self.parameters():
#                 if p.grad is not None:
#                     param_norm = p.grad.data.norm(2)
#                     total_norm += param_norm.item() ** 2
#             total_norm = total_norm ** 0.5
#             if total_norm > 1e4:  # 检测梯度爆炸
#                 print(f"警告：梯度范数异常 ({total_norm:.2f})，应用梯度裁剪")
#                 torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
#         coords = input.indices
        
#         # 添加坐标范围验证（防止体素丢失）
#         # d, h, w = input.spatial_shape
#         # coords[:, 1] = torch.clamp(coords[:, 1], 0, d-1)  # Z轴
#         # coords[:, 2] = torch.clamp(coords[:, 2], 0, h-1)  # Y轴
#         # coords[:, 3] = torch.clamp(coords[:, 3], 0, w-1)  # X轴
        
#         input = spconv.SparseConvTensor(
#             features=input.features,
#             indices=coords.int(),  # 确保坐标类型
#             spatial_shape=input.spatial_shape,
#             batch_size=input.batch_size
#         )
#         return super().forward(input)




# NOTE: 参数化方便应用曼哈顿掩码
class ManhattanSubMConv3d(spconv.SubMConv3d):
    """
    重构版曼哈顿稀疏子流形卷积：
    1. 动态适应权重形状
    2. 自动调整掩码大小
    3. 使用参数化实现约束条件
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manhattan_kernel_template,  # 曼哈顿模板函数
        manhattan_distance: Union[int, Tuple[int, int, int]],  # 曼哈顿距离阈值
        kernel_size: Union[int, Tuple[int, int, int]],  # 卷积核尺寸
        include_non_manhattan: bool = False,
        non_manhattan_weight: float = 0.1,
        non_manhattan_weight_max_scale: float = 1.2,
        scale_decay: Tuple[float, float] = (0.1, 1.0),
        decay_temperature: float = 0.5,
        **kwargs
    ):
        # 规范化参数格式
        if isinstance(manhattan_distance, int):
            manhattan_distance = (manhattan_distance, manhattan_distance, manhattan_distance)
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        
        # 验证参数有效性
        for i in range(3):
            assert kernel_size[i] >= 2 * manhattan_distance[i] + 1, (
                f"维度 {['Z', 'Y', 'X'][i]} 的kernel_size({kernel_size[i]}) "
                f"必须 ≥ 2*manhattan_distance({manhattan_distance[i]}) + 1 = "
                f"{2 * manhattan_distance[i] + 1}"
            )
        
        # 生成偏移并排序
        offsets = manhattan_kernel_template(manhattan_distance)
        offsets.sort(key=lambda x: (abs(x[0])+abs(x[1])+abs(x[2]), abs(x[0]), abs(x[1]), abs(x[2])))
        
        # 调用父类构造函数
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            **kwargs
        )
        
        # 保存参数
        self.manhattan_distance = manhattan_distance
        self.non_manhattan_weight_max_scale = non_manhattan_weight_max_scale
        self.include_non_manhattan = include_non_manhattan
        self.non_manhattan_weight = non_manhattan_weight
        # self.min_decay, self.max_decay = scale_decay
        self.decay_temperature = decay_temperature
        
        # 创建空间掩码和距离张量
        # self.register_buffer("manhattan_spatial_mask", self._create_spatial_mask(offsets))
        # self.register_buffer("manhattan_dist", self._create_distance_tensor())
        self.register_buffer("manhattan_mask", self._create_manhattan_mask(offsets))
        self.register_buffer("distance_map", self._create_distance_map())
        
        # 应用参数化约束
        if include_non_manhattan:
            self._apply_parametrization()
    
    # def _create_spatial_mask(self, offsets):
    #     """创建空间维度的曼哈顿位置掩码"""
    #     kz, ky, kx = self.kernel_size
    #     center_z, center_y, center_x = kz//2, ky//2, kx//2
    #     spatial_mask = torch.zeros(self.kernel_size, device=self.weight.device)
        
    #     for dz, dy, dx in offsets:
    #         kz_pos = center_z + dz
    #         ky_pos = center_y + dy
    #         kx_pos = center_x + dx
    #         if (0 <= kz_pos < kz and 
    #             0 <= ky_pos < ky and 
    #             0 <= kx_pos < kx):
    #             spatial_mask[kz_pos, ky_pos, kx_pos] = 1.0
        
    #     return spatial_mask
    
    # def _create_distance_tensor(self):
    #     """创建曼哈顿距离张量"""
    #     kz, ky, kx = self.kernel_size
    #     center = torch.tensor([kz//2, ky//2, kx//2], device=self.weight.device)
        
    #     # 创建三维网格
    #     grid_z, grid_y, grid_x = torch.meshgrid(
    #         torch.arange(kz, device=self.weight.device),
    #         torch.arange(ky, device=self.weight.device),
    #         torch.arange(kx, device=self.weight.device),
    #         indexing='ij'
    #     )
        
    #     # 计算相对位置
    #     rel_pos = torch.stack([grid_z, grid_y, grid_x], dim=-1) - center
    #     return rel_pos.abs().sum(dim=-1)  # 曼哈顿距离
    
    def _create_manhattan_mask(self, offsets):
        """创建优化的曼哈顿位置掩码"""
        kz, ky, kx = self.kernel_size
        center_z, center_y, center_x = kz//2, ky//2, kx//2
        mask = torch.zeros((kz, ky, kx), dtype=torch.float)
        
        for dz, dy, dx in offsets:
            z, y, x = center_z + dz, center_y + dy, center_x + dx
            if (0 <= z < kz and 0 <= y < ky and 0 <= x < kx):
                mask[z, y, x] = 1.0
                
        return mask
    
    def _create_distance_map(self):
        """高效的曼哈顿距离计算"""
        shape = self.kernel_size
        center = torch.tensor([s//2 for s in shape])
        indices = torch.stack(torch.meshgrid(
            torch.arange(shape[0]),
            torch.arange(shape[1]),
            torch.arange(shape[2]),
            indexing='ij'
        ), -1)
        
        return (indices - center).abs().sum(dim=-1).float()
        
    def _apply_parametrization(self):
        """应用参数化约束"""
        
        # 定义曼哈顿掩码约束
        # class ManhattanConstraint(nn.Module):
        #     def __init__(self, mask, dist, include_non, non_weight, min_decay, max_decay, max_scale):
        #         super().__init__()
        #         self.register_buffer("original_mask", mask)
        #         self.register_buffer("dist", dist)
        #         self.include_non = include_non
        #         self.non_weight = non_weight
        #         self.min_decay = min_decay
        #         self.max_decay = max_decay
        #         self.max_scale = max_scale
            
        #     def forward(self, weight):
        #         # 获取权重张量的形状: [out_channels, kz, ky, kx, in_channels]
        #         out_channels, kz, ky, kx, in_channels = weight.shape
                
        #         # 检查空间维度是否匹配
        #         mask_z, mask_y, mask_x = self.original_mask.shape
                
        #         # 如果空间维度不匹配，调整掩码大小
        #         if (kz, ky, kx) != (mask_z, mask_y, mask_x):
        #             # 创建新的掩码，中心对齐
        #             new_mask = torch.zeros(kz, ky, kx, device=weight.device)
                    
        #             # 计算偏移量
        #             z_offset = (kz - mask_z) // 2
        #             y_offset = (ky - mask_y) // 2
        #             x_offset = (kx - mask_x) // 2
                    
        #             # 复制原始掩码到新掩码中心
        #             z_end = z_offset + mask_z
        #             y_end = y_offset + mask_y
        #             x_end = x_offset + mask_x
                    
        #             new_mask[z_offset:z_end, y_offset:y_end, x_offset:x_end] = self.original_mask
        #         else:
        #             new_mask = self.original_mask
                
        #         # 扩展掩码以匹配权重张量的形状
        #         mask = new_mask.unsqueeze(0).unsqueeze(-1)  # [1, kz, ky, kx, 1]
        #         mask = mask.expand(out_channels, -1, -1, -1, in_channels)  # [out_channels, kz, ky, kx, in_channels]
                
        #         # 应用曼哈顿掩码
        #         masked_weight = weight * mask
                
        #         # 处理非曼哈顿位置
        #         if self.include_non:
        #             # 计算衰减因子
        #             decay = torch.clamp(
        #                 1.0 / (self.dist + 1),
        #                 min=self.min_decay, 
        #                 max=self.max_decay
        #             )
                    
        #             # 计算非曼哈顿位置的最大允许值
        #             max_value = self.non_weight * self.max_scale * decay
                    
        #             # 扩展max_value以匹配权重张量的形状
        #             max_value = max_value.unsqueeze(0).unsqueeze(-1)  # [1, kz, ky, kx, 1]
        #             max_value = max_value.expand(out_channels, -1, -1, -1, in_channels)  # [out_channels, kz, ky, kx, in_channels]
                    
        #             # 应用约束
        #             non_mask = 1 - mask
        #             non_masked = weight * non_mask
        #             constrained = torch.clamp(non_masked, -max_value, max_value)
        #             return masked_weight + constrained
        #         else:
        #             return masked_weight
        
        class ManhattanConstraint(nn.Module):
            def __init__(self, mask, dist_map, base_weight, temperature):
                super().__init__()
                # 预计算衰减因子（固定值，不参与梯度）
                with torch.no_grad():
                    dist_max = dist_map.max()
                    if dist_max == 0: # 安全处理：确保不会除零
                        dist_max = torch.tensor(0.5, device=dist_map.device)
                    
                    # 计算衰减因子
                    decay = torch.exp(-temperature * (dist_map / dist_max)**2)
                    decay_factor = base_weight * decay  # 使用临时变量
                    
                # 注册为buffer (不参与梯度计算)
                self.register_buffer("mask", mask)
                self.register_buffer("decay_factor", decay_factor)  # 直接注册临时变量
            
            def forward(self, weight):
                # 直接在权重上应用约束
                manhattan_part = weight * self.mask.reshape(1, *self.mask.shape, 1)
                
                if not self.training:  # NOTE: 推理模式无需非曼哈顿部分
                    return manhattan_part
                
                # 高效广播应用衰减
                non_manhattan = weight * (1 - self.mask).reshape(1, *self.mask.shape, 1)
                return manhattan_part + non_manhattan * self.decay_factor.reshape(1, *self.decay_factor.shape, 1)
        

        # 创建约束实例
        # constraint = ManhattanConstraint(
        #     mask=self.manhattan_spatial_mask,
        #     dist=self.manhattan_dist,
        #     include_non=self.include_non_manhattan,
        #     non_weight=self.non_manhattan_weight,
        #     min_decay=self.min_decay,
        #     max_decay=self.max_decay,
        #     max_scale=self.non_manhattan_weight_max_scale
        # )
        constraint = ManhattanConstraint(
            mask=self.manhattan_mask,
            dist_map=self.distance_map,
            base_weight=self.non_manhattan_weight,
            temperature=self.decay_temperature
        )

        # 正确应用参数化到模块的权重
        register_parametrization(self, 'weight', constraint)

    def forward(self, input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        # # 坐标范围验证
        # coords = input.indices
        # d, h, w = input.spatial_shape
        # # coords[:, 1] = torch.clamp(coords[:, 1], 0, d-1)  # Z轴
        # # coords[:, 2] = torch.clamp(coords[:, 2], 0, h-1)  # Y轴
        # # coords[:, 3] = torch.clamp(coords[:, 3], 0, w-1)  # X轴
        
        # # input = spconv.SparseConvTensor(
        # #     features=input.features,
        # #     indices=coords.int(),
        # #     spatial_shape=input.spatial_shape,
        # #     batch_size=input.batch_size
        # # )
        # # 使用掩码替代clamp - 更高效
        # valid_mask = (
        #     (coords[:, 1] >= 0) & (coords[:, 1] < d) &
        #     (coords[:, 2] >= 0) & (coords[:, 2] < h) &
        #     (coords[:, 3] >= 0) & (coords[:, 3] < w)
        # )
        # if not valid_mask.all():
        #     # 仅过滤无效点而不是拷贝整个张量
        #     input = input.replace_feature(input.features[valid_mask])
        #     input.indices = coords[valid_mask]

        # 直接调用父类前向传播，参数化会自动应用约束
        return super().forward(input)



''' 曼哈顿是否正确使用上的验证方法
self.conv = ManhattanSubMConv3d(
    in_channels=dim,
    out_channels=dim,
    **(manhattan_conv_cfg[0])
)
# 在初始化后立即检查权重
print("卷积层权重形状:", self.conv.weight.shape)
print("曼哈顿掩码形状:", self.conv.manhattan_spatial_mask.shape)

# 获取未应用约束的原始权重
with torch.no_grad():
    # 获取参数化前的原始权重
    if hasattr(self.conv, 'parametrizations') and 'weight' in self.conv.parametrizations:
        original_weight = self.conv.parametrizations.weight.original
    else:
        original_weight = self.conv.weight.clone()
    
    # 获取应用约束后的权重
    constrained_weight = self.conv.weight.clone()

# 计算曼哈顿位置的数量
manhattan_positions = self.conv.manhattan_spatial_mask.sum().item()
print(f"曼哈顿位置数量: {manhattan_positions}/{self.conv.manhattan_spatial_mask.numel()}")

# 检查曼哈顿位置是否被保留
# 正确扩展掩码以匹配权重张量的形状
out_channels, kz, ky, kx, in_channels = constrained_weight.shape
manhattan_mask_expanded = self.conv.manhattan_spatial_mask.unsqueeze(0)  # [1, kz, ky, kx]
manhattan_mask_expanded = manhattan_mask_expanded.unsqueeze(-1)          # [1, kz, ky, kx, 1]
manhattan_mask_expanded = manhattan_mask_expanded.expand(out_channels, -1, -1, -1, in_channels)  # [out_channels, kz, ky, kx, in_channels]

# 计算曼哈顿位置的变化
manhattan_diff = (constrained_weight - original_weight)[manhattan_mask_expanded.bool()]
print(f"曼哈顿位置权重变化 - 均值: {manhattan_diff.mean().item():.6f}, 标准差: {manhattan_diff.std().item():.6f}")

# 检查非曼哈顿位置是否被约束
non_manhattan_mask = ~manhattan_mask_expanded.bool()
if self.conv.include_non_manhattan:
    non_manhattan_values = constrained_weight[non_manhattan_mask]
    print(f"非曼哈顿位置权重范围: [{non_manhattan_values.min().item():.6f}, {non_manhattan_values.max().item():.6f}]")
else:
    non_manhattan_values = constrained_weight[non_manhattan_mask]
    print(f"非曼哈顿位置是否全为零: {torch.all(non_manhattan_values == 0)}")

卷积层权重形状: torch.Size([128, 3, 3, 3, 128])
曼哈顿掩码形状: torch.Size([3, 3, 3])
曼哈顿位置数量: 7.0/27 -> 这个是重点
曼哈顿位置权重变化 - 均值: 0.000000, 标准差: 0.000000
非曼哈顿位置权重范围: [-0.017010, 0.017010]
'''



class EfficientAttention(nn.Module):
    '''
    Efficient attention is an attention mechanism that substantially optimizes the memory and 
    computational efficiency while retaining exactly the same expressive power as the conventional 
    dot-product attention.
    Reference Paper: https://arxiv.org/abs/1812.01243
    Efficient Attention : -
    '''
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[ :, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[ :, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[ :, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2).contiguous()
            attended_value = (
                context.transpose(1, 2).contiguous() @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention



class EfficientAttention1D(nn.Module):
    '''
    Revised From EfficientAttention with dimension correction
    Reference Paper: https://arxiv.org/abs/1812.01243
    '''
    def __init__(self, in_channels, key_channels, head_count, value_channels,
                 chunk_size_q: int=1024,
                 chunk_size_kv: int=2048):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.chunk_size_q = chunk_size_q
        self.chunk_size_kv = chunk_size_kv
        
        # 确保维度可被头数整除
        assert key_channels % head_count == 0, "key_channels must be divisible by head_count"
        assert value_channels % head_count == 0, "value_channels must be divisible by head_count"
        
        self.keys = nn.Conv1d(in_channels, key_channels, 1)
        self.queries = nn.Conv1d(in_channels, key_channels, 1)
        self.values = nn.Conv1d(in_channels, value_channels, 1)
        
        # 重新投影层输出通道数应与输入通道数匹配
        self.reprojection = nn.Conv1d(value_channels, in_channels, 1)

    def forward(self, query_seq, kv_seq, chunk_size_q=None, chunk_size_kv=None):
        '''
        Input
            query_seq [b, L, dim] or [L, dim]
            kv_seq    [b, L, dim] or [L, dim]
        Output
            output    [b, L, dim]
        '''
        # 预处理输入张量
        if len(query_seq.shape) == 2:
            query_seq = query_seq.unsqueeze(0)
        query_seq_perm = query_seq.permute(0, 2, 1).contiguous()  # [b, in_channels, L_q]

        if len(kv_seq.shape) == 2:
            kv_seq = kv_seq.unsqueeze(0)
        kv_seq_perm = kv_seq.permute(0, 2, 1).contiguous()  # [b, in_channels, L_kv]

        b, _, L_q = query_seq_perm.size()
        _, in_channels_kv, L_kv = kv_seq_perm.size()
        
        # 确保输入通道一致
        assert query_seq_perm.shape[1] == kv_seq_perm.shape[1], \
            f"Input channel mismatch: query_seq_perm {query_seq_perm.shape}, kv_seq_perm {kv_seq_perm.shape}"
            
        # 使用统一通道数（取较小值作为安全措施）
        in_channels = min(query_seq_perm.shape[1], kv_seq_perm.shape[1])
        
        # 确定分块大小
        chunk_size_q = chunk_size_q or min(self.chunk_size_q, L_q)
        chunk_size_kv = chunk_size_kv or min(self.chunk_size_kv, L_kv)
        
        Q = self.queries(query_seq_perm)  # [b, key_channels, L_q]
        K = self.keys(kv_seq_perm)         # [b, key_channels, L_kv]
        V = self.values(kv_seq_perm)       # [b, value_channels, L_kv]
        
        # 检查输出尺寸
        assert Q.size(1) == self.key_channels, f"Q channels {Q.size(1)} != {self.key_channels}"
        assert K.size(1) == self.key_channels, f"K channels {K.size(1)} != {self.key_channels}"
        assert V.size(1) == self.value_channels, f"V channels {V.size(1)} != {self.value_channels}"
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_heads = []
        
        # 处理每个注意力头
        for i in range(self.head_count):
            Q_head = Q[:, i * head_key_channels: (i + 1) * head_key_channels, :]  # [b, C_k_head, L_q]
            K_head = K[:, i * head_key_channels: (i + 1) * head_key_channels, :]  # [b, C_k_head, L_kv]
            V_head = V[:, i * head_value_channels: (i + 1) * head_value_channels, :]  # [b, C_v_head, L_kv]
            
            # 创建正确尺寸的输出张量
            head_output = torch.zeros(b, head_value_channels, L_q, 
                                    device=Q.device, dtype=Q.dtype)
            
            # 分块处理K/V序列
            num_chunks_kv = (L_kv + chunk_size_kv - 1) // chunk_size_kv
            
            for chunk_idx_kv in range(num_chunks_kv):
                kv_start = chunk_idx_kv * chunk_size_kv
                kv_end = min(kv_start + chunk_size_kv, L_kv)
                
                # 提取K/V分块
                K_head_chunk = K_head[:, :, kv_start:kv_end]  # [b, C_k_head, chunk_kv]
                V_head_chunk = V_head[:, :, kv_start:kv_end]  # [b, C_v_head, chunk_kv]
                
                # 计算K_head_chunk的softmax（当前K/V分块）
                K_head_softmax = F.softmax(K_head_chunk, dim=-1)  # [b, C_k_head, chunk_kv]
                
                # 分块处理Q序列
                num_chunks_q = (L_q + chunk_size_q - 1) // chunk_size_q
                
                for chunk_idx_q in range(num_chunks_q):
                    q_start = chunk_idx_q * chunk_size_q
                    q_end = min(q_start + chunk_size_q, L_q)
                    
                    # 提取Q分块
                    Q_head_chunk = Q_head[:, :, q_start:q_end]  # [b, C_k_head, chunk_q]
                    
                    # 计算Q_head_chunk的softmax（当前Q分块）
                    Q_head_softmax = F.softmax(Q_head_chunk, dim=-1)  # [b, C_k_head, chunk_q]
                    
                    # 计算注意力得分 [b, chunk_q, chunk_kv]
                    attention_scores = torch.matmul(
                        Q_head_softmax.permute(0, 2, 1).contiguous(),  # [b, chunk_q, C_k_head]
                        K_head_softmax  # [b, C_k_head, chunk_kv]
                    )  # => [b, chunk_q, chunk_kv]
                    
                    # 计算上下文向量 [b, C_v_head, chunk_q]
                    context_vector = torch.matmul(
                        V_head_chunk,  # [b, C_v_head, chunk_kv]
                        attention_scores.permute(0, 2, 1).contiguous()  # [b, chunk_kv, chunk_q]
                    )  # => [b, C_v_head, chunk_q]
                    
                    # 存储分块结果（累加而不是替换）
                    head_output[:, :, q_start:q_end] += context_vector
            
            attended_heads.append(head_output)
        
        # 确保每个头输出的尺寸一致
        assert all([t.shape == attended_heads[0].shape for t in attended_heads]), \
            f"Head outputs have inconsistent shapes: {[t.shape for t in attended_heads]}"
        
        # 拼接所有头部输出
        aggregated_values = torch.cat(attended_heads, dim=1)  # [b, value_channels, L_q]
        
        # 确保重新投影输入尺寸正确
        assert aggregated_values.size(1) == self.value_channels, \
            f"aggregated_values channels {aggregated_values.size(1)} != {self.value_channels}"
        
        # 重新投影
        reprojected = self.reprojection(aggregated_values)  # [b, in_channels, L_q]
        
        # 确保reprojected与query_seq_perm尺寸一致
        assert reprojected.shape == query_seq_perm.shape, \
            f"Reprojected shape {reprojected.shape} != query_seq_perm shape {query_seq_perm.shape}"
        
        # 残差连接
        output = reprojected + query_seq_perm
        
        # 恢复原始维度
        output = output.permute(0, 2, 1).contiguous()  # [batch, seq_len, in_channels]
        
        return output



class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding



class MLPBlock(nn.Module):
    def __init__(self, input_channel, out_channel, norm_fn):
        super().__init__()
        self.mlp_layer = nn.Sequential(
            nn.Linear(input_channel, out_channel),
            norm_fn(out_channel),
            nn.GELU())

    def forward(self, x):
        mpl_feats = self.mlp_layer(x)
        return mpl_feats



class Sparse1ConvBlock(spconv.SparseModule):
    '''
    A SparseConvBlock, which contain only one (submconv3d + norm + relu) and residual connect
    Reference Paper: https://arxiv.org/abs/2406.10700
    VoxelMamba: -
    '''
    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None, activate='relu', device=None):
        super(Sparse1ConvBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        if activate == 'relu':
            self.activate = nn.ReLU()
        elif activate == 'gelu':
            self.activate = nn.GELU()
        if device is not None:
            self.to(device)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.activate(out.features))

        return out
    


class SparseBasicBlock3D(spconv.SparseModule):
    '''
    A SparseConvBlock, which contain two (submconv3d + norm + relu) and residual connect
    Reference Paper: https://arxiv.org/abs/2310.20234
    HEDNet: -
    '''
    def __init__(self, dim, indice_key, norm_fn=norm1d_fn, bias=True):
        super(SparseBasicBlock3D, self).__init__()

        self.conv1 = spconv.SubMConv3d(dim, dim, 3, 1, 1, bias=bias, indice_key=indice_key)
        self.bn1 = norm_fn(dim)

        self.conv2 = spconv.SubMConv3d(dim, dim, 3, 1, 1, bias=bias, indice_key=indice_key)
        self.bn2 = norm_fn(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + x.features))
        return out



class DownSp(spconv.SparseModule):
    '''
    A Down-Sampling Block, which contain one (SparseConv3d + norm + relu) for down-sampling 
        and some Sparse1ConvBlock
    Reference Paper: https://arxiv.org/abs/2406.10700
    VoxelMamba: -
    '''
    def __init__(self, dim, kernel_size, stride, sub_num, norm_fn, indice_key, type):
        super(DownSp, self).__init__()

        if type == 'down':
            first_block = post_act_block(
                dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                norm_fn=norm_fn, indice_key=f'spconv_down_{indice_key}', conv_type='spconv')
        elif type == 'up':
            first_block = post_act_block(
                dim, dim, kernel_size=kernel_size, indice_key=f'spconv_up_{indice_key}',
                norm_fn=norm_fn, conv_type='inverseconv')
        
        # 1 x
        ## NOTE: The specific component of first_block: ##
        # m = spconv.SparseSequential(
        #     spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        #        bias=False, indice_key=indice_key)
        #     norm_fn(out_channels), # BatchNorm1d
        #     nn.ReLU(),
        # )

        # sub_num x
        ## NOTE: The specific component of Sparse1ConvBlock: ##
        # self.conv1 = spconv.SubMConv3d(
        #     inplanes, planes, kernel_size=3, stride=1, padding=1, bias=bias, indice_key=indice_key
        # )
        # self.bn1 = norm_fn(planes) # BatchNorm1d
        # self.relu = nn.ReLU()

        block_list = [first_block if stride > 1 else nn.Identity()]
        for _ in range(sub_num):
            block_list.append(
                Sparse1ConvBlock(dim, dim, norm_fn=norm_fn, indice_key=indice_key))

        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, x):
        return self.blocks(x)



class DepthwiseDownsample1d(nn.Module):
    """可配置下采样比例的双倍深度可分离卷积"""
    def __init__(self, channels=4, kernel_size=3):
        """
        参数:
            channels (int): 输入通道数
            kernel_size (int): 卷积核大小(奇数)
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
        # 深度卷积层 - 固定步长为1
        self.depthwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,  # 固定为1
            padding=self.padding,
            groups=channels,  # 深度可分离
            bias=False
        )
        
        # 点卷积层
        self.pointwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化卷积核为平均采样模式"""
        with torch.no_grad():
            k = self.kernel_size
            mid = k // 2
            
            # 深度卷积核: 创建[0.25, 0.5, 0.25]类似分布
            weights = torch.zeros_like(self.depthwise.weight)
            weights[:, :, mid] = 0.5
            
            if k > 1:
                # 对称填充周围权重
                weights[:, :, :mid] = 0.25 / mid
                weights[:, :, mid+1:] = 0.25 / mid
            
            self.depthwise.weight.data = weights
            # 点卷积设为恒等变换
            self.pointwise.weight.data = torch.eye(self.channels).unsqueeze(-1)

    def forward(self, x: torch.Tensor, reduction_factor: int) -> torch.Tensor:
        """
        下采样序列长度
        
        参数:
            x: 输入序列 [N, C]
            reduction_factor: 下采样因子(如2, 3, 4等)
            
        返回:
            下采样后的序列 [M, C], M = ceil(N/reduction_factor)
        """
        assert reduction_factor > 0, "下采样因子必须为正整数"
        N, C = x.shape
        assert C == self.channels, f"输入通道数{C}与初始化通道数{self.channels}不匹配"
        
        # 1. 计算需要填充的长度
        target_length = math.ceil(N / reduction_factor) * reduction_factor
        padding_needed = max(0, target_length - N)
        
        # 2. 对序列进行填充(尾部填充0)
        if padding_needed > 0:
            x = F.pad(x, (0, 0, 0, padding_needed), "constant", 0)
        
        # 3. 维度转换: [N', C] -> [1, C, N']
        x = x.permute(1, 0).contiguous().unsqueeze(0)  # [1, C, N']
        
        # 4. 应用深度卷积
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # 5. 下采样: 使用步长卷积
        x = F.avg_pool1d(
            x, 
            kernel_size=reduction_factor, 
            stride=reduction_factor
        )
        
        # 6. 维度转换: [1, C, M] -> [M, C]
        return x.squeeze(0).permute(1, 0).contiguous()



class DepthwiseUpsample1d(nn.Module):
    """上采样模块 - 支持任意缩放因子"""
    def __init__(self, channels=4, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
        # 深度转置卷积
        self.depthwise = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,  # 固定为1
            padding=self.padding,
            groups=channels,
            bias=False
        )
        
        # 点卷积
        self.pointwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化卷积核为插值模式"""
        with torch.no_grad():
            k = self.kernel_size
            mid = k // 2
            
            # 深度转置卷积核: 创建[0.5, 1.0, 0.5]插值核
            weights = torch.zeros_like(self.depthwise.weight)
            weights[:, :, mid] = 1.0
            
            if k > 1:
                weights[:, :, mid-1] = 0.5
                weights[:, :, mid+1] = 0.5
                
            self.depthwise.weight.data = weights
            # 点卷积设为恒等变换
            self.pointwise.weight.data = torch.eye(self.channels).unsqueeze(-1)
    
    def forward(self, 
                x: torch.Tensor, 
                scale_factor: int, 
                original_length: int=None
        ) -> torch.Tensor:
        """
        上采样序列长度
        
        参数:
            x: 输入序列 [M, C]
            scale_factor: 上采样因子(如2, 3, 4等)
            
        返回:
            上采样后的序列 [N, C], N ≈ M * scale_factor
        """
        M, C = x.shape
        assert C == self.channels, f"输入通道数{C}与初始化通道数{self.channels}不匹配"
        
        # 1. 维度转换: [M, C] -> [1, C, M]
        x = x.permute(1, 0).contiguous().unsqueeze(0)  # [1, C, M]
        
        # 2. 应用深度转置卷积
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # 3. 上采样: 使用插值
        x = F.interpolate(
            x, 
            scale_factor=scale_factor, 
            mode='nearest'  # 或 'linear'/'bilinear'根据数据类型
        )

        upsampled = x.squeeze(0).permute(1, 0).contiguous()

        if original_length is not None:
            return upsampled[:original_length, :]
        
        return upsampled



class SequenceReducer(nn.Module):
    def __init__(self, reduction_mode='mean'):
        """
        序列缩减模块
        参数:
            reduction_mode (str): 缩减模式，可选:
                'mean' - 平均池化（默认）
                'max' - 最大池化
                'sum' - 求和池化
                'first' - 取每段第一个元素
                'last' - 取每段最后一个元素
                'random' - 随机取每段一个元素
        """
        super().__init__()
        self.reduction_mode = reduction_mode
        
    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        缩减序列长度
        参数:
            x: 输入序列 [N, 4]
            k: 缩减因子（每k个元素缩减为1个）
        返回:
            缩减后的序列 [M, 4]，其中 M = ceil(N/k)
        """
        assert x.dim() == 2, "输入应为2D张量 [N, 4]"
        N, C = x.shape
        # assert C == 4, "输入序列应有4个特征维度"
        assert k > 0, "缩减因子k必须为正整数"
        
        remainder = (k - N % k) % k
        padded = False
        
        if remainder > 0:
            padded = True
            x = F.pad(x, (0, 0, 0, remainder), "constant", 0)
        
        new_N = x.shape[0] // k
        
        # 根据不同模式进行缩减
        if self.reduction_mode == 'mean':
            result = self._mean_pool(x, k, new_N)
        elif self.reduction_mode == 'max':
            result = self._max_pool(x, k, new_N)
        elif self.reduction_mode == 'sum':
            result = self._sum_pool(x, k, new_N)
        elif self.reduction_mode == 'first':
            result = self._downsample(x, k, new_N, position=0)
        elif self.reduction_mode == 'last':
            result = self._downsample(x, k, new_N, position=k-1)
        elif self.reduction_mode == 'random':
            result = self._random_sample(x, k, new_N)
        else:
            raise ValueError(f"未知缩减模式: {self.reduction_mode}")
        
        return result
    
    def _mean_pool(self, x, k, new_N):
        return x.view(new_N, k, -1).mean(dim=1)
    
    def _max_pool(self, x, k, new_N):
        return x.view(new_N, k, -1).max(dim=1).values
    
    def _sum_pool(self, x, k, new_N):
        return x.view(new_N, k, -1).sum(dim=1)
    
    def _downsample(self, x, k, new_N, position):
        # 获取每段中的指定位置
        indices = torch.arange(position, x.shape[0], k)
        return x[indices]
    
    def _random_sample(self, x, k, new_N):
        # 为每段生成随机索引
        offsets = torch.randint(0, k, (new_N,))
        indices = torch.arange(new_N) * k + offsets
        return x[indices]
    
    def __repr__(self):
        return f"SequenceReducer(mode={self.reduction_mode})"



class SequenceUpsampler(nn.Module):
    def __init__(self, upsample_mode='linear'):
        """
        序列上采样模块
        
        参数:
            upsample_mode (str): 上采样模式，可选:
                'linear' - 线性插值（默认）
                'nearest' - 最近邻插值
                'repeat' - 重复插值
                'deconv' - 转置卷积上采样
        """
        super().__init__()
        self.upsample_mode = upsample_mode
        
    def forward(self, x: torch.Tensor, k: int, original_length: int) -> torch.Tensor:
        """
        上采样序列长度
        
        参数:
            x: 输入序列 [M, 4]
            k: 上采样因子（将序列扩展k倍）
            original_length: 原始序列长度（用于精确恢复）
            
        返回:
            上采样后的序列 [N, 4]，其中 N = original_length
        """
        assert x.dim() == 2, "输入应为2D张量 [M, 4]"
        M, C = x.shape
        # assert C == 4, "输入序列应有4个特征维度"
        assert k > 0, "上采样因子k必须为正整数"
        assert original_length > 0, "原始长度必须为正整数"
        
        # 计算目标长度（确保与原始长度匹配）
        target_length = original_length
        
        # 根据不同模式进行上采样
        if self.upsample_mode == 'linear':
            result = self._linear_upsample(x, k, target_length)
        elif self.upsample_mode == 'nearest':
            result = self._nearest_upsample(x, k, target_length)
        elif self.upsample_mode == 'repeat':
            result = self._repeat_upsample(x, k, target_length)
        elif self.upsample_mode == 'deconv':
            result = self._deconv_upsample(x, k, target_length)
        else:
            raise ValueError(f"未知上采样模式: {self.upsample_mode}")
        
        return result
    
    def _linear_upsample(self, x, k, target_length):
        """线性插值上采样"""
        # 转换为 [1, C, M]
        x = x.permute(1, 0).contiguous().unsqueeze(0)
        
        # 线性插值
        upsampled = F.interpolate(
            x, 
            size=target_length, 
            mode='linear', 
            align_corners=True
        )
        
        # 转换回 [N, C]
        return upsampled.squeeze(0).permute(1, 0).contiguous()
    
    def _nearest_upsample(self, x, k, target_length):
        """最近邻插值上采样"""
        # 转换为 [1, C, M]
        x = x.permute(1, 0).contiguous().unsqueeze(0)
        
        # 最近邻插值
        upsampled = F.interpolate(
            x, 
            size=target_length, 
            mode='nearest'
        )
        
        # 转换回 [N, C]
        return upsampled.squeeze(0).permute(1, 0).contiguous()
    
    def _repeat_upsample(self, x, k, target_length):
        """重复插值上采样"""
        # 计算实际需要的上采样比例
        actual_scale = target_length / x.size(0)
        
        # 转换为 [1, C, M]
        x = x.permute(1, 0).contiguous().unsqueeze(0)
        
        # 使用最近邻插值实现重复
        upsampled = F.interpolate(
            x, 
            scale_factor=actual_scale, 
            mode='nearest'
        )
        
        # 转换回 [N, C]
        return upsampled.squeeze(0).permute(1, 0).contiguous()
    
    def _deconv_upsample(self, x, k, target_length):
        """转置卷积上采样"""
        # 转换为 [1, C, M]
        x = x.permute(1, 0).contiguous().unsqueeze(0)
        
        # 动态创建转置卷积层
        conv = nn.ConvTranspose1d(
            in_channels=4,
            out_channels=4,
            kernel_size=k * 2 - 1,  # 较大的卷积核以获得更好的重建
            stride=k,
            padding=(k - 1),
            output_padding=0,
            groups=4,
            bias=False
        ).to(x.device)
        
        # 初始化权重为插值模式
        with torch.no_grad():
            # 创建类似线性插值的权重
            weights = torch.zeros_like(conv.weight)
            center = k - 1
            weights[:, :, center] = 1.0
            
            # 设置插值权重
            for i in range(1, k):
                weight_val = 1.0 - i / k
                if center - i >= 0:
                    weights[:, :, center - i] = weight_val
                if center + i < weights.size(2):
                    weights[:, :, center + i] = weight_val
            
            conv.weight.data = weights
        
        # 应用转置卷积
        upsampled = conv(x)
        
        # 截取到目标长度
        upsampled = upsampled[:, :, :target_length]
        
        # 转换回 [N, C]
        return upsampled.squeeze(0).permute(1, 0).contiguous()
    
    def __repr__(self):
        return f"SequenceUpsampler(mode={self.upsample_mode})"



class wConv2d(nn.Module):
    """
    Weighted 2D Convolution with spatial density modulation.
    通过密度函数(den)对卷积核进行空间加权调制的二维卷积层，实现中心加权的特征提取。
    参考论文: https://arxiv.org/abs/2505.24527. - Optimal Density Functions for Weighted Convolution in Learning Models (Cammarasana & Patanè, 2025)
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核尺寸 (仅支持奇数尺寸: 1/3/5)
        den (list[float]): 密度函数值，决定卷积核的空间权重分布:
            - kernel_size=1: [] (无调制)
            - kernel_size=3: [a] (建议范围[0.5, 1.5])
            - kernel_size=5: [a, b] (建议[[0.05,1], [0.5,1.5]])
        stride (int/tuple): 步长 (默认1)
        padding (int/tuple): 填充 (默认1)
        groups (int): 分组卷积数 (默认1)
        dilation (int/tuple): 空洞率 (默认1)
        bias (bool): 是否使用偏置 (默认False)
    Input:
        x (Tensor): [batch, in_channels, H, W] 输入张量
    Output:
        Tensor: [batch, out_channels, H_out, W_out] 输出张量
    Example:
        - 1x1卷积,不需要添加系数
          conv = wConv2d(64, 128, kernel_size=1, den=[])
        - 3x3卷积,中心加权系数1.0
          conv = wConv2d(64, 128, kernel_size=3, den=[1.0])
        - 5x5卷积,测试不同密度值
          conv = wConv2d(64, 128, kernel_size=5, den=[0.2, 0.8])
    """
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, dilation=1, bias=False):
        super(wConv2d, self).__init__()       
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size = _pair(kernel_size)
        self.groups = groups
        self.dilation = _pair(dilation)      
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')  
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device),torch.tensor([1.0], device=device),torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
    


class wConv3d(nn.Module):
    """
    Weighted 3D Convolution with spatial density modulation.
    通过三维密度函数调制的三维卷积层，适用于体数据(如医学CT/MRI)的特征提取。
    参考论文: https://arxiv.org/abs/2505.24527. - Optimal Weighted Convolution for Classification and Denoising (Cammarasana & Patanè, 2025)
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核尺寸 (仅支持奇数尺寸: 1/3/5)
        den (list[float]): 密度函数值:
            - kernel_size=1: [] 
            - kernel_size=3: [a] (建议[0.8-1.2])
            - kernel_size=5: [a, b] (建议[0.1, 0.7])
        stride (int/tuple): 步长 (默认1)
        padding (int/tuple): 填充 (默认1)
        groups (int): 分组卷积数 (默认1)
        dilation (int/tuple): 空洞率 (默认1)
        bias (bool): 是否使用偏置 (默认False)
    Input:
        x (Tensor): [batch, in_channels, D, H, W] 输入体数据
    Output:
        Tensor: [batch, out_channels, D_out, H_out, W_out] 输出张量
    Example:
        - 1x1卷积,不需要添加系数
          conv = wConv2d(64, 128, kernel_size=1, den=[])
        - 3x3x3卷积,密度值0.8
          conv = wConv3d(64, 128, kernel_size=3, den=[0.8])
        - 5x5x5卷积优化策略
          conv = wConv3d(64, 128, kernel_size=5, den=[0.1, 0.7])
    """
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, dilation=1, bias=False):
        super(wConv3d, self).__init__()       
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.kernel_size = _triple(kernel_size)
        self.groups = groups
        self.dilation = _triple(dilation)          
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')  
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device),torch.tensor([1.0], device=device),torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.einsum('i,j,k->ijk', self.alfa, self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv3d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)    




##################################################### Funcitons #######################################################


def manhattan_kernel_template(k: int):
    """生成曼哈顿距离≤k的卷积核偏移"""
    offsets = []
    for dx in range(-k, k+1):
        for dy in range(-k, k+1):
            for dz in range(-k, k+1):
                manhattan_dist = abs(dx) + abs(dy) + abs(dz)
                if manhattan_dist <= k:
                    offsets.append((dz, dy, dx))  # 注意：spconv使用(dz, dy, dx)
    return offsets


def manhattan_kernel_template_fast(manhattan_distance: Union[int, Tuple[int, int, int]]):
    """向量化实现的曼哈顿距离≤指定阈值的卷积核偏移"""
    # 规范化参数格式
    if isinstance(manhattan_distance, int):
        kz = ky = kx = manhattan_distance
    else:
        kz, ky, kx = manhattan_distance
    
    # 生成所有可能的偏移组合
    dz_range = torch.arange(-kz, kz+1, dtype=torch.int32)
    dy_range = torch.arange(-ky, ky+1, dtype=torch.int32)
    dx_range = torch.arange(-kx, kx+1, dtype=torch.int32)
    
    dz, dy, dx = torch.meshgrid(dz_range, dy_range, dx_range, indexing='ij')
    
    # 计算曼哈顿距离并过滤
    manhattan_dist = dz.abs() + dy.abs() + dx.abs()
    max_dist = max(kz, ky, kx)
    mask = (dz.abs() <= kz) & (dy.abs() <= ky) & (dx.abs() <= kx) & (manhattan_dist <= max_dist)
    
    # 获取有效偏移并转换为列表
    offsets = torch.stack([dz[mask], dy[mask], dx[mask]], dim=1).tolist()
    return offsets


def spherical_kernel_template(radius: Union[int, Tuple[int, int, int]]):
    """欧氏距离≤radius的球面卷积核"""
    # 规范化参数格式
    if isinstance(radius, int):
        rz = ry = rx = radius
    else:
        rz, ry, rx = radius
    offsets = []
    for dz in range(-rz, rz+1):
        for dy in range(-ry, ry+1):
            for dx in range(-rx, rx+1):
                # 计算欧氏距离
                euclidean_dist = dz**2 + dy**2 + dx**2
                # 检查是否在指定半径内
                if (abs(dz) <= rz and 
                    abs(dy) <= ry and 
                    abs(dx) <= rx and 
                    euclidean_dist <= max(rz, ry, rx)**2):
                    offsets.append((dz, dy, dx))
    return offsets


def cross_kernel_template(radius: Union[int, Tuple[int, int, int]]):
    """3D十字形卷积核，支持各向异性半径
    参数:
        radius: 整数或三维元组，表示各方向的扩展半径
                - 整数: 所有方向使用相同半径
                - 元组: (z_radius, y_radius, x_radius)
    返回:
        偏移量列表 [(dz, dy, dx), ...]
    """
    # 规范化参数格式
    if isinstance(radius, int):
        rz = ry = rx = radius
    else:
        rz, ry, rx = radius
    offsets = [(0, 0, 0)]  # 中心点
    for r in range(1, rz + 1): # Z轴方向
        offsets.extend([(r, 0, 0), (-r, 0, 0)])
    for r in range(1, ry + 1): # Y轴方向
        offsets.extend([(0, r, 0), (0, -r, 0)])
    for r in range(1, rx + 1): # X轴方向
        offsets.extend([(0, 0, r), (0, 0, -r)])
    return offsets


def debug_check_croods_repeated(croods, pre_print=""):
    unique_all_x, inverse_all_x, counts_all_x = torch.unique(
        croods, 
        dim=0, 
        return_inverse=True, 
        return_counts=True
    )
    all_x_has_duplicates = unique_all_x.size(0) < croods.size(0)
    print(pre_print + f"Croods has duplicates: {all_x_has_duplicates}, unique/total: {unique_all_x.size(0)}/{croods.size(0)}")
    return all_x_has_duplicates


def sparse_add(
        f1_sp: spconv.SparseConvTensor, 
        f2_sp: spconv.SparseConvTensor, 
        sparse_shape=None, 
        batch_size=None
    ) -> spconv.SparseConvTensor:

    # 确保稀疏形状和批次大小一致
    if sparse_shape is None:
        sparse_shape = f1_sp.spatial_shape
    if batch_size is None:
        batch_size = f1_sp.batch_size
    
    device = f1_sp.features.device
    dtype = f1_sp.features.dtype
    
    # 类型转换保护
    if f1_sp.features.dtype != f2_sp.features.dtype:
        f2_sp = spconv.SparseConvTensor(
            features=f2_sp.features.to(dtype),
            indices=f2_sp.indices,
            spatial_shape=f2_sp.spatial_shape,
            batch_size=batch_size
        )
    
    coords1 = f1_sp.indices.long()
    coords2 = f2_sp.indices.long()
    features1 = f1_sp.features
    features2 = f2_sp.features
    
    if torch.any(coords1[:, 0] >= batch_size) or torch.any(coords2[:, 0] >= batch_size):
        invalid_batch = torch.cat([
            coords1[coords1[:, 0] >= batch_size],
            coords2[coords2[:, 0] >= batch_size]
        ])
        raise ValueError(f"Batch index exceeds batch_size: {invalid_batch}")
    
    all_coords = torch.cat([coords1, coords2], dim=0)
    unique_coords, inverse_indices, counts = torch.unique(
        all_coords,
        dim=0,
        return_inverse=True,
        return_counts=True,
        sorted=True
    )
    
    # 调试输出
    # print(f"Input coords: {all_coords.shape[0]}, Unique coords: {unique_coords.shape[0]}, "
        #   f"Min counts: {counts.min().item()}, Max counts: {counts.max().item()}")
    
    split_idx = coords1.shape[0]
    indices1 = inverse_indices[:split_idx]
    indices2 = inverse_indices[split_idx:]
    
    combined_features = torch.zeros(
        unique_coords.shape[0], 
        features1.shape[1],
        dtype=dtype,
        device=device
    )
    
    combined_features.scatter_add_(0, indices1.unsqueeze(1).expand(-1, features1.shape[1]), features1)
    combined_features.scatter_add_(0, indices2.unsqueeze(1).expand(-1, features2.shape[1]), features2)
    
    # 计算加权平均值（可选）
    # ...
    
    return spconv.SparseConvTensor(
        features=combined_features,
        indices=unique_coords.int(),
        spatial_shape=sparse_shape,
        batch_size=batch_size
    )


def sparse_multiply(
        f1_sp: spconv.SparseConvTensor, 
        f2_sp: spconv.SparseConvTensor, 
        op="2dx3d",
        mini_batch_size=-1  # NOTE: 8\16 添加mini-batch参数 操作防止峰值显存太高
    ) -> spconv.SparseConvTensor:
    
    coords1 = f1_sp.indices  # [M, 3] (batch_idx, y, x)
    features1 = f1_sp.features  # [M, C1]
    spatial_shape1 = f1_sp.spatial_shape  # [H, W]
    
    coords2 = f2_sp.indices  # [N, 4] (batch_idx, z, y, x)
    features2 = f2_sp.features  # [N, C2]
    spatial_shape2 = f2_sp.spatial_shape  # [Z, H, W]
    batch_size = f2_sp.batch_size

    if op == '2dx3d':
        assert len(spatial_shape1) == 2, f"f1_sp should be 2D (spatial_shape: {spatial_shape1}), got {len(spatial_shape1)}D"
        assert len(spatial_shape2) == 3, f"f2_sp should be 3D (spatial_shape: {spatial_shape2}), got {len(spatial_shape2)}D"
        assert spatial_shape1 == spatial_shape2[1:], (f"Height-Width mismatch: "
                                                        f"2D tensor H,W={spatial_shape1[1:]}, "
                                                        f"3D tensor H,W={spatial_shape2[1:]}")
        assert features1.shape[1] == features2.shape[1], "Feature dimensions must match"

    elif op == '1dx3d':
        assert len(spatial_shape1) == 1, f"f1_sp should be 1D (spatial_shape: {spatial_shape1}), got {len(spatial_shape1)}D"
        assert len(spatial_shape2) == 3, f"f2_sp should be 3D (spatial_shape: {spatial_shape2}), got {len(spatial_shape2)}D"
        assert features1.shape[1] == 1, f"1D tensor must have feature_dim=1, got {features1.shape[1]}"
        assert spatial_shape1[0] == spatial_shape2[0], (f"Z-dimension mismatch: "
                                                        f"1D tensor Z={spatial_shape1[0]}, "
                                                        f"3D tensor Z={spatial_shape2[0]}")
    else:
        raise ValueError(f"Unsupported operation type: {op}")

    # 存储所有结果
    all_multiplied_features = []
    all_matched_coords = []
    
    # 处理 mini_batch_size <= 0 的情况（不使用小批量）
    if mini_batch_size <= 0:
        # 禁用梯度：坐标索引计算和匹配操作
        with torch.no_grad():
            if op == '2dx3d':
                H, W = spatial_shape1[0], spatial_shape1[1]
                # 重新计算相对batch索引（减少键值大小）
                relative_coords1 = coords1.clone()
                relative_coords2 = coords2.clone()
                
                keys1 = relative_coords1[:, 0] * (W * H) + relative_coords1[:, 1] * W + relative_coords1[:, 2]
                keys2 = relative_coords2[:, 0] * (W * H) + relative_coords2[:, 2] * W + relative_coords2[:, 3]
                
                sorted_keys1, sort_idx1 = torch.sort(keys1)
                
                pos = torch.searchsorted(sorted_keys1, keys2)
                
                valid_mask = (pos < len(sorted_keys1)) & (sorted_keys1[pos] == keys2)
                
                matched_coords = coords2[valid_mask]
                matched_idx1 = pos[valid_mask]
            
            elif op == '1dx3d':
                W = spatial_shape1[0]
                _, _, W2 = spatial_shape2
                # 重新计算相对batch索引
                relative_coords1 = coords1.clone()
                relative_coords2 = coords2.clone()
                
                keys1 = relative_coords1[:, 0] * W + relative_coords1[:, 1]
                keys2 = relative_coords2[:, 0] * W + relative_coords2[:, 1]
                
                sorted_keys1, sort_idx1 = torch.sort(keys1)
                
                pos = torch.searchsorted(sorted_keys1, keys2)
                
                valid_mask = (pos < len(sorted_keys1)) & (sorted_keys1[pos] == keys2)
                
                matched_coords = coords2[valid_mask]
                matched_idx1 = pos[valid_mask]
        
        # 特征操作部分需要保留梯度
        if op == '2dx3d':
            # 特征排序需要保留梯度
            sorted_features1 = features1[sort_idx1]
            
            matched_features2 = features2[valid_mask]
            matched_features1 = sorted_features1[matched_idx1]
            multiplied_features = matched_features1 * matched_features2
        elif op == '1dx3d':
            # 特征排序需要保留梯度
            sorted_features1 = features1[sort_idx1].squeeze(1)
            
            matched_features2 = features2[valid_mask]
            matched_features1 = sorted_features1[matched_idx1]
            multiplied_features = matched_features2 * matched_features1.unsqueeze(1)
        
        # 保存结果
        all_multiplied_features.append(multiplied_features)
        all_matched_coords.append(matched_coords)
    
    else:
        # 计算需要多少个mini-batch
        num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size
        
        for mini_idx in range(num_mini_batches):
            # 计算当前mini-batch的batch范围
            batch_start = mini_idx * mini_batch_size
            batch_end = min((mini_idx + 1) * mini_batch_size, batch_size)
            
            # 提取当前mini-batch的数据（保留梯度）
            mask1 = (coords1[:, 0] >= batch_start) & (coords1[:, 0] < batch_end)
            mask2 = (coords2[:, 0] >= batch_start) & (coords2[:, 0] < batch_end)
            
            coords1_mini = coords1[mask1]
            features1_mini = features1[mask1]
            
            coords2_mini = coords2[mask2]
            features2_mini = features2[mask2]
            
            # 如果当前mini-batch有数据，则处理
            if len(coords1_mini) == 0 or len(coords2_mini) == 0:
                continue
            
            # 禁用梯度：坐标索引计算和匹配操作
            with torch.no_grad():
                if op == '2dx3d':
                    H, W = spatial_shape1[0], spatial_shape1[1]
                    # 重新计算相对batch索引（减少键值大小）
                    relative_coords1 = coords1_mini.clone()
                    relative_coords2 = coords2_mini.clone()
                    relative_coords1[:, 0] -= batch_start
                    relative_coords2[:, 0] -= batch_start
                    
                    keys1 = relative_coords1[:, 0] * (W * H) + relative_coords1[:, 1] * W + relative_coords1[:, 2]
                    keys2 = relative_coords2[:, 0] * (W * H) + relative_coords2[:, 2] * W + relative_coords2[:, 3]
                    
                    sorted_keys1, sort_idx1 = torch.sort(keys1)
                    
                    pos = torch.searchsorted(sorted_keys1, keys2)
                    
                    valid_mask = (pos < len(sorted_keys1)) & (sorted_keys1[pos] == keys2)
                    
                    matched_coords_mini = coords2_mini[valid_mask]
                    matched_idx1 = pos[valid_mask]
                
                elif op == '1dx3d':
                    W = spatial_shape1[0]
                    _, _, W2 = spatial_shape2
                    # 重新计算相对batch索引
                    relative_coords1 = coords1_mini.clone()
                    relative_coords2 = coords2_mini.clone()
                    relative_coords1[:, 0] -= batch_start
                    relative_coords2[:, 0] -= batch_start
                    
                    keys1 = relative_coords1[:, 0] * W + relative_coords1[:, 1]
                    keys2 = relative_coords2[:, 0] * W + relative_coords2[:, 1]
                    
                    sorted_keys1, sort_idx1 = torch.sort(keys1)
                    
                    pos = torch.searchsorted(sorted_keys1, keys2)
                    
                    valid_mask = (pos < len(sorted_keys1)) & (sorted_keys1[pos] == keys2)
                    
                    matched_coords_mini = coords2_mini[valid_mask]
                    matched_idx1 = pos[valid_mask]
            
            # 特征操作部分需要保留梯度
            if op == '2dx3d':
                # 特征排序需要保留梯度
                sorted_features1 = features1_mini[sort_idx1]
                
                matched_features2_mini = features2_mini[valid_mask]
                matched_features1_mini = sorted_features1[matched_idx1]
                multiplied_features_mini = matched_features1_mini * matched_features2_mini
            elif op == '1dx3d':
                # 特征排序需要保留梯度
                sorted_features1 = features1_mini[sort_idx1].squeeze(1)
                
                matched_features2_mini = features2_mini[valid_mask]
                matched_features1_mini = sorted_features1[matched_idx1]
                multiplied_features_mini = matched_features2_mini * matched_features1_mini.unsqueeze(1)
            
            # 保存当前mini-batch的结果
            all_multiplied_features.append(multiplied_features_mini)
            all_matched_coords.append(matched_coords_mini)
    
    # 合并所有结果
    if len(all_multiplied_features) == 0:
        # 处理所有批次都没有数据的情况
        multiplied_features = torch.zeros(0, features1.shape[1], device=features1.device)
        matched_coords = torch.zeros(0, coords2.shape[1], dtype=coords2.dtype, device=coords2.device)
    else:
        multiplied_features = torch.cat(all_multiplied_features, dim=0)
        matched_coords = torch.cat(all_matched_coords, dim=0)
    
    return spconv.SparseConvTensor(
        features=multiplied_features,
        indices=matched_coords,
        spatial_shape=spatial_shape2,
        batch_size=batch_size
    )


# @torch.no_grad()
# def get_hilbert_index_3d_mamba_lite( # NOTE 06.28: Z-order 会更快吗？(当前为7~11ms)
#         template, coors, batch_size, 
#         z_dim, hilbert_spatial_size, 
#         shift=(0, 0, 0), origin_coors=None,
#         is_sample_points=False, debug=True):
#     '''
#     coors: (b, z, y, x)
#     shift: (shift_z, shift_y, shift_x)
#     hilbert_spatial_size: [z, y, x]
#     '''
#     hil_size_z, hil_size_y, hil_size_x = hilbert_spatial_size

#     # NOTE int 确保尺寸为整数
#     z_dim = int(z_dim)
#     hil_size_z = int(hil_size_z)
#     hil_size_y = int(hil_size_y)
#     hil_size_x = int(hil_size_x)

#     x = coors[:, 3] + shift[2]
#     y = coors[:, 2] + shift[1]
#     z = coors[:, 1] + shift[0]

#     if is_sample_points == True:

#         Y = hil_size_y
#         X = hil_size_x

#         # total_voxels = z_dim * Y * X
#         # NOTE *#06.12#*: int越界问题
#         total_voxels_tensor = torch.prod(torch.tensor([z_dim, Y, X], dtype=torch.int64))
#         if total_voxels_tensor > torch.iinfo(torch.int64).max: # 验证在 int64 范围内
#             raise ValueError(f"总体素数 {total_voxels_tensor} 超出 int64 范围")
#         total_voxels = total_voxels_tensor.item() # 获取安全的大小值（确保是整数）
#         if not isinstance(total_voxels, int):
#             raise TypeError(f"总体素数 {total_voxels} 不是整数类型")

#         # 1. 计算浮点坐标的linear index (作为浮点数)
#         floating_flat_coors = (
#             coors[:, 0] * Y * X + 
#             coors[:, 1] * X + 
#             coors[:, 2]
#         )
#         original_flat_coors = (
#             origin_coors[:, 0] * Y * X + 
#             origin_coors[:, 1] * X + 
#             origin_coors[:, 2]
#         ).long()  # NOTE *#06.12#*: `.long()` 转换为长整型，用于索引
        
#         # 2. 获取所有空体素的索引
#         # 创建所有可能的索引 (0 到 total_voxels - 1)
#         # all_possible_indices = torch.arange(total_voxels, device=coors.device)
#         all_possible_indices = torch.arange(total_voxels, device=coors.device, dtype=torch.long) # NOTE *#06.12#*: dtype 更改
        
#         # 从所有索引中减去origin_flat_coors得到空体素索引
#         # 使用集合操作找出不在origin_flat_coors中的索引
#         mask = torch.ones(
#             (total_voxels,), # NOTE *#06.12#*: 使用元组形式
#             # total_voxels, 
#             dtype=torch.bool, device=coors.device)
#         mask[original_flat_coors] = False  # 标记被占用的位置为False
#         empty_indices = all_possible_indices[mask]
        
#         # 3. 合并浮点坐标索引和空体素索引
#         # 注意：空体素索引是整数，浮点坐标索引是浮点数
#         combined_coords = torch.cat([floating_flat_coors, empty_indices.float()])
        
#         # 4. 排序所有坐标索引
#         sorted_indices = torch.argsort(combined_coords)
        
#         # 5. 获取浮点坐标在排序后序列中的位置
#         # 浮点坐标在合并序列中的索引范围是0到(N-1)
#         positions = torch.zeros_like(combined_coords).long()
#         positions[sorted_indices] = torch.arange(len(combined_coords), device=coors.device)
        
#         # 提取浮点坐标的位置
#         floating_positions = positions[:len(floating_flat_coors)]
        
#         # 6. 从template获取Hilbert索引
#         template = template.to(coors.device)
#         hil_inds = template[floating_positions.long()].long()

#     else:
#         # 计算非采样点索引顺序
#         flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
#         template = template.to(flat_coors.device) # NOTE: move tempate to cuda 
#         hil_inds = template[flat_coors].long()

#     inds_curt_to_next = {}
#     inds_next_to_curt = {}
#     for i in range(batch_size):
#         batch_mask = coors[:, 0] == i
#         if not batch_mask.any():
#             pass
#         inds_curt_to_next[i] = torch.argsort(hil_inds[batch_mask])
#         inds_next_to_curt[i] = torch.argsort(inds_curt_to_next[i])
#         # 假设 hil_inds_batch 是 [10, 5, 7]，那么 torch.argsort 返回 [1, 2, 0]
#         # inds_curt_to_next[i] 是一个整数数组，表示：原始顺序（当前顺序）中的点，按照希尔伯特索引排序后，应该放置的位置。
#         # inds_next_to_curt[i] 是 inds_curt_to_next[i] 的逆排列。
#         # 如果 inds_curt_to_next[i] 将原始索引映射到新的排序位置，那么 inds_next_to_curt[i] 将排序后的位置映射回原始索引。
#         # 例如，如果 inds_curt_to_next[i] = [1, 2, 0]，那么 torch.argsort([1, 2, 0]) 返回 [2, 0, 1]


#     index_info = {}
#     index_info['inds_curt_to_next'] = inds_curt_to_next
#     index_info['inds_next_to_curt'] = inds_next_to_curt

#     return index_info


@torch.no_grad()
def get_hilbert_index_3d_mamba_lite(
        template, coors, batch_size, 
        z_dim, hilbert_spatial_size, 
        shift=(0, 0, 0), origin_coors=None,
        is_sample_points=False, debug=True
    ):
    '''
    coors: (b, z, y, x)
    shift: (shift_z, shift_y, shift_x)
    hilbert_spatial_size: [z, y, x]
    '''
    hil_size_z, hil_size_y, hil_size_x = hilbert_spatial_size

    # NOTE int 确保尺寸为整数
    z_dim = int(z_dim)
    hil_size_z = int(hil_size_z)
    hil_size_y = int(hil_size_y)
    hil_size_x = int(hil_size_x)

    x = coors[:, 3] + shift[2]
    y = coors[:, 2] + shift[1]
    z = coors[:, 1] + shift[0]

    flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
    template = template.to(flat_coors.device) # NOTE: move tempate to cuda 
    hil_inds = template[flat_coors].long()

    inds_curt_to_next = {}
    inds_next_to_curt = {}
    for i in range(batch_size):
        batch_mask = coors[:, 0] == i
        if not batch_mask.any():
            pass
        inds_curt_to_next[i] = torch.argsort(hil_inds[batch_mask])
        inds_next_to_curt[i] = torch.argsort(inds_curt_to_next[i])

    index_info = {}
    index_info['inds_curt_to_next'] = inds_curt_to_next
    index_info['inds_next_to_curt'] = inds_next_to_curt

    return index_info


# @torch.no_grad()
# def get_morton_index_3d(
#         coors, 
#         batch_size,
#         spatial_size, 
#         shift=(0, 0, 0), 
#         primary_axis='x',  # 'x' or 'y'
#         bits=21  
# ):
#     """
#     高效莫顿排序实现
#     coors: (N, 4) 张量 [batch_idx, z, y, x]
#     spatial_size: [z_size, y_size, x_size]
#     shift: (shift_z, shift_y, shift_x)
#     primary_axis: 主序轴选择，'x'或'y'
#     bits: 编码位数
#     """
#     # 确保主序轴参数有效
#     assert primary_axis in ('x', 'y'), "primary_axis must be 'x' or 'y'"
    
#     # 解包空间尺寸
#     size_z, size_y, size_x = spatial_size
    
#     # 获取坐标值并确保在有效范围内
#     x = (coors[:, 3] + shift[2]).clamp(min=0, max=size_x-1).long()
#     y = (coors[:, 2] + shift[1]).clamp(min=0, max=size_y-1).long()
#     z = (coors[:, 1] + shift[0]).clamp(min=0, max=size_z-1).long()
    
#     # 莫顿编码张量
#     morton_codes = torch.zeros_like(x, dtype=torch.int64)
    
#     # 根据主序轴选择不同的交错方式
#     if primary_axis == 'x':
#         # 传统模式：X 作为主序轴 (XX XX XX... X Y Z Y Z Y Z)
#         for i in range(bits):
#             # 交错的位模式：X,Y,Z
#             morton_codes |= ((x >> i) & 1) << (3*i)      # 最低位：X
#             morton_codes |= ((y >> i) & 1) << (3*i + 1)  # 中间位：Y
#             morton_codes |= ((z >> i) & 1) << (3*i + 2)  # 最高位：Z
            
#     else:  # primary_axis == 'y'
#         # 改进模式：Y 作为主序轴 (YY YY YY... Y X Z X Z X Z)
#         for i in range(bits):
#             # 交错的位模式：Y,X,Z
#             morton_codes |= ((y >> i) & 1) << (3*i)      # 最低位：Y
#             morton_codes |= ((x >> i) & 1) << (3*i + 1)  # 中间位：X
#             morton_codes |= ((z >> i) & 1) << (3*i + 2)  # 最高位：Z

#     # 按批次分组并排序
#     batch_indices = coors[:, 0].long()
#     batch_ids, inverse, counts = batch_indices.unique(
#         return_inverse=True, 
#         return_counts=True, 
#         sorted=True
#     )
    
#     # 空批次处理
#     if len(batch_ids) == 0:
#         return {'inds_curt_to_next': {}, 'inds_next_to_curt': {}}
    
#     # 计算批次边界
#     cum_counts = counts.cumsum(0)
#     starts = torch.cat([torch.tensor([0], device=cum_counts.device), cum_counts[:-1]])
#     ends = cum_counts
    
#     # 为每个批次生成排序索引
#     inds_curt_to_next = {}
#     inds_next_to_curt = {}
    
#     for i, bid in enumerate(batch_ids):
#         start, end = starts[i], ends[i]
#         batch_codes = morton_codes[start:end]
        
#         # 排序当前批次
#         sorted_indices = torch.argsort(batch_codes)
#         # 高效创建逆序索引
#         inv_sorted_indices = torch.zeros_like(sorted_indices)
#         inv_sorted_indices.scatter_(0, sorted_indices, torch.arange(len(sorted_indices)))
        
#         bid = bid.item()
#         inds_curt_to_next[bid] = sorted_indices
#         inds_next_to_curt[bid] = inv_sorted_indices

#     return {
#         'inds_curt_to_next': inds_curt_to_next,
#         'inds_next_to_curt': inds_next_to_curt
#     }


@torch.no_grad()
def get_morton_index_3d(
        coors, 
        batch_size, 
        spatial_size, 
        shift=(0, 0, 0), 
        primary_axis='x',  # 'x' or 'y'
        bits=21  
):
    """
    高效莫顿排序实现 - 优化批处理版本（修复设备不一致问题）
    coors: (N, 4) 张量 [batch_idx, z, y, x]
    batch_size: 总批次数
    spatial_size: [z_size, y_size, x_size]
    shift: (shift_z, shift_y, shift_x)
    primary_axis: 主序轴选择，'x'或'y'
    bits: 编码位数
    """
    assert primary_axis in ('x', 'y'), "primary_axis must be 'x' or 'y'"
    
    size_z, size_y, size_x = spatial_size
    
    x = (coors[:, 3] + shift[2]).clamp(min=0, max=size_x-1).long()
    y = (coors[:, 2] + shift[1]).clamp(min=0, max=size_y-1).long()
    z = (coors[:, 1] + shift[0]).clamp(min=0, max=size_z-1).long()
    
    device = x.device
    
    morton_codes = torch.zeros_like(x, dtype=torch.int64)
    
    if primary_axis == 'x':
        for i in range(bits):
            morton_codes |= ((x >> i) & 1) << (3*i)      # 最低位：X
            morton_codes |= ((y >> i) & 1) << (3*i + 1)  # 中间位：Y
            morton_codes |= ((z >> i) & 1) << (3*i + 2)  # 最高位：Z
    else:  # primary_axis == 'y'
        for i in range(bits):
            morton_codes |= ((y >> i) & 1) << (3*i)      # 最低位：Y
            morton_codes |= ((x >> i) & 1) << (3*i + 1)  # 中间位：X
            morton_codes |= ((z >> i) & 1) << (3*i + 2)  # 最高位：Z

    batch_indices = coors[:, 0].long()
    
    inds_curt_to_next = {}
    inds_next_to_curt = {}
    
    counts = torch.bincount(batch_indices, minlength=batch_size)
    
    cum_counts = counts.cumsum(0)
    starts = torch.cat([torch.tensor([0], device=device), cum_counts[:-1]])
    
    for batch_id in range(batch_size):
        start_idx = starts[batch_id]
        end_idx = cum_counts[batch_id]
        
        if end_idx == start_idx:
            continue
            
        batch_codes = morton_codes[start_idx:end_idx]
        
        sorted_indices = torch.argsort(batch_codes)
        
        inv_sorted_indices = torch.zeros(len(sorted_indices), dtype=sorted_indices.dtype, device=device)
        inv_sorted_indices.scatter_(
            0, 
            sorted_indices, 
            torch.arange(len(sorted_indices), dtype=torch.long, device=device)
        )
        
        inds_curt_to_next[batch_id] = sorted_indices
        inds_next_to_curt[batch_id] = inv_sorted_indices

    return {
        'inds_curt_to_next': inds_curt_to_next,
        'inds_next_to_curt': inds_next_to_curt
    }




@torch.no_grad()
def get_random_index_3d(
        coors, 
        batch_size, 
        spatial_size=None,  # 为保持接口一致，但实际不使用
        shift=(0, 0, 0),   # 为保持接口一致，但实际不使用
        primary_axis=None,  # 为保持接口一致，但实际不使用
        bits=None,          # 为保持接口一致，但实际不使用
        seed=None           # 可选随机种子
):
    """
    随机排序实现 - 为每个批次生成随机索引顺序
    
    参数:
        coors: (N, 4) 张量 [batch_idx, z, y, x]
        batch_size: 总批次数
        spatial_size: 空间尺寸（未使用，仅为接口兼容）
        shift: 偏移量（未使用，仅为接口兼容）
        primary_axis: 主序轴（未使用，仅为接口兼容）
        bits: 编码位数（未使用，仅为接口兼容）
        seed: 随机种子（可选）
    
    返回:
        包含排序索引和逆排序索引的字典
    """
    device = coors.device
    batch_indices = coors[:, 0].long()
    
    # 设置随机种子（如果提供）
    if seed is not None:
        torch.manual_seed(seed)
    
    inds_curt_to_next = {}
    inds_next_to_curt = {}
    
    # 计算每个批次的点数
    counts = torch.bincount(batch_indices, minlength=batch_size)
    cum_counts = counts.cumsum(0)
    starts = torch.cat([torch.tensor([0], device=device), cum_counts[:-1]])
    
    for batch_id in range(batch_size):
        start_idx = starts[batch_id]
        end_idx = cum_counts[batch_id]
        
        if end_idx == start_idx:
            # 空批次处理
            inds_curt_to_next[batch_id] = torch.tensor([], dtype=torch.long, device=device)
            inds_next_to_curt[batch_id] = torch.tensor([], dtype=torch.long, device=device)
            continue
            
        # 当前批次的点数
        num_points = end_idx - start_idx
        
        # 生成随机排列
        rand_indices = torch.randperm(num_points, device=device)
        
        # 创建逆排列
        inv_rand_indices = torch.zeros(num_points, dtype=torch.long, device=device)
        inv_rand_indices.scatter_(
            0, 
            rand_indices, 
            torch.arange(num_points, dtype=torch.long, device=device)
        )
        
        inds_curt_to_next[batch_id] = rand_indices
        inds_next_to_curt[batch_id] = inv_rand_indices

    return {
        'inds_curt_to_next': inds_curt_to_next,
        'inds_next_to_curt': inds_next_to_curt
    }



def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,#
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError
    
    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py

        # 定义可能的激活函数属性名称列表
        activation_attr_names = ['act', 'act_fn', 'activation', 'nonlinearity']

        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                
                # nn.init.kaiming_uniform_(p, a=math.sqrt(5))

                activation_type = None
                for attr_name in activation_attr_names:
                    if hasattr(module, attr_name):
                        act_fn = getattr(module, attr_name)
                        
                        # 处理不同类型的激活函数表示
                        if isinstance(act_fn, nn.ReLU):
                            activation_type = 'relu'
                            break
                        elif isinstance(act_fn, nn.SiLU):
                            activation_type = 'silu'
                            break
                        elif isinstance(act_fn, nn.LeakyReLU):
                            activation_type = 'leaky_relu'
                            a = act_fn.negative_slope
                            break
                        elif callable(act_fn):
                            # 通过函数名识别
                            fn_name = act_fn.__name__.lower()
                            if 'relu' in fn_name:
                                activation_type = 'relu'
                                break
                            elif 'silu' in fn_name:
                                activation_type = 'silu'
                                break
                
                # 2. 如果没找到，尝试模块的默认激活
                if activation_type is None:
                    if hasattr(module, 'default_activation'):
                        if module.default_activation == 'relu':
                            activation_type = 'relu'
                        elif module.default_activation == 'silu':
                            activation_type = 'silu'
                
                # 3. 应用初始化
                if activation_type == 'relu':
                    nn.init.kaiming_uniform_(p, nonlinearity='relu')
                elif activation_type == 'leaky_relu':
                    nn.init.kaiming_uniform_(p, a=a, nonlinearity='leaky_relu')  # 需传递斜率参数 a
                elif activation_type == 'tanh':
                    nn.init.kaiming_uniform_(p, nonlinearity='tanh')
                elif activation_type == 'sigmoid':
                    nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')
                elif activation_type in ['silu', 'gelu', 'mish', 'swish']:
                    nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 用 ReLU 近似
                else:  # 默认回退方案
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))

                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
                
                # print(f"缩放初始化: {name} with act type {activation_type} in {module}")




# @torch.no_grad()
# def get_hilbert_index_3d_mamba_lite(
#         template, coors, batch_size, 
#         z_dim, hilbert_spatial_size, 
#         shift=(0, 0, 0), origin_coors=None,
#         is_sample_points=False):
#     device = coors.device
#     template = template.to(device)
    
#     # 提前计算并转换尺寸为整数
#     hil_size_z, hil_size_y, hil_size_x = [int(s) for s in hilbert_spatial_size]
#     z_dim = int(z_dim)
    
#     # 确保批次大小至少为1
#     if batch_size < 1:
#         batch_size = 1
    
#     # 处理空输入情况
#     if coors.shape[0] == 0:
#         # 创建空结果
#         empty_dict = {
#             0: torch.empty(0, dtype=torch.long, device=device)
#         }
#         return {
#             'inds_curt_to_next': empty_dict,
#             'inds_next_to_curt': empty_dict
#         }
    
#     # 应用坐标偏移
#     batch_idx = coors[:, 0]
#     z = coors[:, 1] + shift[0]
#     y = coors[:, 2] + shift[1]
#     x = coors[:, 3] + shift[2]
    
#     # 提前计算重用值
#     XY = hil_size_y * hil_size_x
#     ZXY = z_dim * XY
    
#     # 处理采样点模式
#     if is_sample_points:
#         if origin_coors is None:
#             raise ValueError("采样点模式需要提供 origin_coors")
        
#         origin_coors = origin_coors.to(device)
#         orig_z = origin_coors[:, 1] if origin_coors.size(1) > 3 else origin_coors[:, 0]
#         orig_y = origin_coors[:, 2] if origin_coors.size(1) > 3 else origin_coors[:, 1]
#         orig_x = origin_coors[:, 3] if origin_coors.size(1) > 3 else origin_coors[:, 2]
        
#         # 原始平铺坐标
#         original_flat_coors = (orig_z * XY + orig_y * hil_size_x + orig_x).long()
        
#         # 采样点浮点坐标
#         floating_flat_coors = (z * XY + y * hil_size_x + x)
        
#         # 计算有效体素总数
#         total_voxels = ZXY
#         if total_voxels > torch.iinfo(torch.int64).max:
#             raise ValueError(f"总素数量 {total_voxels} 超出 int64 范围")
        
#         # 生成空体素索引
#         all_indices = torch.arange(total_voxels, device=device, dtype=torch.long)
#         mask = torch.ones(total_voxels, dtype=torch.bool, device=device)
#         mask.scatter_(0, original_flat_coors, False)
#         empty_indices = all_indices[mask]
        
#         # 合并坐标并排序
#         combined_coords = torch.cat([floating_flat_coors, empty_indices.float()])
#         sorted_indices = combined_coords.argsort()
        
#         # 获取希尔伯特索引
#         positions = torch.empty_like(combined_coords, dtype=torch.long)
#         positions[sorted_indices] = torch.arange(sorted_indices.size(0), device=device)
#         floating_positions = positions[:floating_flat_coors.size(0)]
#         hil_inds = template[floating_positions].long()

#     else:
#         # 非采样点模式
#         flat_coors = (z * XY + y * hil_size_x + x).long()
#         hil_inds = template[flat_coors].long()

#     # 处理可能的空批次情况
#     valid_batches = torch.unique(batch_idx).int().tolist()
#     inds_curt_to_next = {}
#     inds_next_to_curt = {}
    
#     # 对每个批次进行排序
#     for i in range(batch_size):
#         batch_mask = batch_idx == i
#         batch_indices = batch_mask.nonzero(as_tuple=True)[0]
        
#         # 如果批次为空
#         if batch_indices.numel() == 0:
#             # 确保返回1维空张量而不是0维张量
#             inds_curt_to_next[i] = torch.tensor([], dtype=torch.long, device=device)
#             inds_next_to_curt[i] = torch.tensor([], dtype=torch.long, device=device)
#             continue
            
#         batch_hil_inds = hil_inds[batch_mask]
#         sorted_indices = torch.argsort(batch_hil_inds)
        
#         # 确保结果是1维张量
#         if sorted_indices.dim() == 0:
#             sorted_indices = sorted_indices.unsqueeze(0)
        
#         # 创建当前到下一映射
#         inds_curt_to_next[i] = sorted_indices
#         inds_next_to_curt[i] = torch.argsort(sorted_indices)
        
#         # 确保映射是1维张量
#         if inds_next_to_curt[i].dim() == 0:
#             inds_next_to_curt[i] = inds_next_to_curt[i].unsqueeze(0)
#         if inds_curt_to_next[i].dim() == 0:
#             inds_curt_to_next[i] = inds_curt_to_next[i].unsqueeze(0)

#     return {
#         'inds_curt_to_next': inds_curt_to_next,
#         'inds_next_to_curt': inds_next_to_curt
#     }


# NOTE *#06.12#*: 不支持大 batch_size = bs * L, 且会导致计算结果非空体素个数剧减。index_add_ 可能导致某些问题且效率慢。并且后两个传入参数没正确用上
# def sparse_add(f1_sp: spconv.SparseConvTensor, f2_sp: spconv.SparseConvTensor, sparse_shape=None, batch_size=None) -> spconv.SparseConvTensor:

#     sparse_shape = f1_sp.spatial_shape
#     batch_size = f1_sp.batch_size

#     coords1 = f1_sp.indices  # [M, 4]
#     features1 = f1_sp.features  # [M, C]
    
#     coords2 = f2_sp.indices  # [N, 4]
#     features2 = f2_sp.features  # [N, C]
    
#     all_coords = torch.cat([coords1, coords2], dim=0)  # [M+N, 4]
    
#     unique_coords, inverse_indices = torch.unique(all_coords, dim=0, return_inverse=True)
    
#     combined_features = torch.zeros(
#         (unique_coords.size(0), features1.size(1)), 
#         dtype=features1.dtype, 
#         device=features1.device
#     )
    
#     # 将第一个张量的特征添加到对应位置 前M个点来自第一个张量
#     indices1 = inverse_indices[:coords1.size(0)]
#     combined_features.index_add_(0, indices1, features1)
    
#     # 将第二个张量的特征添加到对应位置 后N个点来自第二个张量
#     indices2 = inverse_indices[coords1.size(0):]
#     combined_features.index_add_(0, indices2, features2)
    
#     mixture_sp = spconv.SparseConvTensor(
#         features=combined_features,
#         indices=unique_coords,
#         spatial_shape=sparse_shape,
#         batch_size=batch_size
#     )

#     return mixture_sp


# NOTE #06.12#*: 完全没有相同位置的体素可相乘时可能会出问题, 可能不应该修复这里(顶多报个错)，而是用它的地方
# def sparse_multiply(
#         f1_sp: spconv.SparseConvTensor, 
#         f2_sp: spconv.SparseConvTensor, 
#         op="2dx3d"
#     ) -> spconv.SparseConvTensor:
#     '''
#     2dx3d -> [Z]
#     '''
    
#     coords1 = f1_sp.indices  # [M, 3] (batch_idx, y, x)
#     features1 = f1_sp.features  # [M, C1]
#     spatial_shape1 = f1_sp.spatial_shape  # [H, W]
    
#     coords2 = f2_sp.indices  # [N, 4] (batch_idx, z, y, x)
#     features2 = f2_sp.features  # [N, C2]
#     spatial_shape2 = f2_sp.spatial_shape  # [Z, H, W]
#     batch_size = f2_sp.batch_size

#     if op == '2dx3d':
#         assert len(spatial_shape1) == 2, f"f1_sp should be 2D (spatial_shape: {spatial_shape1}), got {len(spatial_shape1)}D"
#         assert len(spatial_shape2) == 3, f"f2_sp should be 3D (spatial_shape: {spatial_shape2}), got {len(spatial_shape2)}D"
#         assert spatial_shape1 == spatial_shape2[1:], (f"Height-Width mismatch: "
#                                                         f"2D tensor H,W={spatial_shape1[1:]}, "
#                                                         f"3D tensor H,W={spatial_shape2[1:]}")
#         assert features1.shape[1] == features2.shape[1], "Feature dimensions must match"

#         # 索引越界判断, 检查坐标是否在有效范围内
#         # H, W = spatial_shape1[0], spatial_shape1[1]
#         # assert (coords1[:, 1] >= 0).all() and (coords1[:, 1] < H).all(), "coords1 y坐标越界"
#         # assert (coords1[:, 2] >= 0).all() and (coords1[:, 2] < W).all(), "coords1 x坐标越界"
#         # assert (coords2[:, 2] >= 0).all() and (coords2[:, 2] < H).all(), "coords2 y坐标越界"
#         # assert (coords2[:, 3] >= 0).all() and (coords2[:, 3] < W).all(), "coords2 x坐标越界"

#     elif op == '1dx3d':
#         assert len(spatial_shape1) == 1, f"f1_sp should be 1D (spatial_shape: {spatial_shape1}), got {len(spatial_shape1)}D"
#         assert len(spatial_shape2) == 3, f"f2_sp should be 3D (spatial_shape: {spatial_shape2}), got {len(spatial_shape2)}D"
#         assert features1.shape[1] == 1, f"1D tensor must have feature_dim=1, got {features1.shape[1]}"
#         assert spatial_shape1[0] == spatial_shape2[0], (f"Z-dimension mismatch: "
#                                                         f"1D tensor Z={spatial_shape1[0]}, "
#                                                         f"3D tensor Z={spatial_shape2[0]}")
#     else:
#         raise ValueError(f"Unsupported operation type: {op}")

    
#     if op == '2dx3d':
#         # 步骤1: 为2D坐标创建扁平化键值
#         # 使用torch.searchsorted实现高效查找
#         H, W = spatial_shape1[0], spatial_shape1[1]
#         keys1 = coords1[:, 0] * (W * H) + coords1[:, 1] * W + coords1[:, 2]
        
#         # 步骤2: 为3D坐标创建类似的扁平化键值（忽略z坐标）
#         keys2 = coords2[:, 0] * (W * H) + coords2[:, 2] * W + coords2[:, 3]
        
#         # 步骤3: 排序2D键值以便高效搜索
#         sorted_keys1, sort_idx1 = torch.sort(keys1)
#         sorted_features1 = features1[sort_idx1]
        
#         # 步骤4: 在2D键值中搜索3D键值
#         pos = torch.searchsorted(sorted_keys1, keys2)
        
#         # 步骤5: 检查是否存在有效的匹配
#         # 创建一个掩码标记有效的匹配点
#         valid_mask = (pos < len(sorted_keys1)) & (sorted_keys1[pos] == keys2)
        
#         # 步骤6: 只保留有匹配的3D点
#         matched_coords = coords2[valid_mask]
#         matched_features2 = features2[valid_mask]
        
#         # 步骤7: 获取对应的2D特征
#         matched_idx1 = pos[valid_mask]
#         matched_features1 = sorted_features1[matched_idx1]
        
#         # 步骤8: 执行特征乘法
#         multiplied_features = matched_features1 * matched_features2
    
#     elif op == '1dx3d':
#         W = spatial_shape1[0]
#         _, _, W2 = spatial_shape2
#         keys1 = coords1[:, 0] * W + coords1[:, 1]
#         keys2 = coords2[:, 0] * W + coords2[:, 1]
        
#         sorted_keys1, sort_idx1 = torch.sort(keys1)
#         sorted_features1 = features1[sort_idx1].squeeze(1)
        
#         pos = torch.searchsorted(sorted_keys1, keys2)
        
#         valid_mask = (pos < len(sorted_keys1)) & (sorted_keys1[pos] == keys2)
        
#         matched_coords = coords2[valid_mask]
#         matched_features2 = features2[valid_mask]
        
#         matched_idx1 = pos[valid_mask]
#         matched_features1 = sorted_features1[matched_idx1]  # [K] 权重向量
        
#         # 执行权重乘法 (广播)
#         multiplied_features = matched_features2 * matched_features1.unsqueeze(1)


#     result_sp = spconv.SparseConvTensor(
#         features=multiplied_features,
#         indices=matched_coords.int(),
#         spatial_shape=spatial_shape2,
#         batch_size=batch_size
#     )
    
#     return result_sp    
    
# NOTE 06.29: origin without no_grad()
# def sparse_multiply(
#         f1_sp: spconv.SparseConvTensor, 
#         f2_sp: spconv.SparseConvTensor, 
#         op="2dx3d",
#         mini_batch_size=8  # NOTE: 添加mini-batch参数 操作防止峰值显存太高
#     ) -> spconv.SparseConvTensor:
    
#     coords1 = f1_sp.indices  # [M, 3] (batch_idx, y, x)
#     features1 = f1_sp.features  # [M, C1]
#     spatial_shape1 = f1_sp.spatial_shape  # [H, W]
    
#     coords2 = f2_sp.indices  # [N, 4] (batch_idx, z, y, x)
#     features2 = f2_sp.features  # [N, C2]
#     spatial_shape2 = f2_sp.spatial_shape  # [Z, H, W]
#     batch_size = f2_sp.batch_size

#     if op == '2dx3d':
#         assert len(spatial_shape1) == 2, f"f1_sp should be 2D (spatial_shape: {spatial_shape1}), got {len(spatial_shape1)}D"
#         assert len(spatial_shape2) == 3, f"f2_sp should be 3D (spatial_shape: {spatial_shape2}), got {len(spatial_shape2)}D"
#         assert spatial_shape1 == spatial_shape2[1:], (f"Height-Width mismatch: "
#                                                         f"2D tensor H,W={spatial_shape1[1:]}, "
#                                                         f"3D tensor H,W={spatial_shape2[1:]}")
#         assert features1.shape[1] == features2.shape[1], "Feature dimensions must match"

#     elif op == '1dx3d':
#         assert len(spatial_shape1) == 1, f"f1_sp should be 1D (spatial_shape: {spatial_shape1}), got {len(spatial_shape1)}D"
#         assert len(spatial_shape2) == 3, f"f2_sp should be 3D (spatial_shape: {spatial_shape2}), got {len(spatial_shape2)}D"
#         assert features1.shape[1] == 1, f"1D tensor must have feature_dim=1, got {features1.shape[1]}"
#         assert spatial_shape1[0] == spatial_shape2[0], (f"Z-dimension mismatch: "
#                                                         f"1D tensor Z={spatial_shape1[0]}, "
#                                                         f"3D tensor Z={spatial_shape2[0]}")
#     else:
#         raise ValueError(f"Unsupported operation type: {op}")

#     # 存储所有mini-batch的结果
#     all_multiplied_features = []
#     all_matched_coords = []
    
#     # 计算需要多少个mini-batch
#     num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size
    
#     for mini_idx in range(num_mini_batches):
#         # 计算当前mini-batch的batch范围
#         batch_start = mini_idx * mini_batch_size
#         batch_end = min((mini_idx + 1) * mini_batch_size, batch_size)
        
#         # 提取当前mini-batch的数据
#         mask1 = (coords1[:, 0] >= batch_start) & (coords1[:, 0] < batch_end)
#         coords1_mini = coords1[mask1]
#         features1_mini = features1[mask1]
        
#         mask2 = (coords2[:, 0] >= batch_start) & (coords2[:, 0] < batch_end)
#         coords2_mini = coords2[mask2]
#         features2_mini = features2[mask2]
        
#         # 如果当前mini-batch有数据，则处理
#         if len(coords1_mini) == 0 or len(coords2_mini) == 0:
#             continue
        
#         if op == '2dx3d':
#             H, W = spatial_shape1[0], spatial_shape1[1]
#             # 重新计算相对batch索引（减少键值大小）
#             relative_coords1 = coords1_mini.clone()
#             relative_coords2 = coords2_mini.clone()
#             relative_coords1[:, 0] -= batch_start
#             relative_coords2[:, 0] -= batch_start
            
#             keys1 = relative_coords1[:, 0] * (W * H) + relative_coords1[:, 1] * W + relative_coords1[:, 2]
#             keys2 = relative_coords2[:, 0] * (W * H) + relative_coords2[:, 2] * W + relative_coords2[:, 3]
            
#             sorted_keys1, sort_idx1 = torch.sort(keys1)
#             sorted_features1 = features1_mini[sort_idx1]
            
#             pos = torch.searchsorted(sorted_keys1, keys2)
            
#             valid_mask = (pos < len(sorted_keys1)) & (sorted_keys1[pos] == keys2)
            
#             matched_coords_mini = coords2_mini[valid_mask]
#             matched_features2_mini = features2_mini[valid_mask]
            
#             matched_idx1 = pos[valid_mask]
#             matched_features1_mini = sorted_features1[matched_idx1]
            
#             multiplied_features_mini = matched_features1_mini * matched_features2_mini
            
#         elif op == '1dx3d':
#             W = spatial_shape1[0]
#             _, _, W2 = spatial_shape2
#             # 重新计算相对batch索引
#             relative_coords1 = coords1_mini.clone()
#             relative_coords2 = coords2_mini.clone()
#             relative_coords1[:, 0] -= batch_start
#             relative_coords2[:, 0] -= batch_start
            
#             keys1 = relative_coords1[:, 0] * W + relative_coords1[:, 1]
#             keys2 = relative_coords2[:, 0] * W + relative_coords2[:, 1]
            
#             sorted_keys1, sort_idx1 = torch.sort(keys1)
#             sorted_features1 = features1_mini[sort_idx1].squeeze(1)
            
#             pos = torch.searchsorted(sorted_keys1, keys2)
            
#             valid_mask = (pos < len(sorted_keys1)) & (sorted_keys1[pos] == keys2)
            
#             matched_coords_mini = coords2_mini[valid_mask]
#             matched_features2_mini = features2_mini[valid_mask]
            
#             matched_idx1 = pos[valid_mask]
#             matched_features1_mini = sorted_features1[matched_idx1]
            
#             multiplied_features_mini = matched_features2_mini * matched_features1_mini.unsqueeze(1)
        
#         # 保存当前mini-batch的结果
#         all_multiplied_features.append(multiplied_features_mini)
#         all_matched_coords.append(matched_coords_mini)
    
#     # 合并所有mini-batch的结果
#     if len(all_multiplied_features) == 0:
#         # 处理所有批次都没有数据的情况
#         multiplied_features = torch.zeros(0, features1.shape[1], device=features1.device)
#         matched_coords = torch.zeros(0, coords2.shape[1], dtype=coords2.dtype, device=coords2.device)
#     else:
#         multiplied_features = torch.cat(all_multiplied_features, dim=0)
#         matched_coords = torch.cat(all_matched_coords, dim=0)

#     result_sp = spconv.SparseConvTensor(
#         features=multiplied_features,
#         indices=matched_coords.int(),
#         spatial_shape=spatial_shape2,
#         batch_size=batch_size
#     )
    
#     return result_sp





# class XFormersCrossAttention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         attn_type='vanilla',
#         res_norm_fn=None,
#         dropout=0.0,
#         attn_cfg=None  # 统一参数配置字典
#     ):
#         super().__init__()
#         self.dim = dim
#         self.attn_type = attn_type
#         self.dropout = dropout
#         self.attn_cfg = attn_cfg or {
#             # 默认参数配置
#             'window_size': 32,    # 局部窗口注意力
#             'topk': 4,            # 动态稀疏注意力
#             'low_rank': 16,       # 低秩注意力
#             'phi': 'relu',        # 线性注意力核函数
#         }
        
#         assert attn_type in [
#             'vanilla', 'flash', 'sparse', 'block',
#             'linear', 'low_rank', 'window', 'dynamic', 'base'
#         ], f'Invalid attn_type: {attn_type}'

#         if res_norm_fn is not None:
#             self.res_norm_fn = res_norm_fn(dim)
#         else:
#             self.res_norm_fn = None
#         self.scale = 1.0 / math.sqrt(self.dim)

#         # 初始化各注意力类型的特有组件
#         if attn_type == 'linear':
#             # 线性注意力核函数
#             self.phi = {
#                 'relu': nn.ReLU(),
#                 'elu': nn.ELU(),
#                 'softplus': nn.Softplus()
#             }[self.attn_cfg.get('phi', 'relu')]
            
#         elif attn_type == 'low_rank':
#             # 低秩分解矩阵
#             self.U = nn.Linear(dim, self.attn_cfg['low_rank'], bias=False)
#             self.V = nn.Linear(dim, self.attn_cfg['low_rank'], bias=False)
            
#     def forward(self, Q, K, V, residual=None):
#         assert (residual is None and self.res_norm_fn is None) or \
#                (residual is not None and self.res_norm_fn is not None), \
#                'residual and res_norm_fn error'

#         # 根据注意力类型选择实现
#         if self.attn_type in ['vanilla', 'flash', 'sparse', 'block']:
#             # xformers原生实现
#             attn_kwargs = {
#                 'query': Q,
#                 'key': K,
#                 'value': V,
#                 'scale': self.scale,
#                 'p': self.dropout,
#                 **{k: v for k, v in self.attn_cfg.items() 
#                    if k in ['attn_bias', 'causal', 'window_size', 'sparsity_config']}
#             }
#             attn_output = getattr(xops, {
#                 'vanilla': 'memory_efficient_attention',
#                 # 'flash': 'fmha.flash.flash_attn',
#                 # 'sparse': 'fmha.sparse.SparseAttention',
#                 # 'block': 'fmha.block.BlockDiagonalAttention'
#             }[self.attn_type])(**attn_kwargs)

#         elif self.attn_type == 'linear':
#             # 线性注意力
#             Q, K, V = map(self.phi, [Q, K, V])
#             attn_output = torch.matmul(Q, torch.matmul(K.transpose(-2, -1), V)) * self.scale

#         elif self.attn_type == 'low_rank':
#             # 低秩注意力
#             UQ, UK = self.U(Q), self.U(K)
#             attn_output = torch.matmul(UQ, torch.matmul(UK.transpose(-2, -1), V)) * self.scale

#         elif self.attn_type == 'window':
#             # 局部窗口注意力
#             B, L, H, D = Q.shape
#             w = self.attn_cfg['window_size']
#             Q = rearrange(Q, 'b (w n) h d -> b w n h d', w=L//w)
#             K = rearrange(K, 'b (w n) h d -> b w n h d', w=L//w)
#             V = rearrange(V, 'b (w n) h d -> b w n h d', w=L//w)
#             attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
#             attn_output = torch.matmul(F.softmax(attn_weights, dim=-1), V)
#             attn_output = rearrange(attn_output, 'b w n h d -> b (w n) h d')

#         elif self.attn_type == 'dynamic':
#             # 动态稀疏注意力
#             scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
#             topk_scores, topk_indices = torch.topk(scores, self.attn_cfg['topk'], dim=-1)
#             attn_output = torch.zeros_like(V)
#             for i in range(Q.size(0)):
#                 for h in range(Q.size(2)):
#                     attn_output[i, :, h] = torch.matmul(
#                         F.softmax(topk_scores[i, :, h], dim=-1),
#                         V[i, topk_indices[i, :, h], h]
#                     )
#         else: # 'base'
#             attn_output = F.scaled_dot_product_attention(
#                 Q, K, V,
#                 scale=self.scale,
#                 dropout_p=self.dropout if self.training else 0.0,
#                 is_causal=False  # 禁用因果掩码，纯交叉注意力
#             )

#         # 残差连接
#         if residual is not None:
#             return self.res_norm_fn(residual) + attn_output
#         else:
#             return attn_output



# origin: 不含邻近下采样
# class XFormersCrossAttention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         attn_type: str = 'vanilla',
#         proj_type: str = 'linear',
#         res_norm_fn: Optional[nn.Module] = None,
#         dropout: float = 0.0,
#         attn_cfg: Optional[Dict] = None
#     ):
#         """
#         重构的高效跨注意力模块，支持不同长度的Q和K
        
#         新增:
#             - 'flash': 最快的高效注意力实现 (需安装flash_attn库)
        
#         参数:
#             dim: 特征维度
#             attn_type: 注意力类型 
#                 - 'vanilla': xFormers内存优化注意力 (高效)
#                 - 'flash': Flash Attention (当前最快实现，强烈推荐)
#                 - 'base': 基础实现 (稳定)
#                 - 'linear': 线性注意力 (近似)
#                 - 'low_rank': 低秩注意力 (近似)
#                 - 'window': 局部窗口注意力
#                 - 'dynamic': 动态稀疏注意力
#             proj_type: 投影类型
#             res_norm_fn: 残差连接规范化函数
#             dropout: 注意力dropout率
#             attn_cfg: 类型相关配置字典
#         """
#         super().__init__()
#         self.dim = dim
#         self.attn_type = attn_type
#         self.proj_type = proj_type
#         self.dropout = dropout
#         self.attn_cfg = attn_cfg or {}
        
#         if self.proj_type == 'linear':
#             self.q_proj = nn.Linear(dim, dim)    
#             self.k_proj = nn.Linear(dim, dim)    
#             self.v_proj = nn.Linear(dim, dim)    
#         elif self.proj_type == 'conv1d':
#             self.q_proj = nn.Conv1d(dim, dim, 1)
#             self.k_proj = nn.Conv1d(dim, dim, 1)
#             self.v_proj = nn.Conv1d(dim, dim, 1)
#         else:
#             self.q_proj = nn.Identity()
#             self.k_proj = nn.Identity()
#             self.v_proj = nn.Identity()

#         # 设置默认配置
#         defaults = {
#             'window_size': 32,
#             'topk': 0.1, # 小数为动态比例，整数为预设固定个数
#             'topk_q': 0.3,
#             'topk_k': 0.2,
#             'low_rank': max(4, dim // 4), # dim // 8
#             'phi': 'relu',

#             'chunk_size_q': 1024, # 512
#             'chunk_size_k': 4096, # 4096

#             # Flash Attention特定配置
#             'flash_causal': False,
#             'flash_num_heads': 4,  # 默认为单头注意力 1 \ 4
#             'flash_backend': 'triton',
#             'flash_window_size': (128, 128),  # 滑动窗口大小分布表示左右延展每个q_token对K可见长度 (-1,-1) \ (256,256) \ (128, 128)
#             'flash_softcap': 0.0,             # 平滑截断系数 0.0 \ 0.2
#             'flash_use_alibi': True,           # 启用ALiBi位置偏置(搭配多头使用) False \ True
#             'flash_deterministic': False,      # 训练时禁用确定性计算 False
#         }
#         for k, v in defaults.items():
#             if k not in self.attn_cfg:
#                 self.attn_cfg[k] = v
        
#         # 支持的类型列表
#         valid_types = ['vanilla', 'flash', 'base', 'linear', 'low_rank', 'window', 'dynamic']
#         assert attn_type in valid_types, f"无效的注意力类型: {attn_type}，可选: {valid_types}"
        
#         # 残差连接规范化
#         self.res_norm_fn = res_norm_fn(self.dim) if res_norm_fn is not None else nn.Identity()
#         self.scale = 1.0 / sqrt(dim)  # 更精确的缩放系数
        
#         # 特定类型初始化
#         if attn_type == 'linear':
#             # 线性注意力核函数
#             phi_dict = {
#                 'relu': nn.ReLU(),
#                 'elu': nn.ELU(),
#                 'softplus': nn.Softplus(),
#                 'identity': nn.Identity()
#             }
#             self.phi = phi_dict.get(self.attn_cfg['phi'], phi_dict['relu'])
            
#         elif attn_type == 'low_rank':
#             # 低秩分解矩阵
#             rank = min(self.attn_cfg['low_rank'], dim)
#             self.U = nn.Linear(dim, rank, bias=False)
#             self.V = nn.Linear(dim, rank, bias=False)
        
#         # 对于高效注意力类型，检查依赖库可用性
#         if attn_type == 'vanilla':
#             self._check_xformers_available()
#         elif attn_type == 'flash':
#             self._setup_flash_attention()
        
#         self.autocast_enabled = True

#     def _check_xformers_available(self):
#         """验证xFormers可用性"""
#         try:
#             import xformers.ops as xops
#             self.xops = xops
#         except ImportError:
#             raise ImportError("未安装xFormers库。若要使用'vanilla'类型，请先安装: pip install xformers")

#     def _setup_flash_attention(self):
#         """设置Flash Attention需要的配置"""
#         try:
#             import flash_attn
#             from importlib.metadata import version
            
#             # 获取Flash Attention版本
#             flash_version = version('flash_attn')
#             # print(f"检测到Flash Attention版本: {flash_version}")
            
#             # 根据版本选择不同的导入方式
#             if flash_version >= '2.0.0':
#                 from flash_attn.flash_attn_interface import flash_attn_func
#                 self.flash_attn = flash_attn_func
#                 self.flash_version = 2
#             else:
#                 from flash_attn.flash_attn_triton import flash_attn_func
#                 self.flash_attn = flash_attn_func
#                 self.flash_version = 1
            
#             # 验证头数配置
#             self.num_heads = self.attn_cfg['flash_num_heads']
#             assert self.dim % self.num_heads == 0, (
#                 f"特征维度({self.dim})必须能被头数({self.num_heads})整除"
#             )
#             self.head_dim = self.dim // self.num_heads
            
#         except ImportError as e:
#             raise ImportError(
#                 "未安装Flash Attention库。若要使用'flash'类型，请先安装:\n"
#                 "标准版: pip install flash-attn\n"
#                 "Triton后端: pip install flash-attn-triton"
#             ) from e

#     def _reshape_for_flash(self, tensor: torch.Tensor) -> torch.Tensor:
#         """为Flash Attention重塑张量形状"""
#         B, L, _ = tensor.shape
#         # 重塑为 [batch_size, seqlen, num_heads, head_dim]
#         return tensor.view(B, L, self.num_heads, self.head_dim).contiguous()

#     # def _process_flash_chunk(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#     #     """处理单个分块的Flash Attention核心逻辑"""
#     #     # 保存原始数据类型
#     #     orig_dtype = Q.dtype
        
#     #     # 自动转换到半精度
#     #     if orig_dtype not in (torch.float16, torch.bfloat16):
#     #         target_dtype = torch.float16 if Q.device.type == 'cuda' else torch.bfloat16
#     #         Q, K, V = Q.to(target_dtype), K.to(target_dtype), V.to(target_dtype)
        
#     #     # 确保连续内存并重塑形状
#     #     Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
#     #     Q_reshaped = self._reshape_for_flash(Q)
#     #     K_reshaped = self._reshape_for_flash(K)
#     #     V_reshaped = self._reshape_for_flash(V)
        
#     #     # 准备Flash参数
#     #     dropout_p = self.dropout if self.training else 0.0
#     #     causal = self.attn_cfg['flash_causal']
        
#     #     # 根据Flash版本调用不同参数顺序
#     #     try:
#     #         if self.flash_version >= 2:
#     #             attn_output = self.flash_attn(
#     #                 Q_reshaped, K_reshaped, V_reshaped,
#     #                 dropout_p=dropout_p,
#     #                 softmax_scale=self.scale,
#     #                 causal=causal,
                    
#     #             )
#     #         else:
#     #             # Flash v1使用位置参数
#     #             attn_output = self.flash_attn(
#     #                 Q_reshaped, K_reshaped, V_reshaped,
#     #                 dropout_p,
#     #                 self.scale,
#     #                 causal
#     #             )
#     #     except TypeError:
#     #         # 回退到最简调用
#     #         attn_output = self.flash_attn(Q_reshaped, K_reshaped, V_reshaped)
        
#     #     # 恢复形状和数据类型
#     #     B, L_q, _, _ = Q_reshaped.shape
#     #     output = attn_output.reshape(B, L_q, self.dim)
#     #     return output.to(orig_dtype) if orig_dtype != output.dtype else output

#     # def forward_flash_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#     #     """优化的Flash Attention实现，支持chunk_size并提高计算效率"""
#     #     assert K.size(1) == V.size(1), "V序列长度必须与K相同"
        
#     #     # 保存原始数据类型和形状
#     #     # orig_dtype = Q.dtype
#     #     B, L_q, C = Q.shape
#     #     L_k = K.size(1)
        
#     #     # 获取chunk_size配置
#     #     chunk_size_q = self.attn_cfg['chunk_size_q']
#     #     chunk_size_k = self.attn_cfg['chunk_size_k']
        
#     #     # 确定是否启用分块策略
#     #     use_chunking = (chunk_size_q > 0 and L_q > chunk_size_q) or (chunk_size_k > 0 and L_k > chunk_size_k)
        
#     #     # 分块处理策略
#     #     if use_chunking:
#     #         # 实际分块大小（确保不超边界）
#     #         actual_chunk_size_q = min(chunk_size_q, L_q) if chunk_size_q > 0 else L_q
#     #         actual_chunk_size_k = min(chunk_size_k, L_k) if chunk_size_k > 0 else L_k
            
#     #         # 收集分块结果
#     #         outputs = []
#     #         for i in range(0, L_q, actual_chunk_size_q):
#     #             q_start, q_end = i, min(i + actual_chunk_size_q, L_q)
#     #             Q_chunk = Q[:, q_start:q_end, :]
                
#     #             # 选择相关的K/V块
#     #             k_start = max(0, i - actual_chunk_size_k // 2)
#     #             k_end = min(L_k, i + actual_chunk_size_k)
#     #             K_chunk = K[:, k_start:k_end, :]
#     #             V_chunk = V[:, k_start:k_end, :]
                
#     #             # 处理单个分块
#     #             chunk_output = self._process_flash_chunk(Q_chunk, K_chunk, V_chunk)
#     #             outputs.append(chunk_output)
                
#     #         # 合并所有输出
#     #         return torch.cat(outputs, dim=1)
#     #     else:
#     #         # 不分块，处理整个序列
#     #         return self._process_flash_chunk(Q, K, V)

#     def _process_flash_chunk(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """处理单个分块的Flash Attention核心逻辑"""
#         # 保存原始数据类型
#         orig_dtype = Q.dtype
#         self.device = Q.device
        
#         # 自动转换到半精度
#         if orig_dtype not in (torch.float16, torch.bfloat16):
#             target_dtype = torch.float16 if Q.device.type == 'cuda' else torch.bfloat16
#             Q, K, V = Q.to(target_dtype), K.to(target_dtype), V.to(target_dtype)
        
#         # 确保连续内存并重塑形状
#         Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
#         Q_reshaped = self._reshape_for_flash(Q)
#         K_reshaped = self._reshape_for_flash(K)
#         V_reshaped = self._reshape_for_flash(V)
        
#         # 准备Flash参数（使用2.7.0版本的高级参数）
#         params = {
#             'dropout_p': self.dropout if self.training else 0.0,
#             'softmax_scale': self.scale,
#             'causal': self.attn_cfg['flash_causal'],
#             'window_size': self.attn_cfg.get('flash_window_size', (-1, -1)),
#             'softcap': self.attn_cfg.get('flash_softcap', 0.0),
#             'alibi_slopes': self._generate_alibi_slopes(),
#             'deterministic': self.attn_cfg.get('flash_deterministic', False),
#             'return_attn_probs': False
#         }
        
#         # 调用Flash Attention
#         try:
#             attn_output = self.flash_attn(
#                 Q_reshaped, K_reshaped, V_reshaped,
#                 **params
#             )
#         except TypeError as e:
#             print(f"Flash高级调用失败: {e}, 尝试最简模式")
#             # 回退到最简调用
#             attn_output = self.flash_attn(Q_reshaped, K_reshaped, V_reshaped)
        
#         # 恢复形状和数据类型
#         B, L_q, _, _ = Q_reshaped.shape
#         output = attn_output.reshape(B, L_q, self.dim)
#         return output.to(orig_dtype) if orig_dtype != output.dtype else output

#     def _generate_alibi_slopes(self) -> Optional[torch.Tensor]:
#         """生成ALiBi位置偏置斜率"""
#         if not self.attn_cfg.get('flash_use_alibi', False):
#             return None
        
#         # 为每个注意力头生成唯一的斜率
#         num_heads = self.attn_cfg['flash_num_heads']
#         slopes = torch.tensor([1 / (2 ** (i / num_heads)) for i in range(1, num_heads + 1)])
#         return slopes.to(self.device)

#     def forward_flash_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """优化的Flash Attention实现，支持高级参数和chunk_size分块策略"""
#         assert K.size(1) == V.size(1), "V序列长度必须与K相同"
        
#         # 保存原始数据类型和形状
#         B, L_q, C = Q.shape
#         L_k = K.size(1)
        
#         # 获取chunk_size配置
#         chunk_size_q = self.attn_cfg['chunk_size_q']
#         chunk_size_k = self.attn_cfg['chunk_size_k']
        
#         # 确定是否启用分块策略
#         use_chunking = (chunk_size_q > 0 and L_q > chunk_size_q) or (chunk_size_k > 0 and L_k > chunk_size_k)
        
#         # 分块处理策略
#         if use_chunking:
#             # 实际分块大小（确保不超边界）
#             actual_chunk_size_q = min(chunk_size_q, L_q) if chunk_size_q > 0 else L_q
#             actual_chunk_size_k = min(chunk_size_k, L_k) if chunk_size_k > 0 else L_k
            
#             # 收集分块结果
#             outputs = []
#             for i in range(0, L_q, actual_chunk_size_q):
#                 q_start, q_end = i, min(i + actual_chunk_size_q, L_q)
#                 Q_chunk = Q[:, q_start:q_end, :]
                
#                 # 选择相关的K/V块（带上下文重叠）
#                 k_start = max(0, i - actual_chunk_size_k // 2)
#                 k_end = min(L_k, i + actual_chunk_size_k)
#                 K_chunk = K[:, k_start:k_end, :]
#                 V_chunk = V[:, k_start:k_end, :]
                
#                 # 处理单个分块
#                 chunk_output = self._process_flash_chunk(Q_chunk, K_chunk, V_chunk)
#                 outputs.append(chunk_output)
                
#             # 合并所有输出
#             return torch.cat(outputs, dim=1)
#         else:
#             # 不分块，处理整个序列
#             return self._process_flash_chunk(Q, K, V)


#     def forward_vanilla_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """使用xFormers内存优化注意力处理不同长度的Q和K"""
#         # 确保V的序列长度与K一致
#         assert K.size(1) == V.size(1), (
#             f"V的序列长度必须与K相同! "
#             f"K长度: {K.size(1)}, V长度: {V.size(1)}"
#         )
#         with torch.cuda.amp.autocast(enabled=self.autocast_enabled):
#             Q = Q.contiguous()
#             K = K.contiguous()
#             V = V.contiguous()
#             return self.xops.memory_efficient_attention(
#                 Q, K, V,
#                 scale=self.scale,
#                 p=self.dropout if self.training else 0.0,
#                 attn_bias=self.attn_cfg.get('attn_bias', None)
#             )

#     def forward_base_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """基础注意力实现，高效处理不同长度的Q和K"""
#         # 确保V的序列长度与K一致
#         assert K.size(1) == V.size(1), (
#             f"V的序列长度必须与K相同! "
#             f"K长度: {K.size(1)}, V长度: {V.size(1)}"
#         )
        
#         # 计算注意力分数
#         attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
#         # 应用softmax
#         attn_weights = F.softmax(attn_logits, dim=-1)
        
#         # 应用dropout
#         if self.dropout > 0.0 and self.training:
#             attn_weights = F.dropout(attn_weights, p=self.dropout)
        
#         # 加权求和值
#         return torch.matmul(attn_weights, V)

#     def forward_linear_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """带分块的线性注意力实现"""
#         assert K.size(1) == V.size(1), "K和V序列长度必须相同"
        
#         B, Lq, C = Q.shape
#         Lk = K.size(1)
        
#         # 获取配置参数
#         chunk_size_q = self.attn_cfg['chunk_size_q']
#         chunk_size_k = self.attn_cfg['chunk_size_k']
        
#         # 确保分块大小有效
#         chunk_size_q = min(chunk_size_q, Lq) if chunk_size_q > 0 else Lq
#         chunk_size_k = min(chunk_size_k, Lk) if chunk_size_k > 0 else Lk
        
#         # 初始化KV聚合张量
#         kv = torch.zeros(B, C, V.size(-1), device=Q.device, dtype=Q.dtype)
        
#         # ================================
#         # 阶段1: KV聚合 (K和V分块处理)
#         # ================================
#         if chunk_size_k >= Lk:  # 无需分块
#             K_chunk = self.phi(K)
#             kv += torch.einsum('bkc,bkv->bcv', K_chunk, V)
#         else:  # KV分块处理
#             for i in range(0, Lk, chunk_size_k):
#                 # 计算当前分块的起始和结束位置
#                 k_start, k_end = i, min(i + chunk_size_k, Lk)
                
#                 K_chunk = K[:, k_start:k_end, :]
#                 V_chunk = V[:, k_start:k_end, :]
                
#                 # 特征变换和KV聚合
#                 K_chunk = self.phi(K_chunk)
#                 kv += torch.einsum('bkc,bkv->bcv', K_chunk, V_chunk)
        
#         # ================================
#         # 阶段2: 计算输出 (Q分块处理)
#         # ================================
#         outputs = []
        
#         if chunk_size_q >= Lq:  # 无需分块
#             Q_chunk = self.phi(Q)
#             output_chunk = torch.einsum('bqc,bcv->bqv', Q_chunk, kv)
#             outputs.append(output_chunk * self.scale)
#         else:  # Q分块处理
#             for j in range(0, Lq, chunk_size_q):
#                 # 计算当前分块的起始和结束位置
#                 q_start, q_end = j, min(j + chunk_size_q, Lq)
                
#                 Q_chunk = Q[:, q_start:q_end, :]
                
#                 # 特征变换和输出计算
#                 Q_chunk = self.phi(Q_chunk)
#                 output_chunk = torch.einsum('bqc,bcv->bqv', Q_chunk, kv)
#                 outputs.append(output_chunk * self.scale)
        
#         # 合并所有输出分块
#         return torch.cat(outputs, dim=1)

#     def forward_low_rank_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """带分块的低秩注意力实现"""
#         assert K.size(1) == V.size(1), "K和V序列长度必须相同"
        
#         B, Lq, C = Q.shape
#         Lk = K.size(1)
        
#         # 获取配置参数
#         chunk_size_q = self.attn_cfg['chunk_size_q']
#         chunk_size_k = self.attn_cfg['chunk_size_k']
        
#         # 确保分块大小有效
#         chunk_size_q = min(chunk_size_q, Lq) if chunk_size_q > 0 else Lq
#         chunk_size_k = min(chunk_size_k, Lk) if chunk_size_k > 0 else Lk
        
#         # 初始化KV聚合张量
#         kv = torch.zeros(B, self.U.out_features, V.size(-1), 
#                          device=Q.device, dtype=Q.dtype)
        
#         # ================================
#         # 阶段1: KV聚合 (K和V分块处理)
#         # ================================
#         if chunk_size_k >= Lk:  # 无需分块
#             UK_chunk = self.U(K)
#             kv += torch.einsum('bkc,bkv->bcv', UK_chunk, V)
#         else:  # KV分块处理
#             for i in range(0, Lk, chunk_size_k):
#                 # 计算当前分块的起始和结束位置
#                 k_start, k_end = i, min(i + chunk_size_k, Lk)
                
#                 K_chunk = K[:, k_start:k_end, :]
#                 V_chunk = V[:, k_start:k_end, :]
                
#                 # 低秩投影和KV聚合
#                 UK_chunk = self.U(K_chunk)
#                 kv += torch.einsum('bkc,bkv->bcv', UK_chunk, V_chunk)
        
#         # ================================
#         # 阶段2: 计算输出 (Q分块处理)
#         # ================================
#         outputs = []
        
#         if chunk_size_q >= Lq:  # 无需分块
#             UQ_chunk = self.U(Q)
#             output_chunk = torch.einsum('bqc,bcv->bqv', UQ_chunk, kv)
#             outputs.append(output_chunk * self.scale)
#         else:  # Q分块处理
#             for j in range(0, Lq, chunk_size_q):
#                 # 计算当前分块的起始和结束位置
#                 q_start, q_end = j, min(j + chunk_size_q, Lq)
                
#                 Q_chunk = Q[:, q_start:q_end, :]
                
#                 # 低秩投影和输出计算
#                 UQ_chunk = self.U(Q_chunk)
#                 output_chunk = torch.einsum('bqc,bcv->bqv', UQ_chunk, kv)
#                 outputs.append(output_chunk * self.scale)
        
#         # 合并所有输出分块
#         return torch.cat(outputs, dim=1)    
    
#     def forward_window_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """局部窗口注意力处理不同长度的Q和K"""
#         # 确保V的序列长度与K一致
#         assert K.size(1) == V.size(1), (
#             f"V的序列长度必须与K相同! "
#             f"K长度: {K.size(1)}, V长度: {V.size(1)}"
#         )
        
#         B, Lq, C = Q.shape
#         Lk = K.size(1)
#         w = self.attn_cfg['window_size']
        
#         # 对Q和K进行填充以适应窗口
#         pad_len_q = (w - Lq % w) % w
#         pad_len_k = (w - Lk % w) % w
        
#         Q_pad = F.pad(Q, (0, 0, 0, pad_len_q))
#         K_pad = F.pad(K, (0, 0, 0, pad_len_k))
#         V_pad = F.pad(V, (0, 0, 0, pad_len_k))
        
#         # 窗口分割和注意力计算
#         Q_win = rearrange(Q_pad, 'b (w n) c -> b w n c', w=(Lq + pad_len_q) // w)
#         K_win = rearrange(K_pad, 'b (w n) c -> b w n c', w=(Lk + pad_len_k) // w)
#         V_win = rearrange(V_pad, 'b (w n) c -> b w n c', w=(Lk + pad_len_k) // w)
        
#         # 计算注意力
#         attn_logits = torch.einsum('bwik,bwjk->bwij', Q_win, K_win) * self.scale
#         attn_weights = F.softmax(attn_logits, dim=-1)
#         attn_output = torch.einsum('bwij,bwjk->bwik', attn_weights, V_win)
        
#         # 恢复原始形状
#         attn_output = rearrange(attn_output, 'b w n c -> b (w n) c')
#         return attn_output[:, :Lq, :]

#     def forward_dynamic_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """动态稀疏注意力处理不同长度的Q和K"""
#         # 确保V的序列长度与K一致
#         assert K.size(1) == V.size(1), (
#             f"V的序列长度必须与K相同! "
#             f"K长度: {K.size(1)}, V长度: {V.size(1)}"
#         )
        
#         B, Lq, C = Q.shape
#         Lk = K.size(1)
#         # topk = min(self.attn_cfg['topk'], Lk)
#         topk_val = self.attn_cfg['topk']
#         if 0 < topk_val < 1:
#             topk = int(topk_val * Lk)
#         else:
#             topk = int(min(topk_val, Lk))
        
#         # 计算稀疏注意力分数
#         scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
#         # 选择top-k键值对
#         topk_scores, topk_indices = torch.topk(scores, topk, dim=-1)
#         topk_weights = F.softmax(topk_scores, dim=-1)
        
#         # 收集对应的值向量
#         batch_idx = torch.arange(B, device=Q.device)[:, None, None]
#         topk_V = V[batch_idx, topk_indices, :]
        
#         # 加权求和
#         return torch.einsum('bqk,bqkc->bqc', topk_weights, topk_V)

#     def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
#                 residual: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         前向传播
        
#         参数:
#             Q: 查询张量 [B, L_q, C] (序列长度L_q)
#             K: 键张量 [B, L_k, C] (序列长度L_k)
#             V: 值张量 [B, L_v, C] (必须L_v == L_k)
#             residual: 残差连接张量 [B, L_q, C]
            
#         返回:
#             注意力输出张量 [B, L_q, C]
#         """
#         # 验证输入维度
#         assert Q.dim() == 3, f"Q应为3D张量，实际为{Q.dim()}D"
#         assert K.dim() == 3, f"K应为3D张量，实际为{K.dim()}D"
#         assert V.dim() == 3, f"V应为3D张量，实际为{V.dim()}D"
#         assert Q.size(0) == K.size(0) == V.size(0), "批次大小不一致"
#         assert Q.size(2) == K.size(2) == V.size(2), "特征维度不一致"
#         assert K.size(1) == V.size(1), "K和V序列长度必须相同"
        
#         if self.proj_type == 'conv1d':
#             Q = self.q_proj(Q.transpose(1, 2)).transpose(1, 2)
#             K = self.k_proj(K.transpose(1, 2)).transpose(1, 2)
#             V = self.v_proj(V.transpose(1, 2)).transpose(1, 2)
#         else:
#             Q = self.q_proj(Q)
#             K = self.k_proj(K)
#             V = self.v_proj(V)

#         # 根据注意力类型选择实现
#         if self.attn_type == 'vanilla':
#             attn_output = self.forward_vanilla_attention(Q, K, V)
#         elif self.attn_type == 'flash':  # 新增Flash Attention分支
#             attn_output = self.forward_flash_attention(Q, K, V)
#         elif self.attn_type == 'base':
#             attn_output = self.forward_base_attention(Q, K, V)
#         elif self.attn_type == 'linear':
#             attn_output = self.forward_linear_attention(Q, K, V)
#         elif self.attn_type == 'low_rank':
#             attn_output = self.forward_low_rank_attention(Q, K, V)
#         elif self.attn_type == 'window':
#             attn_output = self.forward_window_attention(Q, K, V)
#         elif self.attn_type == 'dynamic':
#             attn_output = self.forward_dynamic_attention(Q, K, V)
        
#         # 确保所有值都有效
#         nan_mask1 = torch.isnan(attn_output) | torch.isinf(attn_output)
#         if nan_mask1.any():
#             print("警告：attn_output 包含NaN/Inf值!")
#             # 替换无效值为0并添加微小噪声
#             attn_output[nan_mask1] = 0
#             noise = torch.randn_like(attn_output) * 1e-8
#             attn_output += noise
            
#         # 残差连接
#         if residual is not None:
#             # 确保残差维度匹配
#             if residual.size() != Q.size():
#                 assert residual.size() == Q.size(), (
#                     f"残差形状{residual.shape}必须与查询形状{Q.shape}匹配"
#                 )

#             # 确保所有值都有效
#             nan_mask2 = torch.isnan(residual) | torch.isinf(residual)
#             if nan_mask2.any():
#                 print("警告：residual 包含NaN/Inf值!")
#                 # 替换无效值为0并添加微小噪声
#                 residual[nan_mask2] = 0
#                 noise = torch.randn_like(residual) * 1e-8
#                 residual += noise

#             if self.res_norm_fn:
#                 return self.res_norm_fn(residual + attn_output)
#             return residual + attn_output
        
#         return attn_output



# NOTE 06.28: origin without flash-attn
# class XFormersCrossAttention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         attn_type: str = 'vanilla',
#         proj_type: str = 'linear',
#         res_norm_fn: Optional[nn.Module] = None,
#         dropout: float = 0.0,
#         attn_cfg: Optional[Dict] = None
#     ):
#         """
#         重构的高效跨注意力模块，支持不同长度的Q和K
        
#         参数:
#             dim: 特征维度
#             attn_type: 注意力类型 
#                 - 'vanilla': xFormers内存优化注意力 (高效)
#                 - 'base': 基础实现 (稳定)
#                 - 'linear': 线性注意力 (近似)
#                 - 'low_rank': 低秩注意力 (近似)
#                 - 'window': 局部窗口注意力
#                 - 'dynamic': 动态稀疏注意力
#             res_norm_fn: 残差连接规范化函数
#             dropout: 注意力dropout率
#             attn_cfg: 类型相关配置字典
#         """
#         super().__init__()
#         self.dim = dim
#         self.attn_type = attn_type
#         self.proj_type = proj_type
#         self.dropout = dropout
#         self.attn_cfg = attn_cfg or {}
        
#         if self.proj_type == 'linear':
#             self.q_proj = nn.Linear(dim, dim)    
#             self.k_proj = nn.Linear(dim, dim)    
#             self.v_proj = nn.Linear(dim, dim)    
#         elif self.proj_type == 'conv1d':
#             self.q_proj = nn.Conv1d(dim, dim, 1)
#             self.k_proj = nn.Conv1d(dim, dim, 1)
#             self.v_proj = nn.Conv1d(dim, dim, 1)
#         else:
#             self.q_proj = nn.Identity()
#             self.k_proj = nn.Identity()
#             self.v_proj = nn.Identity()

#         # 设置默认配置
#         defaults = {
#             'window_size': 32,
#             'topk': 0.1, # TODO: 小数为动态比例，整数为预设固定个数
#             'topk_q': 0.3,
#             'topk_k': 0.2,
#             'low_rank': max(4, dim // 4), # dim // 8
#             'phi': 'relu',
#             'chunk_size_q': 1024,
#             'chunk_size_k': 2048,
#         }
#         for k, v in defaults.items():
#             if k not in self.attn_cfg:
#                 self.attn_cfg[k] = v
        
#         # 支持的类型列表
#         valid_types = ['vanilla', 'base', 'linear', 'low_rank', 'window', 'dynamic']
#         assert attn_type in valid_types, f"无效的注意力类型: {attn_type}，可选: {valid_types}"
        
#         # 残差连接规范化
#         self.res_norm_fn = res_norm_fn(self.dim) if res_norm_fn is not None else nn.Identity()
#         self.scale = 1.0 / math.sqrt(self.dim)
        
#         # 特定类型初始化
#         if attn_type == 'linear':
#             # 线性注意力核函数
#             phi_dict = {
#                 'relu': nn.ReLU(),
#                 'elu': nn.ELU(),
#                 'softplus': nn.Softplus(),
#                 'identity': nn.Identity()
#             }
#             self.phi = phi_dict.get(self.attn_cfg['phi'], phi_dict['relu'])
            
#         elif attn_type == 'low_rank':
#             # 低秩分解矩阵
#             rank = min(self.attn_cfg['low_rank'], dim)
#             self.U = nn.Linear(dim, rank, bias=False)
#             self.V = nn.Linear(dim, rank, bias=False)
        
#         # 对于vanilla类型，检查xFormers可用性
#         if attn_type == 'vanilla':
#             self._check_xformers_available()
        
#         self.autocast_enabled = True

#     def _check_xformers_available(self):
#         """验证xFormers可用性"""
#         try:
#             import xformers.ops as xops
#             self.xops = xops
#         except ImportError:
#             raise ImportError("未安装xFormers库。若要使用'vanilla'类型，请先安装: pip install xformers")

#     def forward_vanilla_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """使用xFormers内存优化注意力处理不同长度的Q和K"""
#         # 确保V的序列长度与K一致
#         assert K.size(1) == V.size(1), (
#             f"V的序列长度必须与K相同! "
#             f"K长度: {K.size(1)}, V长度: {V.size(1)}"
#         )
#         with torch.cuda.amp.autocast(enabled=self.autocast_enabled):
#             Q = Q.contiguous()
#             K = K.contiguous()
#             V = V.contiguous()
#             return self.xops.memory_efficient_attention(
#                 Q, K, V,
#                 scale=self.scale,
#                 p=self.dropout if self.training else 0.0,
#                 attn_bias=self.attn_cfg.get('attn_bias', None)
#             )

#     def forward_base_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """基础注意力实现，高效处理不同长度的Q和K"""
#         # 确保V的序列长度与K一致
#         assert K.size(1) == V.size(1), (
#             f"V的序列长度必须与K相同! "
#             f"K长度: {K.size(1)}, V长度: {V.size(1)}"
#         )
        
#         # 计算注意力分数
#         attn_logits = torch.einsum('bqd,bkd->bqk', Q, K) * self.scale
        
#         # 应用softmax
#         attn_weights = F.softmax(attn_logits, dim=-1)
        
#         # 应用dropout
#         if self.dropout > 0.0 and self.training:
#             attn_weights = F.dropout(attn_weights, p=self.dropout)
        
#         # 加权求和值
#         return torch.einsum('bqk,bkd->bqd', attn_weights, V)

#     # NOTE: origin without chunk dealing
#     # def forward_linear_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#     #     """线性注意力处理不同长度的Q和K"""
#     #     # 确保V的序列长度与K一致
#     #     assert K.size(1) == V.size(1), (
#     #         f"V的序列长度必须与K相同! "
#     #         f"K长度: {K.size(1)}, V长度: {V.size(1)}"
#     #     )
        
#     #     # 特征变换
#     #     Q, K = map(self.phi, [Q, K])
        
#     #     # 高效核计算
#     #     kv = torch.einsum('bkc,bkv->bcv', K, V)  # [B, C, C]
#     #     return torch.einsum('bqc,bcv->bqv', Q, kv) * self.scale

#     # def forward_low_rank_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#     #     """低秩注意力处理不同长度的Q和K"""
#     #     # 确保V的序列长度与K一致
#     #     assert K.size(1) == V.size(1), (
#     #         f"V的序列长度必须与K相同! "
#     #         f"K长度: {K.size(1)}, V长度: {V.size(1)}"
#     #     )
        
#     #     # 低秩投影
#     #     UQ, UK = self.U(Q), self.U(K)
        
#     #     # 高效核计算
#     #     kv = torch.einsum('bkc,bkv->bcv', UK, V)  # [B, C, C]
#     #     return torch.einsum('bqc,bcv->bqv', UQ, kv) * self.scale

    
#     def forward_linear_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """带分块的线性注意力实现"""
#         assert K.size(1) == V.size(1), "K和V序列长度必须相同"
        
#         B, Lq, C = Q.shape
#         Lk = K.size(1)
        
#         # 获取配置参数
#         chunk_size_q = self.attn_cfg['chunk_size_q']
#         chunk_size_k = self.attn_cfg['chunk_size_k']
        
#         # 确保分块大小有效
#         chunk_size_q = min(chunk_size_q, Lq) if chunk_size_q > 0 else Lq
#         chunk_size_k = min(chunk_size_k, Lk) if chunk_size_k > 0 else Lk
        
#         # 初始化KV聚合张量
#         kv = torch.zeros(B, C, V.size(-1), device=Q.device, dtype=Q.dtype)
        
#         # ================================
#         # 阶段1: KV聚合 (K和V分块处理)
#         # ================================
#         if chunk_size_k >= Lk:  # 无需分块
#             K_chunk = self.phi(K)
#             kv += torch.einsum('bkc,bkv->bcv', K_chunk, V)
#         else:  # KV分块处理
#             for i in range(0, Lk, chunk_size_k):
#                 # 计算当前分块的起始和结束位置
#                 k_start, k_end = i, min(i + chunk_size_k, Lk)
                
#                 K_chunk = K[:, k_start:k_end, :]
#                 V_chunk = V[:, k_start:k_end, :]
                
#                 # 特征变换和KV聚合
#                 K_chunk = self.phi(K_chunk)
#                 kv += torch.einsum('bkc,bkv->bcv', K_chunk, V_chunk)
        
#         # ================================
#         # 阶段2: 计算输出 (Q分块处理)
#         # ================================
#         outputs = []
        
#         if chunk_size_q >= Lq:  # 无需分块
#             Q_chunk = self.phi(Q)
#             output_chunk = torch.einsum('bqc,bcv->bqv', Q_chunk, kv)
#             outputs.append(output_chunk * self.scale)
#         else:  # Q分块处理
#             for j in range(0, Lq, chunk_size_q):
#                 # 计算当前分块的起始和结束位置
#                 q_start, q_end = j, min(j + chunk_size_q, Lq)
                
#                 Q_chunk = Q[:, q_start:q_end, :]
                
#                 # 特征变换和输出计算
#                 Q_chunk = self.phi(Q_chunk)
#                 output_chunk = torch.einsum('bqc,bcv->bqv', Q_chunk, kv)
#                 outputs.append(output_chunk * self.scale)
        
#         # 合并所有输出分块
#         return torch.cat(outputs, dim=1)

#     def forward_low_rank_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """带分块的低秩注意力实现"""
#         assert K.size(1) == V.size(1), "K和V序列长度必须相同"
        
#         B, Lq, C = Q.shape
#         Lk = K.size(1)
        
#         # 获取配置参数
#         chunk_size_q = self.attn_cfg['chunk_size_q']
#         chunk_size_k = self.attn_cfg['chunk_size_k']
        
#         # 确保分块大小有效
#         chunk_size_q = min(chunk_size_q, Lq) if chunk_size_q > 0 else Lq
#         chunk_size_k = min(chunk_size_k, Lk) if chunk_size_k > 0 else Lk
        
#         # 初始化KV聚合张量
#         kv = torch.zeros(B, self.U.out_features, V.size(-1), 
#                          device=Q.device, dtype=Q.dtype)
        
#         # ================================
#         # 阶段1: KV聚合 (K和V分块处理)
#         # ================================
#         if chunk_size_k >= Lk:  # 无需分块
#             UK_chunk = self.U(K)
#             kv += torch.einsum('bkc,bkv->bcv', UK_chunk, V)
#         else:  # KV分块处理
#             for i in range(0, Lk, chunk_size_k):
#                 # 计算当前分块的起始和结束位置
#                 k_start, k_end = i, min(i + chunk_size_k, Lk)
                
#                 K_chunk = K[:, k_start:k_end, :]
#                 V_chunk = V[:, k_start:k_end, :]
                
#                 # 低秩投影和KV聚合
#                 UK_chunk = self.U(K_chunk)
#                 kv += torch.einsum('bkc,bkv->bcv', UK_chunk, V_chunk)
        
#         # ================================
#         # 阶段2: 计算输出 (Q分块处理)
#         # ================================
#         outputs = []
        
#         if chunk_size_q >= Lq:  # 无需分块
#             UQ_chunk = self.U(Q)
#             output_chunk = torch.einsum('bqc,bcv->bqv', UQ_chunk, kv)
#             outputs.append(output_chunk * self.scale)
#         else:  # Q分块处理
#             for j in range(0, Lq, chunk_size_q):
#                 # 计算当前分块的起始和结束位置
#                 q_start, q_end = j, min(j + chunk_size_q, Lq)
                
#                 Q_chunk = Q[:, q_start:q_end, :]
                
#                 # 低秩投影和输出计算
#                 UQ_chunk = self.U(Q_chunk)
#                 output_chunk = torch.einsum('bqc,bcv->bqv', UQ_chunk, kv)
#                 outputs.append(output_chunk * self.scale)
        
#         # 合并所有输出分块
#         return torch.cat(outputs, dim=1)    
    
    
#     def forward_window_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """局部窗口注意力处理不同长度的Q和K"""
#         # 确保V的序列长度与K一致
#         assert K.size(1) == V.size(1), (
#             f"V的序列长度必须与K相同! "
#             f"K长度: {K.size(1)}, V长度: {V.size(1)}"
#         )
        
#         B, Lq, C = Q.shape
#         Lk = K.size(1)
#         w = self.attn_cfg['window_size']
        
#         # 对Q和K进行填充以适应窗口
#         pad_len_q = (w - Lq % w) % w
#         pad_len_k = (w - Lk % w) % w
        
#         Q_pad = F.pad(Q, (0, 0, 0, pad_len_q))
#         K_pad = F.pad(K, (0, 0, 0, pad_len_k))
#         V_pad = F.pad(V, (0, 0, 0, pad_len_k))
        
#         # 窗口分割和注意力计算
#         Q_win = rearrange(Q_pad, 'b (w n) c -> b w n c', w=(Lq + pad_len_q) // w)
#         K_win = rearrange(K_pad, 'b (w n) c -> b w n c', w=(Lk + pad_len_k) // w)
#         V_win = rearrange(V_pad, 'b (w n) c -> b w n c', w=(Lk + pad_len_k) // w)
        
#         # 计算注意力
#         attn_logits = torch.einsum('bwik,bwjk->bwij', Q_win, K_win) * self.scale
#         attn_weights = F.softmax(attn_logits, dim=-1)
#         attn_output = torch.einsum('bwij,bwjk->bwik', attn_weights, V_win)
        
#         # 恢复原始形状
#         attn_output = rearrange(attn_output, 'b w n c -> b (w n) c')
#         return attn_output[:, :Lq, :]

#     # TODO 06.13: 修复问题，使其可以分别有 topk_q\topk_k 以及分块大小 chunk_size_q\chunk_size_k
#     def forward_dynamic_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         """动态稀疏注意力处理不同长度的Q和K"""
#         # 确保V的序列长度与K一致
#         assert K.size(1) == V.size(1), (
#             f"V的序列长度必须与K相同! "
#             f"K长度: {K.size(1)}, V长度: {V.size(1)}"
#         )
        
#         B, Lq, C = Q.shape
#         Lk = K.size(1)
#         # topk = min(self.attn_cfg['topk'], Lk)
#         topk_val = self.attn_cfg['topk']
#         if 0 < topk_val < 1:
#             topk = int(topk_val * Lk)
#         else:
#             topk = int(min(topk_val, Lk))
        
#         # 计算稀疏注意力分数
#         scores = torch.einsum('bqc,bkc->bqk', Q, K) * self.scale
        
#         # 选择top-k键值对
#         topk_scores, topk_indices = torch.topk(scores, topk, dim=-1)
#         topk_weights = F.softmax(topk_scores, dim=-1)
        
#         # 收集对应的值向量
#         batch_idx = torch.arange(B, device=Q.device)[:, None, None]
#         topk_V = V[batch_idx, topk_indices, :]
        
#         # 加权求和
#         return torch.einsum('bqk,bqkc->bqc', topk_weights, topk_V)

#     def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
#                 residual: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         前向传播
        
#         参数:
#             Q: 查询张量 [B, L_q, C] (序列长度L_q)
#             K: 键张量 [B, L_k, C] (序列长度L_k)
#             V: 值张量 [B, L_v, C] (必须L_v == L_k)
#             residual: 残差连接张量 [B, L_q, C]
            
#         返回:
#             注意力输出张量 [B, L_q, C]
#         """
#         # 验证输入维度
#         assert Q.dim() == 3, f"Q应为3D张量，实际为{Q.dim()}D"
#         assert K.dim() == 3, f"K应为3D张量，实际为{K.dim()}D"
#         assert V.dim() == 3, f"V应为3D张量，实际为{V.dim()}D"
#         assert Q.size(0) == K.size(0) == V.size(0), "批次大小不一致"
#         assert Q.size(2) == K.size(2) == V.size(2), "特征维度不一致"
#         assert K.size(1) == V.size(1), "K和V序列长度必须相同"
        
#         if self.proj_type == 'conv1d':
#             Q = self.q_proj(Q.transpose(1, 2)).transpose(1, 2)
#             K = self.k_proj(K.transpose(1, 2)).transpose(1, 2)
#             V = self.v_proj(V.transpose(1, 2)).transpose(1, 2)
#         else:
#             Q = self.q_proj(Q)
#             K = self.k_proj(K)
#             V = self.v_proj(V)

#         # 根据注意力类型选择实现
#         if self.attn_type == 'vanilla':
#             attn_output = self.forward_vanilla_attention(Q, K, V)
#         elif self.attn_type == 'base':
#             attn_output = self.forward_base_attention(Q, K, V)
#         elif self.attn_type == 'linear':
#             attn_output = self.forward_linear_attention(Q, K, V)
#         elif self.attn_type == 'low_rank':
#             attn_output = self.forward_low_rank_attention(Q, K, V)
#         elif self.attn_type == 'window':
#             attn_output = self.forward_window_attention(Q, K, V)
#         elif self.attn_type == 'dynamic':
#             attn_output = self.forward_dynamic_attention(Q, K, V)
        
#         # 残差连接
#         if residual is not None:
#             # 确保残差维度匹配
#             assert residual.size() == Q.size(), (
#                 f"残差形状{residual.shape}必须与查询形状{Q.shape}匹配"
#             )
#             if self.res_norm_fn:
#                 return self.res_norm_fn(residual + attn_output)
#             return residual + attn_output
        
#         return attn_output