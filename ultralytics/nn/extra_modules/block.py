import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.modules.utils import _pair
from torch import Tensor
from torch.jit import Final
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union, List
from einops import rearrange, reduce
from collections import OrderedDict


from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose



from .edta import EDTA, LayerNorm


__all__ = ['SPDConv','HPST', 'EGCA']

######################################## Global-to-Local Spatial Aggregation Module start ########################################

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_mul', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    @staticmethod
    def last_zero_init(m: Union[nn.Module, nn.Sequential]) -> None:
        try:
            from mmengine.model import kaiming_init, constant_init
            if isinstance(m, nn.Sequential):
                constant_init(m[-1], val=0)
            else:
                constant_init(m, val=0)
        except ImportError as e:
            pass

    def reset_parameters(self):
        try:
            from mmengine.model import kaiming_init
            if self.pooling_type == 'att':
                kaiming_init(self.conv_mask, mode='fan_in')
                self.conv_mask.inited = True

            if self.channel_add_conv is not None:
                self.last_zero_init(self.channel_add_conv)
            if self.channel_mul_conv is not None:
                self.last_zero_init(self.channel_mul_conv)
        except ImportError as e:
            pass

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()   #拆包：把输入特征图 x 的形状拆开  假设 x 是 [2, 64, 80, 80]，表示： 2张图（batch=2） 每张图64个通道 每张图大小是80×80像素
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)  #展平  把 80×80 的图拉成 6400 个像素点。 现在 input_x 是 [2, 64, 6400]，表示：每张图的每个通道上有6400个像素值
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1) #加维度：变成 [2, 1, 64, 6400]，为了方便后续矩阵乘法。
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)  #输入 x 是 [2, 64, 80, 80]，经过 1×1卷积 后变成 [2, 1, 80, 80]。 这1个通道的图就是“注意力热图”，越亮的地方越重要。
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)  #同样拉平成 [2, 1, 6400]。
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask) #Softmax：把6400个像素值变成“概率”，所有值加起来=1。 相当于“每个像素对全局的贡献权重”。
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask) #矩阵乘法：把“像素值”和“注意力权重”相乘。  结果 context 是 [2, 1, 64, 1]，表示：用注意力加权后的全局特征。
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1) #** reshape**：变成 [2, 64, 1, 1]，就是“一张图的64个通道，每个通道被压缩成一个数”。
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)    #spatial_pool()： 使用 1×1 Conv + Softmax 生成空间注意力图。
        # channel_mul_conv：通过通道乘法增强目标区域  channel_add_conv：可选的通道加法增强。
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class GLSAChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(GLSAChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class GLSASpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(GLSASpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class GLSAConvBranch(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = Conv(in_features, hidden_features, 1, act=nn.ReLU(inplace=True))
        self.conv2 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))#g=hidden_features → groups 数等于通道数，即 depthwise 卷积（每个通道单独卷积）
        self.conv3 = Conv(hidden_features, hidden_features, 1, act=nn.ReLU(inplace=True))
        self.conv4 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))
        self.conv5 = Conv(hidden_features, hidden_features, 1, act=nn.SiLU(inplace=True))
        self.conv6 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ca = GLSAChannelAttention(64)
        self.sa = GLSASpatialAttention()
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1


class MultiScaleResLocal(nn.Module):
    """
    多尺度 3×3 + 5×5 DWConv 堆叠 + 残差 + 掩码加权
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden = hidden_features or in_features
        out   = out_features   or in_features

        # 1×1 统一通道
        self.pw = nn.Conv2d(in_features, hidden, 1, bias=False)

        # ---------- 3×3 残差分支 ----------
        self.dw3_1 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.dw3_2 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)

        # ---------- 5×5 残差分支 ----------
        self.dw5_1 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.dw5_2 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)

        # 融合 & 掩码生成
        self.fuse   = nn.Conv2d(hidden * 2, hidden, 1, bias=False)
        self.mask   = nn.Sequential(
            nn.Conv2d(hidden, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(hidden, out, 1, bias=False)

    def forward(self, x):
        identity = x

        # 统一通道
        x = self.pw(x)

        # 3×3 残差流
        f3 = x + self.dw3_1(x)
        f3 = f3 + self.dw3_2(f3)

        # 5×5 残差流
        f5 = x + self.dw5_1(x)
        f5 = f5 + self.dw5_2(f5)

        # 多尺度融合
        fused = torch.cat([f3, f5], dim=1)
        fused = self.fuse(fused)

        # 掩码加权 & 残差输出
        mask = self.mask(fused)
        out  = identity + fused * mask   # 与原 GLSAConvBranch 风格一致
        out  = self.out_conv(out)
        return out




class LiteMultiScaleMaskLocal(nn.Module):
    def __init__(self, in_features, out_features=None):
        super().__init__()
        out = out_features or in_features

        # 1×1 统一通道
        self.pw = nn.Conv2d(in_features, out, 1, bias=False)

        # 单级 3×3 & 5×5 depthwise
        self.dw3 = nn.Conv2d(out, out, 3, padding=1, groups=out, bias=False)
        self.dw5 = nn.Conv2d(out, out, 5, padding=2, groups=out, bias=False)

        # 融合 & 掩码
        self.fuse = nn.Conv2d(out * 2, out, 1, bias=False)
        self.mask = nn.Sequential(
            nn.Conv2d(out, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(out, out, 1, bias=False)

    def forward(self, x):
        identity = x
        x1 = self.pw(x)                       # [B,out,H,W]

        # 并行单级卷积
        f3 = self.dw3(x)
        # f3 = x + f3
        f5 = self.dw5(x)
        # f5 = x + f5

        # 融合
        fused = self.fuse(torch.cat([f3, f5,x], dim=1))
        mask  = self.mask(fused)

        # 掩码加权 + 残差
        out = identity + fused * mask
        out = self.out_conv(out)
        return out



class EGCA(nn.Module):

    def __init__(self, input_dim=512, embed_dim=32):
        super().__init__()

        self.conv1_1 = Conv(embed_dim*2, embed_dim, 1)
        self.conv1_1_1 = Conv(input_dim//2, embed_dim,1)
        self.local_11conv = nn.Conv2d(input_dim//2,embed_dim,1)
        self.global_11conv = nn.Conv2d(input_dim//2,embed_dim,1)
        self.GlobelBlock = ContextBlock(inplanes= embed_dim, ratio=2)
        # self.local = GLSAConvBranch(in_features = embed_dim, hidden_features = embed_dim, out_features = embed_dim)
        # self.local = LiteMultiScaleMaskLocal(embed_dim, embed_dim)
        self.local = MultiScaleResLocal(embed_dim, embed_dim)
    def forward(self, x):
        b, c, h, w = x.size()
        x_0, x_1 = x.chunk(2,dim = 1)

    # local block
        local = self.local(self.local_11conv(x_0))

    # Globel block
        Globel = self.GlobelBlock(self.global_11conv(x_1))

    # concat Globel + local
        x = torch.cat([local,Globel], dim=1)
        x = self.conv1_1(x)

        return x

######################################## Global-to-Local Spatial Aggregation Module end ########################################

######################################## SPD-Conv start ########################################

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

######################################## SPD-Conv end ########################################

######################################## PyramidSparseTransformer start ########################################



class PSAttn(nn.Module):
    """
    Pyramid Sparse Attention module for efficient multi-scale feature fusion in object detection.

    This module implements a cross-attention mechanism where queries are derived from lower-level features
    and keys/values from higher-level features. It provides a coarse attention output during training and,
    optionally, a fine attention output during inference when `topk > 0`, enhancing performance by focusing
    on key regions across scales.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        q (Conv): Convolution layer for computing queries from the input feature.
        kv (Conv): Convolution layer for computing keys and values from the upper feature.
        proj (Conv): Projection convolution layer for the output.
        pe (Conv): Positional encoding convolution layer.
        gate_conv1d (nn.Conv1d): 1D convolution for computing the gating mechanism.

    Methods:
        forward: Applies pyramid sparse attention to the input tensors.

    Examples:
        >>> attn = PSAttn(dim=256, num_heads=8, topk=4, tau=1.0)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> upper_feat = torch.randn(1, 256, 16, 16)
        >>> output = attn(x, upper_feat)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, topk=4, tau=1.0):
        """
        Initialize the Pyramid Sparse Attention module.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of attention heads.
            topk (int): Number of top tokens to select for fine attention (set to 0 to disable).
            tau (float): Temperature for Gumbel-Softmax (not used in the provided implementation).
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads
        self.topk = topk
        self.tau = tau

        # Convolution layers for queries, keys/values, projection, and positional encoding
        self.q = Conv(dim, all_head_dim, 1, act=False)  # Query convolution
        self.kv = Conv(dim, all_head_dim * 2, 1, act=False)  # Key/Value convolution
        self.proj = Conv(all_head_dim, dim, 1, act=False)  # Output projection
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)  # Positional encoding
        self.gate_conv1d = nn.Conv1d(2 * head_dim, head_dim, kernel_size=1)  # Gating mechanism

    @staticmethod
    def gumbel_softmax(logits):
        """
        Apply Gumbel-Softmax to approximate differentiable top-k selection.

        Args:
            logits (torch.Tensor): Input logits for token scoring.

        Returns:
            torch.Tensor: Soft weights for token selection.
        """
        gumbels = -torch.empty_like(logits).exponential_().log()  # Generate Gumbel noise
        logits = logits + gumbels
        return F.softmax(logits, dim=-1)  # Apply softmax to get soft weights

    def forward(self, x, upper_feat):
        """
        Process the input tensors through pyramid sparse attention.

        This method computes coarse attention using queries from `x` and keys/values from `upper_feat`. During
        inference, if `topk > 0`, it additionally computes fine attention by selecting key regions from `x`
        based on coarse attention scores, then fuses the outputs using a gating mechanism.

        Args:
            x (torch.Tensor): Lower-level feature map; shape [B, C, H, W].
            upper_feat (torch.Tensor): Higher-level feature map; shape [B, C, H/2, W/2].

        Returns:
            torch.Tensor: Fused feature map after attention; shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        N = H * W
        _, _, H_up, W_up = upper_feat.shape

        # Compute queries from lower-level feature
        q = self.q(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # [B, num_heads, N, head_dim]
        # Compute keys and values from higher-level feature
        kv = self.kv(upper_feat).view(B, self.num_heads, 2 * self.head_dim, H_up * W_up).permute(0, 1, 3, 2)
        k, v = kv.split(self.head_dim, dim=3)  # [B, num_heads, H_up*W_up, head_dim] each

        # Compute coarse attention
        sim = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, num_heads, N, H_up*W_up]
        attn = sim.softmax(dim=-1)  # Attention weights
        coarse_out = (attn @ v)  # [B, num_heads, N, head_dim]

        # Fine attention (computed only during inference if topk > 0)
        if 0 < self.topk <= H_up * W_up:
            # Compute fine keys and values from lower-level feature
            f_kv = self.kv(x).view(B, self.num_heads, 2 * self.head_dim, N).permute(0, 1, 3, 2)
            f_k, f_v = f_kv.split(self.head_dim, dim=3)  # [B, num_heads, N, head_dim] each

            # Aggregate similarity scores over query dimension for token selection
            # global_sim = sim.mean(dim=2)  # [B, num_heads, H_up*W_up]
            #global_sim替换为下面这个
            # ---- 热感显著性加权 Top-K ----
            with torch.no_grad():
                heat = F.adaptive_avg_pool2d(x.mean(1, keepdim=True), (H_up, W_up)).flatten(2)  # [B,1,K]
            global_sim = sim.mean(dim=2) * heat.squeeze(1).unsqueeze(1)  # [B,nh,K] * [B,1,K]

            soft_weights = PSAttn.gumbel_softmax(global_sim)  # [B, num_heads, H_up*W_up]
            topk_weights, topk_indices = torch.topk(soft_weights, k=self.topk, dim=-1)  # [B, num_heads, topk]

            # Map selected indices from upper_feat to x (assuming 2x downsampling)
            scale = 2
            h_idx = (topk_indices // W_up) * scale  # Row indices in x
            w_idx = (topk_indices % W_up) * scale   # Column indices in x
            topk_x_indices = []
            for dh in range(scale):
                for dw in range(scale):
                    idx = (h_idx + dh) * W + (w_idx + dw)
                    topk_x_indices.append(idx)
            topk_x_indices = torch.cat(topk_x_indices, dim=-1)  # [B, num_heads, 4*topk]

            # Gather fine keys and values using mapped indices
            topk_k = torch.gather(f_k, dim=2, index=topk_x_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            topk_v = torch.gather(f_v, dim=2, index=topk_x_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            # [B, num_heads, 4*topk, head_dim] each

            # Compute fine attention
            fine_attn = (q @ topk_k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, num_heads, N, 4*topk]
            fine_attn = fine_attn.softmax(dim=-1)
            refined_out = fine_attn @ topk_v  # [B, num_heads, N, head_dim]

            # Fuse coarse and refined outputs using gating
            fusion_input = torch.cat([coarse_out, refined_out], dim=-1)  # [B, num_heads, N, 2*head_dim]
            fusion_input = fusion_input.view(B * self.num_heads, N, -1).transpose(1, 2)  # [B*num_heads, 2*head_dim, N]
            gate = self.gate_conv1d(fusion_input)  # [B*num_heads, head_dim, N]
            gate = torch.sigmoid(gate).transpose(1, 2).view(B, self.num_heads, N, self.head_dim)
            x = gate * refined_out + (1 - gate) * coarse_out  # Gated fusion
        else:
            x = coarse_out  # Use coarse output only if fine attention is disabled

        # Reshape and apply positional encoding
        x = x.transpose(2, 3).reshape(B, self.all_head_dim, H, W)  # [B, all_head_dim, H, W]
        v_reshaped = v.transpose(2, 3).reshape(B, self.all_head_dim, H_up, W_up)  # [B, all_head_dim, H_up, W_up]
        v_pe = self.pe(v_reshaped)  # [B, dim, H_up, W_up]
        v_pe = F.interpolate(v_pe, size=(H, W), mode='bilinear', align_corners=False)  # [B, dim, H, W]
        x = x + v_pe  # Add positional encoding

        # Project back to original dimension
        return self.proj(x)  # [B, C, H, W]

class PSAttnBlock(nn.Module):
    """
    Pyramid Sparse Attention block module for efficient feature fusion.

    This module implements a Pyramid Sparse Attention (PSAttn) mechanism combined with a
    multi-layer perceptron (MLP) to enhance feature representation while maintaining
    computational efficiency. It is designed for feature fusion across different scales
    in computer vision architectures.

    Attributes:
        attn (PSAttn): Pyramid Sparse Attention module for cross-scale feature fusion.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies attention and feed-forward processing to the input tensor.

    Examples:
        >>> block = PSAttnBlock(dim=256, num_heads=8, mlp_ratio=2)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> upper_feat = torch.randn(1, 256, 16, 16)
        >>> output = block(x, upper_feat)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=2, topk = 0):
        """
        Initialize the Pyramid Sparse Attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in the PSAttn module.
            mlp_ratio (float): Expansion ratio for the MLP hidden dimension.
            topk (int): Number of selected token in fine attention, set 0 for training stage.
        """
        super().__init__()
        self.attn = PSAttn(dim, num_heads=num_heads, topk=topk)  # Pyramid Sparse Attention module
        mlp_hidden_dim = int(dim * mlp_ratio)  # Calculate hidden dimension for MLP
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden_dim, 1),  # Expansion convolution
            Conv(mlp_hidden_dim, dim, 1, act=False)  # Projection back to input dimension
        )

        self.apply(self._init_weights)  # Initialize weights

    def _init_weights(self, m):
        """
        Initialize weights using a truncated normal distribution.

        This method ensures that convolutional layers are initialized with weights drawn
        from a truncated normal distribution, aiding in training stability and convergence.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)  # Truncated normal initialization
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Zero initialization for biases

    def forward(self, x, upper_feat):
        """
        Forward pass through the PSAttnBlock.

        Applies the Pyramid Sparse Attention mechanism followed by the MLP to the input tensor,
        using residual connections to preserve information flow.

        Args:
            x (torch.Tensor): Input feature map; shape [B, C, H, W].
            upper_feat (torch.Tensor): Higher-level feature map; shape [B, C, H/2, W/2].

        Returns:
            torch.Tensor: Output feature map after attention and feed-forward processing.
        """
        x = x + self.attn(x, upper_feat)  # Apply attention with residual connection
        return x + self.mlp(x)  # Apply MLP with residual connection

class HPST(nn.Module):
    """
    Pyramid Sparse Transformer (PST) module for enhanced feature fusion with attention mechanisms.

    This module integrates Pyramid Sparse Attention (PSA) blocks to fuse features from different scales,
    leveraging cross-attention and dynamic token selection for efficient computation. It is designed to
    enhance feature representations in tasks such as object detection and image classification.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cvup (Conv): Initial 1x1 convolution layer that reduces input channels from upper-level feature to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        attnlayer_{i} (PSAttnBlock): Stacked Pyramid Sparse Attention blocks for feature fusion.

    Examples:
        >>> m = HPST(512, 512, 256, n=1, mlp_ratio=2.0, e=0.5, k=0)
        >>> x = (torch.randn(1, 512, 32, 32), torch.randn(1, 512, 16, 16))
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, c1, c_up, c2, n=1, mlp_ratio=2.0, e=0.5, k=0):
        """
        Initialize the Pyramid Sparse Transformer module.

        Args:
            c1 (int): Number of input channels.
            c_up (int): Number of input channels from upper-level feature.
            c2 (int): Number of output channels.
            n (int): Number of PSAttnBlock modules to stack.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension in PSAttnBlock.
            e (float): Channel expansion ratio for hidden channels.
            k (int): Number of top-k tokens in fine attention, set to 0 in training phase.
        """
        super().__init__()
        c_ = int(c2 * e)  # Calculate hidden channels
        assert c_ % 32 == 0, "Hidden channels must be a multiple of 32."

        # Initial convolutions to reduce input and upper feature channels
        self.cv1 = Conv(c1, c_, 1, 1)  # Convolution for input feature
        self.cvup = Conv(c_up, c_, 1, 1)  # Convolution for upper-level feature
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # Final convolution to output channels

        self.num_layers = n
        for i in range(n):
            # Stack PSAttnBlock modules for feature fusion
            layer = PSAttnBlock(c_, c_ // 32, mlp_ratio, topk=k)
            self.add_module(f"attnlayer_{i}", layer)

    def forward(self, x):
        """
        Forward pass through the HPST module.

        Processes the input feature and upper-level feature through initial convolutions,
        applies stacked PSAttnBlock modules for feature fusion, and concatenates the outputs
        before a final convolution to produce the output tensor.

        Args:
            x (tuple): Tuple containing two tensors:
                - x[0] (torch.Tensor): Input feature map; shape [B, c1, H, W].
                - x[1] (torch.Tensor): Upper-level feature map; shape [B, c_up, H/2, W/2].

        Returns:
            torch.Tensor: Output feature map after processing; shape [B, c2, H, W].
        """
        # Extract input and upper-level features from tuple
        upper_feat = x[1]
        x = self.cv1(x[0])

        # Apply initial convolution to upper-level feature
        upper_feat = self.cvup(upper_feat)

        # Initialize list to collect outputs from attention blocks
        y = [x]
        for i in range(self.num_layers):
            # Retrieve and apply the i-th attention block
            layer = getattr(self, f"attnlayer_{i}")
            attened = layer(y[-1], upper_feat)
            y.append(attened)

        # Concatenate all outputs and apply final convolution
        y = self.cv2(torch.cat(y, 1))
        return y








######################################## PyramidSparseTransformer end ########################################
