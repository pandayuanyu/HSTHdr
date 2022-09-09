import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
import numpy as np
import torch.utils.checkpoint as checkpoint


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


# 特征图转window
# [B, H, W, C] -> [B*num_windows, Mh, Mw, C]
def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# window转特征图
# [B*num_windows, Mh, Mw, C] -> [B, H, W, C]
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# 输入图像转patch，四倍下采样
# [B, in_c, H_in, W_in] -> [B, C, H_in/4, W_in/4]
# [B, C, H, W] -> [B, HW, C]
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# 二倍下采样
# [B, H*W, C] -> [B, H/2*W/2, 2*C]
# [B, L, C] -> [B, H, W, C] -> [B, H/2*W/2, 4*C] -> [B, H/2*W/2, 2*C]
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


# Transformer里的MLP
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# Transformer里的WMSA
# [batch_size*num_windows, Mh*Mw, C] -> [batch_size*num_windows, Mh*Mw, C]
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Transformer

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])


    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        return x, H, W








# 扩张密集块 DDB - Dilated Dense Block
class DDB(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DDB, self).__init__()
        self.Dilated_conv = Conv_Layer(
            in_channles=in_channels, out_channels=growth_rate,
            padding=2, dilation=2,
        )

    def forward(self, x):
        out = self.Dilated_conv(x)
        return torch.cat([x, out], dim=1)


# 扩张残差密集块 DRDB - Dilated Residual Dense Block
class DRDB(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32, Layer_num=6):
        super(DRDB, self).__init__()
        Current_Channel_num = in_channels

        modules = []
        for i in range(Layer_num):
            modules.append(DDB(in_channels=Current_Channel_num, growth_rate=growth_rate))
            Current_Channel_num += growth_rate
        self.dense_layers = nn.Sequential(*modules)

        self.conv_1x1 = Conv_Layer(
            in_channles=Current_Channel_num, out_channels=in_channels,
            kernel_size=1, padding=0,
        )

    def forward(self, x):
        out1 = self.dense_layers(x)
        out2 = self.conv_1x1(out1)

        return out2 + x








#######################################################################################################
############################### 以上部分为SwinTransformer ##############################################
#######################################################################################################


# 基本卷积模块
class Conv_Layer(nn.Module):
    def __init__(self, in_channles, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1,
                 Activation_required=True
                 ):
        super(Conv_Layer, self).__init__()
        self.conv = nn.Conv2d(
                in_channels=in_channles, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
            )
        self.relu = nn.ReLU(inplace=True)
        self.Activation_required = Activation_required

    def forward(self, x):
        out = self.conv(x)
        if self.Activation_required:
            out = self.relu(out)
        return out


# 注意力模块
class Attention_Module1(nn.Module):
    def __init__(self):
        super(Attention_Module1, self).__init__()
        self.model = nn.Sequential(
            Conv_Layer(in_channles=128, out_channels=64),
            Conv_Layer(in_channles=64, out_channels=64, Activation_required=False),
            nn.Sigmoid()
        )

    def forward(self, Zi, Zr):
        x = torch.cat([Zi, Zr], dim=1)
        out = self.model(x)
        return out


# 注意力网络
# class Attention_Network1(nn.Module):
#     def __init__(self):
#         super(Attention_Network1, self).__init__()
#         self.encoder = Conv_Layer(in_channles=7, out_channels=64)
#         self.Attention_12 = Attention_Module1()
#         self.Attention_21 = Attention_Module1()
#
#     def forward(self, X1, X2):
#         Z1 = self.encoder(X1)
#         Z2 = self.encoder(X2)
#
#         A1 = self.Attention_12(Z1, Z2)
#         A2 = self.Attention_21(Z2, Z1)
#
#         Z1_apo = Z1 * A1
#         Z2_apo = Z2 * A2
#
#         Zs = torch.cat([Z1_apo, Z2, Z2_apo], dim=1)
#
#         return Zs, Z2




# 特征编码层一：dense空洞卷积层
class Can_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(Can_Block, self).__init__()
        self.can_conv = Conv_Layer(in_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        out = self.can_conv(x)
        out = torch.cat([x, out], 1)
        return out


class Can_Encoder(torch.nn.Module):
    def __init__(self, features_num=16, kernel_size=3, stride=1):
        super(Can_Encoder, self).__init__()
        out_channels_def = 16
        self.up_channel_layer = Conv_Layer(7, features_num, 3, 1, 1, 1)
        self.can_block = nn.Sequential(
            Can_Block(features_num, out_channels_def, kernel_size, stride, padding=1, dilation=1),
            Can_Block(features_num + out_channels_def, out_channels_def, kernel_size, stride, padding=2, dilation=2),
            Can_Block(features_num + out_channels_def * 2, out_channels_def, kernel_size, stride, padding=4, dilation=4)
        )

    def forward(self, x):
        x = self.up_channel_layer(x)
        x = self.can_block(x)
        return x



# 通道注意力
class Channel_Attention(nn.Module):
    """ Channel attention module"""

    # https://github.com/junfu1115/DANet/blob/v0.5.0/encoding/nn/attention.py
    def __init__(self, in_dim):
        super(Channel_Attention, self).__init__()
        self.chanel_in = in_dim
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Z1, Zr):
        # Z1 for the first image
        # Zr for the second(reference) image
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = Zr.size()
        # b, c, h, w -> b, c, hxw
        proj_query = Z1.view(m_batchsize, C, -1)
    # b, c, h, w -> b, c, hxw -> b, hxw, c
        proj_key = Zr.view(m_batchsize, C, -1).permute(0, 2, 1)
        # (b, c, hxw) x (b, hxw, c) -> (b, c, c)
        energy = torch.bmm(proj_query, proj_key)
        # (b, c, c)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # b, c, c
        attention = self.softmax(energy_new)
        # b, c, h, w -> b, c, hxw
        proj_value = Zr.view(m_batchsize, C, -1)
        # (b, c, c) x (b, c, hxw) -> (b, c, hxw)
        out = torch.bmm(attention, proj_value)
        # b, c, hxw -> b, c, h, w
        out = out.view(m_batchsize, C, height, width)
        # b, c, h, w
        out = self.beta * out + Zr

        return out


# 空间注意力
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self,  Z1, Zr):

        m_batchsize, _, height, width = Zr.size()

        proj_query = self.query_conv(Z1)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous() \
            .view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous() \
            .view(m_batchsize * height, -1, width).permute(0, 2, 1)

        proj_key = self.key_conv(Zr)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(Zr)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)) \
            .view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        out = self.alpha * (out_H + out_W) + Zr
        return out


# 注意力网络
class Attention_Network(nn.Module):
    def __init__(self):
        super(Attention_Network, self).__init__()
        self.encoder = Can_Encoder(features_num=16)
        self.Attention_Channel = Channel_Attention(in_dim=64)


    def forward(self, X1, X2):
        Z1 = self.encoder(X1)
        Zr = self.encoder(X2)

        # Z_Spatial1 = self.Attention_Spatial(Z1, Zr)
        Z_Channel1 = self.Attention_Channel(Z1, Zr)
       # Z_Channel2 = self.Attention_Channel(Zr, Z1)

        Zs = torch.cat((Z_Channel1, Zr), dim=1)  # 128
        return Zs, Zr

# 用于 HDR 图像估计的合并网络Merging_Network
class Merging_Network(nn.Module):
    def __init__(self):
        super(Merging_Network, self).__init__()
      #   self.conv1 = Conv_Layer(in_channles=64 * 3, out_channels=64)
      #   self.PatchEmbedding = PatchEmbed(
      #       patch_size=1,
      #       in_c=64,
      #       embed_dim=96,
      #       norm_layer=nn.LayerNorm
      #   )
      #   self.SwinTransformer1 = BasicLayer(
      #       dim=96,
      #       depth=2,
      #       num_heads=3,
      #       window_size=8,
      #       mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
      #       drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False
      #   )
      #   self.SwinTransformer2 = BasicLayer(
      #       dim=96,
      #       depth=2,
      #       num_heads=3,
      #       window_size=8,
      #       mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
      #       drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False
      #   )
      #   self.SwinTransformer3 = BasicLayer(
      #       dim=96,
      #       depth=2,
      #       num_heads=3,
      #       window_size=8,
      #       mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
      #       drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False
      #   )
      #   self.conv2 = Conv_Layer(in_channles=96 * 3, out_channels=64)
      #   self.conv3 = Conv_Layer(in_channles=64, out_channels=64)
      # #  self.spatial_att = CrissCrossAttention(in_dim=64)
      #   self.conv4 = Conv_Layer(in_channles=64, out_channels=3, Activation_required=False)
      #   self.tanh = nn.Tanh()
        self.conv1 = Conv_Layer(in_channles=64 * 2, out_channels=64)
        self.DRDB1 = DRDB()
        self.DRDB2 = DRDB()
        self.DRDB3 = DRDB()
        self.conv2 = Conv_Layer(in_channles=64 * 3, out_channels=64)
        self.conv3 = Conv_Layer(in_channles=64, out_channels=64)
        self.conv4 = Conv_Layer(in_channles=64, out_channels=3, Activation_required=False)
        self.tanh = nn.Tanh()

    def forward(self, Zs, Zr):
        # B, 64, 256, 256
        # F0 = self.conv1(Zs)
        # B = F0.shape[0]
        #
        # F0_emd, H, W = self.PatchEmbedding(F0)
        # C = F0_emd.shape[-1]
        #
        # F1, H, W = self.SwinTransformer1(F0_emd, H, W)
        # F2, H, W = self.SwinTransformer2(F1, H, W)
        # F3, H, W = self.SwinTransformer3(F2, H, W)
        #
        # F1 = F1.view(B, H, W, C).permute(0, 3, 1, 2)
        # F2 = F2.view(B, H, W, C).permute(0, 3, 1, 2)
        # F3 = F3.view(B, H, W, C).permute(0, 3, 1, 2)
        #
        # F4 = torch.cat([F1, F2, F3], dim=1)
        # F5 = self.conv2(F4)
        # F6 = self.conv3(F5 + Zr)
        # F7 = self.conv4(F6)

        F0 = self.conv1(Zs)
        F1 = self.DRDB1(F0)
        F2 = self.DRDB2(F1)
        F3 = self.DRDB3(F2)

        F4 = torch.cat([F1, F2, F3], dim=1)

        F5 = self.conv2(F4)
        F6 = self.conv3(F5 + Zr)
        F7 = self.conv4(F6)

        return self.tanh(F7) * 0.5 + 0.5


# AHDRNet
class AHDRNet(nn.Module):
    def __init__(self):
        super(AHDRNet, self).__init__()
        self.A = Attention_Network()
        self.M = Merging_Network()

    def forward(self, X1, X2):
        Zs, Zr = self.A(X1, X2)
        out = self.M(Zs, Zr)
        return out






from torch.utils.tensorboard import SummaryWriter
# 224694
if __name__ == '__main__':
    B = 7
    C = 7
    H = 229
    W = 129

    device = torch.device('cuda')
    model = AHDRNet().to(device)
    image = torch.randn(B, C, H, W).to(device)

    out= model(image, image)
    print(out.shape)

    print("-SwinT构建完成，参数量为： {} ".format(sum(x.numel() for x in model.parameters())))





    # writer = SummaryWriter('./logs/net/PatchEmbed')
    # writer.add_graph(model, image)
    # writer.close()
