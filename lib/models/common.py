# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, depth=4, embed_dim=512, mlp_hidden_dim=2048, length=cfg.DATASET.seqlen, h=8):
        super().__init__()
        drop_rate = 0.1
        drop_path_rate = 0.2
        attn_drop_rate = 0.

        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=True, qk_scale=None,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)]) 
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, v_dim, kv_num, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.kv_num = kv_num
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(v_dim, v_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):

        B, N, C = xq.shape
        v_dim = xv.shape[-1]
        q = self.wq(xq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,N1,C] -> [B,N1,H,(C/H)] -> [B,H,N1,(C/H)]
        k = self.wk(xk).reshape(B, self.kv_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,N2,C] -> [B,N2,H,(C/H)] -> [B,H,N2,(C/H)]
        v = self.wv(xv).reshape(B, self.kv_num, self.num_heads, v_dim // self.num_heads).permute(0, 2, 1, 3)  # [B,N2,C] -> [B,N2,H,(C/H)] -> [B,H,N2,(C/H)]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N1,(C/H)] @ [B,H,(C/H),N2] -> [B,H,N1,N2]
        attn = attn.softmax(dim=-1)
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, v_dim)   # [B,H,N1,N2] @ [B,H,N2,(C/H)] -> [B,H,N1,(C/H)] -> [B,N1,H,(C/H)] -> [B,N1,C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, kv_num, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.2, 
                 attn_drop=0.2, drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.normq = norm_layer(q_dim)
        self.normk = norm_layer(k_dim)
        self.normv = norm_layer(v_dim)
        self.kv_num = kv_num
        self.attn = CrossAttention(q_dim, v_dim, kv_num = kv_num, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(q_dim)
            mlp_hidden_dim = int(q_dim * mlp_ratio)
            self.mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xq, xk, xv):
        xq = xq + self.drop_path(self.attn(self.normq(xq), self.normk(xk), self.normv(xv)))
        if self.has_mlp:
            xq = xq + self.drop_path(self.mlp(self.norm2(xq)))

        return xq
