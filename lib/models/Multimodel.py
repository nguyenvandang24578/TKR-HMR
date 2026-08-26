import os, sys
sys.path.append('./lib')
import matplotlib
matplotlib.use('Agg')   # ← dòng đầu tiên, trước cả import pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp
from geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis, rodrigues

from core.config import cfg
from graph_utils import build_verts_joints_relation
from models.backbones.mesh import Mesh
from functools import partial
from models.smpl_mps import SMPL_MEAN_PARAMS

from models.spin import RegressorSpin
from models.Core_model import PoseImageCrossAttention
from models.hypergcn import HYPERGC
class TemporalConv1D(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, stride=1, padding=1):
        super(TemporalConv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, D) or (B*V, T, D)
        out = x.permute(0, 2, 1).contiguous() # (B, D, T)
        out = self.conv1d(out)
        out = out.permute(0, 2, 1).contiguous() # (B, T, D)
        out = self.norm(out)
        out = self.act(out)
        return out + x # residual connection
from models.Residual import Residual
from models.fusion_module import ComplementSpatial
from math import sqrt
import pickle
import random
BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
from models.smpl_mps import SMPL as smpl
from smpl import SMPL
SMPL_MODEL_DIR = 'data/base_data'
SMPL_MEAN_PARAMS = 'data/base_data/smpl_mean_params.npz'
BASE_DATA_DIR = 'data/base_data'
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
        
        # ======== CODE VISUALIZE ATTENTION ========
        if not self.training and B > 0 and not hasattr(self, '_cross_attn_vis_done'):
            import matplotlib.pyplot as plt
            import os
            os.makedirs('debug_vis', exist_ok=True)
            attn_map = attn[0, 0].detach().cpu().numpy() # [N1, N2]
            plt.figure(figsize=(6, 6))
            plt.imshow(attn_map, cmap='viridis')
            plt.colorbar()
            plt.title('Cross Attention Map')
            plt.savefig(f'debug_vis/cross_attn_shape.png')
            plt.close()
            self._cross_attn_vis_done = True
        # ==========================================
        
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
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
class Pose2Mesh(nn.Module):
    def __init__(self, num_joint, embed_dim=512, smpl_head_hidden_dim: int = 256, smpl_head_depth: int = 3,
                SMPL_MEAN_vertices=osp.join(BASE_DATA_DIR, 'smpl_mean_vertices.npy')):
        super(Pose2Mesh, self).__init__()

        self.mesh = Mesh()
        self.regressorspin = RegressorSpin()
        pretrained_dict = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))['model']
        self.regressorspin.load_state_dict(pretrained_dict, strict=False)
        self.smpl = smpl(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        )
# =========================================================
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
#-------------------------------------------------------------------------------------
        self.projoint = nn.Sequential(
            nn.Linear(57, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        
        self.out_proj = nn.Linear(512, 2048)
        self.inproj_img = nn.Linear(2048, embed_dim)
        self.pose_embed  = nn.Linear(6, embed_dim)
        self.shape_embed  = nn.Linear(10, embed_dim)
        self.fuse_shape = CrossAttentionBlock(q_dim=512, k_dim=512, v_dim=512, kv_num = cfg.DATASET.seqlen, num_heads=8, mlp_ratio=4., qkv_bias=True,
                                        drop=0., attn_drop=0., drop_path=0.2, has_mlp=True)
#-------------------------------------------------------------------------------------
        # CFCer cross-fusion: img ↔ motion mutual attention trước khi merge
        self.cfcer = ComplementSpatial(depths=2, dim=embed_dim)
        # Mamba 1D early fusion
        self.fusion_linear = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
#-------------------------------------------------------------------------------------
        self.residual = Residual(num_joint=num_joint)
        self.node_pe = nn.Embedding(24, embed_dim)
        self.num_hyper_layers = 3
        self.spatial_hypers = nn.ModuleList([
            HYPERGC(embed_dim, embed_dim, vertex_nums=24)
            for _ in range(self.num_hyper_layers)
        ])
#-------------------------------------------------------------------------------------
        self.pose_head = MLP(embed_dim, smpl_head_hidden_dim, 6, smpl_head_depth)
        self.shape_head = MLP(embed_dim, smpl_head_hidden_dim, 10, smpl_head_depth)
        self.kpt_mlp = nn.Sequential(
            nn.Linear(2, embed_dim // 2),   # x, y
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )    
        self.kp_norm = nn.LayerNorm(embed_dim)
        self.kp_map = nn.Parameter(torch.eye(19, 24))  # (19, 24) learnable mapping
        self.shape_token = nn.Embedding(1, embed_dim)
#-------------------------------------------------------------------------------------
        self.pos_embed_cfcer = nn.Parameter(torch.zeros(1, cfg.DATASET.seqlen, embed_dim))
        trunc_normal_(self.pos_embed_cfcer, std=.02)

        self.gamma_proj = nn.Linear(embed_dim, embed_dim)
        self.beta_proj  = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.inject_norm = nn.LayerNorm(embed_dim)
    def forward(self, joints, img_feats, kp2d = None, using_prompt=True, is_train=True, J_regressor=None):
        batch_size = img_feats.shape[0]   # B
        seq_len    = img_feats.shape[1]   # T
        mid = seq_len // 2
        use_kp2d = True

        # 1. Shape gốc từ init_params
        mean_pose  = self.init_pose.view(1, 24, 6)              # (1, 24, 6)
        mean_shape  = self.init_shape.view(1, 10)              # (1, 24, 6)
        pose_emb   = self.pose_embed(mean_pose)                  # (1, 24, embed_dim)
        shape_emb = self.shape_embed(mean_shape) #(1, dim)
        pose_token = pose_emb.unsqueeze(1).expand(
            batch_size, seq_len, 24, -1
        )   
        shape_token = self.shape_token.weight.unsqueeze(0).expand(
            batch_size, seq_len, -1
        ) #(B, T, dim)
        shape_token = shape_token + shape_emb
        if kp2d is not None and use_kp2d:
            kp_emb = self.kpt_mlp(kp2d)          # (B, T, 19, 512)
            kp_emb = self.kp_norm(kp_emb)
            
            # Map 19 keypoints → 24 joints
            kp_add = torch.einsum('btid,ij->btjd', kp_emb, self.kp_map)
            
            # Inject vào pose tokens trực tiếp
            pose_token = pose_token + kp_add     # (B, T, 24, 512)
            pose_token = self.inject_norm(pose_token)  # ← thêm vào đây
        img_feats_proj = self.inproj_img(
            img_feats
        )  # (B, T, 2048) -> # (B, T, 512)

        joints_seq = joints

        # 1: Motion-Centric Refinement
        # motion torch.Size([30, 15, 19, 3])
        motion = joints_seq[:,1:] - joints_seq[:,:-1]
        mean_motion = torch.mean(motion, dim=1,keepdim=True)
        motion = torch.cat([mean_motion, motion], dim=1)
        
        motion_reshaped = motion.reshape(batch_size, seq_len, -1)
        joints_seq_trans = self.projoint(motion_reshaped)
        

        img_feats_pe = img_feats_proj + self.pos_embed_cfcer
        motion_pe = joints_seq_trans + self.pos_embed_cfcer
        img_enhanced, motion_enhanced = self.cfcer(img_feats_pe, joints_seq_trans)

        
        # Early Fusion: Concat enhanced features
        concat_feat = torch.cat([img_enhanced, motion_enhanced], dim=-1) # (B, T, 1024)
        x_fused = self.fusion_linear(concat_feat) # (B, T, 512)
        
        img_feats_trans = self.out_proj(x_fused) + img_feats # (B, T, 2048) + skip

        global_ft = x_fused

        gamma = self.gamma_proj(global_ft).unsqueeze(2) + 1.0
        beta  = self.beta_proj(global_ft).unsqueeze(2)   # (B, T, 1, D)

        out = gamma * pose_token + beta                        # (B,S T, 24, D)

        idx = torch.arange(24, device=out.device)
        dang = self.norm(out) + self.node_pe(idx)
        
        dang_permuted = dang.permute(0, 3, 1, 2).contiguous() # (B, D, T, 24)
        adj_dict = None
        for hyper_layer in self.spatial_hypers:
            dang_permuted, _, adj_dict = hyper_layer(dang_permuted)
        pose_token_op = dang_permuted.permute(0, 2, 3, 1).contiguous() + dang # (B, T, 24, D) + skip around HyperGCN
        
        f_pose  = self.pose_head(pose_token_op) # (B, T, 24, 6)   
        inv_pred2rot6d = f_pose.reshape(batch_size, seq_len, -1)
#---------------------------------------------------------------------------------------------------------------------------------------
        shape_output = self.fuse_shape(shape_token, global_ft, global_ft)
        f_shape  = self.shape_head(shape_output) # (B, T, 24, 6)   
        inv_mesh2shape = f_shape.reshape(batch_size, seq_len, -1)
#---------------------------------------------------------------------------------------------------------------------------------------
        spin_pose = inv_pred2rot6d[:, mid].unsqueeze(1)
        spin_shape = inv_mesh2shape[:, mid].unsqueeze(1)
        spin_img_feat = img_feats_trans[:, mid].unsqueeze(1)
        # print("\n[Pose2Mesh] spin_pose.shape: ", spin_pose.shape)
        # print("\n[Pose2Mesh] spin_shape.shape: ", spin_shape.shape)
        # print("\n[Pose2Mesh] spin_img_feat.shape: ", spin_img_feat.shape)
        output = self.regressorspin(spin_img_feat, init_pose=spin_pose, init_shape=spin_shape, is_train=is_train, J_regressor=J_regressor)

        # attentive addtion
        smpl_vertices_mid = output[-1]['verts'].squeeze(1)
        # print(smpl_vertices_mid.shape)
        residual_joint, residual_mesh = self.residual(joints[:,cfg.DATASET.seqlen // 2], img_feats[:,cfg.DATASET.seqlen // 2])
        smpl_vertices_mid = 0.5 * smpl_vertices_mid + 0.5 * residual_mesh
  
        
        return residual_joint, spin_pose.reshape(batch_size, 144), spin_shape, smpl_vertices_mid, output
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
# ============================================================
# Factory
# ============================================================
def get_model(num_joint, embed_dim):
    model = Pose2Mesh(num_joint, embed_dim)
    return model