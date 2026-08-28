import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from functools import partial

from lib.core.config import cfg
from lib.models.backbones.mesh import Mesh

from lib.models.spin import RegressorSpin
from lib.models.hypergcn import HYPERGCv2
from lib.models.Residual import Residual
from lib.models.fusion_module import ComplementTemporal
from lib.models.shape_features import ShapeFeatureExtractor
from lib.models.common import CrossAttentionBlock

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
SMPL_MEAN_PARAMS_PATH = 'data/base_data/smpl_mean_params.npz'


class TemporalMotionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=3, bidirectional=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, embed_dim // 2, num_layers,
            batch_first=True, bidirectional=bidirectional
        )
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, joints):
        B, T, J, _ = joints.shape
        motion = joints[:, 1:] - joints[:, :-1]
        motion = F.pad(motion, (0, 0, 0, 0, 1, 0))
        motion_flat = motion.reshape(B, T, -1)
        x = self.input_proj(motion_flat)
        x, _ = self.lstm(x)
        x = self.output_proj(x)
        x = self.norm(x)
        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.img_reliability = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, 1), nn.Sigmoid()
        )
        self.mot_reliability = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, 1), nn.Sigmoid()
        )
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_feat, mot_feat):
        w_img = self.img_reliability(img_feat)
        w_mot = self.mot_reliability(mot_feat)
        w_sum = w_img + w_mot + 1e-6
        w_img, w_mot = w_img / w_sum, w_mot / w_sum
        fused = self.fusion(torch.cat([img_feat * w_img, mot_feat * w_mot], dim=-1))
        fused = self.norm(fused)
        return fused, {'w_img': w_img, 'w_mot': w_mot}


class IterativePoseRefiner(nn.Module):
    def __init__(self, embed_dim, num_iter=3, hypergcn=None):
        super().__init__()
        self.num_iter = num_iter
        self.hypergcn = hypergcn
        self.pose_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_iter)
        ])
        self.pose_head = MLP(embed_dim, 256, 6, 3)

    def forward(self, pose_token, global_feat):
        B, T, J, D = pose_token.shape
        intermediate_poses = []
        for i in range(self.num_iter):
            ctx = global_feat.unsqueeze(2).expand(-1, -1, J, -1)
            pose_token = pose_token + self.pose_update[i](torch.cat([pose_token, ctx], dim=-1))
            if self.hypergcn is not None:
                pose_token, _ = self.hypergcn(pose_token)
            intermediate_poses.append(self.pose_head(pose_token))
        return pose_token, intermediate_poses


class PoseShapeCoAdaptation(nn.Module):
    def __init__(self, embed_dim, num_rounds=2):
        super().__init__()
        self.num_rounds = num_rounds
        self.pose_to_shape = CrossAttentionBlock(
            q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim,
            kv_num=24, num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
        )
        self.shape_to_pose = CrossAttentionBlock(
            q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim,
            kv_num=1, num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
        )
        self.shape_head = MLP(embed_dim, 256, 10, 3)
        self.pose_head = MLP(embed_dim, 256, 6, 3)

    def forward(self, pose_tokens, shape_token):
        B, T, J, D = pose_tokens.shape

        for _ in range(self.num_rounds):
            # Reshape về 3D: (Batch * Time, Num_Tokens, Dim)
            shape_token_flat = shape_token.view(B * T, 1, D)
            pose_tokens_flat = pose_tokens.view(B * T, J, D)

            # 1. Pose to Shape (Shape token attends to 24 Pose tokens)
            shape_token_flat = self.pose_to_shape(shape_token_flat, pose_tokens_flat, pose_tokens_flat)
            shape_token = shape_token_flat.view(B, T, D)
            
            # 2. Shape to Pose (24 Pose tokens attend to 1 Shape token)
            pose_tokens_flat = self.shape_to_pose(pose_tokens_flat, shape_token_flat, shape_token_flat)
            pose_tokens = pose_tokens_flat.view(B, T, J, D)

        pose_params = self.pose_head(pose_tokens)
        shape_params = self.shape_head(shape_token)

        return pose_params, shape_params, pose_tokens, shape_token


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


class DifferentiableSPINRefiner(nn.Module):
    def __init__(self, spin_model):
        super().__init__()
        self.spin = spin_model

    def forward(self, img_feats, pred_poses, pred_shapes, is_train=True, J_regressor=None):
        B, T = img_feats.shape[:2]
        outputs = []
        for t in range(T):
            out = self.spin(
                img_feats[:, t:t+1],
                init_pose=pred_poses[:, t:t+1],
                init_shape=pred_shapes[:, t:t+1],
                is_train=is_train,
                J_regressor=J_regressor
            )
            outputs.append(out)
        return outputs


class Pose2Mesh(nn.Module):
    def __init__(
        self,
        num_joint,
        embed_dim=512,
        num_refinement_iters=3,
        num_coadapt_rounds=2,
        temporal_layers=3,
        use_hypergcn=True
    ):
        super(Pose2Mesh, self).__init__()

        self.mesh = Mesh()
        self.regressorspin = RegressorSpin()
        self.num_refinement_iters = num_refinement_iters
        self.use_hypergcn = use_hypergcn

        mean_params = np.load(SMPL_MEAN_PARAMS_PATH)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)

        self.out_proj = nn.Linear(512, 2048)
        self.inproj_img = nn.Linear(2048, embed_dim)
        self.pose_embed = nn.Linear(6, embed_dim)
        self.shape_embed = nn.Linear(10, embed_dim)

        self.fuse_shape = CrossAttentionBlock(
            q_dim=512, k_dim=512, v_dim=512, kv_num=cfg.DATASET.seqlen,
            num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
        )

        self.cfcer = ComplementTemporal(depths=2, dim=embed_dim)

        self.fusion = AdaptiveFusion(embed_dim)

        self.residual = Residual(num_joint=num_joint)
        self.node_pe = nn.Embedding(24, embed_dim)

        self.num_hyper_layers = 3
        self.spatial_hypers = nn.ModuleList([
            HYPERGCv2(embed_dim, embed_dim, num_edges=5)
            for _ in range(self.num_hyper_layers)
        ])

        self.temporal_encoder = TemporalMotionEncoder(
            input_dim=57, embed_dim=embed_dim, num_layers=temporal_layers
        )

        self.iterative_refiner = IterativePoseRefiner(
            embed_dim=embed_dim,
            num_iter=num_refinement_iters,
            hypergcn=self.spatial_hypers[0] if use_hypergcn and len(self.spatial_hypers) > 0 else None
        )

        self.pose_shape_coadapt = PoseShapeCoAdaptation(
            embed_dim=embed_dim,
            num_rounds=num_coadapt_rounds
        )

        self.shape_feat_extractor = ShapeFeatureExtractor(embed_dim=embed_dim)
        self.shape_token = nn.Embedding(1, embed_dim)

        max_seqlen = cfg.DATASET.seqlen
        self.pos_embed_cfcer = nn.Parameter(torch.zeros(1, max_seqlen, embed_dim))
        self.pos_embed_motion = nn.Parameter(torch.zeros(1, max_seqlen, embed_dim))
        trunc_normal_(self.pos_embed_cfcer, std=.2)
        trunc_normal_(self.pos_embed_motion, std=.2)

        self.gamma_proj = nn.Linear(embed_dim, embed_dim)
        self.beta_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.spin_refiner = DifferentiableSPINRefiner(self.regressorspin)

    def load_spin_pretrained(self, ckpt_path):
        pretrained_dict = torch.load(ckpt_path, map_location='cpu')['model']
        self.regressorspin.load_state_dict(pretrained_dict, strict=False)

    def forward(
        self,
        joints,
        img_feats,
        kp2d=None,
        using_prompt=True,
        is_train=True,
        J_regressor=None
    ):
        batch_size = img_feats.shape[0]
        seq_len = img_feats.shape[1]
        mid = seq_len // 2

        mean_pose = self.init_pose.view(1, 24, 6)
        mean_shape = self.init_shape.view(1, 10)

        pose_emb = self.pose_embed(mean_pose)
        shape_emb = self.shape_embed(mean_shape)

        pose_token = pose_emb.unsqueeze(1).expand(batch_size, seq_len, 24, -1)
        shape_token = self.shape_token.weight.unsqueeze(0).expand(batch_size, seq_len, -1)
        shape_token = shape_token + shape_emb

        img_feats_proj = self.inproj_img(img_feats)

        motion_feat = self.temporal_encoder(joints)

        img_enhanced, motion_enhanced = self.cfcer(
            img_feats_proj, motion_feat,
            pe_r=self.pos_embed_cfcer[:, :seq_len],
            pe_d=self.pos_embed_motion[:, :seq_len]
        )

        global_ft, fusion_info = self.fusion(img_enhanced, motion_enhanced)

        img_feats_trans = self.out_proj(global_ft) + img_feats

        gamma = self.gamma_proj(global_ft).unsqueeze(2) + 1.0
        beta = self.beta_proj(global_ft).unsqueeze(2)
        pose_token = gamma * pose_token + beta

        idx = torch.arange(24, device=pose_token.device)
        pose_token = self.norm(pose_token) + self.node_pe(idx)

        pose_token, intermediate_poses = self.iterative_refiner(pose_token, global_ft)

        if kp2d is not None:
            shape_feat = self.shape_feat_extractor(kp2d)
            shape_token = shape_token + shape_feat

        shape_output = self.fuse_shape(shape_token, global_ft, global_ft)

        pred_pose, pred_shape, refined_pose_token, refined_shape_token = self.pose_shape_coadapt(
            pose_token, shape_output
        )

        inv_pred2rot6d = pred_pose.reshape(batch_size, seq_len, -1)
        inv_mesh2shape = pred_shape.reshape(batch_size, seq_len, -1)

        spin_outputs = self.spin_refiner(
            img_feats_trans, inv_pred2rot6d, inv_mesh2shape,
            is_train=is_train, J_regressor=J_regressor
        )

        smpl_vertices_mid = spin_outputs[mid][-1]['verts'].squeeze(1)
        residual_joint, residual_mesh = self.residual(
            joints[:, mid], img_feats[:, mid]
        )
        smpl_vertices_mid = 0.5 * smpl_vertices_mid + 0.5 * residual_mesh

        evo_pose = pred_pose.reshape(batch_size, -1)
        init_smpl_pose = pred_pose[:, mid].reshape(batch_size, -1)
        init_smpl_shape = pred_shape[:, mid].reshape(batch_size, -1)
        
        return (
            evo_pose,
            init_smpl_pose,
            init_smpl_shape,
            smpl_vertices_mid,
            spin_outputs
        )


def get_model(num_joint, embed_dim, **kwargs):
    model = Pose2Mesh(num_joint, embed_dim, **kwargs)
    return model