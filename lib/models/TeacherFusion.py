import os, sys
sys.path.append('./lib')
import torch
import torch.nn as nn 
from core.config import cfg
from models.spin import RegressorSpin
from models.Multimodel import CrossAttentionBlock
import sys
import os.path as osp
sys.path.extend(['./STA-GCN'])
from model.stagcn import STA_GCN, Embeddeding
from graph.coco19 import Graph

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
    
class STAGCN_Backbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(STAGCN_Backbone, self).__init__()
        
        self.graph = Graph()
        A = self.graph.A_binary_with_I
        A_norm = self.graph.A_norm
        num_person = 1
        num_point = 19
        
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.embedder = Embeddeding(in_channels, base_channels)
        
        # stride=1 for ALL layers to preserve temporal sequence length (T=16)
        self.layer1 = STA_GCN(base_channels, base_channels, A_norm, A, residual=False)
        self.layer2 = STA_GCN(base_channels, base_channels, A_norm, A)
        self.layer3 = STA_GCN(base_channels, base_channels, A_norm, A)
        self.layer4 = STA_GCN(base_channels, base_channels, A_norm, A)
        self.layer5 = STA_GCN(base_channels, 2*base_channels, A_norm, A, stride=1) # Modified stride
        self.layer6 = STA_GCN(2*base_channels, 2*base_channels, A_norm, A)
        self.layer7 = STA_GCN(2*base_channels, 2*base_channels, A_norm, A)
        self.layer8 = STA_GCN(2*base_channels, 4*base_channels, A_norm, A, stride=1) # Modified stride
        self.layer9 = STA_GCN(4*base_channels, 4*base_channels, A_norm, A)
        self.layer10 = STA_GCN(4*base_channels, 4*base_channels, A_norm, A)
        
        self.num_point = num_point

    def forward(self, x):
        # x: [B, T, V, C] -> [B, C, T, V, M]
        if len(x.shape) == 4:
            B, T, V, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1) # [B, C, T, V, 1]
            
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.embedder(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)

        # Output shape of layer10 is [N*M, 4*base_channels, T, V]
        # We want to pool spatially (over V) and keep T
        # x shape: [N, 256, T, 19]
        x = x.mean(3) # Spatial average pooling -> [N, 256, T]
        x = x.permute(0, 2, 1).contiguous() # [N, T, 256]
        return x

class TeacherFusion(nn.Module):
    def __init__(self, embed_dim=512, smpl_head_hidden_dim=256, smpl_head_depth=3):
        super(TeacherFusion, self).__init__()
        
        # 1. Feature Extractor cho Skeleton (GT)
        self.skeleton_backbone = STAGCN_Backbone(in_channels=3, base_channels=64)
        
        # Linear layer mapping STAGCN output (256) to embed_dim (512)
        self.skel_proj = nn.Linear(256, embed_dim)
        
        # 2. Linear layer cho Image Features
        self.img_proj = nn.Linear(2048, embed_dim)
        
        # 3. Cross Attention Fusion
        self.mcca = CrossAttentionBlock(q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim, 
                                        kv_num=cfg.DATASET.seqlen, num_heads=8, mlp_ratio=4., 
                                        qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1, drop_path=0.1)
        
        # 4. Project fused features back to 2048 for RegressorSpin
        self.fused_proj = nn.Linear(embed_dim, 2048)
        
        # 5. Motion projection (bổ sung thông tin temporal)
        self.motion_proj = nn.Sequential(
            nn.Linear(19 * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.regressorspin = RegressorSpin()
        pretrained_dict = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))['model']
        self.regressorspin.load_state_dict(pretrained_dict, strict=False)

    def forward(self, gt_pose3d, img_feats):
        """
        gt_pose3d: [B, T, 19, 3] (Ground truth joints)
        img_feats: [B, T, 2048] (Image features from CNN)
        """
        skel_feats = self.skeleton_backbone(gt_pose3d) # [B, T, 256]
        skel_feats = self.skel_proj(skel_feats)        # [B, T, 512]
        
        # Project image features
        img_feats_proj = self.img_proj(img_feats)      # [B, T, 512]
        
        # Fusion: Skeleton query Image
        fused_feats = self.mcca(skel_feats, img_feats_proj, img_feats_proj) # [B, T, 512]
        
        # Project back to 2048 for RegressorSpin
        fused_feats_2048 = self.fused_proj(fused_feats)
        
        # Regress SMPL Parameters
        smpl_output, _, _, _ = self.regressorspin(fused_feats_2048)
        pred_mesh = smpl_output[-1]['verts'][:, cfg.DATASET.seqlen // 2]  # [B, V, 3]
        return fused_feats, pred_mesh, smpl_output, skel_feats

def get_model(embed_dim=512):
    return TeacherFusion(embed_dim=embed_dim)
