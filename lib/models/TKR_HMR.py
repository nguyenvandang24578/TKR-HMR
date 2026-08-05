import torch
import torch.nn as nn
from core.config import cfg as cfg
from models import PoseEstimation
from models import student as StudentFusion

import os
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


class TKR(nn.Module):
    def __init__(self, num_joint, embed_dim, depth):
        super(TKR, self).__init__()
        from models import Multimodel  # Lazy import to avoid circular dependency

        self.num_joint = num_joint
        self.student_kd = get_student_model(num_joint, embed_dim, depth)
        
        # Load pre-trained KD weights
        if hasattr(cfg.MODEL, 'kd_student_path') and cfg.MODEL.kd_student_path:
            if os.path.exists(cfg.MODEL.kd_student_path):
                print(f"===> Loading pretrained KD Student from {cfg.MODEL.kd_student_path}")
                checkpoint = torch.load(cfg.MODEL.kd_student_path, map_location='cpu')
                self.student_kd.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                print(f"===> WARNING: KD Student path {cfg.MODEL.kd_student_path} not found!")

        # Freeze student_kd: no gradient, no training
        for p in self.student_kd.parameters():
            p.requires_grad = False
        self.student_kd.eval()

        # Multimodel should take embed_dim (512), not embed_dim*2 (1024)
        self.pose_mesh_coevo = Multimodel.get_model(num_joint, embed_dim)

    def train(self, mode=True):
        """Override train() to keep student_kd always in eval mode."""
        super().train(mode)
        self.student_kd.eval()
        return self

    def forward(self, pose2d, img_feat, is_train=True):
        # student_kd is frozen: use no_grad and detach outputs
        with torch.no_grad():
            pose3d, fused_feats, pred_mesh, smpl_output, skel_feats = self.student_kd(pose2d, img_feat)
        
        # Detach to completely cut gradient flow back to student_kd
        pose3d = pose3d.detach()
        fused_feats = fused_feats.detach()
        
        # Pass fused_feats as img_feats, and pose3d (converted to meters) as joints
        evo_pose, init_smpl_pose, init_smpl_shape, final_mesh, smploutput = self.pose_mesh_coevo(img_feat, fused_feats, joints=pose3d / 1000.0, kp2d=pose2d, is_train=is_train)
        pose3d = pose3d[:,cfg.DATASET.seqlen // 2]

        return pose3d, evo_pose, init_smpl_pose, init_smpl_shape, final_mesh, smploutput

class Student(nn.Module):
    def __init__(self, num_joint, embed_dim, depth):
        super(Student, self).__init__()

        self.num_joint = num_joint
        self.pose_lifter = PoseEstimation.get_model(num_joint, embed_dim, depth, pretrained=cfg.MODEL.posenet_pretrained)
        self.features_fusion = StudentFusion.get_model(embed_dim)

    def forward(self, pose2d, img_feat, is_train=True):
        pose3d = self.pose_lifter(pose2d, img_feat)
        pose3d = pose3d.reshape(-1, cfg.DATASET.seqlen, self.num_joint, 3)
        
        fused_feats, final_mesh, smploutput, skel_feats = self.features_fusion(pose3d, img_feat)
        return pose3d, fused_feats, final_mesh, smploutput, skel_feats


def get_model(num_joint, embed_dim, depth):
    model = TKR(num_joint, embed_dim, depth)

    return model

def get_student_model(num_joint, embed_dim, depth):
    model = Student(num_joint, embed_dim, depth)

    return model