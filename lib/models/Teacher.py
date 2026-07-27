import torch
import torch.nn as nn
from core.config import cfg as cfg
from models import TeacherFusion, PoseEstimation

import os
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


class Teacher(nn.Module):
    def __init__(self, num_joint, embed_dim, depth):
        super(Teacher, self).__init__()

        self.num_joint = num_joint
        self.pose_lifter = PoseEstimation.get_model(num_joint, embed_dim, depth, pretrained=cfg.MODEL.posenet_pretrained)
        self.teacher_fusion = TeacherFusion.get_model(embed_dim)

    def forward(self, pose2d, img_feat, is_train=True):
        pose3d = self.pose_lifter(pose2d, img_feat)
        pose3d = pose3d.reshape(-1, cfg.DATASET.seqlen, self.num_joint, 3)
        
        fused_feats, final_mesh, smploutput = self.teacher_fusion(pose3d / 1000, img_feat)
        
        pose3d = pose3d

        return pose3d, final_mesh, smploutput


def get_model(num_joint, embed_dim, depth):
    model = Teacher(num_joint, embed_dim, depth)

    return model
