import torch
import torch.nn as nn
from core.config import cfg as cfg
from models import Multimodel, PoseEstimation
from models import student as StudentFusion

import os
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


class TKR(nn.Module):
    def __init__(self, num_joint, embed_dim, depth):
        super(TKR, self).__init__()

        self.num_joint = num_joint
        self.pose_lifter = PoseEstimation.get_model(num_joint, embed_dim, depth, pretrained=cfg.MODEL.posenet_pretrained)
        self.pose_mesh_coevo = Multimodel.get_model(num_joint, embed_dim*2)

    def forward(self, pose2d, img_feat, is_train=True):
        pose3d = self.pose_lifter(pose2d, img_feat)
        pose3d = pose3d.reshape(-1, cfg.DATASET.seqlen, self.num_joint, 3)
        
        final_mesh, smploutput = self.pose_mesh_coevo(pose3d / 1000, img_feat, pose2d, is_train=is_train)
        
        pose3d = pose3d[:,cfg.DATASET.seqlen // 2]

        return pose3d, final_mesh, smploutput

class Student(nn.Module):
    def __init__(self, num_joint, embed_dim, depth):
        super(Student, self).__init__()

        self.num_joint = num_joint
        self.pose_lifter = PoseEstimation.get_model(num_joint, embed_dim, depth, pretrained=cfg.MODEL.posenet_pretrained)
        self.features_fusion = StudentFusion.get_model(embed_dim)

    def forward(self, pose2d, img_feat, is_train=True):
        pose3d = self.pose_lifter(pose2d, img_feat)
        pose3d = pose3d.reshape(-1, cfg.DATASET.seqlen, self.num_joint, 3)
        
        fused_feats, final_mesh, smploutput = self.features_fusion(pose3d, img_feat)
        return pose3d, fused_feats, final_mesh, smploutput


def get_model(num_joint, embed_dim, depth):
    model = TKR(num_joint, embed_dim, depth)

    return model

def get_student_model(num_joint, embed_dim, depth):
    model = Student(num_joint, embed_dim, depth)

    return model