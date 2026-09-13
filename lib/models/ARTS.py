import torch
import torch.nn as nn
from core.config import cfg as cfg
from models import Multimodel, PoseEstimation

import os
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


class ARTS(nn.Module):
    def __init__(self, num_joint, embed_dim, depth):
        super(ARTS, self).__init__()

        self.num_joint = num_joint
        self.pose_lifter = PoseEstimation.get_model(num_joint, embed_dim, depth, pretrained=cfg.MODEL.posenet_pretrained)
        self.pose_mesh_coevo = Multimodel.get_model(num_joint, embed_dim*2)

    def forward(self, pose2d, img_feat, is_train=True):
        pose3d = self.pose_lifter(pose2d, img_feat)
        pose3d = pose3d.reshape(-1, cfg.DATASET.seqlen, self.num_joint, 3)
        
        pred_mesh, pred_pose, pred_shape, smploutput = self.pose_mesh_coevo(pose3d / 1000, img_feat, pose2d, is_train=is_train)
        
        return pred_mesh, pred_pose, pred_shape, smploutput


def get_model(num_joint, embed_dim, depth):
    model = ARTS(num_joint, embed_dim, depth)

    return model