import os, sys
sys.path.append('./lib')
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from core.config import cfg, update_config
import random
import numpy as np
from core.loss import get_loss
from core.base import get_optimizer, get_scheduler, get_dataloader
import models.TKR_HMR as TKR
from models.TeacherFusion import get_model as get_teacher_model
from funcs_utils import save_checkpoint
from models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from core.base import rigid_align, rot6d_to_rotmat, rotation_matrix_to_angle_axis

import warnings
warnings.filterwarnings("ignore")

class KDTrainer:
    def __init__(self, args):
        print("===> Preparations for KD Training...")
        # 1. Load Data
        dataset_names = cfg.DATASET.train_list
        self.train_dataset_list, self.train_loader = get_dataloader(args, dataset_names, is_train=True)
        self.main_dataset = self.train_dataset_list[0]
        self.num_joint = self.main_dataset.joint_num
        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        # 2. Build Models
        print("==> Preparing Teacher MODEL (Frozen)...")
        self.teacher_model = get_teacher_model(embed_dim=cfg.MODEL.hpe_dim).cuda()
        if hasattr(cfg.MODEL, 'teacher_path') and cfg.MODEL.teacher_path and os.path.exists(cfg.MODEL.teacher_path):
            checkpoint = torch.load(cfg.MODEL.teacher_path, map_location='cpu')
            self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded Teacher from {cfg.MODEL.teacher_path}")
        else:
            print(f"WARNING: Teacher checkpoint not found at {getattr(cfg.MODEL, 'teacher_path', 'Not Set')}. Training with untrained Teacher!")
        self.teacher_model.eval() # Freeze Teacher

        print("==> Preparing Student MODEL (TKR)...")
        self.student_model = TKR.get_student_model(self.num_joint, cfg.MODEL.hpe_dim, cfg.MODEL.hpe_dep).cuda()
        if cfg.MODEL.posenet_pretrained:
            print("Student's PoseLifter loaded pretrained weights.")

        # 3. Criterion & Optimizer
        self.loss = get_loss(faces=self.main_dataset.mesh_model.face)
        self.kd_loss_fn = nn.MSELoss()
        
        # Loss Weights
        self.normal_weight = cfg.MODEL.normal_loss_weight
        self.edge_weight = cfg.MODEL.edge_loss_weight
        self.joint_weight = cfg.MODEL.joint_loss_weight
        self.shape_weight = cfg.MODEL.shape_loss_weight
        self.pose_weight = cfg.MODEL.pose_loss_weight
        self.edge_add_epoch = cfg.TRAIN.edge_loss_start
        self.laplacian_weight = 100.0
        self.alpha = getattr(cfg.TRAIN, 'alpha', 0.5) # Default 0.5 KD vs Task
        print(f"KD Alpha: {self.alpha}")

        self.optimizer = get_optimizer(model=self.student_model)
        self.lr_scheduler = get_scheduler(optimizer=self.optimizer)

        # 4. Logger
        self.loss_history = []
        self.print_freq = cfg.TRAIN.print_freq

    def train(self, epoch):
        self.student_model.train()
        self.teacher_model.eval() # Always eval
        
        loader = tqdm(self.train_loader)
        loss_epoch = 0.0

        for i, (inputs, targets, meta) in enumerate(loader):
            input_pose, img_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
            gt_lift3dpose, gt_reg3dpose, gt_mesh = targets['lift_pose3d'].cuda(), targets['reg_pose3d'].cuda(), targets['mesh'].cuda() # Sequence 16 frames (lift_pose3d)
            gt_smplpose, gt_smplshape = targets['smpl_pose'].cuda(), targets['smpl_shape'].cuda()
            val_lift3dpose, val_reg3dpose, val_mesh = meta['lift_pose3d_valid'].cuda(), meta['reg_pose3d_valid'].cuda(), meta['mesh_valid'].cuda()
            
            # --- 1. Pass qua Teacher (KHÔNG tính gradient) ---
            with torch.no_grad():
                t_fused_feats, t_mesh, t_smpl = self.teacher_model(gt_lift3dpose, img_feat)

            # --- B. STUDENT FORWARD ---
            s_pose3d, s_fused_feats, s_mesh, s_smpl = self.student_model(input_pose, img_feat, is_train=True)

            # Root-centering (cùng đơn vị trước, nhân 1000 sau)
            pred_joint = torch.matmul(self.J_regressor[None, :, :], s_mesh)
            gt_joint = torch.matmul(self.J_regressor[None, :, :], gt_mesh)
            
            # Trừ root joint để 2 mesh khớp tuyệt đối tại gốc (0,0,0)
            s_mesh = s_mesh - pred_joint[:, :1, :]
            gt_mesh = gt_mesh - gt_joint[:, :1, :]
            
            # --- C. CALCULATE LOSS ---
            # 1. KD Loss (Feature Distillation)
            loss_kd = self.kd_loss_fn(s_fused_feats, t_fused_feats.detach())

            # 2. Task Loss (nhân 1000 sau khi đã root-center)
            pred_pose = torch.matmul(self.J_regressor[None, :, :], s_mesh * 1000)
            
            # Mesh Vertex Loss
            loss_vertex = self.loss[0](s_mesh, gt_mesh, val_mesh)
            
            # 3D Joint Loss (Regressed from mesh)
            loss_joint = self.joint_weight * self.loss[3](pred_pose, gt_reg3dpose, val_reg3dpose)
            
            # SMPL Parameter Loss (Pose & Shape)
            # Lấy mid-frame trước (theta shape: [B, T, 85]), rồi mới slice features
            mid = cfg.DATASET.seqlen // 2
            theta_mid = s_smpl[-1]['theta'][:, mid, :]  # [B, 85]
            smpl_pose_loss, smpl_shape_loss = self.loss[6](theta_mid[:, 3:75],
                                                           theta_mid[:, 75:],
                                                           gt_smplpose, gt_smplshape, mask_3d=None)
            loss_smpl = self.shape_weight * smpl_shape_loss + self.pose_weight * smpl_pose_loss
            
            # Loss cho 3D Pose lifted từ 2D
            loss_lift = self.joint_weight * self.loss[5](s_pose3d, gt_lift3dpose, val_lift3dpose)
            
            # Tổng Task Loss (Đã bỏ Normal, Laplacian, Edge vì SMPL luôn giữ topology chuẩn)
            loss_task = loss_vertex + loss_joint + loss_smpl + loss_lift

            # Total Loss
            loss = self.alpha * loss_kd + (1.0 - self.alpha) * loss_task

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_epoch += loss.item()
            if i % self.print_freq == 0:
                loader.set_description(f'Ep {epoch} | L_KD: {loss_kd.item():.4f} | L_Task: {loss_task.item():.4f} | Tot: {loss.item():.4f}')

        self.loss_history.append(loss_epoch / len(self.train_loader))
        print(f'Epoch {epoch} Loss: {self.loss_history[-1]:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Distillation for HMR')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--cfg', type=str, help='experiment configure file name')
    
    args = parser.parse_args()
    if args.cfg:
        update_config(args.cfg)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    trainer = KDTrainer(args)
    print("===> Start KD training...")
    
    for epoch in range(1, 30):
        trainer.train(epoch)
        trainer.lr_scheduler.step()
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': trainer.student_model.state_dict(),
            'optim_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
            'train_log': trainer.loss_history,
            'test_log': []
        }, epoch, is_best=True, filename='checkpoint_student_kd.pth.tar')

    print('KD Training Finished!')