import os, sys
sys.path.append('./smplpytorch')
sys.path.append('./data')
sys.path.append('./STA-GCN')

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
from models.smpl_mps import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from core.base import rigid_align, rot6d_to_rotmat, rotation_matrix_to_angle_axis

import math
import warnings
warnings.filterwarnings("ignore")



class KDTrainer:
    def __init__(self, args):
        print("===> Preparations for KD Training...")
        # 1. Load Data
        dataset_names = cfg.DATASET.train_list
        self.train_dataset_list, self.train_loader = get_dataloader(args, dataset_names, is_train=True)
        self.main_dataset = self.train_dataset_list[0]
        
        test_dataset_names = cfg.DATASET.test_list
        self.test_dataset_list, self.test_loader = get_dataloader(args, test_dataset_names, is_train=False)
        self.test_dataset = self.test_dataset_list[0]
        
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
        if cfg.MODEL.posenet_pretrained and hasattr(self.student_model, 'pose_lifter'):
            for param in self.student_model.pose_lifter.parameters():
                param.requires_grad = False
            frozen = sum(p.numel() for p in self.student_model.pose_lifter.parameters())
            trainable = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
            print(f'  [Freeze] PoseLifter frozen: {frozen:,} params')
            print(f'  [Train]  Trainable params: {trainable:,}')

        # 3. Criterion & Optimizer
        self.loss = get_loss(faces=self.main_dataset.mesh_model.face)
        self.kd_loss_fn = nn.MSELoss()
        
        self.loss_history = []
        self.error_history = {'surface': [], 'joint': []}
        
        # Loss Weights
        self.normal_weight = cfg.MODEL.normal_loss_weight
        self.edge_weight = cfg.MODEL.edge_loss_weight
        self.joint_weight = cfg.MODEL.joint_loss_weight
        self.shape_weight = cfg.MODEL.shape_loss_weight
        self.pose_weight = cfg.MODEL.pose_loss_weight
        self.edge_add_epoch = cfg.TRAIN.edge_loss_start
        self.laplacian_weight = 100.0
        self.relation_weight = getattr(cfg.TRAIN, 'relation_weight', 1.0)
        print(f"Relation Weight: {self.relation_weight}")

        # Fixed alpha = 0.5 (50% KD, 50% Task)
        self.alpha = 0.5

        self.optimizer = get_optimizer(model=self.student_model)
        self.lr_scheduler = get_scheduler(optimizer=self.optimizer)

        # 4. Logger
        self.loss_history = []
        self.print_freq = cfg.TRAIN.print_freq

    def train(self, epoch):
        self.student_model.train()
        self.teacher_model.eval()
        
        loader = tqdm(self.train_loader)
        loss_epoch = 0.0

        for i, (inputs, targets, meta) in enumerate(loader):
            input_pose, img_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
            gt_lift3dpose, gt_reg3dpose, gt_mesh = targets['lift_pose3d'].cuda(), targets['reg_pose3d'].cuda(), targets['mesh'].cuda()
            gt_smplpose, gt_smplshape = targets['smpl_pose'].cuda(), targets['smpl_shape'].cuda()
            val_lift3dpose, val_reg3dpose, val_mesh = meta['lift_pose3d_valid'].cuda(), meta['reg_pose3d_valid'].cuda(), meta['mesh_valid'].cuda()
            
            # --- Teacher forward (frozen) ---
            with torch.no_grad():
                t_fused_feats, t_mesh, t_smpl, t_skel_feats = self.teacher_model(gt_lift3dpose, img_feat)

            # --- Student forward ---
            s_pose3d, s_fused_feats, s_mesh, s_smpl, s_skel_feats = self.student_model(input_pose, img_feat, is_train=True)

            # Root-centering
            pred_joint = torch.matmul(self.J_regressor[None, :, :], s_mesh)
            gt_joint = torch.matmul(self.J_regressor[None, :, :], gt_mesh)
            s_mesh = s_mesh - pred_joint[:, :1, :]
            gt_mesh = gt_mesh - gt_joint[:, :1, :]
            
            # --- Compute losses ---
            # KD losses
            loss_feat_kd = self.kd_loss_fn(s_fused_feats, t_fused_feats.detach())
            s_norm = nn.functional.normalize(s_fused_feats, dim=-1)
            t_norm = nn.functional.normalize(t_fused_feats, dim=-1)
            s_sim = s_norm @ s_norm.transpose(-2, -1)  # cosine similarity ∈ [-1, 1]
            t_sim = t_norm @ t_norm.transpose(-2, -1)
            loss_relation = self.relation_weight * self.kd_loss_fn(s_sim, t_sim.detach())
            loss_skel_kd = self.kd_loss_fn(s_skel_feats, t_skel_feats.detach())
            loss_kd = loss_feat_kd + loss_relation + loss_skel_kd

            # Task losses
            pred_pose = torch.matmul(self.J_regressor[None, :, :], s_mesh * 1000)
            loss_vertex = self.loss[0](s_mesh, gt_mesh, val_mesh)
            loss_joint = self.joint_weight * self.loss[3](pred_pose, gt_reg3dpose, val_reg3dpose)
            mid = cfg.DATASET.seqlen // 2
            theta_mid = s_smpl[-1]['theta'][:, mid, :]
            smpl_pose_loss, smpl_shape_loss = self.loss[6](theta_mid[:, 3:75],
                                                           theta_mid[:, 75:],
                                                           gt_smplpose, gt_smplshape, mask_3d=None)
            loss_smpl = self.shape_weight * smpl_shape_loss + self.pose_weight * smpl_pose_loss
            loss_lift = self.joint_weight * self.loss[5](s_pose3d, gt_lift3dpose, val_lift3dpose)
            loss_task = loss_vertex + loss_joint + loss_smpl  # Không cộng loss_lift vì PoseLifter frozen

            # Fixed weighted sum: α*KD + (1-α)*Task with α=0.5
            loss = self.alpha * loss_kd + (1.0 - self.alpha) * loss_task

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_epoch += loss.item()
            if i % self.print_freq == 0:
                loader.set_description(f'Ep{epoch} | SkelKD:{loss_skel_kd.item():.3f} FeatKD:{loss_feat_kd.item():.3f} RelKD:{loss_relation.item():.3f} '
                                       f'Task:{loss_task.item():.3f} | α={self.alpha:.1f}')

        self.loss_history.append(loss_epoch / len(self.train_loader))
        print(f'Epoch {epoch} Loss: {self.loss_history[-1]:.4f}')

    def test(self, epoch):
        self.student_model.eval()
        surface_error = 0.0
        joint_error = 0.0
        avg_skel_sim = 0.0
        avg_feat_sim = 0.0
        
        eval_prefix = f'Epoch{epoch} Test '
        loader = tqdm(self.test_loader[0]) # test_loader returned by get_dataloader is a list
        result = []
        with torch.no_grad():
            for i, (inputs, targets, meta) in enumerate(loader):
                input_pose, input_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
                gt_pose3d, gt_mesh = targets['reg_pose3d'].cuda(), targets['mesh'].cuda()
                gt_lift3dpose = targets['lift_pose3d'].cuda()
                
                # forward Teacher
                t_fused_feats, t_mesh, t_smpl, t_skel_feats = self.teacher_model(gt_lift3dpose, input_feat)
                
                # forward Student
                s_pose3d, s_fused_feats, pred_mesh, smploutput, s_skel_feats = self.student_model(input_pose, input_feat, is_train=False)
                
                skel_sim = nn.functional.cosine_similarity(s_skel_feats, t_skel_feats, dim=-1).mean().item()
                feat_sim = nn.functional.cosine_similarity(s_fused_feats, t_fused_feats, dim=-1).mean().item()
                avg_skel_sim += skel_sim
                avg_feat_sim += feat_sim
                
                pred_mesh, gt_mesh = pred_mesh * 1000, gt_mesh * 1000
                pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
                
                j_error, s_error = self.test_dataset.compute_both_err(pred_mesh, gt_mesh, pred_pose, gt_pose3d)
                
                joint_error += j_error
                surface_error += s_error
                
                # Final Evaluation
                if epoch == cfg.TRAIN.end_epoch:
                    pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy()
                    pred_pose, gt_pose3d = pred_pose.detach().cpu().numpy(), gt_pose3d.detach().cpu().numpy()
                    for j in range(len(input_pose)):
                        out = {}
                        out['mesh_coord'], out['mesh_coord_target'] = pred_mesh[j], target_mesh[j]
                        out['joint_coord'], out['joint_coord_target'] = pred_pose[j], gt_pose3d[j]
                        result.append(out)
                        
            self.surface_error = surface_error / len(self.test_loader[0])
            self.joint_error = joint_error / len(self.test_loader[0])
            avg_skel_sim = avg_skel_sim / len(self.test_loader[0])
            avg_feat_sim = avg_feat_sim / len(self.test_loader[0])
            
            print(f'{eval_prefix}MPVPE: {self.surface_error:.2f}, MPJPE: {self.joint_error:.2f} | Skel Sim: {avg_skel_sim:.3f}, Feat Sim: {avg_feat_sim:.3f}')
            
            if epoch == cfg.TRAIN.end_epoch:
                self.test_dataset.evaluate(result)

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
    
    for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):
        trainer.train(epoch)
        trainer.lr_scheduler.step()
        
        trainer.test(epoch)
        
        if epoch > 1:
            is_best = trainer.joint_error < min(trainer.error_history['joint']) or trainer.surface_error < min(trainer.error_history['surface'])
        else:
            is_best = None
            
        trainer.error_history['surface'].append(trainer.surface_error)
        trainer.error_history['joint'].append(trainer.joint_error)
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': trainer.student_model.state_dict(),
            'optim_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
            'train_log': trainer.loss_history,
            'test_log': trainer.error_history
        }, epoch, is_best=is_best)

    print('===> KD Training finished! Loading BEST checkpoint for final evaluation...')
    best_path = os.path.join(cfg.checkpoint_dir, 'best.pth.tar')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        trainer.student_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"===> Loaded BEST checkpoint from Epoch {checkpoint['epoch']} for final evaluation.")
        trainer.test(cfg.TRAIN.end_epoch)
    else:
        print("===> Best checkpoint not found!")