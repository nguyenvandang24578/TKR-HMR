import os, sys
sys.path.append('./smplpytorch')
sys.path.append('./data')
sys.path.append('./STA-GCN')

sys.path.append('./lib')
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from core.config import cfg, update_config
import random
import numpy as np
from core.base import get_optimizer, get_scheduler, get_dataloader
import models.TKR_HMR as TKR
from models.TeacherFusion import get_model as get_teacher_model
from funcs_utils import save_checkpoint

import math
import warnings
warnings.filterwarnings("ignore")



class KDTrainer:
    def __init__(self, args):
        print("===> Preparations for KD Training (Pure Feature Distillation)...")
        # 1. Load Data
        dataset_names = cfg.DATASET.train_list
        self.train_dataset_list, self.train_loader = get_dataloader(args, dataset_names, is_train=True)
        self.main_dataset = self.train_dataset_list[0]
        
        test_dataset_names = cfg.DATASET.test_list
        self.test_dataset_list, self.test_loader = get_dataloader(args, test_dataset_names, is_train=False)
        self.test_dataset = self.test_dataset_list[0]
        
        self.num_joint = self.main_dataset.joint_num

        # 2. Build Models
        print("==> Preparing Teacher MODEL (Frozen)...")
        self.teacher_model = get_teacher_model(embed_dim=cfg.MODEL.hpe_dim * 2).cuda()
        if hasattr(cfg.MODEL, 'teacher_path') and cfg.MODEL.teacher_path and os.path.exists(cfg.MODEL.teacher_path):
            checkpoint = torch.load(cfg.MODEL.teacher_path, map_location='cpu')
            self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded Teacher from {cfg.MODEL.teacher_path}")
        else:
            print(f"WARNING: Teacher checkpoint not found at {getattr(cfg.MODEL, 'teacher_path', 'Not Set')}. Training with untrained Teacher!")
        self.teacher_model.eval() # Freeze Teacher

        print("==> Preparing Student MODEL (Pure Feature Extractor)...")
        self.student_model = TKR.get_student_model(self.num_joint, cfg.MODEL.hpe_dim, cfg.MODEL.hpe_dep).cuda()
        if cfg.MODEL.posenet_pretrained and hasattr(self.student_model, 'pose_lifter'):
            for param in self.student_model.pose_lifter.parameters():
                param.requires_grad = False
            frozen = sum(p.numel() for p in self.student_model.pose_lifter.parameters())
            trainable = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
            print(f'  [Freeze] PoseLifter frozen: {frozen:,} params')
            print(f'  [Train]  Trainable params: {trainable:,}')

        # 3. KD Losses
        self.kd_loss_fn = nn.MSELoss()
        
        # Learned projector for skeleton features (bridges domain gap)
        embed_dim = cfg.MODEL.hpe_dim * 2  # 512
        self.skel_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        ).cuda()
        
        # KD weights
        self.skel_kd_weight = getattr(cfg.TRAIN, 'skel_kd_weight', 0.3)
        self.relation_weight = getattr(cfg.TRAIN, 'relation_weight', 1.0)
        
        # Noise injection (curriculum) for Teacher input
        self.noise_std_max = getattr(cfg.TRAIN, 'noise_std_max', 0.05)  # 50mm
        self.noise_std_min = getattr(cfg.TRAIN, 'noise_std_min', 0.005)  # 5mm
        
        # Tracking
        self.loss_history = []
        self.error_history = {'feat_sim': [], 'skel_sim': []}
        
        # 4. Optimizer (includes projector params)
        all_params = list(self.student_model.parameters()) + list(self.skel_projector.parameters())
        trainable_params = [p for p in all_params if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=cfg.TRAIN.lr)
        self.lr_scheduler = get_scheduler(optimizer=self.optimizer)

        # 5. Logger
        self.print_freq = cfg.TRAIN.print_freq
        
        print(f"  Noise curriculum: {self.noise_std_max*1000:.0f}mm → {self.noise_std_min*1000:.0f}mm")
        print(f"  Skel KD weight: {self.skel_kd_weight}, Relation weight: {self.relation_weight}")

    def train(self, epoch):
        self.student_model.train()
        self.teacher_model.eval()
        
        # Progressive noise for Teacher input (curriculum: high→low)
        progress = (epoch - cfg.TRAIN.begin_epoch) / max(cfg.TRAIN.end_epoch - cfg.TRAIN.begin_epoch, 1)
        noise_std = self.noise_std_max * (1.0 - progress) + self.noise_std_min * progress
        
        loader = tqdm(self.train_loader)
        loss_epoch = 0.0

        for i, (inputs, targets, meta) in enumerate(loader):
            input_pose, img_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
            gt_lift3dpose = targets['lift_pose3d'].cuda()
            
            # --- Teacher forward (frozen + noise injection) ---
            with torch.no_grad():
                noisy_gt = gt_lift3dpose + torch.randn_like(gt_lift3dpose) * noise_std
                t_fused_feats, _, _, t_skel_feats = self.teacher_model(noisy_gt, img_feat)

            # --- Student forward ---
            s_pose3d, s_fused_feats, s_skel_feats = self.student_model(input_pose, img_feat, is_train=True)

            # --- KD losses ---
            # 1. Feature KD: align fused features (most important)
            loss_feat_kd = self.kd_loss_fn(s_fused_feats, t_fused_feats.detach())
            
            # 2. Skeleton KD via learned projector (bridges domain gap)
            loss_skel_kd = self.skel_kd_weight * self.kd_loss_fn(
                self.skel_projector(s_skel_feats), t_skel_feats.detach()
            )
            
            # 3. Relational KD: preserve inter-sample similarity structure
            s_norm = F.normalize(s_fused_feats, dim=-1)
            t_norm = F.normalize(t_fused_feats, dim=-1)
            s_sim = s_norm @ s_norm.transpose(-2, -1)
            t_sim = t_norm @ t_norm.transpose(-2, -1)
            loss_relation = self.relation_weight * self.kd_loss_fn(s_sim, t_sim.detach())
            
            # Total loss (pure KD, no task loss)
            loss = loss_feat_kd + loss_skel_kd + loss_relation

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_epoch += loss.item()
            if i % self.print_freq == 0:
                loader.set_description(
                    f'Ep{epoch} | FeatKD:{loss_feat_kd.item():.4f} '
                    f'SkelKD:{loss_skel_kd.item():.4f} '
                    f'RelKD:{loss_relation.item():.4f} '
                    f'noise:{noise_std*1000:.1f}mm'
                )

        self.loss_history.append(loss_epoch / len(self.train_loader))
        print(f'Epoch {epoch} Loss: {self.loss_history[-1]:.4f}')

    def test(self, epoch):
        self.student_model.eval()
        self.teacher_model.eval()
        avg_skel_sim = 0.0
        avg_feat_sim = 0.0
        avg_feat_mse = 0.0
        
        eval_prefix = f'Epoch{epoch} Test '
        loader = tqdm(self.test_loader[0])
        with torch.no_grad():
            for i, (inputs, targets, meta) in enumerate(loader):
                input_pose, input_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
                gt_lift3dpose = targets['lift_pose3d'].cuda()
                
                # Teacher forward (no noise during test)
                t_fused_feats, _, _, t_skel_feats = self.teacher_model(gt_lift3dpose, input_feat)
                
                # Student forward
                s_pose3d, s_fused_feats, s_skel_feats = self.student_model(input_pose, input_feat, is_train=False)
                
                # Cosine similarity metrics
                skel_sim = F.cosine_similarity(
                    self.skel_projector(s_skel_feats), t_skel_feats, dim=-1
                ).mean().item()
                feat_sim = F.cosine_similarity(s_fused_feats, t_fused_feats, dim=-1).mean().item()
                feat_mse = F.mse_loss(s_fused_feats, t_fused_feats).item()
                
                avg_skel_sim += skel_sim
                avg_feat_sim += feat_sim
                avg_feat_mse += feat_mse
                
            n_batches = len(self.test_loader[0])
            avg_skel_sim /= n_batches
            avg_feat_sim /= n_batches
            avg_feat_mse /= n_batches
            
            self.feat_sim = avg_feat_sim  # Used for best model selection
            
            print(f'{eval_prefix}Feat Sim: {avg_feat_sim:.4f} | Skel Sim: {avg_skel_sim:.4f} | Feat MSE: {avg_feat_mse:.6f}')
            
            self.error_history['feat_sim'].append(avg_feat_sim)
            self.error_history['skel_sim'].append(avg_skel_sim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Distillation for HMR (Pure Feature)')
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
    print("===> Start KD training (Pure Feature Distillation)...")
    
    for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):
        trainer.train(epoch)
        trainer.lr_scheduler.step()
        
        trainer.test(epoch)
        
        # Best model = highest feat_sim (cosine similarity)
        if epoch > cfg.TRAIN.begin_epoch:
            is_best = trainer.feat_sim > max(trainer.error_history['feat_sim'][:-1])
        else:
            is_best = True  # First epoch is always best so far
            
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': trainer.student_model.state_dict(),
            'projector_state_dict': trainer.skel_projector.state_dict(),
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
        trainer.skel_projector.load_state_dict(checkpoint['projector_state_dict'])
        print(f"===> Loaded BEST checkpoint from Epoch {checkpoint['epoch']} for final evaluation.")
        trainer.test(cfg.TRAIN.end_epoch)
    else:
        print("===> Best checkpoint not found!")