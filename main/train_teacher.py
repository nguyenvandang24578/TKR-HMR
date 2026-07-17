import os, sys
sys.path.append('./lib')
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.config import cfg, update_config
import __init_path
import random
import numpy as np
from core.loss import get_loss
from core.base import get_optimizer, get_scheduler, get_dataloader
from models.TeacherFusion import get_model
from funcs_utils import save_checkpoint

import warnings
warnings.filterwarnings("ignore")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TeacherTrainer:
    def __init__(self, args, load_dir=''):
        print("===> Preparations for Teacher Training...")
        # 1. Load Data
        dataset_names = cfg.DATASET.train_list
        self.train_dataset_list, self.train_loader = get_dataloader(args, dataset_names, is_train=True)
        self.main_dataset = self.train_dataset_list[0]
        self.num_joint = 19 # COCO joints in PW3D dataset

        # 2. Build Model
        print(f"==> Preparing Teacher MODEL...")
        self.model = get_model(embed_dim=cfg.MODEL.hpe_dim)
        print('# of model parameters: {}'.format(count_parameters(self.model)))
        self.model = self.model.cuda()

        # 3. Criterion & Optimizer
        self.criterion = get_loss(faces=self.main_dataset.mesh_model.face)
        self.optimizer = get_optimizer(model=self.model)
        self.lr_scheduler = get_scheduler(optimizer=self.optimizer)

        # 4. Logger
        self.loss_history = []
        self.print_freq = cfg.TRAIN.print_freq

    def train(self, epoch):
        self.model.train()
        loader = tqdm(self.train_loader)
        loss_epoch = 0.0

        for i, (inputs, targets, meta) in enumerate(loader):
            img_feat = inputs['img_feature'].cuda()
            # Lấy Ground Truth 3D joints (chuẩn COCO 19 khớp)
            gt_skeleton = targets['lift_pose3d'].cuda() 
            
            # Lấy Ground Truth Mesh
            gt_mesh = targets['mesh'].cuda() * 1000.0 # meter -> mm

            # Forward
            fused_feats, smpl_output = self.model(gt_skeleton, img_feat)

            # Extract predicted mesh and joints
            pred_mesh = smpl_output[-1]['verts'] * 1000.0
            # pred_joints = smpl_output[-1]['joints'] * 1000.0

            # Compute Loss
            # For simplicity, we just use mesh loss and SMPL parameter loss if available
            loss_dict = self.criterion(
                pred_mesh=pred_mesh,
                gt_mesh=gt_mesh,
                pred_pose=None, # Teacher focuses on mesh/SMPL fitting
                gt_pose=None
            )
            
            loss = loss_dict['mesh'] # Basic mesh loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_epoch += loss.item()
            if i % self.print_freq == 0:
                loader.set_description(f'Epoch {epoch} | Loss: {loss.item():.4f}')

        self.loss_history.append(loss_epoch / len(self.train_loader))
        print(f'Epoch {epoch} Loss: {self.loss_history[-1]:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Teacher Fusion Model')
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
    
    trainer = TeacherTrainer(args)
    print("===> Start training Teacher...")
    
    for epoch in range(1, 10): # Quick train test
        trainer.train(epoch)
        trainer.lr_scheduler.step()
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optim_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
            'train_log': trainer.loss_history,
            'test_log': []
        }, epoch, is_best=True, filename='checkpoint_teacher.pth.tar')

    print('Teacher Training Finished!')
