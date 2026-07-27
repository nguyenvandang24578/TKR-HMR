import numpy as np
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
import os
import Human36M.dataset, COCO.dataset, PW3D.dataset, MPII3D.dataset, MPII.dataset

import models
import models.TKR_HMR as TKR
from core.config import cfg
from core.loss import get_loss
from multiple_datasets import MultipleDatasets
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters, lr_check
import time
from coord_utils import rigid_align

from geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis
import math
from smpl import SMPL
mesh_model = SMPL()


def get_dataloader(args, dataset_names, is_train):
    dataset_split = 'TRAIN' if is_train else 'TEST'
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    dataset_list, dataloader_list = [], []

    print(f"==> Preparing {dataset_split} Dataloader...")
    for name in dataset_names:
        dataset = eval(f'{name}.dataset')(dataset_split.lower(), args=args)
        print("# of {} {} data: {}".format(dataset_split, name, len(dataset)))
        dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle,
                                num_workers=cfg.DATASET.workers,
                                pin_memory=False)
        dataset_list.append(dataset)
        dataloader_list.append(dataloader)

    if not is_train:
        return dataset_list, dataloader_list
    else:
        trainset_loader = MultipleDatasets(dataset_list, make_same_len=True)
        batch_generator = DataLoader(dataset=trainset_loader,
                          batch_size=batch_per_dataset * len(dataset_names),
                          shuffle=cfg[dataset_split].shuffle,
                          num_workers=cfg.DATASET.workers, pin_memory=False)
        return dataset_list, batch_generator


def prepare_network(args, load_dir='', is_train=True):
    dataset_names = cfg.DATASET.train_list if is_train else cfg.DATASET.test_list
    dataset_list, dataloader = get_dataloader(args, dataset_names, is_train)
    model, criterion, optimizer, lr_scheduler = None, None, None, None
    loss_history, test_error_history = [], {'surface': [], 'joint': []}

    main_dataset = dataset_list[0]
    J_regressor = eval(f'torch.Tensor(main_dataset.joint_regressor_{cfg.DATASET.input_joint_set})')
    if is_train or load_dir:
        print(f"==> Preparing Student MODEL...")
        model = TKR.get_student_model(
            num_joint=main_dataset.joint_num,
            embed_dim=cfg.MODEL.hpe_dim,
            depth=cfg.MODEL.hpe_dep
        )
        print('# of model parameters: {}'.format(count_parameters(model)))

        # Freeze PoseLifter nếu đã load pretrained weights
        if cfg.MODEL.posenet_pretrained and hasattr(model, 'pose_lifter'):
            for param in model.pose_lifter.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in model.pose_lifter.parameters())
            print(f'  [Freeze] PoseLifter frozen: {frozen:,} params')
            print(f'  [Train]  Trainable params: {trainable:,}')

    if is_train:
        criterion = get_loss(faces=main_dataset.mesh_model.face)
        # Chỉ optimize các params có requires_grad=True (bỏ qua PoseLifter đã freeze)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=cfg.TRAIN.lr)
        lr_scheduler = get_scheduler(optimizer=optimizer)

    if load_dir and (not is_train or args.resume_training):
        print('==> Loading checkpoint')
        checkpoint = load_checkpoint(load_dir=load_dir, pick_best=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if is_train:
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            curr_lr = 0.0

            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']

            lr_state = checkpoint['scheduler_state_dict']
            # update lr_scheduler
            lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
            lr_scheduler.load_state_dict(lr_state)

            loss_history = checkpoint['train_log']
            test_error_history = checkpoint['test_log']
            cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
            print('===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}'
                  .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return dataloader, dataset_list, model, criterion, optimizer, lr_scheduler, loss_history, test_error_history


class Trainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history\
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.main_dataset = self.dataset_list[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        self.model = self.model.cuda()
        self.normal_weight = cfg.MODEL.normal_loss_weight
        self.edge_weight = cfg.MODEL.edge_loss_weight
        self.joint_weight = cfg.MODEL.joint_loss_weight
        self.edge_add_epoch = cfg.TRAIN.edge_loss_start
        self.shape_weight = cfg.MODEL.shape_loss_weight
        self.pose_weight = cfg.MODEL.pose_loss_weight
        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='Student/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)
        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            # convert to cuda
            input_pose, input_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
            gt_lift3dpose, gt_reg3dpose, gt_mesh = targets['lift_pose3d'].cuda(), targets['reg_pose3d'].cuda(), targets['mesh'].cuda()
            gt_smplpose, gt_smplshape = targets['smpl_pose'].cuda(), targets['smpl_shape'].cuda()
            val_lift3dpose, val_reg3dpose, val_mesh = meta['lift_pose3d_valid'].cuda(), meta['reg_pose3d_valid'].cuda(), meta['mesh_valid'].cuda()

            # forward Student: returns (pose3d, fused_feats, pred_mesh, smploutput)
            s_pose3d, s_fused_feats, pred_mesh, smploutput = self.model(input_pose, input_feat, is_train=True)

            pred_joint = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
            gt_joint = torch.matmul(self.J_regressor[None, :, :], gt_mesh)

            # Trừ root joint để 2 mesh khớp tuyệt đối tại gốc (0,0,0)
            pred_mesh = pred_mesh - pred_joint[:, :1, :]
            gt_mesh = gt_mesh - gt_joint[:, :1, :]

            # Tính pred_pose để áp dụng hàm loss MPJPE
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh * 1000.0)

            # Khởi tạo loss: vertex, normal, joint
            loss1 = self.loss[0](pred_mesh, gt_mesh, val_mesh)
            loss2 = self.normal_weight * self.loss[1](pred_mesh, gt_mesh)
            loss4 = self.joint_weight * self.loss[3](pred_pose, gt_reg3dpose, val_reg3dpose)

            # Loss cho 3D Pose lifted từ 2D (đặc trưng của Student có PoseLifter)
            loss_lift = self.joint_weight * self.loss[5](s_pose3d, gt_lift3dpose, val_lift3dpose)

            # Tính SMPL loss (pose + shape)
            # Student smploutput[-1]['theta'] có shape [B, T, 85], lấy mid-frame
            mid = cfg.DATASET.seqlen // 2
            theta_mid = smploutput[-1]['theta'][:, mid, :]  # [B, 85]
            smpl_pose_loss, smpl_shape_loss = self.loss[6](theta_mid[:, 3:75], theta_mid[:, 75:], gt_smplpose, gt_smplshape, mask_3d=None)
            smpl_loss = self.shape_weight * smpl_shape_loss + self.pose_weight * smpl_pose_loss

            # Tổng loss
            loss = loss1 + loss2 + loss4 + loss_lift + smpl_loss

            if epoch > self.edge_add_epoch:
                loss3 = self.edge_weight * self.loss[2](pred_mesh, gt_mesh)
                loss += loss3

            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # log
            running_loss += float(loss.detach().item())
            if cfg.TRAIN.wandb:
                wandb_loss1, wandb_loss2, wandb_loss4 = loss1.detach(), loss2.detach(), loss4.detach()
                wandb_loss3 = loss3.detach() if epoch > self.edge_add_epoch else 0
                wandb_loss_lift = loss_lift.detach()
                wandb.log(
                    {
                        'train_loss/vertex_loss': wandb_loss1,
                        'train_loss/normal_loss': wandb_loss2,
                        'train_loss/edge_loss': wandb_loss3,
                        'train_loss/joint_loss': wandb_loss4,
                        'train_loss/lift_loss': wandb_loss_lift,
                    }
                )

            if i % self.print_freq == 0:
                loss1, loss2, loss4 = loss1.detach(), loss2.detach(), loss4.detach()
                loss3 = loss3.detach() if epoch > self.edge_add_epoch else 0
                loss_lift_val = loss_lift.detach()
                total_loss = loss.detach()

                batch_generator.set_description(f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                                                f'mesh: {loss1:.3f} '
                                                f'norm: {loss2:.3f} '
                                                f'edge: {loss3:.3f} '
                                                f'joint: {loss4:.3f} '
                                                f'lift: {loss_lift_val:.3f} '
                                                f'total: {total_loss:.3f} ')

        self.loss_history.append(running_loss / len(batch_generator))
        for i, pg in enumerate(self.optimizer.param_groups):
            group_name = ['SPIN', 'Fresh'][i] if i < 2 else f'Group{i}'
            grads = [p.grad.norm().item() for p in pg['params'] if p.grad is not None]
            if grads:
                avg_grad = sum(grads) / len(grads)
                print(f"  [{group_name}] grad_norm={avg_grad:.4f}")
        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')


class Tester:
    def __init__(self, args, load_dir=''):
        self.val_loader, self.val_dataset, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)

        self.val_loader, self.val_dataset = self.val_loader[0], self.val_dataset[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.val_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        if self.model:
            self.model = self.model.cuda()

        # initialize error value
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        surface_error = 0.0
        joint_error = 0.0

        result = []
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (inputs, targets, meta) in enumerate(loader):
                input_pose, input_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
                gt_pose3d, gt_mesh = targets['reg_pose3d'].cuda(), targets['mesh'].cuda()

                # forward Student (is_train=False)
                s_pose3d, s_fused_feats, pred_mesh, smploutput = self.model(input_pose, input_feat, is_train=False)

                pred_mesh, gt_mesh = pred_mesh * 1000, gt_mesh * 1000

                pred_joint = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
                gt_joint = torch.matmul(self.J_regressor[None, :, :], gt_mesh)

                # Trừ root joint để 2 mesh khớp tuyệt đối tại gốc (0,0,0)
                pred_mesh = pred_mesh - pred_joint[:, :1, :]
                gt_mesh = gt_mesh - gt_joint[:, :1, :]

                pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)

                j_error, s_error = self.val_dataset.compute_both_err(pred_mesh, gt_mesh, pred_pose, gt_pose3d)

                joint_error += j_error
                surface_error += s_error

                # Final Evaluation
                if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                    pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy()
                    pred_pose, gt_pose3d = pred_pose.detach().cpu().numpy(), gt_pose3d.detach().cpu().numpy()
                    # len(input_pose)=batchsize
                    for j in range(len(input_pose)):
                        out = {}
                        out['mesh_coord'], out['mesh_coord_target'] = pred_mesh[j], target_mesh[j]
                        out['joint_coord'], out['joint_coord_target'] = pred_pose[j], gt_pose3d[j]
                        result.append(out)

            self.surface_error = surface_error / len(self.val_loader)
            self.joint_error = joint_error / len(self.val_loader)

            print(f'{eval_prefix}MPVPE: {self.surface_error:.2f}, MPJPE: {self.joint_error:.2f}')

            if cfg.TRAIN.wandb:
                wandb_joint_error = self.joint_error
                wandb_verts_error = self.surface_error
                wandb.log(
                    {
                        'epoch': epoch,
                        'error/MPJPE': wandb_joint_error,
                        'error/MPVPE': wandb_verts_error,
                    }
                )

            # Final Evaluation
            if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                self.val_dataset.evaluate(result)
