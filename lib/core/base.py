import numpy as np
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
import os

import models
import Human36M.dataset, COCO.dataset, PW3D.dataset, MPII3D.dataset, MPII.dataset
from core.config import cfg
from core.loss import get_loss
from multiple_datasets import MultipleDatasets
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters, lr_check
from models.backbones.mesh import Mesh
import time
from coord_utils import rigid_align
import torch.nn.functional as F

from geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis
import math
from smpl import SMPL
import trimesh

mesh_model = SMPL()
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from thop import profile
from thop import clever_format
def plot_attention_map(attn_map, frame_idx=0):
    """
    attn_map: Tensor có shape (B*T, num_heads, 24, 24)
    """
    # 1. Trích xuất frame cần xem (Ví dụ: Frame 0)
    # Shape: (num_heads, 24, 24)
    single_frame_attn = attn_map[frame_idx] 
    
    # 2. Trung bình cộng các Heads lại với nhau để lấy tổng thể 
    # Shape: (24, 24)
    mean_attn = single_frame_attn.mean(dim=0).cpu().numpy()
    
    # Tên của 24 khớp xương (Dựa theo chuẩn SMPL)
    joint_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
        'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 
        'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
        'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]

    # 3. Vẽ biểu đồ
    plt.figure(figsize=(12, 10))
    sns.heatmap(mean_attn, 
                xticklabels=joint_names, 
                yticklabels=joint_names, 
                cmap='viridis', # Dùng thang màu Viridis rực rỡ
                cbar_kws={'label': 'Attention Score'})
    
    plt.title(f"Joint Self-Attention Map (Frame {frame_idx})", fontsize=16)
    plt.xlabel("Key", fontsize=12)
    plt.ylabel("Query", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Lưu lại thành file ảnh
    plt.savefig(f"attention_heatmap_frame_{frame_idx}.png", dpi=300)
    plt.show()
    print("✅ Đã xuất ảnh Heatmap!")
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
        batch_generator = DataLoader(dataset=trainset_loader, \
                          batch_size=batch_per_dataset * len(dataset_names), \
                          shuffle=cfg[dataset_split].shuffle, \
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
        print(f"==> Preparing {cfg.MODEL.name} MODEL...")
        if cfg.MODEL.name == 'ARTS':
            model = models.ARTS.get_model(num_joint=main_dataset.joint_num, embed_dim=cfg.MODEL.hpe_dim, depth=cfg.MODEL.hpe_dep)
        elif cfg.MODEL.name == 'PoseEst':
            model = models.PoseEstimation.get_model(num_joint=main_dataset.joint_num, embed_dim=cfg.MODEL.hpe_dim, depth=cfg.MODEL.hpe_dep, pretrained=False)
        print('# of model parameters: {}'.format(count_parameters(model)))

    if is_train:
        criterion = get_loss(faces=main_dataset.mesh_model.face)
        optimizer = get_optimizer(model=model)
        lr_scheduler = get_scheduler(optimizer=optimizer)

    if load_dir and (not is_train or args.resume_training):
        print('==> Loading checkpoint')
        checkpoint = load_checkpoint(load_dir=load_dir, pick_best=(cfg.MODEL.name == 'PoseEst'))
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

        self.model = torch.nn.DataParallel(self.model).cuda()

        self.mesh = Mesh()
        self.normal_weight = cfg.MODEL.normal_loss_weight
        self.edge_weight = cfg.MODEL.edge_loss_weight
        self.joint_weight = cfg.MODEL.joint_loss_weight
        self.edge_add_epoch = cfg.TRAIN.edge_loss_start
        self.shape_weight = cfg.MODEL.shape_loss_weight
        self.pose_weight = cfg.MODEL.pose_loss_weight

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='ARTS/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)
        for i, pg in enumerate(self.optimizer.param_groups):
            group_name = ['SPIN', 'Fresh'][i] if i < 2 else f'Group{i}'
            print(f"  [{group_name}] lr={pg['lr']:.2e} | params={sum(p.numel() for p in pg['params']):,}")
        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            # convert to cuda
            input_pose, input_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
            gt_lift3dpose, gt_reg3dpose, gt_mesh = targets['lift_pose3d'].cuda(), targets['reg_pose3d'].cuda(), targets['mesh'].cuda()
            gt_smplpose, gt_smplshape = targets['smpl_pose'].cuda(), targets['smpl_shape'].cuda()
            val_lift3dpose, val_reg3dpose, val_mesh = meta['lift_pose3d_valid'].cuda(), meta['reg_pose3d_valid'].cuda(), meta['mesh_valid'].cuda()
            
            pose3d, evo_pose, init_smpl_pose, init_smpl_shape, pred_mesh, smploutput = self.model(input_pose, input_feat, is_train=True) 
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh * 1000)
            loss1, loss2, loss4, loss5, loss6 = self.loss[0](pred_mesh, gt_mesh, val_mesh),  \
                                         self.normal_weight * self.loss[1](pred_mesh, gt_mesh), \
                                         self.joint_weight * self.loss[3](pred_pose, gt_reg3dpose, val_reg3dpose), \
                                         self.joint_weight * self.loss[4](evo_pose, gt_lift3dpose, val_lift3dpose), \
                                         self.joint_weight * self.loss[5](pose3d, gt_lift3dpose, val_lift3dpose)
            
            pa_loss = 0
            for n in range(pred_pose.shape[0]):
                pose_aligned = rigid_align(pred_pose[n].detach().cpu().numpy(), gt_reg3dpose[n].detach().cpu().numpy()) # perform rigid 
                pose_aligned = torch.from_numpy(pose_aligned).cuda()
                pa_loss += self.loss[3](pose_aligned, gt_reg3dpose[n], val_reg3dpose)
            pa_loss = self.joint_weight * (pa_loss / pred_pose.shape[0])

            init_rotmat = rot6d_to_rotmat(init_smpl_pose).view(init_smpl_pose.shape[0], 24, 3, 3)
            init_axis = rotation_matrix_to_angle_axis(init_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
            init_smpl_pose_loss, init_smpl_shape_loss = self.loss[6](init_axis,\
                                                                    init_smpl_shape,\
                                                                    gt_smplpose,\
                                                                    gt_smplshape,\
                                                                    mask_3d=None)
            init_smpl_loss = self.shape_weight * init_smpl_shape_loss + self.pose_weight * init_smpl_pose_loss

            smpl_pose_loss, smpl_shape_loss = self.loss[6](smploutput[-1]['theta'][:, 3:75],\
                                                           smploutput[-1]['theta'][:, 75:],\
                                                            gt_smplpose,\
                                                            gt_smplshape,\
                                                            mask_3d=None)
            mid_smpl_loss = self.shape_weight * smpl_shape_loss + self.pose_weight * smpl_pose_loss 
            # Directly supervise the pose/shape predicted before the SPIN
            # refinement so gradients reach pose_head and shape_head.
            loss = loss1 + loss4 + mid_smpl_loss + init_smpl_loss
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
                wandb_loss1, wandb_loss2, wandb_loss4, wandb_loss6 = loss1.detach(), loss2.detach(), loss4.detach(), loss6.detach()
                wandb_loss3 = loss3.detach() if epoch > self.edge_add_epoch else 0
                wandb.log(
                    {
                        'train_loss/vertex_loss': wandb_loss1,
                        'train_loss/normal_loss': wandb_loss2,
                        'train_loss/edge_loss': wandb_loss3,
                        'train_loss/mesh2joint3d_loss': wandb_loss4,
                        'train_loss/liftjoint3d_loss': wandb_loss6
                    }
                )

            if i % self.print_freq == 0:
                loss1, loss2, loss4, loss5 = loss1.detach(), loss2.detach(), loss4.detach(), loss5.detach()
                loss3 = loss3.detach() if epoch > self.edge_add_epoch else 0
                init_loss = init_smpl_loss.detach()
                smpl_loss = mid_smpl_loss.detach()
                loss6 = loss6.detach()
                pa_loss = pa_loss.detach()
                total_loss = loss.detach()
                alpha_val = torch.sigmoid((self.model.module if hasattr(self.model, 'module') else self.model).pose_mesh_coevo.blend_weight).item()

# Lấy giá trị an toàn (nếu là Tensor thì gọi .item(), nếu là số thì giữ nguyên)
                l1 = loss1.item() if hasattr(loss1, 'item') else loss1
                l2 = loss2.item() if hasattr(loss2, 'item') else loss2
                l3 = loss3.item() if hasattr(loss3, 'item') else loss3
                l4 = loss4.item() if hasattr(loss4, 'item') else loss4
                s_loss = smpl_loss.item() if hasattr(smpl_loss, 'item') else smpl_loss
                t_loss = total_loss.item() if hasattr(total_loss, 'item') else total_loss
                a_val = alpha_val.item() if hasattr(alpha_val, 'item') else alpha_val

                # In ra log
                batch_generator.set_description(f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                                                f'mesh: {l1:.3f} '
                                                f'normal: {l2:.3f} '
                                                f'edge: {l3:.3f} '
                                                f'mpjpe_loss: {l4:.3f} '
                                                f'spin%: {a_val*100:.1f} '
                                                f'res%: {(1-a_val)*100:.1f} '
                                                f'smpl: {s_loss:.3f} '
                                                f'tl: {t_loss:.3f}')

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
            self.model = torch.nn.DataParallel(self.model).cuda()

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
                
                gt_lift_pose3d = targets['lift_pose3d'].cuda()

                # # ==========================================
                # # ĐOẠN CODE THÊM VÀO ĐỂ TÍNH FLOPS/PARAMS
                # # ==========================================
                # if i == 0: 
                #     print("\n[INFO] Đang tính toán FLOPs và Parameters...")
                #     # Truyen dung tuple cac input ban dua vao self.model
                #     macs, params = profile(self.model, inputs=(input_pose, input_feat), verbose=False)
                    
                #     # Chuyển đổi sang định dạng dễ đọc (M, G)
                #     macs_format, params_format = clever_format([macs, params], "%.2f")
                #     print(f"==========> MACs (GFLOPs): {macs_format} | Params: {params_format} <==========")
                # # ==========================================
                pose3d, evo_pose, init_smpl_pose, init_smpl_shape, pred_mesh, smploutput = self.model(input_pose, input_feat, is_train=False)

                # ==========================================
                # MODEL DIAGNOSTIC — chạy 1 lần ở batch đầu
                # ==========================================
                # if i == 0:
                #     self._run_diagnostic(input_pose, input_feat, pred_mesh, smploutput)
                pred_mesh, gt_mesh = pred_mesh * 1000, gt_mesh * 1000

                pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)

                j_error, s_error = self.val_dataset.compute_both_err(pred_mesh, gt_mesh, pred_pose, gt_pose3d)

                if i % self.print_freq == 0:
                    loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => surface error: {s_error:.4f}, joint error: {j_error:.4f}')

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

    def _run_diagnostic(self, input_pose, input_feat, pred_mesh, smploutput):
        """One-shot diagnostic: chạy 1 lần ở batch đầu của test."""
        # Lấy model gốc (bỏ DataParallel wrapper)
        m = self.model.module if hasattr(self.model, 'module') else self.model
        # Nếu ARTS wrapper → lấy pose_mesh_coevo (Pose2Mesh)
        pm = m.pose_mesh_coevo if hasattr(m, 'pose_mesh_coevo') else m

        print("\n" + "=" * 70)
        print("📊 MODEL DIAGNOSTIC (batch đầu tiên)")
        print("=" * 70)

        # ── 1. HyperGCN ──
        if hasattr(pm, 'spatial_hypers'):
            print("\n┌─── HYPERGCN ───")
            for idx, layer in enumerate(pm.spatial_hypers):
                a_c = layer.alpha_chain_raw.item()
                a_h = layer.alpha_hyper_raw.item()
                ratio = abs(a_c) / (abs(a_c) + abs(a_h) + 1e-8) * 100

                ah = layer.adaptive_hyper
                b0 = F.softplus(ah.beta0_raw).item()
                b1 = F.softplus(ah.beta1_raw).item()
                b2 = F.softplus(ah.beta2_raw).item()
                bt = b0 + b1 + b2

                M = F.softplus(ah.M_raw)
                m_diff = (M - ah.H_init).abs().mean().item()
                root_leak = ah.H_init[0].sum().item()

                # Weight norms
                rc_norm = sum(p.data.norm().item() for p in layer.root_chain.parameters()) / max(1, sum(1 for _ in layer.root_chain.parameters()))
                ah_norm = sum(p.data.norm().item() for p in layer.adaptive_hyper.parameters()) / max(1, sum(1 for _ in layer.adaptive_hyper.parameters()))

                flag_alpha = "⚠️" if abs(a_c) < 0.01 or abs(a_h) < 0.01 else "✅"
                flag_root = "⚠️" if abs(root_leak) > 0.01 else "✅"

                print(f"│  Layer {idx}: α_chain={a_c:.4f} ({ratio:.0f}%) | α_hyper={a_h:.4f} ({100-ratio:.0f}%) {flag_alpha}")
                print(f"│    β₀(fixed)={b0:.3f}({b0/bt*100:.0f}%) β₁(learn)={b1:.3f}({b1/bt*100:.0f}%) β₂(dynamic)={b2:.3f}({b2/bt*100:.0f}%)")
                print(f"│    |M-H_init|={m_diff:.4f}  root_leak={root_leak:.6f} {flag_root}  W_norm: RC={rc_norm:.2f} AH={ah_norm:.2f}")

        # ── 2. Node PE ──
        if hasattr(pm, 'node_pe'):
            pe = pm.node_pe.weight.data
            norms = pe.norm(dim=1)
            print(f"│  node_pe: norm min={norms.min():.3f} max={norms.max():.3f} mean={norms.mean():.3f}")

        # ── 3. Post-HyperGCN Norm ──
        if hasattr(pm, 'post_hyper_norm'):
            phn = pm.post_hyper_norm
            print(f"│  post_hyper_norm: γ_mean={phn.weight.data.mean():.4f} β_mean={phn.bias.data.mean():.4f}")
        print("└───")

        # ── 4. CFCer / Cross-Attention ──
        print("\n┌─── FUSION (CFCer) ───")
        if hasattr(pm, 'use_cfcer') and pm.use_cfcer:
            pe_cfcer_norm = pm.pos_embed_cfcer.data.norm().item()
            pe_motion_norm = pm.pos_embed_motion.data.norm().item()
            print(f"│  CFCer pos_embed: img_norm={pe_cfcer_norm:.3f}, motion_norm={pe_motion_norm:.3f}")
            # Attention stats
            for d_idx, (anm, ank) in enumerate(pm.cfcer.att_nets):
                q_norm = anm.q.weight.data.norm().item()
                k_norm = anm.k.weight.data.norm().item()
                print(f"│  Depth {d_idx}: ANM q_norm={q_norm:.3f} k_norm={k_norm:.3f}")
        else:
            print("│  Dùng CrossAttentionBlock (không phải CFCer)")
        print("└───")

        # ── 5. FiLM conditioning ──
        print("\n┌─── FiLM CONDITIONING ───")
        gamma_w = pm.gamma_proj.weight.data.norm().item()
        beta_w = pm.beta_proj.weight.data.norm().item()
        print(f"│  gamma_proj W_norm={gamma_w:.3f}  beta_proj W_norm={beta_w:.3f}")
        print("└───")

        # ── 6. SPIN output analysis ──
        print("\n┌─── SPIN REGRESSOR ───")
        spin_verts = smploutput[-1]['verts']
        spin_theta = smploutput[-1]['theta']
        print(f"│  verts: shape={tuple(spin_verts.shape)}, mean={spin_verts.mean():.4f}, std={spin_verts.std():.4f}")
        print(f"│  theta: shape={tuple(spin_theta.shape)}, mean={spin_theta.mean():.4f}, std={spin_theta.std():.4f}")
        # pose vs shape magnitude
        if spin_theta.dim() == 3:
            cam_part = spin_theta[:, :, :3]
            pose_part = spin_theta[:, :, 3:75]
            shape_part = spin_theta[:, :, 75:]
        else:
            cam_part = spin_theta[:, :3]
            pose_part = spin_theta[:, 3:75]
            shape_part = spin_theta[:, 75:]
        print(f"│  cam:   mean={cam_part.mean():.4f} std={cam_part.std():.4f}")
        print(f"│  pose:  mean={pose_part.mean():.4f} std={pose_part.std():.4f}")
        print(f"│  shape: mean={shape_part.mean():.4f} std={shape_part.std():.4f}")
        print("└───")

        # ── 7. Blend & Residual ──
        print("\n┌─── RESIDUAL & BLEND ───")
        blend_raw = pm.blend_weight.item()
        blend_sig = torch.sigmoid(pm.blend_weight).item()
        print(f"│  blend_weight raw={blend_raw:.4f}  sigmoid={blend_sig:.4f}")
        print(f"│  (Lưu ý: forward đang hardcode 0.5/0.5, KHÔNG dùng blend_weight)")
        print(f"│  pred_mesh: mean={pred_mesh.mean():.4f}, std={pred_mesh.std():.4f}")
        print("└───")

        # ── 8. Parameter count per module ──
        print("\n┌─── PARAMETER COUNT ───")
        modules_count = {}
        total_params = sum(p.numel() for p in pm.parameters())

        module_groups = {
            'CFCer/CrossAttn': ['cfcer', 'cross_attn_img', 'cross_attn_motion', 'pos_embed_cfcer', 'pos_embed_motion'],
            'Fusion Linear': ['fusion_linear', 'conv1d_fusion'],
            'FiLM': ['gamma_proj', 'beta_proj'],
            'HyperGCN': ['spatial_hypers', 'node_pe', 'post_hyper_norm'],
            'Pose/Shape Head': ['pose_head', 'shape_head', 'fuse_shape', 'shape_token', 'shape_embed', 'pose_embed'],
            'KP2D Injection': ['kpt_mlp', 'kp_norm', 'kp_map', 'inject_norm'],
            'SPIN Regressor': ['regressorspin'],
            'Residual CoEvo': ['residual'],
            'Projection': ['projoint', 'inproj_img', 'out_proj'],
        }

        for group_name, attr_names in module_groups.items():
            count = 0
            for attr in attr_names:
                if hasattr(pm, attr):
                    obj = getattr(pm, attr)
                    if isinstance(obj, torch.nn.Module):
                        count += sum(p.numel() for p in obj.parameters())
                    elif isinstance(obj, torch.nn.Parameter):
                        count += obj.numel()
            if count > 0:
                print(f"│  {group_name:20s}: {count:>10,} ({count/total_params*100:5.1f}%)")

        print(f"│  {'TOTAL':20s}: {total_params:>10,}")
        print("└───")
        print("=" * 70 + "\n")


class LiftTrainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history \
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.loss = self.loss[0]
        self.main_dataset = self.dataset_list[0]
        self.num_joint = self.main_dataset.joint_num
        # self.num_joint = 16
        self.print_freq = cfg.TRAIN.print_freq

        self.model = self.model.cuda()

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='PoseEst/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)

        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (img_joint, cam_joint, joint_valid, img_features) in enumerate(batch_generator):
            img_joint, cam_joint = img_joint.cuda().float(), cam_joint.cuda().float()
            joint_valid = joint_valid.cuda().float()
            img_features = img_features.cuda().float()

            pred_joint = self.model(img_joint, img_features)
            pred_joint = pred_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)
            cam_joint = cam_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)

            mpjpe_loss = self.loss(pred_joint, cam_joint, joint_valid)
            
            loss = mpjpe_loss

            self.optimizer.zero_grad()
            loss.backward()  
            # loss.sum().backward()
            self.optimizer.step()

            running_loss += float(loss.detach().item())
            if cfg.TRAIN.wandb:
                wandb_loss = loss.detach()
                wandb.log(
                    {
                        'train_loss/total_loss': wandb_loss
                    }
                )

            if i % self.print_freq == 0:
                batch_generator.set_description(f'Epoch{epoch}_({i}/{len(self.batch_generator)}) => '
                                                f'total loss: {loss.detach():.4f} ')

        self.loss_history.append(running_loss / len(self.batch_generator))

        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')


class LiftTester:
    def __init__(self, args, load_dir=''):
        self.val_loader, self.val_dataset, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)
        self.val_dataset = self.val_dataset[0]
        self.val_loader = self.val_loader[0]

        self.num_joint = self.val_dataset.joint_num
        # self.num_joint = 16
        self.print_freq = cfg.TRAIN.print_freq

        if self.model:
            self.model = self.model.cuda()

        # initialize error value
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        

        result = []
        joint_error = 0.0
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (img_joint, cam_joint, _, img_features) in enumerate(loader):
                img_joint, cam_joint = img_joint.cuda().float(), cam_joint.cuda().float()
                img_features = img_features.cuda().float()

                pred_joint = self.model(img_joint, img_features)
                pred_joint = pred_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)

                mpjpe = self.val_dataset.compute_joint_err(pred_joint, cam_joint)
                joint_error += mpjpe

                if i % self.print_freq == 0:
                    loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => joint error: {mpjpe:.4f}')

                # Final Evaluation
                if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                    pred_joint, target_joint = pred_joint.detach().cpu().numpy(), cam_joint.detach().cpu().numpy()
                    for j in range(len(pred_joint)):
                        out = {}
                        out['joint_coord'], out['joint_coord_target'] = pred_joint[j], target_joint[j]
                        result.append(out)

        self.joint_error = joint_error / len(self.val_loader)
        print(f'{eval_prefix}MPJPE: {self.joint_error:.4f}')

        if cfg.TRAIN.wandb:
                wandb_error = self.joint_error
                wandb.log(
                    {
                        'epoch': epoch,
                        'error/MPJPE': wandb_error
                    }
                )

        # Final Evaluation
        if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
            self.val_dataset.evaluate_joint(result)