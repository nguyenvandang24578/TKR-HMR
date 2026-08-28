"""
Pose2Mesh — bản đã đơn giản hóa / sửa lỗi.

So với bản gốc, các thay đổi chính:

1. BUG FIX: import đúng `HYPERGCv2` từ `models.hypergcn` (bản gốc import
   `HYPERGC, A_19` từ `models.original_hypergcn` nhưng lại dùng
   `HYPERGCv2` — NameError khi khởi tạo model).
2. CẮT BỚT: bỏ 2/3 layer trong `spatial_hypers` không bao giờ được gọi
   (bản gốc tạo ModuleList 3 HYPERGCv2 nhưng chỉ dùng `[0]`, dùng lặp lại
   qua tất cả các vòng refine — 2 layer còn lại chỉ tốn tham số). Giờ chỉ
   giữ đúng 1 module `HYPERGCv2`, weight-shared qua các vòng lặp một cách
   tường minh (giống kiểu "universal transformer"), không còn param chết.
3. GIẢM ĐỘ SÂU MẶC ĐỊNH: `num_refinement_iters` 3->2, `num_coadapt_rounds`
   2->1. Nhiều tầng lặp hơn không có bằng chứng là cần thiết trong bản
   gốc — để mặc định gọn hơn, vẫn có thể tăng lại qua tham số nếu ablation
   cho thấy cần.
4. THÊM: deep-supervision tùy chọn — `forward(..., return_aux=True)` sẽ
   trả thêm `intermediate_poses` (từ từng vòng refine) và `fusion_info`
   (trọng số reliability ảnh/motion) để tính loss phụ hoặc log/debug,
   thay vì luôn luôn bị bỏ đi như bản gốc.
5. AN TOÀN HƠN: việc load checkpoint SPIN được tách thành hàm riêng
   `load_pretrained_spin()`, có try/except + log rõ ràng, thay vì crash
   cứng trong `__init__` nếu thiếu file hoặc mismatch key.
6. GAMMA/BETA (FiLM) được chặn biên bằng tanh để tránh gamma nổ về giá trị
   lớn khi global_ft có scale bất thường lúc đầu train.
7. Dọn import không dùng (os, Mlp, DropPath, partial, A_19).

Các phần mình KHÔNG có source (RegressorSpin, CrossAttentionBlock,
Residual, ShapeFeatureExtractor, Mesh, core.config) được giữ nguyên
interface gọi như bản gốc — chưa thể chạy end-to-end thử nghiệm vì thiếu
các file đó, cần bạn tự chạy lại toàn bộ pipeline để xác nhận.
"""
import sys
sys.path.append('./lib')
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from core.config import cfg
from models.backbones.mesh import Mesh

from models.spin import RegressorSpin
from models.hypergcn import HYPERGCv2
from models.Residual import Residual
from models.fusion_module import ComplementTemporal
from models.shape_features import ShapeFeatureExtractor
from models.common import CrossAttentionBlock

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
SMPL_MEAN_PARAMS_PATH = 'data/base_data/smpl_mean_params.npz'
NUM_SMPL_JOINTS = 24  # model nay gan chat voi topology SMPL (24 khop),
                       # tham so `num_joint` trong __init__ CHI dung cho
                       # nhanh Residual, khong lam thay doi so khop noi tai.


class TemporalMotionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=3, bidirectional=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, embed_dim // 2, num_layers,
            batch_first=True, bidirectional=bidirectional
        )
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, joints):
        B, T, J, _ = joints.shape
        motion = joints[:, 1:] - joints[:, :-1]
        motion = F.pad(motion, (0, 0, 0, 0, 1, 0))
        motion_flat = motion.reshape(B, T, -1)
        x = self.input_proj(motion_flat)
        x, _ = self.lstm(x)
        x = self.output_proj(x)
        x = self.norm(x)
        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.img_reliability = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, 1), nn.Sigmoid()
        )
        self.mot_reliability = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, 1), nn.Sigmoid()
        )
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_feat, mot_feat):
        w_img = self.img_reliability(img_feat)
        w_mot = self.mot_reliability(mot_feat)
        w_sum = w_img + w_mot + 1e-6
        w_img, w_mot = w_img / w_sum, w_mot / w_sum
        fused = self.fusion(torch.cat([img_feat * w_img, mot_feat * w_mot], dim=-1))
        fused = self.norm(fused)
        return fused, {'w_img': w_img, 'w_mot': w_mot}


class IterativePoseRefiner(nn.Module):
    """
    Tinh chinh pose token qua nhieu vong lap, moi vong: tron voi global
    context (qua MLP) roi (tuy chon) truyen qua HYPERGCv2 de mix thong tin
    giua cac khop theo kinematic tree + hyperedge hoc duoc.

    hypergcn duoc weight-share qua tat ca cac vong lap (mot instance duy
    nhat) — day la lua chon co chu dich (giong "universal transformer":
    giam tham so, khuyen khich hoc mot phep bien doi on dinh, co the ap
    dung lap lai), khong phai bug nhu ban goc (ban goc tao 3 instance
    nhung chi dung 1, 2 cai con lai chet).
    """
    def __init__(self, embed_dim, num_iter=2, hypergcn=None):
        super().__init__()
        self.num_iter = num_iter
        self.hypergcn = hypergcn
        self.pose_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_iter)
        ])
        self.pose_head = MLP(embed_dim, 256, 6, 3)

    def forward(self, pose_token, global_feat):
        B, T, J, D = pose_token.shape
        intermediate_poses = []
        for i in range(self.num_iter):
            ctx = global_feat.unsqueeze(2).expand(-1, -1, J, -1)
            pose_token = pose_token + self.pose_update[i](torch.cat([pose_token, ctx], dim=-1))
            if self.hypergcn is not None:
                pose_token, _ = self.hypergcn(pose_token)
            intermediate_poses.append(self.pose_head(pose_token))
        return pose_token, intermediate_poses


class PoseShapeCoAdaptation(nn.Module):
    def __init__(self, embed_dim, num_rounds=1):
        super().__init__()
        self.num_rounds = num_rounds
        self.pose_to_shape = CrossAttentionBlock(
            q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim,
            kv_num=24, num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
        )
        self.shape_to_pose = CrossAttentionBlock(
            q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim,
            kv_num=1, num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
        )
        self.shape_head = MLP(embed_dim, 256, 10, 3)
        self.pose_head = MLP(embed_dim, 256, 6, 3)

    def forward(self, pose_tokens, shape_token):
        B, T, J, D = pose_tokens.shape

        for _ in range(self.num_rounds):
            shape_token_flat = shape_token.view(B * T, 1, D)
            pose_tokens_flat = pose_tokens.view(B * T, J, D)

            # 1. Pose -> Shape (shape token attends to 24 pose tokens)
            shape_token_flat = self.pose_to_shape(shape_token_flat, pose_tokens_flat, pose_tokens_flat)
            shape_token = shape_token_flat.view(B, T, D)

            # 2. Shape -> Pose (24 pose tokens attend to 1 shape token)
            pose_tokens_flat = self.shape_to_pose(pose_tokens_flat, shape_token_flat, shape_token_flat)
            pose_tokens = pose_tokens_flat.view(B, T, J, D)

        pose_params = self.pose_head(pose_tokens)
        shape_params = self.shape_head(shape_token)

        return pose_params, shape_params, pose_tokens, shape_token


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class Pose2Mesh(nn.Module):
    def __init__(
        self,
        num_joint,
        embed_dim=512,
        num_refinement_iters=2,
        num_coadapt_rounds=1,
        temporal_layers=3,
        use_hypergcn=True,
        spin_checkpoint_name='spin_model_checkpoint.pth.tar',
    ):
        super(Pose2Mesh, self).__init__()

        self.mesh = Mesh()
        self.num_refinement_iters = num_refinement_iters
        self.use_hypergcn = use_hypergcn

        mean_params = np.load(SMPL_MEAN_PARAMS_PATH)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)

        self.out_proj = nn.Linear(512, 2048)
        self.inproj_img = nn.Linear(2048, embed_dim)
        self.pose_embed = nn.Linear(6, embed_dim)
        self.shape_embed = nn.Linear(10, embed_dim)

        self.fuse_shape = CrossAttentionBlock(
            q_dim=512, k_dim=512, v_dim=512, kv_num=cfg.DATASET.seqlen,
            num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
        )

        self.cfcer = ComplementTemporal(depths=2, dim=embed_dim)

        self.fusion = AdaptiveFusion(embed_dim)

        self.residual = Residual(num_joint=num_joint)
        self.node_pe = nn.Embedding(NUM_SMPL_JOINTS, embed_dim)

        # Chi giu DUNG MOT layer HYPERGCv2 (thay vi 3 layer nhung chi dung
        # [0] nhu ban goc), weight-share qua cac vong refine.
        self.spatial_hyper = HYPERGCv2(embed_dim, embed_dim, num_edges=5) if use_hypergcn else None

        self.temporal_encoder = TemporalMotionEncoder(
            input_dim=57, embed_dim=embed_dim, num_layers=temporal_layers
        )

        self.iterative_refiner = IterativePoseRefiner(
            embed_dim=embed_dim,
            num_iter=num_refinement_iters,
            hypergcn=self.spatial_hyper
        )

        self.pose_shape_coadapt = PoseShapeCoAdaptation(
            embed_dim=embed_dim,
            num_rounds=num_coadapt_rounds
        )

        self.shape_feat_extractor = ShapeFeatureExtractor(embed_dim=embed_dim)
        self.shape_token = nn.Embedding(1, embed_dim)

        max_seqlen = cfg.DATASET.seqlen
        self.pos_embed_cfcer = nn.Parameter(torch.zeros(1, max_seqlen, embed_dim))
        self.pos_embed_motion = nn.Parameter(torch.zeros(1, max_seqlen, embed_dim))
        trunc_normal_(self.pos_embed_cfcer, std=.2)
        trunc_normal_(self.pos_embed_motion, std=.2)

        self.gamma_proj = nn.Linear(embed_dim, embed_dim)
        self.beta_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.regressorspin = RegressorSpin()
        self._spin_checkpoint_name = spin_checkpoint_name
        self.load_pretrained_spin()

    def load_pretrained_spin(self):
        """
        Nap checkpoint SPIN pretrained mot cach an toan.

        FIX (so voi ban goc): ban goc goi torch.load(...)['model'] truc
        tiep trong __init__, khong try/except -> neu thieu file hoac file
        hong, TOAN BO model khong khoi tao duoc, ke ca khi chi can test
        kien truc / export ma khong can trong so pretrained. Gio tach
        thanh ham rieng, bat loi va log ro, cho phep model van khoi tao
        duoc (voi RegressorSpin dung random init) neu thieu checkpoint.
        """
        ckpt_path = osp.join(BASE_DATA_DIR, self._spin_checkpoint_name)
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            pretrained_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            missing, unexpected = self.regressorspin.load_state_dict(pretrained_dict, strict=False)
            if missing:
                print(f"[Pose2Mesh] SPIN checkpoint: {len(missing)} key thieu trong checkpoint "
                      f"(vd: {missing[:3]}...)")
            if unexpected:
                print(f"[Pose2Mesh] SPIN checkpoint: {len(unexpected)} key du thua trong checkpoint "
                      f"(vd: {unexpected[:3]}...)")
        except FileNotFoundError:
            print(f"[Pose2Mesh][WARNING] Khong tim thay checkpoint SPIN tai '{ckpt_path}'. "
                  f"RegressorSpin se dung random init — chi phu hop de test kien truc, "
                  f"KHONG dung de train/eval that.")
        except Exception as e:
            print(f"[Pose2Mesh][WARNING] Loi khi load checkpoint SPIN tu '{ckpt_path}': {e}. "
                  f"RegressorSpin se dung random init.")

    def forward(
        self,
        joints,
        img_feats,
        kp2d=None,
        using_prompt=True,
        is_train=True,
        J_regressor=None,
        return_aux=False,
    ):
        batch_size = img_feats.shape[0]
        seq_len = img_feats.shape[1]
        mid = seq_len // 2

        mean_pose = self.init_pose.view(1, NUM_SMPL_JOINTS, 6)
        mean_shape = self.init_shape.view(1, 10)

        pose_emb = self.pose_embed(mean_pose)
        shape_emb = self.shape_embed(mean_shape)

        pose_token = pose_emb.unsqueeze(1).expand(batch_size, seq_len, NUM_SMPL_JOINTS, -1)
        shape_token = self.shape_token.weight.unsqueeze(0).expand(batch_size, seq_len, -1)
        shape_token = shape_token + shape_emb

        img_feats_proj = self.inproj_img(img_feats)

        motion_feat = self.temporal_encoder(joints)

        img_enhanced, motion_enhanced = self.cfcer(
            img_feats_proj, motion_feat,
            pe_r=self.pos_embed_cfcer[:, :seq_len],
            pe_d=self.pos_embed_motion[:, :seq_len]
        )

        global_ft, fusion_info = self.fusion(img_enhanced, motion_enhanced)

        img_feats_trans = self.out_proj(global_ft) + img_feats

        # FiLM modulation cho pose_token, chan bien gamma bang tanh de
        # tranh gamma bung no khi global_ft co scale bat thuong luc dau
        # train (vd chua warmup, init ngau nhien).
        gamma = 1.0 + torch.tanh(self.gamma_proj(global_ft)).unsqueeze(2)
        beta = self.beta_proj(global_ft).unsqueeze(2)
        pose_token = gamma * pose_token + beta

        idx = torch.arange(NUM_SMPL_JOINTS, device=pose_token.device)
        pose_token = self.norm(pose_token) + self.node_pe(idx)

        pose_token, intermediate_poses = self.iterative_refiner(pose_token, global_ft)

        if kp2d is not None:
            shape_feat = self.shape_feat_extractor(kp2d)
            shape_token = shape_token + shape_feat

        shape_output = self.fuse_shape(shape_token, global_ft, global_ft)

        pred_pose, pred_shape, refined_pose_token, refined_shape_token = self.pose_shape_coadapt(
            pose_token, shape_output
        )

        inv_pred2rot6d = pred_pose.reshape(batch_size, seq_len, -1)
        inv_mesh2shape = pred_shape.reshape(batch_size, seq_len, -1)

        spin_outputs = self.regressorspin(
            img_feats_trans, inv_pred2rot6d, inv_mesh2shape,
            is_train=is_train, J_regressor=J_regressor
        )

        smpl_vertices = spin_outputs[-1]['verts']
        smpl_vertices_mid = smpl_vertices[:, mid]
        residual_joint, residual_mesh = self.residual(
            joints[:, mid], img_feats[:, mid]
        )
        smpl_vertices_mid = 0.5 * smpl_vertices_mid + 0.5 * residual_mesh

        evo_pose = residual_joint
        init_smpl_pose = pred_pose[:, mid].reshape(batch_size, -1)
        init_smpl_shape = pred_shape[:, mid].reshape(batch_size, -1)

        outputs = (
            evo_pose,
            init_smpl_pose,
            init_smpl_shape,
            smpl_vertices_mid,
            spin_outputs
        )

        if return_aux:
            # De xuat: dung intermediate_poses de tinh them deep-supervision
            # loss (vd MSE giua tung buoc refine va GT pose 6D), va
            # fusion_info de log/debug ty trong dong gop cua anh vs motion.
            aux = {
                'intermediate_poses': intermediate_poses,
                'fusion_info': fusion_info,
                'refined_pose_token': refined_pose_token,
                'refined_shape_token': refined_shape_token,
            }
            return outputs + (aux,)

        return outputs


def get_model(num_joint, embed_dim, **kwargs):
    model = Pose2Mesh(num_joint, embed_dim, **kwargs)
    return model