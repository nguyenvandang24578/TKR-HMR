import os, sys
sys.path.append('./lib')
import torch
from core.config import cfg, update_config
from models.TeacherFusion import get_model as get_teacher_model

update_config('config/train_kd.yml')

# Load teacher
embed_dim = cfg.MODEL.hpe_dim * 2
teacher = get_teacher_model(embed_dim=embed_dim).cuda()

checkpoint_path = './experiment/teacher_512/checkpoint/best.pth.tar'
if os.path.exists(checkpoint_path):
    print(f"Loading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    teacher.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Checkpoint not found!")
    sys.exit(1)

teacher.eval()

# Create dummy input in mm (around 500.0)
dummy_pose = torch.randn(2, 16, 19, 3).cuda() * 50.0 + 500.0
dummy_img = torch.randn(2, 16, 2048).cuda()

with torch.no_grad():
    fused_feats, pred_mesh, smpl_output, skel_feats = teacher(dummy_pose, dummy_img)

print(f"t_skel_feats shape: {skel_feats.shape}")
print(f"t_skel_feats mean: {skel_feats.mean().item():.4f}, std: {skel_feats.std().item():.4f}")
print(f"t_skel_feats max: {skel_feats.max().item():.4f}, min: {skel_feats.min().item():.4f}")

print(f"t_fused_feats shape: {fused_feats.shape}")
print(f"t_fused_feats mean: {fused_feats.mean().item():.4f}, std: {fused_feats.std().item():.4f}")
print(f"t_fused_feats max: {fused_feats.max().item():.4f}, min: {fused_feats.min().item():.4f}")
