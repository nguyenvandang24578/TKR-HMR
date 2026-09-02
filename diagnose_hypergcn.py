"""
HyperGCN Diagnostic Script
===========================
Chạy: python diagnose_hypergcn.py --ckpt <path_to_checkpoint> --cfg <config.yml>
Hoặc không cần checkpoint (chỉ check random init):
       python diagnose_hypergcn.py --cfg config/train_mesh_3dpw.yml
"""
import os, sys
sys.path.append('./lib')

import argparse
import torch
import numpy as np
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='', help='Path to checkpoint')
parser.add_argument('--cfg', type=str, default='', help='Config yaml file')
parser.add_argument('--gpu', type=str, default='0', help='GPU id')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.cfg:
    from core.config import cfg, update_config
    update_config(args.cfg)
else:
    from core.config import cfg

import models
from models.Multimodel import Pose2Mesh
from models.hypergcn import HYPERGCv2, build_H_init_no_root

# ============================================================
# 1. TẠO MODEL
# ============================================================
print("=" * 70)
print("HYPERGCN DIAGNOSTIC TOOL")
print("=" * 70)

num_joint = 19  # COCO
embed_dim = cfg.MODEL.hpe_dim * 2  # giống ARTS
print(f"\n[CONFIG] num_joint={num_joint}, embed_dim={embed_dim}, seqlen={cfg.DATASET.seqlen}")

model = Pose2Mesh(num_joint=num_joint, embed_dim=embed_dim)

# Load checkpoint nếu có
if args.ckpt and os.path.exists(args.ckpt):
    print(f"\n[LOAD] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # Handle DataParallel prefix
    cleaned = {}
    for k, v in state_dict.items():
        # ARTS wraps as pose_mesh_coevo.xxx
        k = k.replace('pose_mesh_coevo.', '')
        k = k.replace('module.pose_mesh_coevo.', '')
        k = k.replace('module.', '')
        cleaned[k] = v
    
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    
    hyper_missing = [k for k in missing if 'hyper' in k or 'node_pe' in k or 'post_hyper' in k]
    hyper_loaded = [k for k in cleaned.keys() if 'hyper' in k or 'node_pe' in k or 'post_hyper' in k]
    
    if hyper_missing:
        print(f"\n⚠️  HyperGCN keys MISSING from checkpoint (will use random init):")
        for k in hyper_missing:
            print(f"    - {k}")
    if hyper_loaded:
        print(f"\n✅ HyperGCN keys LOADED from checkpoint:")
        for k in hyper_loaded[:10]:
            print(f"    - {k}")
        if len(hyper_loaded) > 10:
            print(f"    ... và {len(hyper_loaded)-10} keys nữa")
    
    if not hyper_loaded and hyper_missing:
        print("\n🔴 Checkpoint KHÔNG có weights cho HyperGCN!")
        print("   → HyperGCN đang dùng random init, chưa được train.")
else:
    print("\n[INFO] Không load checkpoint — dùng random init để kiểm tra cấu trúc.")

model = model.cuda()
model.eval()

# ============================================================
# 2. KIỂM TRA THAM SỐ TĨNH (Static Parameters)
# ============================================================
print("\n" + "=" * 70)
print("PHẦN 1: THAM SỐ TĨNH CỦA HYPERGCN")
print("=" * 70)

for layer_idx, hyper_layer in enumerate(model.spatial_hypers):
    print(f"\n--- Layer {layer_idx} ---")
    
    # Alpha weights
    a_chain = hyper_layer.alpha_chain_raw.item()
    a_hyper = hyper_layer.alpha_hyper_raw.item()
    print(f"  α_chain = {a_chain:.4f}  |  α_hyper = {a_hyper:.4f}")
    ratio = abs(a_chain) / (abs(a_chain) + abs(a_hyper) + 1e-8)
    print(f"  → RootChain chiếm {ratio*100:.1f}%, AdaptiveHyper chiếm {(1-ratio)*100:.1f}%")
    
    if abs(a_chain) < 0.01 or abs(a_hyper) < 0.01:
        print(f"  ⚠️  MỘT NHÁNH GẦN NHƯ BỊ TẮT (α ≈ 0)!")
    
    # Beta coefficients (adaptive hyper)
    ah = hyper_layer.adaptive_hyper
    b0 = F.softplus(ah.beta0_raw).item()
    b1 = F.softplus(ah.beta1_raw).item()
    b2 = F.softplus(ah.beta2_raw).item()
    print(f"  β₀(H_init)={b0:.4f}  β₁(M_learn)={b1:.4f}  β₂(S_dynamic)={b2:.4f}")
    total_beta = b0 + b1 + b2
    print(f"  → H_init: {b0/total_beta*100:.1f}%, M_learnable: {b1/total_beta*100:.1f}%, S_dynamic: {b2/total_beta*100:.1f}%")
    
    # M (learnable membership)
    M = F.softplus(ah.M_raw)
    print(f"  M matrix: min={M.min():.4f}, max={M.max():.4f}, mean={M.mean():.4f}")
    
    # H_init vs M difference
    H_init = ah.H_init
    diff = (M - H_init).abs().mean().item()
    print(f"  |M - H_init| mean = {diff:.4f} (0=chưa học gì, lớn=đã adapt)")
    
    # Root masking check
    print(f"  H_init[root=0] sum = {H_init[0].sum().item():.6f} (phải = 0)")

# Node PE
node_pe_weight = model.node_pe.weight.data
pe_norms = node_pe_weight.norm(dim=1)
print(f"\n--- Node Positional Embedding ---")
print(f"  Shape: {tuple(node_pe_weight.shape)}")
print(f"  Norm per joint: min={pe_norms.min():.4f}, max={pe_norms.max():.4f}, mean={pe_norms.mean():.4f}")
print(f"  Cosine sim (joint 0 vs 1): {F.cosine_similarity(node_pe_weight[0:1], node_pe_weight[1:2]).item():.4f}")
print(f"  Cosine sim (joint 16 vs 17): {F.cosine_similarity(node_pe_weight[16:17], node_pe_weight[17:18]).item():.4f} (L-Arm vs R-Arm)")

# post_hyper_norm
if hasattr(model, 'post_hyper_norm'):
    phn = model.post_hyper_norm
    print(f"\n--- Post-HyperGCN LayerNorm ---")
    print(f"  gamma: mean={phn.weight.data.mean():.4f}, std={phn.weight.data.std():.4f}")
    print(f"  beta:  mean={phn.bias.data.mean():.4f}, std={phn.bias.data.std():.4f}")

# ============================================================
# 3. FORWARD PASS VỚI DUMMY DATA
# ============================================================
print("\n" + "=" * 70)
print("PHẦN 2: FORWARD PASS DIAGNOSTICS")
print("=" * 70)

B, T = 2, cfg.DATASET.seqlen
dummy_joints = torch.randn(B, T, 19, 3).cuda()
dummy_img = torch.randn(B, T, 2048).cuda()
dummy_kp2d = torch.randn(B, T, 19, 2).cuda()

# Hook để capture intermediate values
hyper_inputs = []
hyper_outputs = []
h_tildes = []

def make_hook_in(layer_idx):
    def hook_fn(module, input, output):
        hyper_inputs.append(input[0].detach())
        out_tensor, aux = output
        hyper_outputs.append(out_tensor.detach())
        h_tildes.append(aux['H_tilde'])
    return hook_fn

hooks = []
for i, layer in enumerate(model.spatial_hypers):
    hooks.append(layer.register_forward_hook(make_hook_in(i)))

with torch.no_grad():
    try:
        _ = model(dummy_joints, dummy_img, kp2d=dummy_kp2d, is_train=False)
        forward_ok = True
    except Exception as e:
        print(f"\n🔴 FORWARD PASS FAILED: {e}")
        import traceback
        traceback.print_exc()
        forward_ok = False

for h in hooks:
    h.remove()

if forward_ok:
    print(f"\n✅ Forward pass thành công!")
    
    for i in range(len(hyper_inputs)):
        inp = hyper_inputs[i]
        out = hyper_outputs[i]
        
        print(f"\n--- HyperGCN Layer {i} ---")
        print(f"  Input:  mean={inp.mean():.4f}, std={inp.std():.4f}, shape={tuple(inp.shape)}")
        print(f"  Output: mean={out.mean():.4f}, std={out.std():.4f}, shape={tuple(out.shape)}")
        
        # Feature change
        diff = (out - inp).norm(dim=-1).mean().item()
        inp_norm = inp.norm(dim=-1).mean().item()
        print(f"  ‖output - input‖ / ‖input‖ = {diff:.4f} / {inp_norm:.4f} = {diff/(inp_norm+1e-8):.4f}")
        print(f"  → Relative change: {diff/(inp_norm+1e-8)*100:.1f}%")
        
        if diff / (inp_norm + 1e-8) < 0.01:
            print(f"  ⚠️  HyperGCN gần như KHÔNG thay đổi features (<1%)!")
        elif diff / (inp_norm + 1e-8) > 2.0:
            print(f"  ⚠️  HyperGCN thay đổi features QUÁ MẠNH (>200%)!")
        
        # H_tilde analysis
        H = h_tildes[i]  # (K, 24, 5)
        print(f"  H_tilde: shape={tuple(H.shape)}, min={H.min():.4f}, max={H.max():.4f}")
        print(f"  H_tilde[root=0] sum = {H[:, 0, :].abs().sum().item():.6f} (phải ≈ 0)")
        
        # Per-edge membership
        H_mean = H.mean(dim=0)  # (24, 5)
        edge_names = ['Torso+Head', 'L-Arm', 'R-Arm', 'L-Leg', 'R-Leg']
        for e_idx, e_name in enumerate(edge_names):
            members = (H_mean[:, e_idx] > 0.1).nonzero(as_tuple=True)[0].tolist()
            strength = H_mean[:, e_idx].sum().item()
            print(f"    Edge {e_idx} ({e_name}): active joints={members}, total strength={strength:.3f}")
        
        # Check NaN/Inf
        if torch.isnan(out).any():
            print(f"  🔴 NaN DETECTED trong output!")
        if torch.isinf(out).any():
            print(f"  🔴 Inf DETECTED trong output!")

    # ============================================================
    # 4. GRADIENT FLOW CHECK
    # ============================================================
    print("\n" + "=" * 70)
    print("PHẦN 3: GRADIENT FLOW CHECK")
    print("=" * 70)
    
    model.train()
    dummy_joints2 = torch.randn(B, T, 19, 3).cuda()
    dummy_img2 = torch.randn(B, T, 2048).cuda()
    dummy_kp2d2 = torch.randn(B, T, 19, 2).cuda()
    
    model.zero_grad()
    out = model(dummy_joints2, dummy_img2, kp2d=dummy_kp2d2, is_train=True)
    # out = (residual_joint, spin_pose, spin_shape, smpl_vertices_mid, output)
    mesh_out = out[3]  # smpl_vertices_mid
    loss = mesh_out.sum()
    loss.backward()
    
    print("\nGradient norms per HyperGCN sub-module:")
    for i, layer in enumerate(model.spatial_hypers):
        print(f"\n  Layer {i}:")
        
        # RootChain
        rc_grads = []
        for name, p in layer.root_chain.named_parameters():
            if p.grad is not None:
                rc_grads.append(p.grad.abs().mean().item())
        rc_avg = sum(rc_grads) / len(rc_grads) if rc_grads else 0
        
        # AdaptiveHyper
        ah_grads = []
        for name, p in layer.adaptive_hyper.named_parameters():
            if p.grad is not None:
                ah_grads.append(p.grad.abs().mean().item())
        ah_avg = sum(ah_grads) / len(ah_grads) if ah_grads else 0
        
        print(f"    RootChain avg |grad| = {rc_avg:.6f}  ({'✅' if rc_avg > 1e-7 else '🔴 DEAD'})")
        print(f"    AdaptiveHyper avg |grad| = {ah_avg:.6f}  ({'✅' if ah_avg > 1e-7 else '🔴 DEAD'})")
        
        # Alpha gradients
        if layer.alpha_chain_raw.grad is not None:
            print(f"    α_chain grad = {layer.alpha_chain_raw.grad.item():.6f}")
        if layer.alpha_hyper_raw.grad is not None:
            print(f"    α_hyper grad = {layer.alpha_hyper_raw.grad.item():.6f}")
        
        # Beta gradients
        ah = layer.adaptive_hyper
        if ah.beta0_raw.grad is not None:
            print(f"    β₀ grad={ah.beta0_raw.grad.item():.6f}, β₁ grad={ah.beta1_raw.grad.item():.6f}, β₂ grad={ah.beta2_raw.grad.item():.6f}")

    # node_pe gradient
    if model.node_pe.weight.grad is not None:
        pe_grad = model.node_pe.weight.grad.abs().mean().item()
        print(f"\n  node_pe avg |grad| = {pe_grad:.6f}  ({'✅' if pe_grad > 1e-7 else '🔴 DEAD'})")

# ============================================================
# 5. PARAMETER COUNT
# ============================================================
print("\n" + "=" * 70)
print("PHẦN 4: PARAMETER COUNT")
print("=" * 70)

total = sum(p.numel() for p in model.parameters())
hyper_total = 0
for i, layer in enumerate(model.spatial_hypers):
    n = sum(p.numel() for p in layer.parameters())
    hyper_total += n
    print(f"  HyperGCN Layer {i}: {n:,} params")

node_pe_n = model.node_pe.weight.numel()
print(f"  node_pe: {node_pe_n:,} params")
post_hyper_norm = getattr(model, 'post_hyper_norm', None)
post_hyper_norm_params = sum(p.numel() for p in post_hyper_norm.parameters()) if post_hyper_norm is not None else 0
print(f"  post_hyper_norm: {post_hyper_norm_params:,} params")
print(f"\n  HyperGCN total: {hyper_total + node_pe_n:,} / {total:,} = {(hyper_total+node_pe_n)/total*100:.1f}% of model")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
