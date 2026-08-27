import os
import sys
import argparse
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lib'))

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

# Cài đặt style chuẩn bài báo khoa học với font fallback an toàn cho Linux
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'serif'],
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

# 17 khớp Human3.6M chuẩn xác theo đúng dataset.py / base.py
H36M_JOINTS = [
    'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 
    'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 
    'R_Shoulder', 'R_Elbow', 'R_Wrist'
]

# Ánh xạ tên đầy đủ -> tên trong danh sách
JOINT_ALIASES = {
    'right wrist': 'R_Wrist', 'r wrist': 'R_Wrist', 'r_wrist': 'R_Wrist',
    'left wrist': 'L_Wrist', 'l wrist': 'L_Wrist', 'l_wrist': 'L_Wrist',
    'right ankle': 'R_Ankle', 'r ankle': 'R_Ankle', 'r_ankle': 'R_Ankle',
    'left ankle': 'L_Ankle', 'l ankle': 'L_Ankle', 'l_ankle': 'L_Ankle',
    'right knee': 'R_Knee', 'r knee': 'R_Knee', 'r_knee': 'R_Knee',
    'left knee': 'L_Knee', 'l knee': 'L_Knee', 'l_knee': 'L_Knee',
    'right shoulder': 'R_Shoulder', 'r shoulder': 'R_Shoulder',
    'left shoulder': 'L_Shoulder', 'l shoulder': 'L_Shoulder',
    'right elbow': 'R_Elbow', 'r elbow': 'R_Elbow',
    'left elbow': 'L_Elbow', 'l elbow': 'L_Elbow',
    'head': 'Head', 'neck': 'Neck', 'pelvis': 'Pelvis'
}

def load_j_regressor():
    """Tải ma trận J_regressor (17, 6890) chuẩn như trong base.py / smpl.py"""
    paths = [
        'data/Human36M/J_regressor_h36m_correct.npy',
        'data/base_data/J_regressor_h36m.npy',
        'data/base_data/J_regressor_extra.npy'
    ]
    for p in paths:
        full_p = ROOT / p
        if full_p.exists():
            reg = np.load(str(full_p)).astype(np.float32)
            print(f"[INFO] Đã tải J_regressor từ: {full_p} (shape: {reg.shape})")
            return reg
            
    print("[WARNING] Không tìm thấy file J_regressor_h36m_correct.npy, kiểm tra lại thư mục data/")
    return None

def compute_acceleration(joints):
    """
    Tính gia tốc chuyển động: a_t = J_{t+1} - 2*J_t + J_{t-1}
    Khớp 100% với hàm compute_error_accel trong lib/eval_utils.py
    """
    return joints[2:] - 2 * joints[1:-1] + joints[:-2]

def extract_joints_from_pkl(pkl_path, j_regressor=None):
    """
    Trích xuất chuỗi khớp 3D (T, 17, 3) tính bằng mm, y hệt dòng 194 trong base.py:
    pred_pose = torch.matmul(self.J_regressor, pred_mesh * 1000)
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Không tìm thấy file: {pkl_path}")
    
    data = joblib.load(pkl_path)
    first_person_id = list(data.keys())[0]
    person_data = data[first_person_id]
    
    pred_mesh = person_data['mesh'] # Shape: (T, 6890, 3) tính bằng mét
    T = pred_mesh.shape[0]
    
    # Đổi sang mm giống như base.py dòng 194
    pred_mesh_mm = pred_mesh * 1000.0
    
    if j_regressor is not None:
        # Nhân ma trận: (17, 6890) @ (T, 6890, 3) -> (T, 17, 3)
        joints_3d = np.matmul(j_regressor, pred_mesh_mm)
    else:
        sample_indices = np.linspace(0, 6889, 17).astype(int)
        joints_3d = pred_mesh_mm[:, sample_indices, :]
        
    return joints_3d, T

def plot_motion_trajectory(pred_joints, baseline_joints=None, 
                           joint_idx=16, joint_name="Right Wrist", save_path="output/video_trajectory.png"):
    """
    Vẽ 4 biểu đồ trực quan:
    1. Quỹ đạo trục Z (chiều sâu) theo thời gian
    2. Quỹ đạo trục X và Y theo thời gian
    3. Đồ thị gia tốc / độ rung giật (Acceleration Jitter)
    4. Quỹ đạo 3D trong không gian
    """
    T = pred_joints.shape[0]
    time_axis = np.arange(T)
    
    p_traj = pred_joints[:, joint_idx, :] if pred_joints.ndim == 3 else pred_joints
    base_traj = baseline_joints[:, joint_idx, :] if (baseline_joints is not None and baseline_joints.ndim == 3) else baseline_joints

    fig = plt.figure(figsize=(15, 9))
    
    # (a) Trục Z
    ax1 = fig.add_subplot(2, 2, 1)
    if base_traj is not None:
        ax1.plot(time_axis, base_traj[:, 2], label='Baseline (w/o Mamba)', color='#D95F02', linewidth=1.5, alpha=0.8)
    ax1.plot(time_axis, p_traj[:, 2], label='Ours (Mamba - Smooth)', color='#1B9E77', linewidth=2.0)
    ax1.set_title(f'(a) {joint_name} - Z-axis Trajectory', fontweight='bold')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Z Position (mm)')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # (b) Trục X & Y
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_axis, p_traj[:, 0], label='X-axis (Ours)', color='#7570B3', linewidth=1.8)
    ax2.plot(time_axis, p_traj[:, 1], label='Y-axis (Ours)', color='#E7298A', linewidth=1.8)
    ax2.set_title(f'(b) {joint_name} - X and Y Trajectories', fontweight='bold')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Position (mm)')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # (c) Đồ thị Gia tốc rung giật (Acceleration)
    ax3 = fig.add_subplot(2, 2, 3)
    p_acc = np.linalg.norm(compute_acceleration(p_traj), axis=-1)
    acc_time = np.arange(1, T - 1)
    
    if base_traj is not None:
        base_acc = np.linalg.norm(compute_acceleration(base_traj), axis=-1)
        ax3.plot(acc_time, base_acc, label=f'Baseline (Avg Accel: {np.mean(base_acc):.1f})', color='#D95F02', linewidth=1.5, alpha=0.8)
    ax3.plot(acc_time, p_acc, label=f'Ours Mamba (Avg Accel: {np.mean(p_acc):.1f})', color='#1B9E77', linewidth=2.0)
    
    ax3.set_title(f'(c) {joint_name} - Acceleration (Jitter / Smoothness)', fontweight='bold')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Acceleration (mm/s²)')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # (d) Không gian 3D
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    if base_traj is not None:
        ax4.plot(base_traj[:, 0], base_traj[:, 1], base_traj[:, 2], label='Baseline', color='#D95F02', alpha=0.7, linewidth=1.5)
    ax4.plot(p_traj[:, 0], p_traj[:, 1], p_traj[:, 2], label='Ours Mamba', color='#1B9E77', linewidth=2.2)
    ax4.scatter(p_traj[0, 0], p_traj[0, 1], p_traj[0, 2], color='blue', s=60, label='Start')
    ax4.scatter(p_traj[-1, 0], p_traj[-1, 1], p_traj[-1, 2], color='red', s=60, label='End')

    ax4.set_title(f'(d) 3D Space Trajectory Trail', fontweight='bold')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_zlabel('Z (mm)')
    ax4.legend(loc='best', framealpha=0.8, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"\n[THÀNH CÔNG] Đã xuất biểu đồ:")
    print(f"  -> Ảnh PNG: {save_path}")
    print(f"  -> Vector PDF: {pdf_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Motion Trajectory from Video Demo output PKL')
    parser.add_argument('--pkl', type=str, default='', help='Đường dẫn file ARTS_output.pkl (ví dụ: output/demo_output/dance/ARTS_output.pkl)')
    parser.add_argument('--base_pkl', type=str, default='', help='(Tùy chọn) File pkl của baseline cũ để so sánh')
    parser.add_argument('--joint', type=str, default='Right Wrist', help='Tên khớp (Right Wrist, Left Wrist, Right Ankle, Head, etc.)')
    parser.add_argument('--out', type=str, default='output/video_trajectory.png', help='Đường dẫn file ảnh đầu ra')
    args = parser.parse_args()

    # 1. Tìm index khớp chuẩn Human3.6M
    joint_query = args.joint.strip()
    normalized_name = JOINT_ALIASES.get(joint_query.lower(), joint_query)
    
    if normalized_name in H36M_JOINTS:
        joint_idx = H36M_JOINTS.index(normalized_name)
        display_name = normalized_name
    else:
        print(f"[WARN] Khớp '{args.joint}' không có trong 17 khớp H36M, mặc định chọn R_Wrist (index 16)")
        joint_idx = 16
        display_name = 'R_Wrist'

    # 2. Tải J_regressor chuẩn
    j_regressor = load_j_regressor()

    # 3. Trích xuất và vẽ
    if args.pkl and os.path.exists(args.pkl):
        pred_joints, T = extract_joints_from_pkl(args.pkl, j_regressor)
        print(f"[INFO] Đã trích xuất chuỗi {T} frames thành công.")
        
        base_joints = None
        if args.base_pkl and os.path.exists(args.base_pkl):
            base_joints, _ = extract_joints_from_pkl(args.base_pkl, j_regressor)
            
        plot_motion_trajectory(pred_joints, base_joints, joint_idx=joint_idx, joint_name=display_name, save_path=args.out)
    else:
        demo_dir = Path(ROOT / 'output' / 'demo_output')
        found_pkls = list(demo_dir.glob('**/ARTS_output.pkl')) if demo_dir.exists() else []
        if found_pkls:
            latest_pkl = str(found_pkls[0])
            print(f"[AUTO] Tự động chọn file output: {latest_pkl}")
            pred_joints, T = extract_joints_from_pkl(latest_pkl, j_regressor)
            plot_motion_trajectory(pred_joints, None, joint_idx=joint_idx, joint_name=display_name, save_path=args.out)
        else:
            print("[INFO] Vui lòng truyền đường dẫn file pkl:")
            print("  python main/visualize_trajectory.py --pkl output/demo_output/<video>/ARTS_output.pkl")
