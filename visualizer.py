import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

def visualize_hypergcn(model, save_dir='./visualizations'):
    """
    Vẽ ma trận H_tilde của HyperGCN để xem mức độ kết nối của 24 khớp với 5 vùng cơ thể.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    target_model = model.pose_mesh_coevo if hasattr(model, 'pose_mesh_coevo') else model
    
    # target_model.spatial_hypers là ModuleList trong Pose2Mesh
    for i, hyper_layer in enumerate(target_model.spatial_hypers):
        if not hasattr(hyper_layer, 'last_aux'):
            print(f"Chưa có last_aux ở Hyper layer {i}. Chạy forward trước nhé!")
            continue
            
        aux = hyper_layer.last_aux
        H_tilde = aux['H_tilde'][0].cpu() # Lấy mẫu đầu tiên trong batch, shape (24, 5)
        
        plt.figure(figsize=(6, 10))
        sns.heatmap(H_tilde.numpy(), annot=False, cmap='viridis')
        
        title_str = (f'HyperGCN Layer {i} - H_tilde (24 Joints x 5 Hyperedges)\n'
                     f'Alpha Chain: {aux["alpha_chain"]:.3f}, Alpha Hyper: {aux["alpha_hyper"]:.3f}')
        print(f"\n[Thông số Layer {i}] Alpha Chain (Nhánh tĩnh): {aux['alpha_chain']:.4f} | Alpha Hyper (Nhánh động): {aux['alpha_hyper']:.4f}")
        plt.title(title_str)
        plt.xlabel('Hyperedges (0: Torso, 1: LArm, 2: RArm, 3: LLeg, 4: RLeg)')
        plt.ylabel('24 SMPL Joints')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'hypergcn_layer_{i}.png'))
        plt.close()
        print(f"Đã lưu: {os.path.join(save_dir, f'hypergcn_layer_{i}.png')}")


def visualize_cfcer_attention(model, save_dir='./visualizations'):
    """
    Vẽ Attention Map (T x T) của các khối Cross-Attention trong CFCer.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    target_model = model.pose_mesh_coevo if hasattr(model, 'pose_mesh_coevo') else model
    
    # target_model.cfcer là ComplementTemporal chứa att_nets
    for i, att_net_pair in enumerate(target_model.cfcer.att_nets):
        # att_net_pair[0] nhánh 1, att_net_pair[1] nhánh 2
        for j, branch_name in enumerate(['Branch1_M', 'Branch2_K']):
            att_net = att_net_pair[j]
            if not hasattr(att_net, 'last_attn'):
                print(f"Chưa có last_attn ở CFCer Layer {i} - {branch_name}")
                continue
                
            attn = att_net.last_attn[0] # (Heads, T, T)
            attn_mean = attn.mean(dim=0).numpy() # Tính trung bình qua các heads -> (T, T)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(attn_mean, annot=False, cmap='magma')
            plt.title(f'CFCer Layer {i} - {branch_name} Attention Map (T x T)')
            plt.xlabel('Source Frames (Keys)')
            plt.ylabel('Target Frames (Queries)')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'cfcer_layer_{i}_{branch_name}_attn.png'))
            plt.close()
            print(f"Đã lưu: {os.path.join(save_dir, f'cfcer_layer_{i}_{branch_name}_attn.png')}")


def plot_gating_scores(model, save_dir='./visualizations'):
    """
    Vẽ biểu đồ đường (Line chart) xem Gating Score của 2 nhánh thay đổi thế nào qua thời gian (T).
    Lưu ý: EnhanceModule nằm trong ComplementSpatial, nhưng trong code Multimodel.py bạn không dùng ComplementSpatial mà chỉ gọi thẳng ComplementTemporal (CFCer).
    Do đó, hàm này sẽ hữu ích nếu bạn có dùng temp_enhance_module ở file gốc CrossFusionNet.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Nếu mô hình có EnhanceModule (ví dụ nằm trong model.cfcer hoặc đâu đó)
    # Tùy theo nơi bạn đặt EnhanceModule, bạn thay đổi code này:
    # Ở đây tôi làm mẫu với giả định bạn xài ở CrossFusionNet
    target_model = model.pose_mesh_coevo if hasattr(model, 'pose_mesh_coevo') else model
    if hasattr(target_model, 'temp_enhance_module') and hasattr(target_model.temp_enhance_module, 'last_score_rgb'):
        score_rgb = target_model.temp_enhance_module.last_score_rgb[0].mean(dim=-1).numpy() # (T,)
        score_depth = target_model.temp_enhance_module.last_score_depth[0].mean(dim=-1).numpy() # (T,)
        
        print(f"\n[Gating Score Trung bình] RGB/Image: {score_rgb.mean():.4f} | Depth/Motion: {score_depth.mean():.4f}")
        
        T = len(score_rgb)
        
        plt.figure(figsize=(8, 4))
        plt.plot(range(T), score_rgb, label='Score RGB/Image', marker='o')
        plt.plot(range(T), score_depth, label='Score Depth/Motion', marker='s')
        plt.title('Gating Scores Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Confidence Score (0-1)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gating_scores.png'))
        plt.close()
        print(f"Đã lưu: {os.path.join(save_dir, f'gating_scores.png')}")

# Hướng dẫn sử dụng:
if __name__ == "__main__":
    print("--- HƯỚNG DẪN DÙNG ---")
    print("1. Chạy model forward trên tập test như bình thường.")
    print("2. Sau khi chạy model(joints, img_feats, kp2d), gọi 2 hàm này:")
    print("   visualize_hypergcn(model)")
    print("   visualize_cfcer_attention(model)")
    print("3. Kiểm tra kết quả trong thư mục ./visualizations")
