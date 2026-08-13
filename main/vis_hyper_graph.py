import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('./lib')
from core.config import cfg, update_config
from core.base import prepare_network

def vis_heatmap(adj_matrix, title="Hyper-Graph Adjacency Matrix", save_path="hyper_heatmap.png"):
    V = adj_matrix.shape[0]
    
    joint_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
        'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 
        'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
        'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]
        
    plt.figure(figsize=(14, 12))
    sns.heatmap(adj_matrix, 
                xticklabels=joint_names, 
                yticklabels=joint_names, 
                cmap='magma', 
                cbar_kws={'label': 'Connection Strength'})
    
    plt.title(title, fontsize=18)
    plt.xlabel("Target Node", fontsize=14)
    plt.ylabel("Source Node", fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved Heatmap to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize Anatomical Hyper-GCN')
    parser.add_argument('--cfg', type=str, required=True, help='experiment configure file name')
    parser.add_argument('--load_dir', type=str, required=True, help='checkpoint path')
    args = parser.parse_args()
    
    update_config(args.cfg)
    
    print("==> Loading model and dataset...")
    args.resume_training = False
    dataloaders, dataset_list, model, _, _, _, _, _ = prepare_network(args, load_dir=args.load_dir, is_train=False)
    
    if not model:
        print("Failed to load model!")
        return
        
    model = model.cuda()
    model.eval()
    
    test_loader = dataloaders[0]
    
    print("==> Running inference on 1 batch...")
    with torch.no_grad():
        for i, (inputs, targets, meta) in enumerate(test_loader):
            input_pose = inputs['pose2d'].cuda()
            input_feat = inputs['img_feature'].cuda()
            
            *_, output_dict = model(input_pose, input_feat, is_train=False)
            
            hyper_adj = output_dict[-1].get('hyper_adj', None)
            
            if hyper_adj is None or not isinstance(hyper_adj, dict):
                print("Error: 'hyper_adj' not found or is not a dict. Check HYPERGC return format.")
                break
                
            # Plot the 3 anatomical scales
            for scale_name, matrix in hyper_adj.items():
                if matrix is None:
                    continue
                # matrix is a PyTorch tensor (or numpy array)
                if isinstance(matrix, torch.Tensor):
                    matrix_to_plot = matrix.cpu().numpy()
                else:
                    matrix_to_plot = np.array(matrix)
                
                # Check shape, if it has batch size (b, v, v) take the first one
                if len(matrix_to_plot.shape) == 3:
                    matrix_to_plot = matrix_to_plot[0]
                    
                vis_heatmap(
                    matrix_to_plot, 
                    title=f"Anatomical Hyper-Graph: {scale_name.capitalize()} Scale", 
                    save_path=f"hyper_{scale_name}_heatmap.png"
                )
            break

if __name__ == '__main__':
    main()
