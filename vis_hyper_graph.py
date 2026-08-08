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

def vis_heatmap(adj_matrix, save_path="hyper_heatmap.png"):
    # adj_matrix shape expected: (V, V) where V = 24 + 3 = 27
    V = adj_matrix.shape[0]
    
    joint_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
        'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 
        'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
        'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]
    # Add virtual joints
    for i in range(V - 24):
        joint_names.append(f'Virtual_{i+1}')
        
    plt.figure(figsize=(14, 12))
    sns.heatmap(adj_matrix, 
                xticklabels=joint_names, 
                yticklabels=joint_names, 
                cmap='magma', 
                cbar_kws={'label': 'Connection Strength (alpha * G)'})
    
    plt.title("Hyper-Graph Adaptive Adjacency Matrix", fontsize=18)
    plt.xlabel("Target Node", fontsize=14)
    plt.ylabel("Source Node", fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved Heatmap to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize Hyper-GCN Adjacency Matrix')
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
    
    test_loader = dataloaders[0] # Use the first test dataset
    
    print("==> Running inference on 1 batch...")
    with torch.no_grad():
        for i, (inputs, targets, meta) in enumerate(test_loader):
            input_pose = inputs['pose2d'].cuda()
            input_feat = inputs['img_feature'].cuda()
            
            # Forward pass
            # Output of Multimodel is: residual_joint, final_pose, pred_mean_shape, smpl_vertices_mid, output_dict
            *_, output_dict = model(input_pose, input_feat, is_train=False)
            
            # Extract hyper_adj
            hyper_adj = output_dict[-1].get('hyper_adj', None)
            
            if hyper_adj is None:
                print("Error: 'hyper_adj' not found in model output. Check if G_scaled is returned properly.")
                break
                
            # hyper_adj shape: [Batch, V, V] (assuming A is returned like that, or [num_subset, V, V])
            # Let's check the shape in hypergcn.py: A shape is (num_subset, V, V) or (N, num_subset, V, V)?
            # In HYPERGC, t_x is (N, C, V), distance_x is (N, num_subset, V, V), H is (N, num_subset, V, V)
            # wait, H is (N, num_subset, V, V), W is (N, num_subset, V, V), G is (N, num_subset, V, V)
            # alpha is (1,), G_scaled is (N, num_subset, V, V)
            
            print(f"Extracted hyper_adj shape: {hyper_adj.shape}")
            
            # We just take the first sample in the batch
            if len(hyper_adj.shape) == 4: # (N, num_subset, V, V)
                matrix_to_plot = hyper_adj[0].mean(dim=0).cpu().numpy() # mean over num_subsets
            elif len(hyper_adj.shape) == 3: # (N, V, V) or (num_subset, V, V)
                # If it's (num_subset, V, V) because the user returned self.A or something
                matrix_to_plot = hyper_adj[0].cpu().numpy()
            else:
                matrix_to_plot = hyper_adj.cpu().numpy()
                
            vis_heatmap(matrix_to_plot, save_path="hyper_heatmap.png")
            break # Just need 1 batch

if __name__ == '__main__':
    main()
