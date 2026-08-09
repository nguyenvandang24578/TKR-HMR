import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# Spatial connections for physical adj (kinematic tree of SMPL)
CONNECTIONS = {
    0: [1, 2, 3],
    1: [0, 4],
    2: [0, 5],
    3: [0, 6],
    4: [1, 7],
    5: [2, 8],
    6: [3, 9],
    7: [4, 10],
    8: [5, 11],
    9: [6, 12, 13, 14],
    10: [7],
    11: [8],
    12: [9, 15],
    13: [9, 16],
    14: [9, 17],
    15: [12],
    16: [13, 18],
    17: [14, 19],
    18: [16, 20],
    19: [17, 21],
    20: [18, 22],
    21: [19, 23],
    22: [20],
    23: [21]
}

class HYPERGC(nn.Module):
    def __init__(self, in_channels, out_channels, vertex_nums=24):
        super(HYPERGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = vertex_nums
        
        self.U = nn.Linear(self.in_channels, self.out_channels)
        self.V = nn.Linear(self.in_channels, self.out_channels)
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)
        
        self.relu = nn.ReLU()
        
        # Initialize the hyper-graph adjacency matrices
        self.adj = self._init_spatial_adj()
        
        # Body Scale (5 Parts)
        G_body = self._init_body_adj()
        self.b_adj = nn.Parameter(torch.from_numpy(G_body.astype(np.float32)), requires_grad=False)
        self.conv_body = nn.Conv2d(self.in_channels, self.out_channels, 1)
        self.alpha3 = nn.Parameter(torch.ones(1))
        
        # Part Scale (10 Parts)
        G_part = self._init_part_adj()
        self.p_adj = nn.Parameter(torch.from_numpy(G_part.astype(np.float32)), requires_grad=False)
        self.conv_part = nn.Conv2d(self.in_channels, self.out_channels, 1)
        self.alpha2 = nn.Parameter(torch.ones(1))
        
        # Spatial Scale
        self.alpha1 = nn.Parameter(torch.ones(1))

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            connected_nodes = CONNECTIONS.get(i, [])
            for j in connected_nodes:
                adj[i, j] = 1
        return adj
        
    def _init_body_adj(self):
        # 5 Body parts
        H = np.zeros((self.num_nodes, 5))
        # 1. Torso: 0, 3, 6, 9, 12, 13, 14, 15
        for i in [0, 3, 6, 9, 12, 13, 14, 15]: H[i][0] = 1
        # 2. Left Arm: 16, 18, 20, 22
        for i in [16, 18, 20, 22]: H[i][1] = 1
        # 3. Right Arm: 17, 19, 21, 23
        for i in [17, 19, 21, 23]: H[i][2] = 1
        # 4. Left Leg: 1, 4, 7, 10
        for i in [1, 4, 7, 10]: H[i][3] = 1
        # 5. Right Leg: 2, 5, 8, 11
        for i in [2, 5, 8, 11]: H[i][4] = 1
        return self._compute_G(H)

    def _init_part_adj(self):
        # 10 Parts
        H = np.zeros((self.num_nodes, 10))
        # 1. Pelvis, Spine1, Spine2 (0, 3, 6)
        for i in [0, 3, 6]: H[i][0] = 1
        # 2. Spine3, Neck, Head (9, 12, 15)
        for i in [9, 12, 15]: H[i][1] = 1
        # 3. L_Collar, L_Shoulder (13, 16)
        for i in [13, 16]: H[i][2] = 1
        # 4. R_Collar, R_Shoulder (14, 17)
        for i in [14, 17]: H[i][3] = 1
        # 5. L_Elbow, L_Wrist, L_Hand (18, 20, 22)
        for i in [18, 20, 22]: H[i][4] = 1
        # 6. R_Elbow, R_Wrist, R_Hand (19, 21, 23)
        for i in [19, 21, 23]: H[i][5] = 1
        # 7. L_Hip, L_Knee (1, 4)
        for i in [1, 4]: H[i][6] = 1
        # 8. R_Hip, R_Knee (2, 5)
        for i in [2, 5]: H[i][7] = 1
        # 9. L_Ankle, L_Foot (7, 10)
        for i in [7, 10]: H[i][8] = 1
        # 10. R_Ankle, R_Foot (8, 11)
        for i in [8, 11]: H[i][9] = 1
        return self._compute_G(H)

    def _compute_G(self, H):
        H = np.array(H)
        n_edge = H.shape[1]
        W = np.ones(n_edge)
        DV = np.sum(H * W, axis=1)
        DE = np.sum(H, axis=0)
        
        invDE = np.asmatrix(np.diag(np.power(DE + 1e-8, -1)))
        DV2 = np.asmatrix(np.diag(np.power(DV + 1e-8, -0.5)))
        W = np.asmatrix(np.diag(W))
        H = np.asmatrix(H)
        HT = H.T
        
        G = DV2 * H * W * invDE * HT * DV2
        return np.array(G)
        
    def norm(self, A):
        D_list = torch.sum(A, 0).view(1, self.num_nodes)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = torch.eye(self.num_nodes).to(device=A.device) * D_list_12
        A = torch.matmul(A, D_12)
        return A
        
    @staticmethod
    def normalize_digraph(adj):
        b, n, c = adj.shape 
        node_degrees = adj.detach().sum(dim=-1)
        deg_inv_sqrt = node_degrees ** -0.5
        norm_deg_matrix = torch.eye(n).to(adj.device)
        norm_deg_matrix = norm_deg_matrix.view(1, n, n) * deg_inv_sqrt.view(b, n, 1)
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)
        return norm_adj
        
    def change_adj_device_to_cuda(self, adj):
        dev = self.V.weight.get_device()
        if dev >= 0 and adj.get_device() < 0:
            adj = adj.to(dev)
        return adj

    def forward(self, x):
        # Input shape: (B, C, T, 24)
        b, c, t, v = x.shape
        
        # Part features
        G_part = self.p_adj.to(x.device)
        x_part = rearrange(x, 'b c t v -> b (c t) v')
        x_part = torch.matmul(x_part, self.norm(G_part))
        x_part = rearrange(x_part, 'b (c t) v -> b c t v', c=c)
        aggregate2 = self.conv_part(x_part) # b c t v
        aggregate2 = rearrange(aggregate2, 'b c t v -> (b t) v c')
        
        # Body features
        G_body = self.b_adj.to(x.device)
        x_body = rearrange(x, 'b c t v -> b (c t) v')
        x_body = torch.matmul(x_body, self.norm(G_body))
        x_body = rearrange(x_body, 'b (c t) v -> b c t v', c=c)
        aggregate3 = self.conv_body(x_body) # b c t v
        aggregate3 = rearrange(aggregate3, 'b c t v -> (b t) v c')
        
        # Spatial joint features
        x_flat = rearrange(x, 'b c t v -> (b t) v c')
        adj = self.change_adj_device_to_cuda(self.adj)
        adj = adj.repeat(b * t, 1, 1)
        norm_adj = self.normalize_digraph(adj)
        aggregate1 = norm_adj @ self.V(x_flat)
        
        aggregate = aggregate1 * self.alpha1 + aggregate2 * self.alpha2 + aggregate3 * self.alpha3
        
        if self.in_channels == self.out_channels:
            out = self.relu(x_flat + self.batch_norm(aggregate + self.U(x_flat)))
        else:
            out = self.relu(self.batch_norm(aggregate + self.U(x_flat)))
            
        out = rearrange(out, '(b t) v c -> b c t v', b=b, t=t)
        
        # Return format matching Multimodel.py: `hyper_out, _, G_scaled = self.spatial_hyper(...)`
        adj_dict = {
            'part': G_part,
            'body': G_body,
            'spatial': self.adj
        }
        return out, None, adj_dict
