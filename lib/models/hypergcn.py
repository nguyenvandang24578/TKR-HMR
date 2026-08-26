import torch
import torch.nn as nn
from math import sqrt
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

# ============================================================
# 6-part body incidence matrix (N=24 joints, E=6 hyperedges)
# ============================================================
def build_H_init(num_nodes=24, num_edges=6):
    """
    Tầng 1: H_init — "Giữ lại bản ngã" (kiến thức sinh học).
    Ma trận cố định {0,1}, 6 bộ phận cơ thể:
      0: Đầu    (Head, Neck)
      1: Thân   (Pelvis, Spine1-3, L/R Collar)
      2: Tay trái (L_Shoulder, L_Elbow, L_Wrist, L_Hand)
      3: Tay phải (R_Shoulder, R_Elbow, R_Wrist, R_Hand)
      4: Chân trái (L_Hip, L_Knee, L_Ankle, L_Foot)
      5: Chân phải (R_Hip, R_Knee, R_Ankle, R_Foot)
    """
    H = torch.zeros(num_nodes, num_edges)
    # 0: Head — 12(Neck), 15(Head)
    for i in [12, 15]:           H[i, 0] = 1.0
    # 1: Torso — 0(Pelvis), 3(Spine1), 6(Spine2), 9(Spine3), 13(L_Collar), 14(R_Collar)
    for i in [0, 3, 6, 9, 13, 14]: H[i, 1] = 1.0
    # 2: Left Arm — 16(L_Shoulder), 18(L_Elbow), 20(L_Wrist), 22(L_Hand)
    for i in [16, 18, 20, 22]:  H[i, 2] = 1.0
    # 3: Right Arm — 17(R_Shoulder), 19(R_Elbow), 21(R_Wrist), 23(R_Hand)
    for i in [17, 19, 21, 23]:  H[i, 3] = 1.0
    # 4: Left Leg — 1(L_Hip), 4(L_Knee), 7(L_Ankle), 10(L_Foot)
    for i in [1, 4, 7, 10]:     H[i, 4] = 1.0
    # 5: Right Leg — 2(R_Hip), 5(R_Knee), 8(R_Ankle), 11(R_Foot)
    for i in [2, 5, 8, 11]:     H[i, 5] = 1.0
    return H


def compute_G_batched(H, eps=1e-8):
    """
    Tính HGNN Laplacian batched: G = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
    Args:
        H: (B, N, E) — incidence matrix (có thể chứa giá trị thực, không nhất thiết binary)
    Returns:
        G: (B, N, N) — normalized hypergraph adjacency
    """
    # Dv: degree of each vertex = sum over hyperedges
    Dv = H.sum(dim=-1) + eps                      # (B, N)
    Dv_inv_sqrt = Dv.pow(-0.5)                     # (B, N)

    # De: degree of each hyperedge = sum over vertices
    De = H.sum(dim=1) + eps                        # (B, E)
    De_inv = De.pow(-1)                            # (B, E)

    # G = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
    # Step 1: H * De^{-1} → (B, N, E)
    H_De = H * De_inv.unsqueeze(1)                 # broadcast (B,1,E) * (B,N,E)

    # Step 2: (H * De^{-1}) @ H^T → (B, N, N)
    G = torch.bmm(H_De, H.transpose(1, 2))        # (B, N, N)

    # Step 3: Dv^{-1/2} * G * Dv^{-1/2} (symmetric normalization)
    G = Dv_inv_sqrt.unsqueeze(-1) * G * Dv_inv_sqrt.unsqueeze(1)  # (B, N, N)

    return G


class HYPERGC(nn.Module):
    """
    Adaptive Hypergraph Convolution với 3 tầng tri thức:
      - H_init (Tầng 1): Cấu trúc sinh học cố định — 6 bộ phận cơ thể
      - M      (Tầng 2): Learnable global prior — quy luật chung của dataset
      - S      (Tầng 3): Dynamic per-sample — ứng biến theo từng video/hành động

    Song song với 1 nhánh spatial (kinematic tree 24×24) giữ nguyên.
    """
    def __init__(self, in_channels, out_channels, vertex_nums=24, num_edges=6):
        super(HYPERGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = vertex_nums
        self.num_edges = num_edges

        # === Spatial branch (giữ nguyên — kinematic tree vật lý) ===
        self.adj = self._init_spatial_adj()
        self.V_spatial = nn.Linear(in_channels, out_channels)
        self.alpha_spatial = nn.Parameter(torch.ones(1))

        # === Adaptive Hyper branch ===
        # Tầng 1: H_init — cố định, register_buffer (không train)
        H_init = build_H_init(vertex_nums, num_edges)
        self.register_buffer('H_init', H_init)  # (N, E)

        # Tầng 2: M — learnable global prior, khởi tạo từ H_init
        self.M = nn.Parameter(H_init.clone())   # (N, E)

        # Tầng 3: S — dynamic, tính từ features
        self.phi1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.phi2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable mixing weights (beta0 lớn, beta1/beta2 nhỏ → ưu tiên H_init ban đầu)
        self.beta0 = nn.Parameter(torch.tensor(1.0))   # H_init weight
        self.beta1 = nn.Parameter(torch.tensor(0.1))   # M weight
        self.beta2 = nn.Parameter(torch.tensor(0.1))   # S weight

        self.alpha_hyper = nn.Parameter(torch.ones(1))
        self.conv_hyper = nn.Conv2d(in_channels, out_channels, 1)

        # === Shared output ===
        self.U = nn.Linear(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)
        self.relu = nn.ReLU()

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            connected_nodes = CONNECTIONS.get(i, [])
            for j in connected_nodes:
                adj[i, j] = 1
        return adj

    @staticmethod
    def normalize_digraph(adj):
        b, n, c = adj.shape
        node_degrees = adj.detach().sum(dim=-1)
        deg_inv_sqrt = node_degrees ** -0.5
        norm_deg_matrix = torch.eye(n).to(adj.device)
        norm_deg_matrix = norm_deg_matrix.view(1, n, n) * deg_inv_sqrt.view(b, n, 1)
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)
        return norm_adj

    def _compute_S(self, x):
        """
        Tầng 3: S — Dynamic incidence matrix, tính từ features.
        x: (B, C, T, V)
        Returns: S (B, N, E) — soft assignment mỗi joint → 6 bộ phận
        """
        b, c, t, v = x.shape
        q = rearrange(self.phi1(x), 'b c t v -> b v (c t)')   # (B, N, CT)
        k = rearrange(self.phi2(x), 'b c t v -> b (c t) v')   # (B, CT, N)

        # Pairwise similarity (attention-like)
        sim = torch.bmm(q, k) / sqrt(c * t)                   # (B, N, N)

        # Project similarity → hyperedge space qua H_init
        H_init_expanded = self.H_init.unsqueeze(0).expand(b, -1, -1)  # (B, N, E)
        S = torch.bmm(sim, H_init_expanded)                    # (B, N, E)

        # Softmax theo E: mỗi joint phân bố xác suất thuộc về 6 bộ phận
        S = torch.softmax(S, dim=-1)                           # (B, N, E)

        return S

    def forward(self, x):
        """
        Input:  x — (B, C, T, 24)
        Output: out — (B, C, T, 24), None, adj_dict
        """
        b, c, t, v = x.shape

        # ============================================================
        # Nhánh 1: Spatial (kinematic tree — giữ nguyên)
        # ============================================================
        x_flat = rearrange(x, 'b c t v -> (b t) v c')         # (BT, N, C)
        adj = self.adj.to(x.device).repeat(b * t, 1, 1)       # (BT, N, N)
        norm_adj = self.normalize_digraph(adj)
        aggregate_spatial = norm_adj @ self.V_spatial(x_flat)  # (BT, N, C)

        # ============================================================
        # Nhánh 2: Adaptive Hyper (H_init + M + S)
        # ============================================================
        # Tính S dynamic từ features
        S = self._compute_S(x)                                 # (B, N, E)

        # Ghép 3 tầng: H̃ = β₀·H_init + β₁·M + β₂·S
        H_init_expanded = self.H_init.unsqueeze(0).expand(b, -1, -1)  # (B, N, E)
        M_expanded = self.M.unsqueeze(0).expand(b, -1, -1)            # (B, N, E)

        H_tilde = (self.beta0 * H_init_expanded
                  + self.beta1 * M_expanded
                  + self.beta2 * S)                            # (B, N, E)

        # Tính HGNN Laplacian batched: G = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
        G_hyper = compute_G_batched(H_tilde)                   # (B, N, N)

        # Hypergraph convolution
        x_hyper = rearrange(x, 'b c t v -> b (c t) v')        # (B, CT, V)
        x_hyper = torch.bmm(x_hyper, G_hyper)                  # (B, CT, V)
        x_hyper = rearrange(x_hyper, 'b (c t) v -> b c t v', c=c)
        aggregate_hyper = self.conv_hyper(x_hyper)             # (B, C, T, V)
        aggregate_hyper = rearrange(aggregate_hyper, 'b c t v -> (b t) v c')  # (BT, N, C)

        # ============================================================
        # Aggregate: spatial + adaptive hyper
        # ============================================================
        aggregate = (self.alpha_spatial * aggregate_spatial
                   + self.alpha_hyper * aggregate_hyper)

        if self.in_channels == self.out_channels:
            out = self.relu(x_flat + self.batch_norm(aggregate + self.U(x_flat)))
        else:
            out = self.relu(self.batch_norm(aggregate + self.U(x_flat)))

        out = rearrange(out, '(b t) v c -> b c t v', b=b, t=t)

        adj_dict = {
            'H_tilde': H_tilde.detach(),   # (B, N, E) — có thể visualize
            'G_hyper': G_hyper.detach(),    # (B, N, N)
            'spatial': self.adj,
            'beta': [self.beta0.item(), self.beta1.item(), self.beta2.item()],
        }
        return out, None, adj_dict
