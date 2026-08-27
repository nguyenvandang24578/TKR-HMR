import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class HYPERGC(nn.Module):
    """
    Original HYPERGC block from Hyper-GCN/model/hypergcn_base.py
    Takes in (N, C, T, V)
    """
    def __init__(self, in_channels, out_channels, vertex_nums, virtual_num, A, hyper=True, num_subset=3, rel_reduction=4):
        super(HYPERGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vertex_nums = vertex_nums
        self.virtual_num = virtual_num
        self.rel_reduction = rel_reduction
        self.num_subset = num_subset
        self.hyper = hyper
        mid_in_channels = in_channels // num_subset
        mid_out_channels = out_channels // num_subset
        self.mid_in_channels = mid_in_channels
        self.mid_out_channels = mid_out_channels

        if self.hyper:
            self.hidden_channels = max(1, mid_in_channels // rel_reduction)
            self.to_V = nn.Conv1d(in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset)
            self.to_W = nn.Sequential(
                nn.Conv1d(in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset),
                nn.LeakyReLU(),
                nn.Conv1d(num_subset * self.hidden_channels, num_subset, kernel_size=1),
                nn.Tanh()
            )
            self.hyper_joint = nn.Parameter(torch.randn(1, in_channels).repeat(self.virtual_num, 1))
            self.alpha = nn.Parameter(torch.ones(1))
            self.softmax = nn.Softmax(dim=-1)

        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=num_subset)
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.edge_importance = nn.Parameter(torch.ones(A.shape))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        if self.hyper:
            conv_init(self.to_V)
            conv_init(self.to_W[0])
            conv_init(self.to_W[2])
        conv_init(self.conv_d)

    def hyper_norm(self, H, W):
        w = torch.diag_embed(W)
        norm_w = torch.norm(H, 1, dim=2, keepdim=True) + 1e-8
        w_ = w / norm_w

        H_w = H @ w
        norm_v = torch.norm(H_w, 1, dim=3, keepdim=True) + 1e-8
        h_ = H_w / norm_v
        A = h_ @ w_ @ H.transpose(3, 2)
        return A

    def a_norm(self, A):
        d_r = torch.norm(A, 1, dim=2, keepdim=True) + 1e-8
        return A / d_r

    def forward(self, x):
        N, C, T, V = x.size()

        h_x = self.hyper_joint
        h_x = (h_x.T).unsqueeze(1)
        x = torch.cat([x, h_x.repeat(N, 1, T, 1)], dim=-1)
        V += self.virtual_num
        A = self.PA.cuda(x.get_device())
        A = self.edge_importance * A
        A = self.a_norm(A)

        if self.hyper:
            t_x = x.mean(2)

            v_x = self.to_V(t_x)

            dis_v_x = v_x.view(N, self.num_subset, self.hidden_channels, V)
            dis_v_x = dis_v_x.permute(0, 1, 3, 2).contiguous()
            distance_x = torch.cdist(dis_v_x, dis_v_x)
            H = torch.zeros_like(distance_x)

            # topk is min distance (nearest neighbors)
            topk_v, topk_indices = torch.topk(distance_x, min(9, V), largest=False)
            topk_v = self.softmax(-topk_v)
            H = torch.scatter(H, 3, topk_indices, topk_v)

            W = self.to_W(t_x)

            G = self.hyper_norm(H, W)
            alpha = self.alpha
            alpha = self.relu(alpha)
            A = A + alpha * G

        d_x = self.conv_d(x)
        d_x = d_x.view(N, self.num_subset, self.mid_out_channels, T, V)
        y = torch.einsum('nkuv,nkctv->nkctu', A, d_x).contiguous()
        y = y.view(N, self.out_channels, T, V)

        x = x[..., :self.vertex_nums]
        y = y[..., :self.vertex_nums]

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y, self.hyper_joint


# ============================================================
# Build A_19 spatial graph for 19 joints (COCO + Pelvis + Neck)
# 19 joints: 
# 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LSho, 6:RSho, 7:LElb, 8:RElb, 9:LWri, 10:RWri
# 11:LHip, 12:RHip, 13:LKne, 14:RKne, 15:LAnk, 16:RAnk, 17:Pelvis, 18:Neck
# ============================================================
def get_spatial_graph_19():
    num_node = 19
    self_link = [(i, i) for i in range(num_node)]
    
    # Parent (closer to root) -> Child (further from root)
    inward_ori_index = [
        (17, 11), (17, 12), (17, 18),   # Pelvis -> Hips, Neck
        (11, 13), (13, 15),             # L Leg
        (12, 14), (14, 16),             # R Leg
        (18, 5), (18, 6), (18, 0),      # Neck -> Shoulders, Nose
        (5, 7), (7, 9),                 # L Arm
        (6, 8), (8, 10),                # R Arm
        (0, 1), (0, 2),                 # Nose -> Eyes
        (1, 3), (2, 4)                  # Eyes -> Ears
    ]
    
    inward = [(i, j) for (i, j) in inward_ori_index]
    outward = [(j, i) for (i, j) in inward_ori_index]
    
    def edge2mat(link, num_node):
        A = np.zeros((num_node, num_node))
        for i, j in link:
            A[j, i] = 1
        return A
        
    I = edge2mat(self_link, num_node)
    In = edge2mat(inward, num_node)
    Out = edge2mat(outward, num_node)
    
    def normalize_digraph(A):
        Dl = np.sum(A, 0)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD
        
    A = np.stack((I, normalize_digraph(In), normalize_digraph(Out)))
    return A

A_19 = get_spatial_graph_19()
