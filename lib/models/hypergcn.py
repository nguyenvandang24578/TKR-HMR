import torch
import torch.nn as nn
import numpy as np

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def build_smpl_hypergraph_adjacency(virtual_num=3, num_subset=8):
    """
    Builds the adjacency matrix for SMPL 24 joints with virtual joints.
    """
    num_node = 24
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    
    self_link = [(i, i) for i in range(num_node)]
    inward = []
    outward = []
    for i, p in enumerate(parents):
        if p != -1:
            inward.append((i, p))
            outward.append((p, i))
            
    I = np.pad(edge2mat(self_link, num_node), ((0, virtual_num), (0, virtual_num)))
    In = np.pad(edge2mat(inward, num_node), ((0, virtual_num), (0, virtual_num)))
    Out = np.pad(edge2mat(outward, num_node), ((0, virtual_num), (0, virtual_num)))
    
    # Fully connect virtual nodes to all physical nodes and to themselves
    for i in range(virtual_num):
        I[num_node + i, num_node + i] = 1
        In[:num_node, num_node + i] = 1
        Out[num_node + i, :num_node] = 1
        
    A = I + In + Out
    # Replicate for num_subset
    A = np.repeat(A[np.newaxis, :], num_subset, axis=0)
    return A

class HYPERGC(nn.Module):
    def __init__(self, in_channels, out_channels, vertex_nums=24, virtual_num=3, A=None, hyper=True, num_subset=8, rel_reduction=4):
        super(HYPERGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vertex_nums = vertex_nums
        self.virtual_num = virtual_num
        self.rel_reduction = rel_reduction
        self.num_subset = num_subset
        self.hyper = hyper
        
        if A is None:
            A = build_smpl_hypergraph_adjacency(virtual_num, num_subset)
            
        mid_in_channels = in_channels // num_subset
        mid_out_channels = out_channels // num_subset
        self.mid_in_channels = mid_in_channels
        self.mid_out_channels = mid_out_channels

        if self.hyper:
            self.hidden_channels = mid_in_channels // rel_reduction
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
        A = self.PA.cuda(x.get_device()) if x.is_cuda else self.PA
        A = self.edge_importance * A
        A = self.a_norm(A)

        if self.hyper:
            t_x = x.mean(2)

            v_x = self.to_V(t_x)

            dis_v_x = v_x.view(N, self.num_subset, self.hidden_channels, V)
            dis_v_x = dis_v_x.permute(0, 1, 3, 2).contiguous()
            distance_x = torch.cdist(dis_v_x, dis_v_x)
            H = torch.zeros_like(distance_x)

            topk_v, topk_indices = torch.topk(distance_x, 9, largest=False)
            topk_v = self.softmax(-topk_v)
            H = torch.scatter(H, 3, topk_indices, topk_v)

            W = self.to_W(t_x)

            G = self.hyper_norm(H, W)
            alpha = self.alpha
            alpha = self.relu(alpha)
            G_scaled = alpha * G
            A = A + G_scaled
        else:
            G_scaled = None

        d_x = self.conv_d(x)
        d_x = d_x.view(N, self.num_subset, self.mid_out_channels, T, V)
        y = torch.einsum('nkuv,nkctv->nkctu', A, d_x).contiguous()
        y = y.view(N, self.out_channels, T, V)

        x = x[..., :self.vertex_nums]
        y = y[..., :self.vertex_nums]

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y, self.hyper_joint, G_scaled
