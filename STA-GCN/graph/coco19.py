import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 19
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), 
    (6, 8), (8, 10), (5, 7), (7, 9), 
    (12, 14), (14, 16), (11, 13), (13, 15), 
    (17, 11), (17, 12), (17, 18), 
    (18, 5), (18, 6), (18, 0)
]
inward = inward_ori_index
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary_with_I)
