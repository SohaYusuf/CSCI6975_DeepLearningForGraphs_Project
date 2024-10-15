import datetime
import random
from typing import List

import numpy as np
import torch
import torch_geometric
from torch_sparse import SparseTensor, spadd, spspmm

# Global variables and device settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ddtype = torch.float64
torch.set_default_dtype(ddtype)

# Results folder
folder = f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

def set_random_seed(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def display_graph_info(graph, name, print_mat=False):
    print(f'------------------- {name} -------------------------')
    print(' ====== SHAPES =======')
    print(f'N: {graph.n}')
    print(f'Number of edges: {graph.num_edges}')
    print(f'Number of nodes: {graph.num_nodes}')
    print(f'Node features shape: {graph.x.shape}')
    print(f'Edge features shape: {graph.edge_attr.shape}')
    
    if print_mat:
        for edge, attr in zip(graph.edge_index.t().tolist(), graph.edge_attr.tolist()):
            print(f'Edge {edge} = {attr}')
        print(f'Node attributes of the graph: {graph.x}')