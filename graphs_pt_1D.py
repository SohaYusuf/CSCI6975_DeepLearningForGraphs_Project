import argparse
import os
import json
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
import time
import scipy

# unsteady
# python graphs_pt_1D.py --data_path data/advection_1D/unsteady/ --check 1 --num_b 10000 --mode "train" --data_type "advection_1D"
# python graphs_pt_1D.py --data_path data/advection_1D/unsteady/ --check 1 --num_b 1 --mode "test" --data_type "advection_1D" --reorder True

# steady
# python graphs_pt_1D.py --data_path data/advection_1D/steady/ --check 1 --num_b 10000 --mode "train" --data_type "advection_1D"
# python graphs_pt_1D.py --data_path data/advection_1D/steady/ --check 1 --num_b 1 --mode "test" --data_type "advection_1D" --reorder True

def check_solution_error_numpy(A, s, b):
    """
    Calculate the residual error ||b - As|| using numpy
    """
    print(f'A shape: {A.shape}\nb shape: {b.shape}\ns shape: {s.shape}') 
    residual = np.linalg.norm(b.reshape(-1, 1) - A.dot(s.reshape(-1, 1))) 
    print(f'residual for 1 numpy: {residual}')

def check_solution_error_torch(graph):
    """
    Calculate the residual error ||b - As|| after linear system is converted to graph
    """
    A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_attr.squeeze(), requires_grad=False)
    residual_ = torch.linalg.norm(graph.b.reshape(-1,1) - torch.sparse.mm(A, graph.s.reshape(-1,1)))
    print('Residual error 2 torch: ',residual_)
    del A, residual_

def matrix_to_graph(A, ddtype):
    """
    Convert a matrix A to a PyTorch Geometric Data object.
    """
    num_nodes = A.shape[0]
    # Create edge indices
    indices = A._indices()
    values = A._values()
    node_features = torch.zeros(num_nodes,dtype=ddtype)
    # Create a PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=indices, edge_attr=values)
    return data

def read_A_file(txt_file_path, N):
    """
    Read sparse matrix text file and return coo numpy matrix A and torch coo tensor A_tensor.
    """
    # Read sparse matrix entries from the text file
    sparse_matrix_entries_ = []
    with open(txt_file_path, 'r') as file:
        for line in file:
            row, col, val = map(float, line.split())
            sparse_matrix_entries_.append((int(row), int(col), val))
    # Extract row indices, column indices, and values
    rows, cols, vals = zip(*sparse_matrix_entries_)
    # Construct sparse coo matrix in numpy
    A =  coo_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64)
    # Construct sparse coo tensor in torch
    vals_tensor = torch.tensor(vals, dtype=torch.float64)
    rows_tensor = torch.tensor(rows, dtype=torch.int64)
    cols_tensor = torch.tensor(cols, dtype=torch.int64)
    A_tensor = torch.sparse_coo_tensor(indices=torch.stack([rows_tensor, cols_tensor]), values=vals_tensor, size=(N, N), dtype=torch.float64)
    return A, A_tensor

def load_data(mode, N, p, num_b, path, reorder, method=None, check=True, random_seed=42):
    """
    Load matrix A from text file and return sparse tensor A with randomly generated vector b and solution vector u.
    """
    rng = np.random.RandomState(random_seed)
    current_dir = f'{path}/{mode}/matrices/'   # Get the directory of the current file
    txt_file_path = os.path.join(current_dir, f"{N}_{method}.txt" if reorder else f"{N}.txt")
    # Construct numpy coo matrix A and torch coo tensor A_tensor
    A, A_tensor = read_A_file(txt_file_path, N)

    if mode=="test":
        b = rng.uniform(-1, 1, size=N)
        b_mfem = b/np.linalg.norm(b)
        u_mfem = scipy.sparse.linalg.spsolve(A, b_mfem)
        # mfem_u_path = os.path.join(current_dir, f"u_{N}_{method}.txt" if reorder else f"u_{N}.txt")
        # u_mfem = np.loadtxt(mfem_u_path)
        # b_mfem = A.dot(u_mfem)
        print('b_mfem: ', b_mfem) 
        print('error in ||A*u_mfem - b_mfem||: ', np.linalg.norm(b_mfem - A.dot(u_mfem)))
        b_mfem = torch.tensor(b_mfem, dtype=torch.float64)
        u_mfem = torch.tensor(u_mfem, dtype=torch.float64)
        return A_tensor, b_mfem, u_mfem
    else:
        b_list = []
        x_list = []
        for k in range(num_b):
            b = rng.uniform(-1, 1, size=N)
            b = b/np.linalg.norm(b)
            x = scipy.sparse.linalg.spsolve(A, b)
            if check:
                check_solution_error_numpy(A, x, b)
            b_list.append(torch.tensor(b, dtype=torch.float64))
            x_list.append(torch.tensor(x, dtype=torch.float64))
        return A_tensor, b_list, x_list

def save_pt_file(A, b, u, path, ddtype, check):
    """
    Construct a graph and save the graph as .pt file.
    """
    #save the matrix as a graph in .pt file
    graph = matrix_to_graph(A, ddtype)
    graph.s, graph.b, graph.n = u, b, A.shape[0]
    if check:
        check_solution_error_torch(graph)
    torch.save(graph, path)
    # Verify .pt file is written successfully
    g = torch.load(path)
    # change dtype to double
    g.x = g.x.to(ddtype)
    g.edge_attr = g.edge_attr.to(ddtype)
    print(f'\n .pt file is generated at \n {path}')

def generate_data(mode, path, check, num_b, reorder):
    """
    Generate data from given metadata and save as PyTorch Geometric Data objects.
    """
    ddtype = torch.float64
    meta_path = os.path.join(path, f"{mode}/matrices/meta_{mode}.json")  # meta file path
    graph_path = os.path.join(path, f"{mode}/graphs/")                   # folder where graphs will be saved as .pt files   
    os.makedirs(graph_path, exist_ok=True)                               # Create graphs directory if it does not exist
    
    # Read the meta file
    with open(meta_path, 'r') as meta_file:
        meta_data = json.load(meta_file)
    N_list = meta_data["N"]
    p_list = meta_data["p"]
    r_list = meta_data["reordering"] if reorder else None
    num_graphs = len(N_list)*len(p_list)*len(r_list)*num_b if reorder else len(N_list)*len(p_list)*num_b
    
    for N in N_list:
        for p in p_list:
            if reorder:
                for r,method in enumerate(r_list):
                    print(f'\n---------------- N = {N}, p = {p}, method = {method} ------------------')
                    # testing dataset for reordered matrices
                    if mode=="test":
                        graph_path_test = f"{graph_path}/p{p}_{method}/"
                        os.makedirs(graph_path_test, exist_ok=True)
                        save_path = os.path.join(graph_path_test, f"{N}_p{p}_r{r}.pt")
                        A, b_mfem, u_mfem  = load_data(mode, N, p, num_b, path, reorder, method)
                        save_pt_file(A, b_mfem, u_mfem, save_path, ddtype, check)
                    # training and validation datasets for reordered matrices
                    else:
                        A, b_list, s_list = load_data(mode, N, p, num_b, path, reorder, method)
                        for k in range(num_b):
                            print(f'----------- {k} -----------')
                            b = b_list[k]
                            s = s_list[k]
                            save_path = os.path.join(graph_path, f"{N}_p{p}_b{k}_r{r}.pt")
                            save_pt_file(A, b, s, save_path, ddtype, check)
            # for no reordering
            else:
                # testing dataset
                if mode=="test":
                    graph_path_test = f"{graph_path}/p{p}/"
                    os.makedirs(graph_path_test, exist_ok=True)
                    save_path = os.path.join(graph_path_test, f"{N}_p{p}.pt")
                    A, b_mfem, u_mfem  = load_data(mode, N, p, num_b, path, reorder)
                    save_pt_file(A, b_mfem, u_mfem, save_path, ddtype, check)
                # training and validation datasets for reordered matrices
                else:
                    A, b_list, s_list = load_data(mode, N, p, num_b, path, reorder)
                    for k in range(num_b):
                        print(f'----------- {k} -----------')
                        b = b_list[k]
                        s = s_list[k]
                        save_path = os.path.join(graph_path, f"{N}_p{p}_b{k}.pt")
                        save_pt_file(A, b, s, save_path, ddtype, check)

    return num_graphs

def main(path, check, num_b, split, mode, data_type, reorder):
    """
    Main function to generate data from metadata and save as PyTorch Geometric Data objects.
    """
    if mode=="train":
        num_graphs_train = generate_data(mode="train", path=path, check=check, num_b=int(split*num_b), reorder=False)
        num_graphs_val = generate_data(mode="val", path=path, check=check, num_b=int((1-split)*num_b), reorder=False)
        if reorder:
            print(f'--------------------------------------------------------- reordering ---------------------------------------------')
            num_graphs_train_reorder = generate_data(mode="train", path=path, check=check, num_b=int(split*num_b), reorder=True)
            num_graphs_val_reorder = generate_data(mode="val", path=path, check=check, num_b=int((1-split)*num_b), reorder=True)

        total_graphs_train = num_graphs_train + num_graphs_train_reorder if reorder else num_graphs_train
        total_graphs_val = num_graphs_val + num_graphs_val_reorder if reorder else num_graphs_val
        print(f'Total train graphs: {total_graphs_train}, Total val graphs: {total_graphs_val}')

    if mode=="test":
        num_graphs_test = generate_data(mode, path, check, num_b, reorder=False)
        if reorder:
            print(f'--------------------------------------------------------- reordering ---------------------------------------------')
            num_graphs_test_reorder = generate_data(mode, path, check, num_b, reorder=reorder)

        total_graphs_test = num_graphs_test + num_graphs_test_reorder if reorder else num_graphs_test
        print(f'Total test graphs: {total_graphs_test}')
                        
if __name__ == "__main__":
    
    # get the start time
    st = time.time()
    
    parser = argparse.ArgumentParser(description='Generate and save PyTorch Geometric Data objects.')
    parser.add_argument('--data_path', type=str, default="data/advection_1D/", help='Path to data folder')
    parser.add_argument('--check', type=int, default=1)
    parser.add_argument('--num_b', type=int, default=100)
    parser.add_argument('--split', type=float, default=0.96)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_type', type=str, default='advection_1D')
    parser.add_argument('--reorder', type=str, default=False)
    
    args = parser.parse_args()
    main(args.data_path, args.check, args.num_b, args.split, args.mode, args.data_type, args.reorder)
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')