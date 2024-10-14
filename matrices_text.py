import os
import numpy as np
from sklearn.model_selection import train_test_split
import json
import argparse
import time

# unsteady
# python matrices_text.py --N_list 100 --mode "train" --num_CFL 1.386 --steady 0
# python matrices_text.py --N_list 100 1000 10000 100000 1000000 --mode "test" --num_CFL 1.386 --steady 0

# steady
# python matrices_text.py --N_list 100 --mode "train" --steady 1
# python matrices_text.py --N_list 100 1000 10000 100000 1000000 --mode "test" --steady 1

def generate_and_save_matrix(N_list, h, folder, ddtype, num_CFL, steady):
    
    """Generate matrix A for each h in h_values and save it to the specified folder."""

    if steady:

        """ A = [ 1 0 0
                -1/2h 0 1/2h
                0 -1/h 1/h ] """
    
        for i,N in enumerate(N_list):
            
            print(f'\n---------------- N = {N} ------------------')
        
            print(' Saving .txt file for N: ', N)

            # Create diagonals for the sparse matrix
            main_diag = np.zeros(N, dtype=ddtype)
            lower_diag = np.full(N-1, -1/(2*h), dtype=ddtype)
            upper_diag = np.full(N-1, 1/(2*h), dtype=ddtype)

            # Modify the diagonal entries
            main_diag[0] = 1/h
            main_diag[-1] = 1/h
            upper_diag[0] = 1/h
            lower_diag[-1] = -1/h
            
            # Combine row and column indices for each diagonal entry
            indices = np.arange(N)
            upper_diag_indices = list(zip(indices[:-1], indices[1:]))
            diag_indices = list(zip(indices, indices))
            lower_diag_indices = list(zip(indices[1:], indices[:-1]))

            # Combine row and column indices and values for each diagonal
            sparse_matrix_entries = list(zip(upper_diag_indices, upper_diag)) + \
                                    list(zip(diag_indices, main_diag)) + \
                                    list(zip(lower_diag_indices, lower_diag))

            file_name = os.path.join(folder, f"{N}.txt")
            with open(file_name, 'w') as file:
                for entry in sparse_matrix_entries:
                    file.write(f"{entry[0][0]} {entry[0][1]} {entry[1]}\n")

    else:

        """ A = [ 1 0 0
                -cfl/4 1 cfl/4
                0 -cfl/2 (1 + (cfl/2)) ] """

        for i,N in enumerate(N_list):
            
            print(f'\n---------------- N = {N} ------------------')
        
            print(' Saving .txt file for N: ', N)

            # Create diagonals for the sparse matrix
            main_diag = np.ones(N, dtype=ddtype)
            lower_diag = np.full(N-1, -num_CFL/(4), dtype=ddtype)
            upper_diag = np.full(N-1, num_CFL/(4), dtype=ddtype)

            # Modify the diagonal entries
            # main_diag[0] = 1
            main_diag[-1] = 1 + (num_CFL/2)
            upper_diag[0] = 0
            lower_diag[-1] = - num_CFL/2
            
            # Combine row and column indices for each diagonal entry
            indices = np.arange(N)
            upper_diag_indices = list(zip(indices[:-1], indices[1:]))
            diag_indices = list(zip(indices, indices))
            lower_diag_indices = list(zip(indices[1:], indices[:-1]))

            # Combine row and column indices and values for each diagonal
            sparse_matrix_entries = list(zip(upper_diag_indices, upper_diag)) + \
                                    list(zip(diag_indices, main_diag)) + \
                                    list(zip(lower_diag_indices, lower_diag))

            file_name = os.path.join(folder, f"{N}.txt")
            with open(file_name, 'w') as file:
                for entry in sparse_matrix_entries:
                    file.write(f"{entry[0][0]} {entry[0][1]} {entry[1]}\n")

                    

def save_meta_file(N_list, h, folder, file_name, steady, num_CFL):
    
    """Save metadata about the generated matrices to a JSON file."""

    if steady:
        formula = "A = [[1, 0, 0], [-1/(2*h), 0, 1/(2*h)], [0, -1/h, 1/h]]"
    else:
        formula = "A = [[1, 0, 0], [cfl/2, 1, -cfl/2], [0, cfl/2, 1 - cfl/2]]"
    
    metadata = {
        "num_examples": len(N_list),
        "formula": formula,
        "N": N_list,
        "h": h,
        "steady": steady,
        "p": [0],
        "reordering": ["RCM"],
        "CFL": num_CFL
    }
    with open(os.path.join(folder, file_name), 'w') as f:
        json.dump(metadata, f, indent=4)


def main(N_list, train_size, random_state, data_path, num_CFL, steady, mode):
    
    """Main function to generate and save matrices for train and test sets."""
    
    ddtype = np.float64
    
    # Generate h values
    print('N_list: ',N_list)
    print('Random seed: ', random_state)
    print('dtype: ', ddtype)
    
    if steady:
        current_path = f"{data_path}/steady/"
    else:
        current_path = f"{data_path}/unsteady/"
    
    if mode=="train":
        print('training examples: ', len(N_list))
        print('validation examples: ', len(N_list))
        for mode in ["train", "val"]:
            dirr = f"{mode}/matrices/"
            pathh = os.path.join(current_path, dirr)
            os.makedirs(pathh, exist_ok=True)
            print(f'\n-------------- Generating {mode} data -----------------')
            # Generate and save matrices for the train set
            h=1
            generate_and_save_matrix(N_list, h, pathh, ddtype, num_CFL, steady)
            # Save metadata JSON files for train and test sets
            save_meta_file(N_list, h, pathh, f"meta_{mode}.json", steady, num_CFL)
            print(f'\n {len(N_list)} {mode} A matrices have been saved as .txt files at \n{pathh}')
                
            print('\n =========== Finished! =========== \n') 
    
    if mode=="test":
        print('testing examples: ', len(N_list), "\n")
        dirr = f"{mode}/matrices/"
        pathh = os.path.join(current_path, dirr)
        os.makedirs(pathh, exist_ok=True)
        print(f'\n-------------- Generating {mode} data -----------------')
        # Generate and save matrices for the train set
        h=1
        generate_and_save_matrix(N_list, h, pathh, ddtype, num_CFL, steady)
        # Save metadata JSON files for train and test sets
        save_meta_file(N_list, h, pathh, f"meta_{mode}.json", steady, num_CFL)
        print(f'\n {len(N_list)} {mode} A matrices have been saved as .txt files at \n{pathh}')
            
        print('\n =========== Finished! =========== \n')

    
if __name__ == "__main__":
    
    # get the start time
    st = time.time()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate and save matrices for train and test sets.')
    parser.add_argument('--N_list', metavar='N', type=int, nargs='+', default=[1000])
    parser.add_argument('--train_size', type=float, default=0.7, help='Size of the training set (default: 0.7)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for train_test_split (default: 42)')
    parser.add_argument('--data_path', type=str, default="data/advection_1D/", help='Path where data will be stored (default: current directory)')
    parser.add_argument('--steady', type=int, default=False, help='Path where data will be stored (default: current directory)')
    parser.add_argument('--num_CFL', type=float, default=1.386, help='CFL number for unsteady 1D advection equation')
    parser.add_argument('--mode', type=str, default="train", help='generate text files (A matrices) for train or test')
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args.N_list, args.train_size, args.random_state, args.data_path, args.num_CFL, args.steady, args.mode)
    
    # get the end time
    et = time.time()
    
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')