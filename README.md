# CSCI6975_DeepLearningForGraphs_Project

Dependencies
1. Pytorch
2. torch_geometric
3. scipy
4. matplotlib

Create conda environment using the following commands:
```
-------- Install torch 2.0 for CUDA 11.8 -----------
conda create --name torch-env python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

Generate the training and validation dataset:
```
python matrices_text.py --N_list 100 --mode "train" --steady 1
```
```
python graphs_pt_1D.py --data_path data/advection_1D/steady/ --check 1 --num_b 10000 --mode "train" --data_type "advection_1D"
```

Generate the test dataset:
```
python matrices_text.py --N_list 100 1000 --mode "test" --steady 1
```
```
python graphs_pt_1D.py --data_path data/advection_1D/steady/ --check 1 --num_b 1 --mode "test" --data_type "advection_1D"
```
