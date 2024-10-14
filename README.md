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
