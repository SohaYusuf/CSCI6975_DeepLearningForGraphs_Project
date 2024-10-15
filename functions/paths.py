import numpy as np
import json

def advection_data_path(path, dataset, steady):
    paths = {}
    if steady:
        paths["train"] = f"{path}/{dataset}/steady/train/graphs/"
        paths["val"] = f"{path}/{dataset}/steady/val/graphs/"
        paths["test"] = f"{path}/{dataset}/steady/test/graphs/"
    else:
        paths["train"] = f"{path}/{dataset}/unsteady/train/graphs/"
        paths["val"] = f"{path}/{dataset}/unsteady/val/graphs/"
        paths["test"] = f"{path}/{dataset}/unsteady/test/graphs/"
    return paths