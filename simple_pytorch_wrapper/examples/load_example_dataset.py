import numpy as np
import torch
from pkg_resources import resource_filename

def load_language_digit_example_dataset():
    """
    Dataset from https://github.com/ardamavi/Sign-Language-Digits-Dataset
    """
    # Locate the data files inside the installed package
    np_X_path = resource_filename("simple_pytorch_wrapper", "examples/data/X.npy")
    np_Y_path = resource_filename("simple_pytorch_wrapper", "examples/data/Y.npy")
    
    # Load data
    np_X = np.load(np_X_path)
    np_Y = np.load(np_Y_path)
    X = torch.from_numpy(np_X) 
    Y = torch.from_numpy(np_Y)
    return X, Y