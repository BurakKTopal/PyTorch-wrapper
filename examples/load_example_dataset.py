import torch
import numpy as np

def load_language_digit_example_dataset():
    """
    Dataset from https://github.com/ardamavi/Sign-Language-Digits-Dataset
    """
    np_X = np.load(f"./examples/data/X.npy")
    np_Y = np.load(f"./examples/data/Y.npy")
    X = torch.from_numpy(np_X) 
    Y = torch.from_numpy(np_Y)
    return X, Y