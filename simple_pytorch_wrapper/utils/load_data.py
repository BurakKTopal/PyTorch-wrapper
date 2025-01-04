import torch
import numpy as np

# Loading data function:
def load_data_from_numpy_files(folder_name):
    np_X = np.load(f"{folder_name}/X.npy")
    np_Y = np.load(f"{folder_name}/Y.npy")
    X = torch.from_numpy(np_X) 
    Y = torch.from_numpy(np_Y)
    return X, Y