import random
import numpy as np
import torch

def set_seed(seed):
    Warning("To ensure reproducibility, set_seed() has to be first called, before any other function call is made.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return