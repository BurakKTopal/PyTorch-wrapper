import random
import numpy as np
import torch
from simple_pytorch_wrapper.utils.display_warning import display_warning

def set_seed(seed):
    display_warning("set_seed()", "To ensure reproducibility, be sure to call set_seed() before you generate your pytorch network.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return