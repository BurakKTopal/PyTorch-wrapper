"""
Simple PyTorch Wrapper Package
A lightweight PyTorch wrapper for fast and easy neural network training and evaluation.
"""

# Metadata
__version__ = "0.0.1"
__author__ = "Burakktopal"
__description__ = "A lightweight PyTorch wrapper for fast and easy neural network training and evaluation."

# Import and expose key functionality from submodules
from .examples import (
    CNN_example_run,
    FNN_example_run,
    load_language_digit_example_dataset,
)
from .utils import (
    set_seed,
    FNNGenerator,
    CNNGenerator,
    load_data_from_numpy_files,
    NetworkType,
)
from .wrapper import PytorchWrapper

# Define public API
__all__ = [
    # Examples
    "CNN_example_run",
    "FNN_example_run",
    "load_language_digit_example_dataset",
    
    # Utils
    "set_seed",
    "FNNGenerator",
    "CNNGenerator",
    "load_data_from_numpy_files",
    "NetworkType",
    
    # Wrapper
    "PytorchWrapper",
]
