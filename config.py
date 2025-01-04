from enum import Enum
import torch.nn as nn
from helpers import CNNGenerator, FNNGenerator
from wrappers.BasePytorchWrapper import NetworkType

"""
Config file containing setup for the training and network configuration
"""

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Network configuration
NETWORK_TYPE = NetworkType.CNN

# CNN configuration
NETWORK = CNNGenerator(
    input_channels=1,
    conv_layers=[{
        'out_channels': 16,
        'kernel_size': (2, 2),
        'stride': 1,
        'padding': 0
    }],
    fc_layers=[500],
    output_size=10,
    batch_size=BATCH_SIZE,
    use_pooling=False
)

# Alternative FNN configuration (commented out by default)
"""
NETWORK = FNNGenerator(
    input_size=64*64,  # flattened value of image
    output_size=10,    # number of possible outputs
    hidden_layers=[100, 500],  # sizes of hidden layers
    hidden_activations=[nn.ReLU(), nn.ReLU()],  # activation functions for each hidden layer
)
"""