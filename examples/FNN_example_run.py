from utils.NN_generators import  FNNGenerator
from wrappers.network_types import NetworkType
from examples.xNN_example_run import x_NN_example_run

import torch.nn as nn

def FNN_example_run(learning_rate, batch_size, epochs, plot):

    # Generating network
    network = FNNGenerator(
    input_size=64*64,  # flattened value of image
    output_size=10,    # number of possible outputs
    hidden_layers=[100, 500],  # sizes of hidden layers
    hidden_activations=[nn.ReLU(), nn.ReLU()],  # activation functions for each hidden layer
    )
    network_type = NetworkType.FNN
    x_NN_example_run(learning_rate, batch_size, epochs, plot, network, network_type)
    return