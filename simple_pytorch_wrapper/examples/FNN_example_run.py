from simple_pytorch_wrapper.utils.NN_generators import  FNNGenerator
from simple_pytorch_wrapper.utils.network_types import NetworkType
from simple_pytorch_wrapper.examples.xNN_example_run import xNN_example_run
import torch.nn as nn

def FNN_example_run(learning_rate, batch_size, epochs, plot, seed=None):
    network = FNNGenerator(
    input_size=64*64,  
    output_size=10,  
    hidden_layers=[100, 500],  
    hidden_activations=[nn.ReLU(), nn.ReLU()], 
    )
    network_type = NetworkType.FNN
    xNN_example_run(learning_rate, batch_size, epochs, plot, network, network_type, seed)
    return