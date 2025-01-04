from simple_pytorch_wrapper import PytorchWrapper, FNNGenerator, NetworkType
from simple_pytorch_wrapper.utils import set_seed, load_language_digit_example_dataset
import torch.nn as nn

def main():
    # For reproducibility
    set_seed(0)

    # Example data
    X, Y = load_language_digit_example_dataset()

    # Transform data accordingly (vectorizing) + squeezing for batching
    X, Y = PytorchWrapper.vectorize_data(X, Y, NetworkType.FNN) # Needed to transform the data to the correct format for the input

    # Pytorch wrapper
    wrapper = PytorchWrapper(X, Y)  
    

    network = FNNGenerator(
    input_size=64*64,  # flattened value of image
    output_size=10,    # number of possible outputs
    hidden_layers=[100, 500],  # sizes of hidden layers
    hidden_activations=[nn.ReLU(), nn.ReLU()],  # activation functions for each hidden layer
    )
    
    # Create custom pytorch network
    wrapper.upload_pyTorch_network(network) 

    wrapper.setup_training(batch_size=32, learning_rate=0.01, epochs=10)

    # Training
    wrapper.train_network(plot=True)  # Enable plotting

    # Training ended
    wrapper.visualize()

if __name__ == "__main__":
    main()