# Assess fast and easy your neural network

## Importance of easy and quick assessment of quality
A lightweight PyTorch wrapper that can be used to fasten process of training and setting up arbitrary Neural Network to quickly test an idea/setup. The wrapper provides an interface for both standard neural networks and CNNs, but can be extended to any architecture, with built-in visualization and performance tracking capabilities. The wrapper is customizable and aims to be used on any dataset. 

## Usage
`pip install simple_pytorch_wrapper`

This package is aimed to be as simple as possible. The following example trains on the example dataset in this package, using a simple feedforward network.

```python
from simple_pytorch_wrapper import PytorchWrapper, FNNGenerator, NetworkType
from simple_pytorch_wrapper import set_seed, load_language_digit_example_dataset
import torch.nn as nn

def main():
    # For reproducibility
    set_seed(0)

    # Example data
    X, Y = load_language_digit_example_dataset()

    # Transform shape of *image* data accordingly
    X, Y = PytorchWrapper.vectorize_image_data(X, Y, NetworkType.FNN) 

    # Pytorch wrapper
    wrapper = PytorchWrapper(X, Y, classification=True) 
    
    # Generate a feedforward neural network
    network = FNNGenerator(
    input_size=64*64,  
    output_size=10,   
    hidden_layers=[100, 500],  
    hidden_activations=[nn.ReLU(), nn.ReLU()]
    )
    
    # Upload python network to wrapper
    wrapper.upload_pytorch_network(network) 

    # Training
    wrapper.setup_training(batch_size=32, learning_rate=0.001, epochs=10)
    wrapper.train_network(plot=False)

    # Visuals
    wrapper.visualize()

if __name__ == "__main__":
    main()
```
You see that training a network from start to finish, with a clean visualization in less than 15 effective lines!

## Upload your own network!
The `network` variable above is there to be defined by you. This is aimed to take in any Neural network design. The only caveat is to correctly shape your input. In case of image analysis, this is already implemented for a feedforward and CNN, these are already implemented by the `vectorize_image_data(*args)` function. If you already have correctly formatted your image_data, then this static function call is not necessary. 

To get you started on the networks, there has been provided already NN generator classes for a feed-forward NN and a convolutional one. These can be used respectively by calling `FNNGenerator(*args)` and `CNNGenerator(*args)`. 

The function signature of the FNNGenerator is: 
```python
network = FNNGenerator(input_size: int,
              output_size: int,
              hidden_layers: List[int],
              hidden_activations: Optional[List[nn.Module]]
              )
```

While for the `CNNGenerator`, this is: 
```python 
network = CNNGenerator(
                input_channels: int,
                conv_layers: List[Dict[str, int]],  
                fc_layers: List[int], 
                output_size: int,
                batch_size: int,
                image_height: int, 
                image_width: int,
                seed: Optional[int] = None,
                use_pooling: bool = False
            )
```
The `image_height` and `image_width`are necessary to get the dimensionality of the forward layers in the CNN right.

## Turning off warnings 
There has been implemented several warnings to ensure proper use. To turn off these warnings, place `suppress_warnings()` at the top of your code.

## Example runs
It can be hard to know the power of a tool, without having a solid example. By calling `FNN_example_run()` with the training params as function arguments, you get a run on the example data set for a FNN architecture. The `CNN_example_run()` does it with a CNN architecture.


## Dataset
The data used for the example is the Sign Language Digits Dataset from the Turkey Ankara Ayrancı Anadolu High School students. This dataset is available at [ardamavi/Sign-Language-Digits-Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset).
