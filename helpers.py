import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# Loading data function:
def loading_data(folder_name):
    np_X = np.load(f"{folder_name}/X.npy")
    np_Y = np.load(f"{folder_name}/Y.npy")
    X = torch.from_numpy(np_X) 
    Y = torch.from_numpy(np_Y)
    return X, Y

def FNNGenerator(input_size, output_size, hidden_layers, hidden_activations = None):
    """
    Generates a PyTorch neural network with specified hidden layers, widths, and activation functions.
    """
    if not hidden_activations:
        # Relu is default
        hidden_activations = [nn.ReLU() for _ in range(hidden_layers)]
    if len(hidden_layers) != len(hidden_activations):
        raise ValueError("The length of hidden_layers and hidden_activations must be the same.")

    layers = []
    
    # Input to the first hidden layer
    current_input_size = input_size
    for hidden_units, activation in zip(hidden_layers, hidden_activations):
        layers.append(nn.Linear(current_input_size, hidden_units))
        if activation:
            layers.append(activation)
        current_input_size = hidden_units
    
    # Final layer (hidden to output)
    layers.append(nn.Linear(current_input_size, output_size))
    
    # Create the Sequential model
    network = nn.Sequential(*layers)
    
    return network

class CNNGenerator(nn.Module):
    """
    Dynamic CNN class to construct convolutional layers and fully connected layers
    based on input configurations.
    """
    def __init__(self, input_channels, conv_layers, fc_layers, output_size, batch_size, use_pooling=False):
        """
        Arguments:
            input_channels (int): Number of input channels
            conv_layers (list of dict): List of dictionaries specifying conv layer configurations.
                                      Each dict should contain keys: `out_channels`, `kernel_size`, 
                                      `stride`, and `padding`.
            fc_layers (list of int): List specifying the number of neurons in each fully connected layer.
            output_size (int): Number of output classes for the final layer.
            batch_size: the size of the used batch so that neural network can process one batch at once.
            use_pooling (bool): Whether to use MaxPool2d after each conv layer
        """
        super(CNNGenerator, self).__init__()
        self.batch_size = batch_size
        self.use_pooling = use_pooling
        self.flattened_size_per_element_in_batch = None
        
        # Constructing the convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        print(input_channels, conv_layers, fc_layers, output_size, batch_size)
        
        for conv in conv_layers:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv['out_channels'],
                    kernel_size=conv['kernel_size'],
                    stride=conv['stride'],
                    padding=conv['padding']
                )
            )
            in_channels = conv['out_channels']
        
        # Pooling layer if requested
        self.pool = nn.MaxPool2d(kernel_size=(2, 2)) if use_pooling else None
        
        # Calculate the flattened size using dummy input
        dummy_input = torch.zeros(batch_size, input_channels, 64, 64)
        with torch.no_grad():
            x = dummy_input
            for conv in self.conv_layers:
                x = F.relu(conv(x))
                if self.use_pooling:
                    x = self.pool(x)
                    
            if (x.numel() % batch_size != 0):
                raise ValueError("the number of inputs must be multiple of batch size")
            self.flattened_size_per_element_in_batch = x.numel()//batch_size
            
        # Constructing the regular NN structure
        self.fc_layers = nn.ModuleList()
        in_features = self.flattened_size_per_element_in_batch
        
        for fc_units in fc_layers:
            print(in_features, fc_units)
            self.fc_layers.append(nn.Linear(in_features, fc_units))
            in_features = fc_units
        
        # The output layer
        self.output_layer = nn.Linear(in_features, output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Pass input through convolutional layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            if self.use_pooling:
                x = self.pool(x)

        # Flatten the output
        x = x.view(-1, self.flattened_size_per_element_in_batch)
        
        # Pass through feedforward layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        
        # Output layer
        x = self.output_layer(x)
        return x