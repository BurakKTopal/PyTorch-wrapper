import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_pytorch_wrapper.utils.display_warning import display_warning

class FNNGenerator(nn.Module):
    """
    A customizable feedforward neural network generator.
    """
    def __init__(self, input_size, output_size, hidden_layers, hidden_activations=None):
        """
        Initializes the feedforward neural network.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            hidden_layers (list of int): List of integers specifying the number of units in each hidden layer.
            hidden_activations (list of nn.Module or None): List of activation functions for each hidden layer.
                                                            Defaults to ReLU for all layers if None.
        """
        super(FNNGenerator, self).__init__()

        if not hidden_activations:
            hidden_activations = [nn.ReLU() for _ in hidden_layers]

        if len(hidden_layers) != len(hidden_activations):
            raise ValueError("The length of hidden_layers and hidden_activations must be the same.")

        layers = []
        current_input_size = input_size

        for hidden_units, activation in zip(hidden_layers, hidden_activations):
            layers.append(nn.Linear(current_input_size, hidden_units)) 
            if activation:
                layers.append(activation)  
            current_input_size = hidden_units

        layers.append(nn.Linear(current_input_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        return self.network(x)

class CNNGenerator(nn.Module):
    """
    Dynamic CNN class to construct convolutional layers and fully connected layers
    based on input configurations.
    """
    def __init__(self, input_channels, conv_layers, fc_layers, output_size, batch_size, image_height, image_width, seed=None, use_pooling=False):
        """
        Arguments:
            input_channels (int): Number of input channels.
            conv_layers (list of dict): List of dictionaries specifying conv layer configurations.
                                      Each dict should contain keys: `out_channels`, `kernel_size`, 
                                      `stride`, and `padding`.
            fc_layers (list of int): List specifying the number of neurons in each fully connected layer.
            output_size (int): Number of output classes for the final layer.
            batch_size (int): The size of the batch to be processed at once.
            image_height (int): Height of the input image.
            image_width (int): Width of the input image.
            seed (int, optional): Random seed for reproducibility.
            use_pooling (bool): Whether to use MaxPool2d after each conv layer. The kernel size is fixed at (2, 2).
        """
        super(CNNGenerator, self).__init__()
        
        if not seed:
            display_warning("__init__() CNNGenerator", "You have not chosen a seed, the CNN network will be initialized randomly.")
        self.batch_size = batch_size
        self.use_pooling = use_pooling
        self.flattened_size_per_element_in_batch = None
        
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
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
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2)) if use_pooling else None
        
        dummy_input = torch.zeros(batch_size, input_channels, image_height, image_width)
        with torch.no_grad():
            x = dummy_input
            for conv in self.conv_layers:
                x = F.relu(conv(x))
                if self.use_pooling:
                    x = self.pool(x)
                    
            if (x.numel() % batch_size != 0):
                raise ValueError("The number of inputs must be a multiple of the batch size.")
            self.flattened_size_per_element_in_batch = x.numel() // batch_size
        
        self.fc_layers = nn.ModuleList()
        in_features = self.flattened_size_per_element_in_batch
        
        for fc_units in fc_layers:
            self.fc_layers.append(nn.Linear(in_features, fc_units))
            in_features = fc_units
        
        self.output_layer = nn.Linear(in_features, output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Forward pass for the CNNGenerator.
        """
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            if self.use_pooling:
                x = self.pool(x)

        x = x.view(-1, self.flattened_size_per_element_in_batch)
        
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        
        x = self.output_layer(x)
        return x