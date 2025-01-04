import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.display import clear_output

from enum import Enum

class NetworkType(Enum):
    FNN = "FNN"
    CNN = "CNN"

class BasePytorchWrapper():
    """
    This class serves as a wrapper for the general structure for the digit recoginition assignment.
    By the use of upload_pyTorch_network(), which provides a 'gateway' to setup any neural network and train it with abstracted commands
    Whether it is a regular neural network, or a CNN, the class won't need extra adaptations (hence in that sense it is a 'wrapper')
    Note that this is in the assumption that the data (X, Y) was initialized conform to 
    the neural network given in upload_pyTorch_network().
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.train_data = None
        self.test_data = None
        self._generate_train_and_test(X, Y)
        self.optimizer = None
        self.loader = None
        self.batch_size  = None
        self.learning_rate = None
        self.epochs = None
        self.loss_function = None
        self.batch_L2 = []
        self.train_L2 = []
        self.test_L2 = []
        self.pyTorch_network = None

    @staticmethod
    def transform_image_data(X, Y, NN_type):
        match NN_type:
            case NetworkType.FNN:
                X = X.reshape(X.shape[0], -1) # We feed the image as a vector
            case NetworkType.CNN:
                X = X.unsqueeze(1) # For (mini-)batch processing
            case _:
                raise ValueError("Unsupported network type: only FNN and CNN are supported. You can add more if you want...")
        return X, Y

    def setup_training(self, batch_size, learning_rate, epochs, loss_function='LSE', reset_training=True):
        """
        This will setup the params for the following training, moreover:
        -The splitting of the bigger (train) set into smaller pieces (= batch size)
        -The rate/step size with which the Stochastic Gradient Descent will be performed (= learning rate)
        -The number of repitititons that we ran through all batches once (epochs)
        -The type of loss function to be used (default is LSE)
        """
        if reset_training:
            self.reset_training() # Reset any training data from the previous iterations, option available in case we want to continue from last training
        self.pyTorch_network.parameters()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # Dataloader gives random batches
        self.loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)

        # Setting up the SGD optimizer and give the parateters of our network.
        self.optimizer = torch.optim.SGD(self.pyTorch_network.parameters(), lr=learning_rate)
        
        # Setting up the loss function
        if loss_function == 'LSE':
            self.loss_function = lambda outputs, targets: torch.sum((targets - outputs).pow(2))
        elif loss_function == 'CrossEntropy':
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function. Please use 'LSE' or 'CrossEntropy'.")
        return
    
    def _generate_train_and_test(self, X, Y):
        """
        Generates test and train data for the neural net to train. 
        Can be used multiple times on one instance.
        """
        # Creating data set of X and Y
        dataset = torch.utils.data.TensorDataset(X, Y)

        # Determining sizes of sets: 80% train, and 20% test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # Generating test and train set according to proportions
        self.train_data, self.test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
        return self.train_data, self.test_data
    
    def reset_pyTorch_network_params(self):
        """
        Resets the params of the pyTorch network to start clean off
        """
        # Reset params of neural network
        for m in self.pyTorch_network.children():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
    
    def calculate_accuracy(self):
        """
        Calculates the percentage of correct guesses by the NN
        """
        Xtest, Ytest = self.test_data[:] # retrieve the test data
        correct = 0
        m = nn.Softmax(dim=1)
        for index in range(len(Ytest)):
            self.pyTorch_network.eval() 
            y_test = torch.argmax(Ytest[index])
            y_pred = torch.argmax(self.predict(Xtest[index]))
            if y_test == y_pred:
                correct +=1
        return round(correct/len(Ytest),2)*100

    def upload_pyTorch_network(self, network):
        """
        IMPORTANT: this function serves as a sort of 'gateway', providing the wrapper the (PyTorch)
        network that it will uses to train
        """
        self.pyTorch_network = network
        return
    

    def train_network_in_epoch(self):
        # Training network
        self.pyTorch_network.train()
        for inputs, targets in self.loader:
            self.optimizer.zero_grad()
            outputs = self.predict(inputs)
            loss = self.loss_function(outputs, targets)
            self.batch_L2.append(loss.item() / self.batch_size)
            loss.backward()
            self.optimizer.step()

        # Switch the network to evaluation mode
        self.pyTorch_network.eval()

        # Calculate and store the average training loss
        average_train_loss = np.mean(self.batch_L2[-len(self.loader):])
        self.train_L2.append(average_train_loss)

        # Evaluate the model on the test data
        test_inputs, test_targets = self.test_data[:]
        test_outputs = self.predict(test_inputs)

        # Compute the average test loss
        average_test_loss = self.loss_function(test_outputs, test_targets) / len(self.test_data)
        self.test_L2.append(average_test_loss.item())
            
    def train_network(self, plot=False):
        """
        Training the network within the frame of the assignment. Mostly adapted from the examples as given by the lecturers.
        """
        for epoch in range(0, self.epochs-1):
            self.train_network_in_epoch()
            
            if plot:  # Only plotting if wishing too
                self.visualize()
                
            if not plot:
                # If no plot is shown, then at least some kind of progression bar
                print(f"Progress: {round(epoch / self.epochs, 2) * 100}%")

    def visualize(self):
        """
        Visualization with three subplots showing:
        1. Loss metrics (batch, training, and testing loss)
        2. Accuracy over epochs
        3. CPU cycle time per epoch
        """
        clear_output(wait=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
            
        # Plot 1: Losses
        ax.set_yscale('log')
        ax.plot(self.train_L2, label="training loss")
        ax.plot(self.test_L2, label="testing loss")
        ax.legend()
        ax.set_title('Loss Metrics')
        
        # Add text for test loss
        x_anchor = 0.5 * ax.get_xlim()[0] + 0.5 * ax.get_xlim()[1]
        y_anchor = 10**(0.2*math.log10(ax.get_ylim()[0]) + 0.8*math.log10(ax.get_ylim()[1]))
        ax.text(x_anchor, y_anchor, f"Test loss: {self.test_L2[-1]:.2e}", ha='center', fontsize=12)
    
        plt.tight_layout()
        plt.show()
    
    def predict(self, x):
        return self.pyTorch_network(x)
    
    def reset_training(self):
        """
        This will reset any training done, and serves to give clean plots when trying to train again.
        """
        self.batch_L2 = []
        self.train_L2 = []
        self.test_L2 = []
        self.reset_pyTorch_network_params()
        self.optimizer = None
        self.loader = None
