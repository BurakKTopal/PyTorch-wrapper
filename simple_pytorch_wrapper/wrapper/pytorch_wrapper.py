import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from simple_pytorch_wrapper.utils.network_types import NetworkType
from simple_pytorch_wrapper.utils.warnings import display_warning


class PytorchWrapper():
    """
    This class serves as a wrapper for the general structure for the digit recoginition assignment.
    By the use of upload_pytorch_network(), which provides a 'gateway' to setup any neural network and train it with abstracted commands
    Whether it is a regular neural network, or a CNN, the class won't need extra adaptations (hence in that sense it is a 'wrapper')
    Note that this is in the assumption that the data (X, Y) was initialized conform to 
    the neural network given in upload_pytorch_network(). The `classification` variable should be true (default) if the neural network
    is used for classification, and false if it is used for regression. 
    """
    def __init__(self, X, Y, classification = True):
        self.classification = classification
        if not classification:
            display_warning("__init__() PytorchWrapper", "No accuracy calculations will be performed, as the network is used for regression.")
        self.X = X
        self.Y = Y
        self.train_data = None
        self.test_data = None
        self.generate_train_and_test(X, Y)
        self.optimizer = None
        self.loader = None
        self.batch_size  = None
        self.learning_rate = None
        self.epochs = None
        self.loss_function = None
        self.avg_train_loss = []
        self.avg_test_loss = []
        self.pytorch_network = None
        self.l_accuracy = []
        self.cpu_cycles = []

    @staticmethod
    def vectorize_image_data(X, Y, NN_type):
        """ 
        Needed to vectorize image data, only if you already did not do so
        """
        display_warning("vectorize_image_data()", "This is a custom vectorization only meant for images. Please be aware of this.")
        match NN_type:
            case NetworkType.FNN:
                X = X.reshape(X.shape[0], -1) # We feed the image as a vector
            case NetworkType.CNN:
                X = X.unsqueeze(1) # For (mini-)batch processing
            case _:
                raise ValueError("Unsupported iamge data conversion.")
        return X, Y

    def display_progress(self, curr_epoch, plot):
        """
        Displays the progress of the training, and optionally plots the results.
        """
        print(f"Progress: {round(curr_epoch/self.epochs, 2)*100}%")
        if plot and (curr_epoch % max(2, self.epochs // 10) == 0):  # Only plotting if wishing to, and at most 10 times
            self.visualize()
        return

    def contains_nan(self, value, label):
        """
        Checks if value contains NaN values. If so, a warning is displayed.
        """
        value = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
        if torch.isnan(value).any():
            display_warning("contains_nan()", f"NaN values detected in variable '{label}'. \n Please check your network architecture.")
            return True
        return False
    
    def setup_training(self, batch_size, learning_rate, epochs, loss_function='LSE', reset_training=True):
        """
        This will setup the params for the following training, moreover:
        -The splitting of the bigger (train) set into smaller pieces (= batch size)
        -The rate/step size with which the Stochastic Gradient Descent will be performed (= learning rate)
        -The number of repetitions that we ran through all batches once (epochs)
        -The type of loss function to be used (default is LSE)
        """
        if reset_training:
            # Reset any training data from the previous iterations, option available in case we want to continue from last training
            self.reset_training() 
        self.pytorch_network.parameters()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)

        # Setting up the SGD optimizer and give feed parameters of network.
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=learning_rate)
        
        # Setting up the loss function
        if loss_function == 'LSE':
            self.loss_function = lambda outputs, targets: torch.sum((targets - outputs).pow(2))
        elif loss_function == 'CrossEntropy':
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function. Please use 'LSE' or 'CrossEntropy'.")
        return

    
    def generate_train_and_test(self, X, Y):
        """
        Generates test and train data for the neural net to train. 
        Can be used multiple times on one instance.
        Standard splitting is 80% train and 20% test.
        """
        dataset = torch.utils.data.TensorDataset(X, Y)

        # Determining sizes of sets: 80% train, and 20% test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        self.train_data, self.test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
        return self.train_data, self.test_data
    
    def reset_pytorch_network_params(self):
        """
        Resets the params of the pytorch network to start clean off
        """
        for m in self.pytorch_network.children():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        return
    
    def calculate_accuracy(self):
        """
        Calculates the percentage of correct guesses by the NN. 
        Only sensible for binary and multi-class classification.
        """
        if not self.classification:
            display_warning("calculate_accuracy()", "Accuracy calculations makes only sense with classification of the prediction. ")
            return
        
        Xtest, Ytest = self.test_data[:] 
        correct = 0
        number_of_nans = 0 
        for index in range(len(Ytest)):
            self.pytorch_network.eval() 
            y_pred = self.predict(Xtest[index])

            if self.contains_nan(y_pred, f"y_pred{index}"): 
                number_of_nans+=1
                if number_of_nans > len(Ytest)//100:
                    display_warning("calculate_accuracy()", "More than 1% of the predictions contain NaN values, cannot give representative accuracy. Please check your network architecture.")
                    break
                continue 

            if (y_pred.dim() == 1 and y_pred.numel() == 1): # Binary classification
                y_test = round(Ytest[index].item())
                y_pred = round(y_pred.item())

            else: # Multi-class classification
                y_test = torch.argmax(Ytest[index])
                y_pred = torch.argmax(y_pred)

            if y_test == y_pred:
                correct +=1
        return round(correct/len(Ytest), 2)*100

    def upload_pytorch_network(self, network):
        """
        IMPORTANT: this function serves as a sort of 'gateway', providing the wrapper the (PyTorch)
        network that it will uses to train
        """
        self.pytorch_network = network
        return
    

    def train_network_in_epoch(self):
        """
        Stands in for the training of the network in one epoch.
        """
        batch_loss = []
        self.pytorch_network.train()
        for inputs, targets in self.loader:
            self.optimizer.zero_grad()
            outputs = self.predict(inputs)
            loss = self.loss_function(outputs, targets)
            batch_loss.append(loss.item() / self.batch_size)
            loss.backward()
            self.optimizer.step()

        self.pytorch_network.eval()

        # Calculate and store the average training loss
        average_train_loss = np.mean(batch_loss)
        self.avg_train_loss.append(average_train_loss) if not self.contains_nan(average_train_loss, "average_train_loss") else None

        # Evaluate the model on the test data
        test_inputs, test_targets = self.test_data[:]
        test_outputs = self.predict(test_inputs)
        
        # Compute the average test loss
        average_test_loss = self.loss_function(test_outputs, test_targets) / len(self.test_data) 
        
        self.avg_test_loss.append(average_test_loss.item()) if not self.contains_nan(average_test_loss, 'average_test_loss') else None
        

    def train_network(self, plot=False):
        """
        Trains the network for the specified number of epochs.
        """
        for epoch in range(1, self.epochs+1):
            start_cpu_time = time.process_time() * 1000

            self.train_network_in_epoch()

            end_cpu_time = time.process_time() * 1000
            self.cpu_cycles.append(end_cpu_time - start_cpu_time)
            self.l_accuracy.append(self.calculate_accuracy()) if self.classification else None

            self.display_progress(epoch, plot) 
        return
    
    def visualize(self):
        """
        Visualization with three separate figures showing:
        1. Loss metrics (batch, training, and testing loss)
        2. Accuracy over epochs
        3. CPU cycle time per epoch
        Only displays the figures if the corresponding data is available/non-empty.
        """        
        # Figure 1: Losses
        if not len(self.avg_test_loss) == 0:
            _, ax1 = plt.subplots(figsize=(10, 4))
            ax1.set_yscale('log')
            ax1.plot(self.avg_train_loss, label="training loss")
            ax1.plot(self.avg_test_loss, label="testing loss")
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.set_title('Loss Metrics')
            x_anchor = 0.5 * ax1.get_xlim()[0] + 0.5 * ax1.get_xlim()[1]
            y_anchor = 10**(0.2*math.log10(ax1.get_ylim()[0]) + 0.8*math.log10(ax1.get_ylim()[1]))
            ax1.text(x_anchor, y_anchor, f"Test loss: {self.avg_test_loss[-1]:.2e}", ha='center', fontsize=12)
            plt.tight_layout()
        
        # Figure 2: Accuracy
        if not len(self.l_accuracy) == 0:
            _, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(self.l_accuracy, 'g-', label='Accuracy')
            ax2.set_title('Accuracy over Epochs')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_xlabel('Epoch')
            ax2.grid(True)
            ax2.text(0.02, 0.95, f"Current accuracy: {self.l_accuracy[-1]:.2f}%", 
                    transform=ax2.transAxes, fontsize=12)
            plt.tight_layout()
        
        # Figure 3: CPU Cycles
        if not len(self.cpu_cycles) == 0:
            _, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(self.cpu_cycles, 'r-', label='CPU Cycles')
            ax3.set_title('CPU Cycles per Epoch')
            ax3.set_ylabel('CPU Cycles (ms)')
            ax3.set_xlabel('Epoch')
            ax3.grid(True)
            ax3.text(0.02, 0.95, f"Last epoch cycles: {self.cpu_cycles[-1]:,} ms", 
                    transform=ax3.transAxes, fontsize=12)
            plt.tight_layout()

        plt.show()
    
    def predict(self, x):
        """
        Predicts the output of the network for a given input.
        """
        return self.pytorch_network(x)
    
    def reset_training(self):
        """
        This will reset any training done, and serves to give clean plots when trying to train again.
        """
        self.avg_train_loss = []
        self.avg_test_loss = []
        self.reset_pytorch_network_params()
        self.optimizer = None
        self.loader = None
        self.l_accuracy = []
        self.cpu_cycles = []
