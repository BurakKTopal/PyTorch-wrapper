import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
import os
from wrappers.base_pytorch_wrapper import BasePytorchWrapper

class ExtendedPytorchWrapper(BasePytorchWrapper):
    def __init__(self, X, Y, seed=None):
        self.generator = None
        if seed is not None:
            self.seed = seed
            self.set_seed()

        super().__init__(X, Y) # First need to setup the seed before continue

        self.l_accuracy = []
        self.cpu_cycles = []

    def set_seed(self):
        return
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

    def setup_training(self, batch_size, learning_rate, epochs, loss_function='LSE',  reset_training=True):
        super().setup_training(batch_size, learning_rate, epochs, loss_function, reset_training)
        # Override only the loader to use the generator
        self.loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, generator=self.generator)

    def _generate_train_and_test(self, X, Y):
        dataset = torch.utils.data.TensorDataset(X, Y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_data, self.test_data = torch.utils.data.random_split(dataset, [train_size, test_size], 
                                                                        generator=self.generator)
        
        for i, (data, target) in enumerate(self.train_data):
            print(f"Sample {i}:")
            print(f"Data: {data}")
            print(f"Target: {target}")
            print("-" * 20)
            if i==10:
                break
        return self.train_data, self.test_data

    def train_network(self, plot=False):
        for epoch in range(1, self.epochs + 1):
            start_cpu_time = time.process_time() * 1000
            self.set_seed()
            super().train_network_in_epoch()

            end_cpu_time = time.process_time() * 1000
            self.cpu_cycles.append(end_cpu_time - start_cpu_time)
            self.l_accuracy.append(self.calculate_accuracy())

            self._handle_plot(epoch, plot)


    def visualize(self):
        """
        Visualization with three separate figures showing:
        1. Loss metrics (batch, training, and testing loss)
        2. Accuracy over epochs
        3. CPU cycle time per epoch
        """
        clear_output(wait=True)
        
        # Figure 1: Losses
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.set_yscale('log')
        ax1.plot(self.train_L2, label="training loss")
        ax1.plot(self.test_L2, label="testing loss")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Loss Metrics')
        x_anchor = 0.5 * ax1.get_xlim()[0] + 0.5 * ax1.get_xlim()[1]
        y_anchor = 10**(0.2*math.log10(ax1.get_ylim()[0]) + 0.8*math.log10(ax1.get_ylim()[1]))
        ax1.text(x_anchor, y_anchor, f"Test loss: {self.test_L2[-1]:.2e}", ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Accuracy
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(self.l_accuracy, 'g-', label='Accuracy')
        ax2.set_title('Accuracy over Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_xlabel('Epoch')
        ax2.grid(True)
        if self.l_accuracy:  # Only add text if we have accuracy data
            ax2.text(0.02, 0.95, f"Current accuracy: {self.l_accuracy[-1]:.2f}%", 
                     transform=ax2.transAxes, fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Figure 3: CPU Cycles
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(self.cpu_cycles, 'r-', label='CPU Cycles')
        ax3.set_title('CPU Cycles per Epoch')
        ax3.set_ylabel('CPU Cycles (ms)')
        ax3.set_xlabel('Epoch')
        ax3.grid(True)
        if self.cpu_cycles:  # Only add text if we have CPU cycle data
            ax3.text(0.02, 0.95, f"Last epoch cycles: {self.cpu_cycles[-1]:,} ms", 
                     transform=ax3.transAxes, fontsize=12)
        plt.tight_layout()
        plt.show()
        
        
    def reset_training(self):
        super().reset_training()
        self.l_accuracy = []
        self.cpu_cycles = []
        self.generator = None
        if self.seed:
            self.set_seed()  