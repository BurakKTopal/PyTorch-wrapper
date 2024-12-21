# PyTorch-wrapper

A flexible PyTorch wrapper that simplifies the process of training and setting up arbitrary Neural Network architectures. This wrapper provides an interface for both standard neural networks and CNNs, but can be extended to any architecture, with built-in visualization and performance tracking capabilities.

## Features

- Easy setup of neural network architectures (both standard and CNN)
- Built-in train/test split functionality (80/20)
- Configurable training parameters via config.py
- Real-time visualization of:
  - Loss metrics (batch, training, and testing loss)
  - Accuracy over epochs
  - CPU cycle time per epoch
- Support for different loss functions (LSE, CrossEntropy)
- Reproducible results with seed setting
- Progress tracking during training

## Configuration

All network and training parameters can be configured in `config.py`. This includes:
- Network architecture (FNN or CNN)
- Network parameters (layers, neurons, etc.)
- Training parameters:
  - Batch size
  - Learning rate
  - Number of epochs
  - Loss function

## Installation

### Requirements
- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- IPython (for visualization)

## Quick Start

1. Clone the repository
2. Modify `config.py` to set your desired network architecture and training parameters
3. Run `main.py` to start training

## Dataset

The data used for the example is the Sign Language Digits Dataset from the Turkey Ankara Ayrancı Anadolu High School students. This dataset is available at [ardamavi/Sign-Language-Digits-Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset).
