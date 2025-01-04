from simple_pytorch_wrapper.examples.load_example_dataset import load_language_digit_example_dataset
from simple_pytorch_wrapper.wrapper.pytorch_wrapper import PytorchWrapper

def xNN_example_run(learning_rate, batch_size, epochs, plot, network, network_type, seed=None):

    # Upload data
    X, Y = load_language_digit_example_dataset()

    # Transform data accordingly (vectorizing) + squeezing for batching
    X, Y = PytorchWrapper.vectorize_data(X, Y, network_type) # Needed to transform the data to the correct format for the input

    # Extended wrapper
    wrapper = PytorchWrapper(X, Y)  # Using the extended wrapper with a seed
    
    wrapper.upload_pyTorch_network(network) # CNN network is also possible

    wrapper.setup_training(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)

    # Training
    wrapper.train_network(plot=plot)  # Enable plotting

    # Training ended
    wrapper.visualize()

    # Evaluating accuracy
    accuracy = wrapper.calculate_accuracy()
    print(f"The final accuracy is: {accuracy}%")
    
    return

