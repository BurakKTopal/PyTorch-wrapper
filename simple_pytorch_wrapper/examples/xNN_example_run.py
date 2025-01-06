from simple_pytorch_wrapper.examples.load_example_dataset import load_language_digit_example_dataset
from simple_pytorch_wrapper.wrapper.pytorch_wrapper import PytorchWrapper

def xNN_example_run(learning_rate, batch_size, epochs, plot, network, network_type, seed=None):

    X, Y = load_language_digit_example_dataset()

    X, Y = PytorchWrapper.vectorize_image_data(X, Y, network_type) 

    wrapper = PytorchWrapper(X, Y)  
    
    wrapper.upload_pytorch_network(network) 

    wrapper.setup_training(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)

    wrapper.train_network(plot=plot) 

    wrapper.visualize()

    accuracy = wrapper.calculate_accuracy()
    print(f"The final accuracy is: {accuracy}%")
    
    return