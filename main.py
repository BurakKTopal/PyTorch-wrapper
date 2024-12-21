from helpers import loading_data
from wrappers.ExtendedPytorchWrapper import ExtendedPytorchWrapper
from wrappers.BasePytorchWrapper import BasePytorchWrapper
from config import *

def main():
    folder_name = "./sign-language-digits-dataset"
    X, Y = loading_data(folder_name)

    # REMOVE IF NOT USING IMAGES
    X, Y = BasePytorchWrapper.transform_image_data(X, Y, NETWORK_TYPE) # Needed to transform the data to the correct format for the input

    # Base wrapper
    # wrapper = BasePytorchWrapper(X, Y)  # Using the extended wrapper with a seed

    # Extended wrapper
    wrapper = ExtendedPytorchWrapper(X, Y, seed=42)  # Using the extended wrapper with a seed
    
    wrapper.upload_pyTorch_network(NETWORK) # CNN network is also possible

    wrapper.setup_training(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS)

    # Training
    wrapper.train_network(plot=False)  # Enable plotting

    # Training ended
    wrapper.visualize()

    # Evaluating accuracy
    accuracy = wrapper.calculate_accuracy()
    print(f"The accuracy is: {accuracy}%")


if __name__ == "__main__":
    main()