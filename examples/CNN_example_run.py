from utils.NN_generators import  CNNGenerator
from examples.xNN_example_run import x_NN_example_run
from wrappers.network_types import NetworkType
def CNN_example_run(learning_rate, batch_size, epochs, plot):

    network = CNNGenerator(
    input_channels=1,
    conv_layers=[{
        'out_channels': 16,
        'kernel_size': (2, 2),
        'stride': 1,
        'padding': 0
    }],
    fc_layers=[500],
    output_size=10,
    batch_size=batch_size,
    use_pooling=False
    )
    network_type = NetworkType.CNN
    x_NN_example_run(learning_rate, batch_size, epochs, plot, network, network_type)
    return
