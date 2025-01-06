from simple_pytorch_wrapper.utils.NN_generators import  CNNGenerator
from simple_pytorch_wrapper.examples.xNN_example_run import xNN_example_run
from simple_pytorch_wrapper.utils.network_types import NetworkType
from simple_pytorch_wrapper.utils.warnings import suppress_warnings

def CNN_example_run(learning_rate, batch_size, epochs, plot):
    suppress_warnings()
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
    image_height=64,
    image_width=64,
    use_pooling=False
    )
    network_type = NetworkType.CNN
    xNN_example_run(learning_rate, batch_size, epochs, plot, network, network_type)
    return
