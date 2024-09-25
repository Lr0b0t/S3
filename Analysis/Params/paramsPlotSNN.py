import torch
import torch.nn as nn
from models.snns import SNN

# Define parameters
nb_layers = 3
nb_hiddens_list = [128, 512]
dataset_name = "SHD"
batch_size = 128
nb_inputs = 700
nb_outputs = 20 if dataset_name.lower() == "shd" else 35
model_types = ["LIF", "adLIF", "RLIF", "RadLIF", "LIFcomplex", "RLIFcomplex1MinAlpha"]

# Function to calculate the number of parameters
def get_nb_params(input_shape, layer_sizes, neuron_type):
    net = SNN(
        input_shape=input_shape,
        layer_sizes=layer_sizes,
        neuron_type=neuron_type,
        dropout=0.1,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        use_readout_layer=True,
    )
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

# Prepare dictionary to store number of parameters for each model type and hidden size
nb_params_hiddens = {model_type: [] for model_type in model_types}

# Iterate over model types and hidden layer sizes
for model_type in model_types:
    for nb_hiddens in nb_hiddens_list:
        # Define layer sizes (same hidden size for all layers except the last)
        layer_sizes = [nb_hiddens] * (nb_layers - 1) + [nb_outputs]
        input_shape = (batch_size, None, nb_inputs)
        
        # Get number of parameters for the current configuration
        nb_params = get_nb_params(input_shape, layer_sizes, model_type)
        
        # Store the result in the dictionary
        nb_params_hiddens[model_type].append(nb_params)
        
        # Print the result
        print(f"{model_type}: {nb_layers} layers, {nb_hiddens} hidden units, {nb_params/ 1e6 } M parameters")

# If needed, you can store the results in a DataFrame or output it to a CSV file as well
