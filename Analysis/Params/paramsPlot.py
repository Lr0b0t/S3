import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Adjust the path to include the project_root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sparch.models.snns import SNN

dataset_name = "SHD"
batch_size = 128
nb_inputs = 700
nb_outputs = 20 if dataset_name == "shd" else 35
model_types = ["LIF", "adLIF", "RLIF","RadLIF", "LIFcomplex", "RLIFcomplex1MinAlpha"]

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

# Data for the first plot (number of parameters vs number of layers)
nb_hiddens = 128
nb_layers_list = [2, 3, 4, 5, 6, 7]
nb_params_layers = {model_type: [] for model_type in model_types}

for model_type in model_types:
    for nb_layers in nb_layers_list:
        layer_sizes = [nb_hiddens] * (nb_layers - 1) + [nb_outputs]
        input_shape = (batch_size, None, nb_inputs)
        nb_params = get_nb_params(input_shape, layer_sizes, model_type)
        nb_params_layers[model_type].append(nb_params)

# Data for the second plot (number of parameters vs number of hidden units)
nb_layers = 3
nb_hiddens_list = [64, 128, 256, 512, 1024, 2048, 3072, 4096]
nb_params_hiddens = {model_type: [] for model_type in model_types}

for model_type in model_types:
    for nb_hiddens in nb_hiddens_list:
        layer_sizes = [nb_hiddens] * (nb_layers - 1) + [nb_outputs]
        input_shape = (batch_size, None, nb_inputs)
        nb_params = get_nb_params(input_shape, layer_sizes, model_type)
        nb_params_hiddens[model_type].append(nb_params)
        print(model_type+' '+str(nb_layers)+' '+str(nb_hiddens)+' '+str(nb_params))

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# First subplot
for model_type in model_types:
    ax1.plot(nb_layers_list, nb_params_layers[model_type], marker='o', label=model_type)
ax1.set_xlabel("Number of Layers")
ax1.set_ylabel("Number of Parameters")
ax1.set_title("Number of Parameters vs Number of Layers")
ax1.legend()

# Second subplot
for model_type in model_types:
    ax2.plot(nb_hiddens_list, nb_params_hiddens[model_type], marker='o', label=model_type)
ax2.set_xlabel("Number of Hidden Units")
ax2.set_ylabel("Number of Parameters")
ax2.set_title("Number of Parameters vs Number of Hidden Units")
ax2.legend()

plt.tight_layout()
plt.savefig("plots/SHD/params_plot.png")
plt.show()
