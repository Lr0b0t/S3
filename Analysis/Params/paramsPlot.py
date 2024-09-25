import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Adjust the path to include the project_root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sparch')))

from models.snns import SNN

dataset_name = "SHD"
batch_size = 128
nb_inputs = 700 
nb_outputs = 20 if dataset_name == "shd" else 35
model_types = ["LIFcomplex"]
extra_features = {
    "thr": 1,               # Threshold value
    "bRand": "RandN",         # Bias initialization method: "RandN" for normal distribution, "Rand" for uniform
    "superspike": False,      # Boolean to use SuperSpike spike function
    "slayer": False,           # Boolean to use SLAYER spike function (only one of superspike or slayer should be True)
    "xavier_init": False,      # Use Xavier initialization for weights
    "dt_min": 0.01,          # Minimum value for log(dt)
    "dt_max": 1,           # Maximum value for log(dt)
    "c_discr": False,         # Whether to discretize c values (use continuous if False)
    "c_param": False,          # Whether to use complex-valued c parameters
    "reset": "half_reset",    # Reset type: "no_reset", "half_reset", or "reset" (1.0)
    "rst_detach": False,      # Boolean for resetting with gradient detachment
    "s_GLU": False,
    "time_offset":0                        # Boolean for using GLU in the output layer
}

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
        extra_features=extra_features
    )
    
    param_info = {}
    total_params = 0
    
    for name, param in net.named_parameters():
        if param.requires_grad:
            param_info[name] = param.numel()  # Store parameter name and its count
            total_params += param.numel()
    
    return param_info, total_params

# Data for the second plot (number of parameters vs number of hidden units)
nb_layers = 3
nb_hiddens_list = [128,  512, ]
nb_params_hiddens = {model_type: [] for model_type in model_types}

# Assuming 'model_types', 'nb_hiddens_list', 'nb_layers', 'nb_outputs', 'batch_size', 'nb_inputs', and 'nb_params_hiddens' are defined

for model_type in model_types:
    for nb_hiddens in nb_hiddens_list:
        # Define layer sizes: [hidden_size, ..., hidden_size, output_size]
        layer_sizes = [nb_hiddens] * (nb_layers - 1) + [nb_outputs]
        input_shape = (batch_size, None, nb_inputs)  # Shape of the input

        # Get the parameter names and count
        param_info, total_params = get_nb_params(input_shape, layer_sizes, model_type)

        # Store the total parameter count in nb_params_hiddens
        nb_params_hiddens[model_type].append(total_params)

        # Print the model type, number of layers, hidden size, and total parameters
        print(f"{model_type} | Layers: {nb_layers} | Hidden units: {nb_hiddens} | Total params: {total_params/1000000}M")

        # Print the names and count of each parameter
        for param_name, count in param_info.items():
            print(f"  {param_name}: {count}")

# Plotting
'''
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
'''