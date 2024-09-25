import sys
import os
import torch
import torch.nn as nn
import pandas as pd

# Adjust the path to include the directory containing sparch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sparch')))

from models.snns import S4Model

# Define the range of values for each parameter
nb_layers_values = [2, 3]
nb_state_values = [2, 4, 64, 128]
nb_hiddens_values = [128, 512]

dataset_name = "SHD"
batch_size = 128
nb_inputs = 700 / 5 
nb_outputs = 20 if dataset_name == "shd" else 35
nb_state = 2

# Prepare a list to store the results
results = []

# Function to count the number of parameters in millions
def get_nb_params(input_shape):
    net = S4Model(
        d_input=700,  # example value, replace with actual value
        d_output=35,  # example value, replace with actual value
        d_model=nb_hiddens,  # use current value
        d_state=nb_state,  # use current value
        n_layers=nb_layers,  # use current value
        dropout=0.2,  # example value, replace with actual value
        prenorm=False,  # example value, replace with actual value
        lr=0.01,  # example value, replace with actual value
        batch_size=32,  # example value, replace with actual value
        normalization="batchnorm",  # example value, replace with actual value
        extra_features={
            "pure_complex": True,  # example value, replace with actual value
            "dt_min": 0.001,  # example value, replace with actual value
            "dt_max": 0.1,  # example value, replace with actual value
            "activation": "GELU",  # example value, replace with actual value
            "premix": False,  # example value, replace with actual value
            "mix": "GLU",  # example value, replace with actual value
            "residual1": True,  # example value, replace with actual value
            "residual2": True,  # example value, replace with actual value
            "drop2": True,  # example value, replace with actual value
            "use_readout_layer": True,  # example value, replace with actual value
            "time_offset": 0
        }
    ).to(torch.device("cuda"))
    
    param_info = {}
    total_params = 0
    
    for name, param in net.named_parameters():
        if param.requires_grad:
            param_info[name] = param.numel()  # Store parameter name and its count
            total_params += param.numel()
    
    return param_info, total_params
# Iterate over all combinations of parameters
nb_layers = 2
nb_hiddens_list = [128,  512, ]
for nb_hiddens in nb_hiddens_list:
    # Define layer sizes: [hidden_size, ..., hidden_size, output_size]
    input_shape = (batch_size, None, nb_inputs)  # Shape of the input

    # Get the parameter names and count
    param_info, total_params = get_nb_params(input_shape)


    # Print the model type, number of layers, hidden size, and total parameters
    print(f" | Layers: {nb_layers} | Hidden units: {nb_hiddens} | Total params: {total_params/1000000}M")

    # Print the names and count of each parameter
    for param_name, count in param_info.items():
        print(f"  {param_name}: {count}")

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the results to a CSV file
df.to_csv('s4model_parameters_M.csv', index=False)

# Print the DataFrame
print(df)
