import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
from matplotlib.patches import Circle

# Adjust the path to include the project_root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sparch.models.snns import SNN
from sparch.models.snns import LIFcomplexLayer

dataset_name = "SHD"
batch_size = 128
nb_inputs = 700
nb_outputs = 20 if dataset_name == "shd" else 35
neuron_type = "LIFcomplex"

nb_hiddens = 512
nb_layers = 3

layer_sizes = [nb_hiddens] * (nb_layers - 1) + [nb_outputs]
input_shape = (batch_size, None, nb_inputs)

extra_features = {
    'superspike': False,      # Use SuperSpike spike function
    'slayer': False,          # Use SLAYER spike function (set both to False to use Boxcar)
    'xavier_init': False,      # Use Xavier initialization
    'no_reset': False,        # Disable reset (if True, reset_factor = 0)
    'half_reset': True,      # Use half reset (if True, reset_factor = 0.5)
    'rst_detach': False,      # Detach the reset during gradient computation
    'dt_min': 0.01,           # Minimum time constant for neuron dynamics
    'dt_max': 0.4,           # Maximum time constant for neuron dynamics
    's_GLU': False,            # Whether to use Gated Linear Units in output layer
    'time_offset':10,
    'thr':1,
    'bRand':'Rand',
    'c_discr':False,
    'c_param':False,
    'reset':'half_reset'
}
net = SNN(
    input_shape=input_shape,
    layer_sizes=layer_sizes,
    neuron_type=neuron_type,
    dropout=0.1,
    normalization="batchnorm",
    use_bias=False,
    bidirectional=False,
    use_readout_layer=True,
    extra_features=extra_features  # Pass the extra features here
)

# Load the trained model
model_path = '../../exp/test_exps/shd_LIFcomplex_3lay512_drop0_1_batchnorm_nobias_udir_noreg_lr0_01/checkpoints/best_model.pth'
trained_net = torch.load(model_path)

def compute_alpha(log_log_alpha, alpha_img, log_dt):
    # Compute alpha as per your formula
    alpha = torch.exp((-torch.exp(log_log_alpha) + 1j * alpha_img) * torch.exp(log_dt))
    return alpha

def plot_alpha_distribution(layer, trained_layer, i):
    # Extract the values of log_log_alpha, alpha_img, and log_dt from each layer
    log_log_alpha = layer.log_log_alpha.detach().cpu()
    alpha_img = layer.alpha_img.detach().cpu()
    log_dt = layer.log_dt.detach().cpu()

    # Compute alpha and its conjugate for the initial model
    alpha = compute_alpha(log_log_alpha, alpha_img, log_dt)
    alpha_conjugate = alpha.conj()

    # Compute continuous alpha values and their conjugates
    continuous_alpha = (-torch.exp(log_log_alpha) + 1j * alpha_img).detach().cpu()
    continuous_alpha_conjugate = continuous_alpha.conj()

    # Extract the values of log_log_alpha, alpha_img, and log_dt from the trained layer
    trained_log_log_alpha = trained_layer.log_log_alpha.detach().cpu()
    trained_alpha_img = trained_layer.alpha_img.detach().cpu()
    trained_log_dt = trained_layer.log_dt.detach().cpu()

    # Compute alpha and its conjugate for the trained model
    trained_alpha = compute_alpha(trained_log_log_alpha, trained_alpha_img, trained_log_dt)
    trained_alpha_conjugate = trained_alpha.conj()

    # Compute continuous alpha values and their conjugates for trained model
    trained_continuous_alpha = (-torch.exp(trained_log_log_alpha) + 1j * trained_alpha_img).detach().cpu()
    trained_continuous_alpha_conjugate = trained_continuous_alpha.conj()

    # Convert to numpy arrays with resolve_neg()
    alpha_real = alpha.real.resolve_neg().numpy()
    alpha_imag = alpha.imag.resolve_neg().numpy()
    alpha_conjugate_real = alpha_conjugate.real.resolve_neg().numpy()
    alpha_conjugate_imag = alpha_conjugate.imag.resolve_neg().numpy()

    trained_alpha_real = trained_alpha.real.resolve_neg().numpy()
    trained_alpha_imag = trained_alpha.imag.resolve_neg().numpy()
    trained_alpha_conjugate_real = trained_alpha_conjugate.real.resolve_neg().numpy()
    trained_alpha_conjugate_imag = trained_alpha_conjugate.imag.resolve_neg().numpy()

    continuous_alpha_real = continuous_alpha.real.resolve_neg().numpy()
    continuous_alpha_imag = continuous_alpha.imag.resolve_neg().numpy()
    continuous_alpha_conjugate_real = continuous_alpha_conjugate.real.resolve_neg().numpy()
    continuous_alpha_conjugate_imag = continuous_alpha_conjugate.imag.resolve_neg().numpy()

    trained_continuous_alpha_real = trained_continuous_alpha.real.resolve_neg().numpy()
    trained_continuous_alpha_imag = trained_continuous_alpha.imag.resolve_neg().numpy()
    trained_continuous_alpha_conjugate_real = trained_continuous_alpha_conjugate.real.resolve_neg().numpy()
    trained_continuous_alpha_conjugate_imag = trained_continuous_alpha_conjugate.imag.resolve_neg().numpy()

    # Compute histograms
    hist_log_dt = torch.exp(log_dt).detach().cpu().numpy()
    hist_trained_log_dt = torch.exp(trained_log_dt).detach().cpu().numpy()

    # Create a figure and axis for alpha distributions
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    # Create separate unit circle instances
    unit_circle1 = Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1.5)
    unit_circle2 = Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1.5)

    # Plot the initial alpha values in the complex plane
    axs[0, 0].scatter(alpha_real, alpha_imag, color='green', alpha=0.3, marker='o', label='Alpha')
    axs[0, 0].scatter(alpha_conjugate_real, alpha_conjugate_imag, color='red', alpha=0.3, marker='o', label='Alpha Conjugate')
    axs[0, 0].axhline(0, color='black', linewidth=0.5)
    axs[0, 0].axvline(0, color='black', linewidth=0.5)
    axs[0, 0].add_patch(unit_circle1)
    axs[0, 0].set_aspect('equal', 'box')
    axs[0, 0].set_xlim([-1.1, 1.1])
    axs[0, 0].set_ylim([-1.1, 1.1])
    axs[0, 0].set_title(f'Discrete Alpha Distribution for Initial Model Layer {i}')
    axs[0, 0].set_xlabel('Real part')
    axs[0, 0].set_ylabel('Imaginary part')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Plot the trained alpha values in the complex plane
    axs[0, 1].scatter(trained_alpha_real, trained_alpha_imag, color='green', alpha=0.6, marker='o', label='Alpha')
    axs[0, 1].scatter(trained_alpha_conjugate_real, trained_alpha_conjugate_imag, color='red', alpha=0.6, marker='o', label='Alpha Conjugate')
    axs[0, 1].axhline(0, color='black', linewidth=0.5)
    axs[0, 1].axvline(0, color='black', linewidth=0.5)
    axs[0, 1].add_patch(unit_circle2)
    axs[0, 1].set_aspect('equal', 'box')
    axs[0, 1].set_xlim([-1.1, 1.1])
    axs[0, 1].set_ylim([-1.1, 1.1])
    axs[0, 1].set_title(f'Discrete Alpha Distribution for Trained Model Layer {i}')
    axs[0, 1].set_xlabel('Real part')
    axs[0, 1].set_ylabel('Imaginary part')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Plot the continuous alpha values for the initial model
    axs[1, 0].scatter(continuous_alpha_real, continuous_alpha_imag, color='blue', alpha=0.6, marker='o', label='Continuous Alpha')
    axs[1, 0].scatter(continuous_alpha_conjugate_real, continuous_alpha_conjugate_imag, color='orange', alpha=0.6, marker='o', label='Continuous Alpha Conjugate')
    axs[1, 0].axhline(0, color='black', linewidth=0.5)
    axs[1, 0].axvline(0, color='black', linewidth=0.5)
    axs[1, 0].set_aspect('auto')
    axs[1, 0].set_title(f'Continuous Alpha Distribution for Initial Model Layer {i}')
    axs[1, 0].set_xlabel('Real part')
    axs[1, 0].set_ylabel('Imaginary part')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Plot the continuous alpha values for the trained model
    axs[1, 1].scatter(trained_continuous_alpha_real, trained_continuous_alpha_imag, color='blue', alpha=0.6, marker='o', label='Continuous Alpha')
    axs[1, 1].scatter(trained_continuous_alpha_conjugate_real, trained_continuous_alpha_conjugate_imag, color='orange', alpha=0.6, marker='o', label='Continuous Alpha Conjugate')
    axs[1, 1].axhline(0, color='black', linewidth=0.5)
    axs[1, 1].axvline(0, color='black', linewidth=0.5)
    axs[1, 1].set_aspect('auto')
    axs[1, 1].set_title(f'Continuous Alpha Distribution for Trained Model Layer {i}')
    axs[1, 1].set_xlabel('Real part')
    axs[1, 1].set_ylabel('Imaginary part')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # Save the plot of alpha distributions
    plt.savefig(f"../../plots/SHD/Params/params_plot{i}.png")
    plt.close()

    # Create a separate figure for the histogram of log_dt
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    ax_hist.hist(hist_trained_log_dt, bins=30, color='purple', alpha=0.5, label='Trained log_dt exp')  # Adjust alpha to 0.5
    ax_hist.hist(hist_log_dt, bins=30, color='orange', alpha=0.5, label='Initial log_dt exp')  # Adjust alpha to 0.5
    ax_hist.set_title(f'dt distribution for Layer {i}')
    ax_hist.set_xlabel('Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.legend()
    ax_hist.grid(True)


    # Save the histogram plot
    plt.savefig(f"../../plots/SHD/Params/histogram_plot{i}.png")
    plt.close()


# Plot both initial and trained alpha values
for i, (layer, trained_layer) in enumerate(zip(net.snn, trained_net.snn)):
    if isinstance(layer, LIFcomplexLayer) and isinstance(trained_layer, LIFcomplexLayer):
        plot_alpha_distribution(layer, trained_layer, i)

print(net)
