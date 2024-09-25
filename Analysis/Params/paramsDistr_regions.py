import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from sympy import symbols, Eq, solve, sqrt, pi

# Adjust the path to include the project_root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sparch.models.snns import SNN
from sparch.models.snns import LIFcomplexLayer, adLIFclampLayer

dataset_name = "SHD"
batch_size = 128
nb_inputs = 700
nb_outputs = 20 if dataset_name == "shd" else 35
neuron_model = 'LIFcomplex'
nb_hiddens = 512
nb_layers = 3

layer_sizes = [nb_hiddens] * (nb_layers - 1) + [nb_outputs]
input_shape = (batch_size, None, nb_inputs)

extra_features = {
    'superspike': False,
    'slayer': False,
    'xavier_init': False,
    'no_reset': False,
    'half_reset': True,
    'rst_detach': False,
    'dt_min': 0.01,
    'dt_max': 0.4,
    's_GLU': False,
    'time_offset': 10,
    'thr': 1,
    'bRand': 'Rand',
    'c_discr': False,
    'c_param': False,
    'reset': 'half_reset'
}

net = SNN(
    input_shape=input_shape,
    layer_sizes=layer_sizes,
    neuron_type=neuron_model,
    dropout=0.1,
    normalization="batchnorm",
    use_bias=False,
    bidirectional=False,
    use_readout_layer=True,
    extra_features=extra_features
)

# Load the trained model
if neuron_model == 'adLIFclamp':
    print('path')
    model_path = '../../exp/test_exps/shd_adLIFclamp_3lay512_drop0_1_batchnorm_nobias__/checkpoints/best_model.pth'
elif neuron_model == 'LIFcomplex':
    model_path = '../../exp/test_exps/shd_LIFcomplex_3lay512_drop0_1_batchnorm_nobias_udir_noreg_lr0_01/checkpoints/best_model.pth'


trained_net = torch.load(model_path)
def find_system_eigenvalues_numeric(tau_m, tau_w, R, a_w):
    A = np.array([[-1/tau_m, -R/tau_m], [a_w/tau_w, -1/tau_w]])
    eigenvalues_matrix, eigenvectors = np.linalg.eig(A)
    eigenvalues = np.diag(eigenvalues_matrix)
    return eigenvalues, eigenvectors

def plot_eigenvalue_distribution():
    R = 1  # You may adjust this value for different systems
    tau_m_values = np.linspace(-1.0001, -0.94, 500) 
    a_R_values = np.linspace(-1.004, -0.998, 500)

    # Set up the plot
    plt.figure(figsize=(8, 6))

    for tau_m_over_tau_w in tau_m_values:
        for a_R in a_R_values:
            tau_m = 1  # Arbitrary tau_m value
            tau_w = tau_m / tau_m_over_tau_w
            a_w = a_R / R
            eigenvalues, _ = find_system_eigenvalues_numeric(tau_m, tau_w, R, a_w)
            
            # Compute the Nullstelle (1/tau_w)
            nullstelle = -1 / tau_w
            eigenvalues = [eigenvalues[0][0],eigenvalues[1][1]]
            if (eigenvalues[0]>0 and eigenvalues[0]!=nullstelle) or (eigenvalues[1]>0 and eigenvalues[1]!=nullstelle):
                plt.plot(tau_m_over_tau_w, a_R, 'gs', alpha=0.2)
            # If eigenvalues are complex
            elif np.iscomplex(eigenvalues).any():
                plt.plot(tau_m_over_tau_w, a_R, 'bs', alpha=0.2)  # Blue dots for complex
            else:
                # Compare Nullstelle with min(eigenvalue)
                if nullstelle > eigenvalues[0] and nullstelle > eigenvalues[1]:
                    plt.plot(tau_m_over_tau_w, a_R, 'rs', alpha=0.2)  # Red dots if Nullstelle < min(eigenvalue)
                else:
                    plt.plot(tau_m_over_tau_w, a_R, 'ys', alpha=0.2)  # Yellow dots if Nullstelle >= min(eigenvalue)

    # Create custom legend handles with correct colors
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, alpha=1, label='Oscillatory Dynamics'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, alpha=1, label='Non-oscillatory (Mixed) Dynamics'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=10, alpha=1, label='Non-oscillatory (integrator) Dynamics'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, alpha=1, label='Unstable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, alpha=1, label='Neuron parameters after training'),
    ]

    # Add legend
    plt.legend(handles=legend_elements)

    # Add labels and title
    plt.xlabel(r'$\tau_m / \tau_w$')
    plt.ylabel(r'$a_R$')
    plt.title('Eigenvalues in function of $\\tau_m / \\tau_w$ and $a_R$')
    plt.grid(True)

def compute_alpha(log_log_alpha, alpha_img, log_dt):
    alpha = torch.exp((-torch.exp(log_log_alpha) + 1j * alpha_img) * torch.exp(log_dt))
    return alpha

def compute_rat_R_a(real_value, img_value, tau_m=0.01):
    # Define symbols
    rat, R_a = symbols('rat R_a')

    # Define the equations
    eq1 = Eq(-(rat + 1) / (2 * tau_m), real_value)
    eq2 = Eq(sqrt((rat**2 - 2*rat + 1 - 4*R_a * rat)*(-1)) / (2 * tau_m), img_value)

    # Solve the system of equations
    solutions = solve([eq1, eq2], (rat, R_a))
    return solutions

# Function to plot rat and R_a over the eigenvalue distribution plot
def plot_rat_vs_R_a(layer, trained_layer, i, tau_m=0.01):
    log_log_alpha = layer.log_log_alpha.detach().cpu()
    alpha_img = layer.alpha_img.detach().cpu()
    log_dt = layer.log_dt.detach().cpu()

    trained_log_log_alpha = trained_layer.log_log_alpha.detach().cpu()
    trained_alpha_img = trained_layer.alpha_img.detach().cpu()
    trained_log_dt = trained_layer.log_dt.detach().cpu()

    trained_continuous_alpha = (-torch.exp(trained_log_log_alpha) + 1j * trained_alpha_img).detach().cpu()

    rat_values = []
    R_a_values = []
    for alpha_val in trained_continuous_alpha:
        real_value = alpha_val.real.item()
        img_value = alpha_val.imag.item()
        solutions = compute_rat_R_a(real_value, img_value, tau_m)

        for solution in solutions:
            rat_values.append(solution[0])  # Add all solutions for rat
            R_a_values.append(solution[1])  # Add all solutions for R_a


    # Superimpose the rat vs R_a plot on the eigenvalue distribution
    plt.scatter(rat_values, R_a_values, color='black', marker='o', label=f'rat vs R_a (layer {i})', alpha=1, zorder=3)
    plt.xlabel(' $\\tau_m / \\tau_w$')
    plt.ylabel('aR')
    plt.title(f'subthreshold parameters of complexLIF for layer {i}')
    plt.grid(True)

# Main Plot Function
def plot_combined_eigenvalues_and_rat_R_a(net, trained_net):
    plot_eigenvalue_distribution()  # First plot the eigenvalue distribution
    
    # Loop over the layers and plot rat vs R_a
    for i, (layer, trained_layer) in enumerate(zip(net.snn, trained_net.snn)):
        if isinstance(layer, LIFcomplexLayer) and isinstance(trained_layer, LIFcomplexLayer):
            plot_rat_vs_R_a(layer, trained_layer, i)
            if neuron_model == 'adLIFclamp':
                plt.savefig(f"../../plots/SHD/Params/adLIFclamprat_vs_R_a_layer{i}.png")
            elif neuron_model == 'LIFcomplex':
                plt.savefig(f"../../plots/SHD/Params/cmplxLIFrat_vs_R_a_layer{i}.png")
            plt.show()

if neuron_model == 'adLIFclamp':
    print("plot")
    # Plot both initial and trained alpha values and corresponding rat and R_a values
    for i, trained_layer in enumerate( trained_net.snn):
        print(trained_layer)
        if isinstance(trained_layer, adLIFclampLayer):
            print("plot1")
            plot_combined_eigenvalues_and_rat_R_a(net, trained_net)
elif neuron_model == 'LIFcomplex':
    # Plot both initial and trained alpha values and corresponding rat and R_a values
    for i, (layer, trained_layer) in enumerate(zip(net.snn, trained_net.snn)):
        if isinstance(layer, LIFcomplexLayer) and isinstance(trained_layer, LIFcomplexLayer):
            plot_combined_eigenvalues_and_rat_R_a(net, trained_net)

print(net)
