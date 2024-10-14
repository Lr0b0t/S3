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
from sparch.models.snns import LIFcomplexLayer, adLIFclampLayer, BRFLayer, ResonateFireLayer

dataset_name = "ssc"
batch_size = 128
nb_inputs = 700
nb_outputs = 20 if dataset_name == "shd" else 35
model_type = "LIFcomplex"

nb_hiddens = 512
nb_layers = 3
layer_sizes = [nb_hiddens] * (nb_layers - 1) + [nb_outputs]
input_shape = (batch_size, None, nb_inputs)
extra_features = {
    'alpha_max': 1,
    'alpha_min': 0.1,
    'batch_size': 256,
    'bidirectional': False,
    'c_sum': False,
    'clamp_alpha': False,
    'complex_reset': False,
    'continuous': False,
    'data_folder': "//Local/datasets/ssc/",
    'dataset_name': "ssc",
    'debug': False,
    'dt_max': 25.2,
    'dt_min': 0.01,
    'dt_train': False,
    'dt_uniform': False,
    'exp_factor': 2,
    'extra_b': False,
    'gamma_norm': False,
    'gpu_device': 0,
    'half_reset': True,
    'jaxreadout': False,
    'layernorm_readout': False,
    'load_exp_folder': None,
    'log_tofile': True,
    'lr': 0.005,
    'LRU_b': False,
    'LRU_img_init': True,
    'LRU_max_phase': 3.14/2,
    'LRU_no_dt': True,
    'LRU_norm': False,
    'LRU_r_max': 0.9,
    'LRU_r_min': 0.9,
    'LRU_re_init': True,
    'max_phase': 6.28,
    'max_time': 1.2,
    'model_type': "LIFcomplex",
    'nb_epochs': 100,
    'nb_hiddens': 512,
    'nb_layers': 3,
    'nb_steps': 300,
    'new_exp_folder': None,
    'no_b': False,
    'no_reset': False,
    'normalization': "batchnorm",
    'num_workers': 8,
    'only_do_testing': False,
    'pdrop': 0.55,
    'r_max': 0.9,
    'r_min': 0,
    'recurrent': False,
    'reg_factor': 0.5,
    'reg_fmax': 0.5,
    'reg_fmin': 0.01,
    'relu_spike': False,
    'reparam': False,
    'residual': False,
    'ro_int': 0,
    'rst_detach': False,
    's4_init': False,
    's4_opt': False,
    'save_best': False,
    'scheduler_factor': 0.7,
    'scheduler_patience': 10,
    'seed': 13,
    'shifted_relu': False,
    'slayer': False,
    'snnax_optim': False,
    'spatial_bin': 5,
    'start_epoch': 0,
    'superspike': False,
    'sweep_id': None,
    'tau_m': 0.10768,
    'taylor': False,
    'time_offset': 0,
    'use_augm': False,
    'use_bias': False,
    'use_pretrained_model': False,
    'use_regularizers': False,
    'weight_norm': False,
    'xavier_init': False,
    'zero_init': False
}

net = SNN(
    input_shape=input_shape,
    layer_sizes=layer_sizes,
    neuron_type=model_type,
    use_readout_layer=True,
    extra_features = extra_features
)

'''
model_path = '../../exp/test_exps/shd_LIFcomplex_3lay512_drop0_1_batchnorm_nobias_udir_noreg_lr0_01/checkpoints/best_model.pth'
#model_path = '../../exp/paper_models/ssc_LIFcomplex_3lay512_/checkpoints/best_model.pth'
trained_net_cmplx = torch.load(model_path)
model_path = '../../exp/test_exps/shd_adLIFclamp_3lay512_drop0_1_batchnorm_nobias__/checkpoints/best_model.pth'
trained_net_adLIF = torch.load(model_path)
model_path = '../../SHD_runs/exp/paper_models/shd_ResonateFire_3lay512_/checkpoints/best_model.pth'
trained_net_rf = torch.load(model_path)
'''

def find_system_eigenvalues_numeric(tau_m_array, tau_w_array, R_array, a_w_array):
    eigenvalue_list = []

    # Assuming tau_m_array, tau_w_array, R_array, a_w_array are all numpy arrays
    for tau_m, tau_w, R, a_w in zip(tau_m_array, tau_w_array, R_array, a_w_array):
        A = np.array([[-1/tau_m, -R/tau_m], [a_w/tau_w, -1/tau_w]])  # Example system matrix
        eigenvalues = np.linalg.eigvals(A)
        eigenvalue_list.append(eigenvalues)
    
    return np.array(eigenvalue_list) 


def compute_rat_R_a(real_value, img_value, tau_m=0.01):
    rat, R_a = symbols('rat R_a')
    eq1 = Eq(-(rat + 1) / (2 * tau_m), real_value)
    eq2 = Eq(sqrt((rat**2 - 2*rat + 1 - 4*R_a * rat)*(-1)) / (2 * tau_m), img_value)
    solutions = solve([eq1, eq2], (rat, R_a))
    return solutions

discreteEV_complex = np.zeros((2,512,2)).astype(complex)
discreteEV_adLIF = np.zeros((2,512,2)).astype(complex)
discreteEV_rf = np.zeros((2,512,2)).astype(complex)
discreteEV_complex_init = np.zeros((2,512)).astype(complex)
'''
for i, trained_layer in enumerate(trained_net_cmplx.snn):
    if isinstance(trained_layer, LIFcomplexLayer):
        alpha_img = trained_layer.alpha_img.detach().cpu().numpy()
        log_dt = trained_layer.log_dt.detach().cpu().numpy()
        log_log_alpha = trained_layer.log_log_alpha.detach().cpu().numpy()
        dt =  np.exp(log_dt)
        alpha_real = -np.exp(log_log_alpha)
        discreteEV_complex[i,:,0] = np.exp((-np.exp(log_log_alpha)+1j*alpha_img)*dt)
        discreteEV_complex[i,:,1] = np.exp((-np.exp(log_log_alpha)-1j*alpha_img)*dt)


for i, trained_layer in enumerate(trained_net_adLIF.snn):
    print(trained_layer)
    if isinstance(trained_layer, adLIFclampLayer):
        
        alpha = trained_layer.alpha.detach().cpu().numpy()
        beta = trained_layer.beta.detach().cpu().numpy()
        dt = 0.001
        tau_m = - dt/np.log(alpha)
        tau_w = - dt/np.log(beta)
        a = trained_layer.a.detach().cpu().numpy()
        R = np.ones(alpha.shape)
        a_w = a
        eigenvalues = find_system_eigenvalues_numeric(tau_m, tau_w, R, a_w)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        discreteEV_adLIF[i,:] = np.exp((real_parts+1j*imag_parts)*dt)


for i, trained_layer in enumerate(trained_net_rf.snn):
    print(trained_layer)
    if isinstance(trained_layer, ResonateFireLayer):
        
        alpha_im = trained_layer.alpha_im.detach().cpu().numpy()
        alpha_real = trained_layer.alpha_real.detach().cpu().numpy()
        alpha_real = np.clip(alpha_real, max = -0.1)
        dt = 0.004

        discreteEV_rf[i,:,0] = np.exp((alpha_real+1j*alpha_im)*dt)
        discreteEV_rf[i,:,1] = np.exp((alpha_real-1j*alpha_im)*dt)

'''
for i, trained_layer in enumerate(net.snn):
    print(trained_layer)
    if isinstance(trained_layer, LIFcomplexLayer):
        alpha_img = trained_layer.alpha_img.detach().cpu().numpy()
        log_dt = trained_layer.log_dt.detach().cpu().numpy()
        log_log_alpha = trained_layer.log_log_alpha.detach().cpu().numpy()
        LRU_img_init = trained_layer.LRU_img_init
        LRU_no_dt = trained_layer.LRU_no_dt
        dt =  np.exp(log_dt) 

        if LRU_img_init :
            alpha_img = np.exp(alpha_img)
        else: 
            alpha_img = alpha_img

        alpha_cont = -2*np.exp(log_log_alpha)+1j*alpha_img
        if not LRU_no_dt:
            alpha_cont *=np.exp(log_dt)
        alpha = np.exp(alpha_cont)
        discreteEV_complex_init[i,:] = alpha


for i in range(2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

   ## ax[0].scatter(discreteEV_adLIF[i,:,:].real, discreteEV_adLIF[i,:,:].imag, color='red',marker='o',s = 1)
   # ax[0].scatter(discreteEV_rf[i,:].real, discreteEV_rf[i,:].imag, color='green', marker='o', s = 1)
   # ax[0].scatter(discreteEV_complex[i,:].real, discreteEV_complex[i,:].imag, color='blue', marker='o',s = 1,)
    ax[0].scatter(discreteEV_complex_init[i,:].real, discreteEV_complex_init[i,:].imag, color='blue', marker='o',s = 1,)


    circle = plt.Circle((0, 0), radius=1, color='black', fill=False)
    ax[0].add_artist(circle)

    # Label axes
    ax[0].set_xlabel(r'Re($\lambda$)')
    ax[0].set_ylabel(r'Im($\lambda$)')
    #ax[0].set_title('Scatter Plot of Eigenvalues on Complex Plane')
    ax[0].axhline(0, color='black',linewidth=0.5)
    ax[0].axvline(0, color='black',linewidth=0.5)
    ax[0].grid(True)

    ax[0].set_xlim((-1, 1))
    ax[0].set_ylim((-1, 1))

    ax[0].grid(True)
    #ax[0].legend()

   


    plt.savefig(f"../../plots/SSC/EV/discr/DiscreteAllEVs_layer_init{i}.png")
    plt.show()
    plt.clf()  # Clear the figure for the next plot

