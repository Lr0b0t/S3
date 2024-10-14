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


model_path = '../../exp/test_exps/shd_LIFcomplex_3lay512_drop0_1_batchnorm_nobias_udir_noreg_lr0_01/checkpoints/best_model.pth'
#model_path = '../../exp/paper_models/ssc_LIFcomplex_3lay512_/checkpoints/best_model.pth'
trained_net_cmplx = torch.load(model_path)
model_path = '../../exp/test_exps/shd_adLIFclamp_3lay512_drop0_1_batchnorm_nobias__/checkpoints/best_model.pth'
trained_net_adLIF = torch.load(model_path)
model_path = '../../SHD_runs/exp/paper_models/shd_ResonateFire_3lay512_/checkpoints/best_model.pth'
trained_net_rf = torch.load(model_path)


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


for i in range(2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

   ## ax[0].scatter(discreteEV_adLIF[i,:,:].real, discreteEV_adLIF[i,:,:].imag, color='red',marker='o',s = 1)
   # ax[0].scatter(discreteEV_rf[i,:].real, discreteEV_rf[i,:].imag, color='green', marker='o', s = 1)
    ax[0].scatter(discreteEV_complex[i,:].real, discreteEV_complex[i,:].imag, color='blue', marker='o',s = 1,)

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

    # Right: Zoomed-in subplot
    ax[1].scatter(discreteEV_complex[i,:].real, discreteEV_complex[i,:].imag, color='blue', marker='o', s=2)
    ax[1].scatter(discreteEV_adLIF[i,:,:].real, discreteEV_adLIF[i,:,:].imag, color='red', marker='o', s=2)
    ax[1].scatter(discreteEV_rf[i,:].real, discreteEV_rf[i,:].imag, color='green', marker='o', s=2)

    # Zoomed-in limits for the subplot
    ax[1].set_xlim((0.75, 1))
    ax[1].set_ylim((-0.1, 0.1))
    #ax[1].set_xlabel('Real Part')
    #ax[1].set_ylabel('Imaginary Part')
    ax[1].grid(True)
    #ax[1].set_title('Zoomed-in Region')


    plt.savefig(f"../../plots/SHD/EV/discr/DiscreteAllEVs_layer{i}.svg")
    plt.show()
    plt.clf()  # Clear the figure for the next plot

