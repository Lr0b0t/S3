import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_eigenvalues(alpha, beta, a):
    # Construct the 2x2 matrix
    matrix = torch.tensor([[alpha, alpha - 1], [(1 - beta) * a, beta]], dtype=torch.float32)
    
    # Compute the eigenvalues
    eigenvalues = torch.linalg.eigvals(matrix)
    return eigenvalues

def plot_eigenvalues_distribution(layer_data, save_dir):
    colors = plt.cm.get_cmap('tab10', len(layer_data))  # Use a colormap for distinct colors

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, data in enumerate(layer_data):
        real_evs, complex_real_evs, complex_imag_evs, name, real_percentage, complex_percentage = data
        color = colors(idx)

        axes[0].hist(real_evs, bins=30, alpha=0.5, edgecolor='black', label=f'{name} ({real_percentage:.2f}% Real)', color=color)
        axes[0].set_title('Real Eigenvalues Distribution')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')

        axes[1].hist(complex_real_evs, bins=30, alpha=0.5, edgecolor='black', label=f'{name} ({complex_percentage:.2f}% Complex)', color=color)
        axes[1].set_title('Complex Eigenvalues Real Part Distribution')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')

        positive_imag_evs = [imag_ev for imag_ev in complex_imag_evs if imag_ev > 0]
        axes[2].hist(positive_imag_evs, bins=30, alpha=0.5, edgecolor='black', label=f'{name} ({complex_percentage:.2f}% Complex)', color=color)
        axes[2].set_title('Complex Eigenvalues Positive Imaginary Part Distribution')
        axes[2].set_xlabel('Value')
        axes[2].set_ylabel('Frequency')

    for ax in axes:
        ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'adLIF_3lay256_EVdistr.png')
    plt.savefig(save_path)
    plt.close()

def analyze_eigenvalues(model_path, save_dir):
    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # If the model is wrapped in a checkpoint dictionary, extract the actual model
    if isinstance(model, dict) and 'model' in model:
        model = model['model']

    print(f"Analyzing eigenvalues of model from {model_path}:")
    layer_data = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and hasattr(module, 'beta') and hasattr(module, 'a'):
            alpha = module.alpha.detach().cpu().numpy()
            beta = module.beta.detach().cpu().numpy()
            a = module.a.detach().cpu().numpy()
            
            real_evs = []
            complex_real_evs = []
            complex_imag_evs = []
            complex_count = 0
            real_count = 0
            total_count = 0
            
            for i in range(len(alpha)):
                eigenvalues = compute_eigenvalues(alpha[i], beta[i], a[i])
                total_count += len(eigenvalues)
                for ev in eigenvalues:
                    ev = ev.item()  # Ensure ev is a Python complex number
                    if ev.imag != 0:
                        complex_real_evs.append(ev.real)
                        complex_imag_evs.append(ev.imag)
                        complex_count += 1
                    else:
                        real_evs.append(ev.real)
                        real_count += 1
            
            real_percentage = (real_count / total_count) * 100
            complex_percentage = (complex_count / total_count) * 100
            
            print(f"Layer {name}:")
            print(f"  Real eigenvalues: {real_percentage:.2f}%")
            print(f"  Complex eigenvalues: {complex_percentage:.2f}%")
            
            layer_data.append((real_evs, complex_real_evs, complex_imag_evs, name, real_percentage, complex_percentage))
    
    # Plot the distribution for collected layers
    plot_eigenvalues_distribution(layer_data, save_dir)

def main():
    # Define the path to the model file
    model_path1 = '../../exp/test_exps/shd_adLIF_3lay256_drop0_1_batchnorm_nobias_udir_noreg_lr0_01/checkpoints/best_model.pth'
    
    # Define the directory to save plots
    save_dir = './plots/SHD'
    os.makedirs(save_dir, exist_ok=True)
    
    # Analyze eigenvalues for the first model and save plots
    analyze_eigenvalues(model_path1, save_dir)

if __name__ == "__main__":
    main()
