import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def compute_alpha(log_log_alpha, alpha_img, log_dt):
    log_log_alpha = log_log_alpha.float()
    alpha_img = alpha_img.float()
    log_dt = log_dt.float()
    
    alpha = torch.exp((-torch.exp(log_log_alpha) + 1j * alpha_img) * torch.exp(log_dt))
    return alpha

def plot_distribution(layer_data, save_path):
    plt.figure(figsize=(18, 6))
    
    for data in layer_data:
        alpha_real, alpha_imag, dt, layer_name = data
        plt.subplot(1, 3, 1)
        plt.hist(alpha_real, bins=30, alpha=0.5, edgecolor='black', label=f'{layer_name}')
        plt.title('Alpha Real Part Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    
        plt.subplot(1, 3, 2)
        plt.hist(alpha_imag, bins=30, alpha=0.5, edgecolor='black', label=f'{layer_name}')
        plt.title('Alpha Imaginary Part Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        plt.subplot(1, 3, 3)
        plt.hist(dt, bins=30, alpha=0.5, edgecolor='black', label=f'{layer_name}')
        plt.title('Dt Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
    
    plt.subplot(1, 3, 1)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_model_parameters(model_path):
    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # If the model is wrapped in a checkpoint dictionary, extract the actual model
    if isinstance(model, dict) and 'model' in model:
        model = model['model']

    print(f"Parameters of model from {model_path}:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

def analyze_alpha_parameters(model_path, save_dir):
    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # If the model is wrapped in a checkpoint dictionary, extract the actual model
    if isinstance(model, dict) and 'model' in model:
        model = model['model']

    print(f"Analyzing alpha parameters of model from {model_path}:")
    layer_data = []
    for name, module in model.named_modules():
        if hasattr(module, 'log_log_alpha') and hasattr(module, 'alpha_img') and hasattr(module, 'log_dt'):
            log_log_alpha = module.log_log_alpha
            alpha_img = module.alpha_img
            log_dt = module.log_dt
            
            alpha = compute_alpha(log_log_alpha, alpha_img, log_dt)
            dt = torch.exp(log_dt)
            
            # Calculate the mean and std for the real part of alpha
            alpha_real_mean = alpha.real.mean().item()
            alpha_real_std = alpha.real.std().item()
            
            # Calculate the mean and std for the imaginary part of alpha
            alpha_img_mean = alpha.imag.mean().item()
            alpha_img_std = alpha.imag.std().item()

            dt_mean = dt.mean().item()
            dt_std = dt.std().item()
            
            print(f"Layer {name}:")
            print(f"  alpha (real part) - mean: {alpha_real_mean}, std: {alpha_real_std}")
            print(f"  alpha (imaginary part) - mean: {alpha_img_mean}, std: {alpha_img_std}")
            print(f"  dt - mean: {dt_mean}, std: {dt_std}")

            # Collect data for plotting
            layer_data.append((alpha.real.detach().cpu().numpy(), alpha.imag.detach().cpu().numpy(), dt.detach().cpu().numpy(), name))

    # Plot the distribution for collected layers
    save_path = os.path.join(save_dir, 'LIFcomplex_3lay256_EVdistr.png')
    plot_distribution(layer_data, save_path)

def main():
    # Define the paths to the model files
    model_path1 = './exp/test_exps/shd_adLIF_3lay256_drop0_1_batchnorm_nobias_udir_noreg_lr0_01/checkpoints/best_model.pth'
    model_path2 = './exp/test_exps/shd_LIFcomplex_3lay256_drop0_1_batchnorm_nobias_udir_noreg_lr0_01/checkpoints/best_model.pth'
    
    # Define the directory to save plots
    save_dir = './plots/SHD'
    os.makedirs(save_dir, exist_ok=True)
    
    # Print the parameters of each model
    print_model_parameters(model_path1)
    print_model_parameters(model_path2)

    # Analyze alpha parameters for the second model and save plots
    analyze_alpha_parameters(model_path2, save_dir)

if __name__ == "__main__":
    main()
