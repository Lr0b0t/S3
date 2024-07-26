import torch

def compute_eigenvalues(alpha, beta, a):
    # Construct the 2x2 matrix
    matrix = torch.tensor([[alpha, alpha - 1], [(1 - beta) * a, beta]], dtype=torch.float32)
    
    # Compute the eigenvalues
    eigenvalues = torch.linalg.eigvals(matrix)
    return eigenvalues

def find_problematic_values(model_path):
    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # If the model is wrapped in a checkpoint dictionary, extract the actual model
    if isinstance(model, dict) and 'model' in model:
        model = model['model']

    print(f"Analyzing eigenvalues of model from {model_path}:")

    problematic_values = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and hasattr(module, 'beta') and hasattr(module, 'a'):
            alpha = module.alpha.detach().cpu().numpy()
            beta = module.beta.detach().cpu().numpy()
            a = module.a.detach().cpu().numpy()
            
            for i in range(len(alpha)):
                eigenvalues = compute_eigenvalues(alpha[i], beta[i], a[i])
                for ev in eigenvalues:
                    ev = ev.item()  # Ensure ev is a Python complex number
                    if ev.real > 1:
                        problematic_values.append((alpha[i], beta[i], a[i], ev.real, ev.imag, name))

    if problematic_values:
        print("Found problematic values:")
        for alpha, beta, a, real, imag, name in problematic_values:
            print(f"Layer {name}: alpha = {alpha}, beta = {beta}, a = {a} -> Eigenvalue = {real} + {imag}j")
    else:
        print("No problematic values found.")

def main():
    # Define the path to the model file
    model_path1 = './exp/test_exps/shd_adLIF_3lay256_drop0_1_batchnorm_nobias_udir_noreg_lr0_01/checkpoints/best_model.pth'
    
    # Find and print problematic values
    find_problematic_values(model_path1)

if __name__ == "__main__":
    main()
