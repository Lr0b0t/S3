import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re

# Function to save results to JSON
def save_results_to_json(result_entry, filename='resultsSHD.json'):
    try:
        with open(filename, 'r') as file:
            results = json.load(file)
    except FileNotFoundError:
        results = []

    results.append(result_entry)
    
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)


# Plot
def plot_results(df, model_types_to_plot,  save_dir, plotMax):
    filename = generate_filename(model_types_to_plot)
    plt.figure(figsize=(16, 6))

    # First subplot: Accuracy over number of layers for entries with number_neurons = 128
    plt.subplot(1, 2, 1)
    df_neurons_128 = df[df['number_neurons'] == 128]
    colors = cm.get_cmap('tab20', len(model_types_to_plot))

    for i, model_type in enumerate(model_types_to_plot):
        subset = df_neurons_128[df_neurons_128['model_type'] == model_type].sort_values('number_layers')
        if not subset.empty:
            plt.plot(subset['number_layers'], subset['test_acc'], label=model_type, color=colors(i), marker='o')

            if plotMax:
                # Find and plot the maximum accuracy
                max_acc = subset['test_acc'].max()
                max_acc_layer = subset.loc[subset['test_acc'].idxmax(), 'number_layers']
                plt.axhline(y=max_acc, color=colors(i), linestyle='--', linewidth=0.8)
                plt.annotate(f'{max_acc:.4f} @ {max_acc_layer}', xy=(max_acc_layer, max_acc), 
                            xytext=(max_acc_layer, max_acc + 0.001), color=colors(i))

    plt.xlabel('Number of Layers')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy over Number of Layers (Neurons=128)')
    plt.legend()
    plt.xticks(subset['number_layers'].unique())
    #plt.ylim(0.7, 1.0)  # Set the y-axis limits

    # Second subplot: Accuracy over number of neurons for entries with 3 layers
    plt.subplot(1, 2, 2)
    df_layers_3 = df[df['number_layers'] == 3]

    for i, model_type in enumerate(model_types_to_plot):
        subset = df_layers_3[df_layers_3['model_type'] == model_type].sort_values('number_neurons')
        x_positions = range(len(subset))
        if not subset.empty:
            plt.plot(x_positions, subset['test_acc'], label=model_type, color=colors(i), marker='o')

            if plotMax:
                # Find and plot the maximum accuracy
                max_acc = subset['test_acc'].max()
                max_acc_index = subset['test_acc'].idxmax()
                max_acc_neurons = subset.loc[max_acc_index, 'number_neurons']
                plt.axhline(y=max_acc, color=colors(i), linestyle='--', linewidth=0.8)
                plt.annotate(f'{max_acc:.4f} @ {max_acc_neurons}', xy=(subset.index.get_loc(max_acc_index), max_acc), 
                            xytext=(subset.index.get_loc(max_acc_index), max_acc + 0.001), color=colors(i))

    plt.xlabel('Number of Neurons')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy over Number of Neurons (Layers=3)')
    plt.legend()
    plt.xticks(ticks=x_positions, labels=subset['number_neurons'])
    #plt.ylim(0.7, 1.0)  # Set the y-axis limits

    plt.tight_layout()
    plt.savefig(save_dir+filename)
    plt.close()
    print(f"Plot saved as {filename}")


def load_results_from_json(filename='resultsSHD.json'):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    
def generate_filename(model_types_to_plot):
    sanitized_names = [re.sub(r'\W+', '', model) for model in model_types_to_plot]
    return f"plot_{'_'.join(sanitized_names)}.png"

# Example usage
def main():
    ds = "SSC"
    save_dir = '../../plots/'+str(ds)+'/Acc/'
    results = load_results_from_json(filename='../../results'+str(ds)+'.json')
    df = pd.DataFrame(results)

    plotMax = False
    
    model_types_to_plot = [ "LIFcomplex"]

    plot_results(df, model_types_to_plot, save_dir, plotMax)


if __name__ == '__main__':
    main()