import matplotlib.pyplot as plt
import pandas as pd
import json
import re
import numpy as np
import matplotlib.cm as cm
import random

# Define a color mapping for specific model types
color_map = {
    "LIFcomplex": "#9467bd",
    "RLIFcomplex": "#7f7f7f",
    "LIF": "#d62728",
    "RLIF": "#2ca02c",
    "adLIF": "#1f77b4",
    "RadLIF": "#8c564b"
}
plt.rcParams.update({
    'figure.figsize': (10, 6),   # Standard figure size
    'figure.dpi': 100,           # Resolution
    'font.size': 12,             # General font size
    'axes.grid': True,           # Enable grid
    'grid.linestyle': '--',      # Grid line style
    'grid.alpha': 0.7,           # Grid line transparency
    'lines.linewidth': 3,        # Line width
    'lines.markersize': 6,       # Marker size
    'axes.titlesize': 16,        # Title font size
    'axes.labelsize': 14,        # Axis label font size
    'xtick.labelsize': 12,       # X-axis tick label size
    'ytick.labelsize': 12,       # Y-axis tick label size
    'legend.fontsize': 12,       # Legend font size
    'figure.autolayout': True,   # Enable automatic layout
    'axes.spines.top': False,    # Turn off top spine
    'axes.spines.right': False,  # Turn off right spine
    'axes.prop_cycle': plt.cycler(color=[
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]), # Custom color cycle
})

def random_color():
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

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

# Function to combine LIFcomplex and LIFcomplexDiscr by taking the maximum accuracy for each X value
def combine_lif_complex(df):
    lif_complex = df[df['model_type'] == 'LIFcomplex']
    lif_complex_discr = df[df['model_type'] == 'LIFcomplexDiscr']
    
    combined = pd.concat([lif_complex, lif_complex_discr])
    combined = combined.groupby(['number_layers', 'number_neurons']).agg(
        {'test_acc': 'max'}).reset_index()
    combined['model_type'] = 'LIFcomplex_combined'
    
    return combined

def remove_duplicates_preserve_order(original_list):
    seen = set()
    result = []
    for item in original_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Plot function
def plot_results(df, model_types_to_plot, save_dir, plotMax, layers_to_plot, neurons_to_plot,broader_linewidth):
    # Combine "LIFcomplexDiscr" and "LIFcomplex" into a single set of data points
    combined_lif_complex = combine_lif_complex(df)
    df = df[~df['model_type'].isin(['LIFcomplex', 'LIFcomplexDiscr'])]
    df = pd.concat([df, combined_lif_complex])
    df['combined_LIFcomplex'] = df['model_type'].apply(lambda x: 'LIFcomplex_combined' if x in ['LIFcomplex', 'LIFcomplexDiscr'] else x)
    
    # Print column names and first few rows for debugging
    print("Columns in DataFrame:", df.columns)
    print("First few rows of DataFrame:")
    print(df.head())
    
    # Update the list of model types to plot
    model_types_to_plot = ['LIFcomplex_combined' if model in ['LIFcomplex', 'LIFcomplexDiscr'] else model for model in model_types_to_plot]
    model_types_to_plot = remove_duplicates_preserve_order(model_types_to_plot)  # Remove duplicates
    
    filename = generate_filename(model_types_to_plot)
    plt.figure(figsize=(16, 6))

    # Define color cycling based on the fixed color map
    def get_color(model_type):
        if model_type in color_map:
            return color_map[model_type]
        else:
            return random_color()  # Generate a random color for unknown model types

    # First subplot: Accuracy over number of layers for entries with number_neurons = 128
    plt.subplot(1, 2, 1)
    neurons_condition = df['number_neurons'] == 128
    layers_condition = df['number_layers'].isin(layers_to_plot)

    # Apply the conditions to filter the dataframe
    df_neurons_128 = df[neurons_condition & layers_condition]
    
    print("Filtered DataFrame (neurons = 128):")
    print(df_neurons_128.head())

    handle_dict = {} 

    for model_type in model_types_to_plot:
        subset = df_neurons_128[df_neurons_128['combined_LIFcomplex'] == model_type].sort_values('number_layers')
        label = 'LIFcomplex' if model_type == 'LIFcomplex_combined' else model_type
        label = 'RLIFcomplex' if label == 'RLIFcomplex1MinAlpha' else label

        if not subset.empty:
            color = get_color(label)
            linewidth = 4 if broader_linewidth and model_type == 'LIFcomplex_combined' else 3
            handle, = plt.plot(subset['number_layers'], subset['test_acc'], label=label, color=color, marker='o', linewidth=linewidth)
            handle_dict[label] = handle

            if plotMax:
                # Find and plot the maximum accuracy
                max_acc = subset['test_acc'].max()
                max_acc_layer = subset.loc[subset['test_acc'].idxmax(), 'number_layers']
                plt.axhline(y=max_acc, color=color, linestyle='--', linewidth=0.8)
                plt.annotate(f'{max_acc:.4f} @ {max_acc_layer}', xy=(max_acc_layer, max_acc), 
                            xytext=(max_acc_layer, max_acc + 0.001), color=color)
                print(label + str(max_acc) + '  ' + str(max_acc_layer))

    plt.xlabel('Number of Layers')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy over Number of Layers (128 neurons per layer)')
    ordered_handles = [handle_dict[label] for label in model_types_to_plot if label in handle_dict]  # Ensure only labels present in the plot are added to legend
    plt.legend()#handles=ordered_handles
    plt.xticks(subset['number_layers'].unique())
    plt.ylim(bottom=0.65)

    # Second subplot: Accuracy over number of neurons for entries with 3 layers
    plt.subplot(1, 2, 2)
    neurons_condition = df['number_layers'] == 3
    layers_condition = df['number_neurons'].isin(neurons_to_plot)

    # Apply the conditions to filter the dataframe
    df_layers_3 = df[neurons_condition & layers_condition]

    for model_type in model_types_to_plot:
        subset = df_layers_3[df_layers_3['combined_LIFcomplex'] == model_type].sort_values('number_neurons')
        label = 'LIFcomplex' if model_type == 'LIFcomplex_combined' else model_type
        label = 'RLIFcomplex' if label == 'RLIFcomplex1MinAlpha' else label
        x_positions = range(len(subset))
        linewidth = 4 if broader_linewidth and model_type == 'LIFcomplex_combined' else 3
        if not subset.empty:
            color = get_color(label)
            
            plt.plot(x_positions, subset['test_acc'], label=label, color=color, marker='o', linewidth=linewidth)

            if plotMax:
                # Find and plot the maximum accuracy
                max_acc = subset['test_acc'].max()
                max_acc_index = subset['test_acc'].idxmax()
                max_acc_neurons = subset.loc[max_acc_index, 'number_neurons']
                plt.axhline(y=max_acc, color=color, linestyle='--', linewidth=0.8)
                plt.annotate(f'{max_acc:.4f} @ {max_acc_neurons}', xy=(x_positions[subset.index.get_loc(max_acc_index)], max_acc), 
                            xytext=(x_positions[subset.index.get_loc(max_acc_index)], max_acc + 0.001), color=color)
                print("MAX: " + label + str(max_acc) + '  ' + str(max_acc_neurons))

    plt.xlabel('Number of Neurons per Layer')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy over Number of Neurons per Layer (3 layers)')
    plt.legend()
    plt.xticks(ticks=x_positions, labels=subset['number_neurons'])

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

    plotMax = True
    broader_linewidth = True
    model_types_to_plot = [ "LIF", "RLIF", "LIFcomplex","LIFcomplexDiscr","RLIFcomplex1MinAlpha","adLIF","RadLIF",]
        #"LIFcomplex","LIFcomplexDiscr","adLIF", "adLIFclamp", "adLIFnoClamp"]
        #"LIFcomplex","LIFcomplexDiscr","adLIF","LIF"]
                           #"RLIFcomplex1MinAlpha","adLIF", "RadLIF","LIF","RLIF"]
    # "RLIFcomplex1MinAlpha", "LIF","RLIF",
                           #,"LIFcomplexDiscr","RadLIF","adLIF","RLIF","LIF", "RLIFcomplex1MinAlpha", "adLIFclamp"]
        #"adLIF","RadLIF"]
        #"LIF", "RLIF"]
        #"LIFcomplex","LIFcomplexDiscr", "adLIF","LIF"]
                           # "RadLIF", "LIFcomplex", "RLIFcomplex1MinAlpha", "adLIFclamp", "LIFcomplexDiscr"]
    layers_to_plot = [2, 3, 4, 5, 6,7,8,9,10]
    neurons_to_plot = [64,128,256,512,1024, 2048,3072,4096]
    plot_results(df, model_types_to_plot, save_dir, plotMax, layers_to_plot,neurons_to_plot,broader_linewidth)


if __name__ == '__main__':
    main()
