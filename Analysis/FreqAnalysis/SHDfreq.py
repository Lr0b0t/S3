import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from sparch.dataloaders.spiking_datasets import load_shd_or_ssc

dataset_name = "shd"
data_folder = os.path.expanduser('~/local/SHD')
batch_size = 128
nb_steps = 250 
spatial_bin = 1

train_loader = load_shd_or_ssc(
    dataset_name=dataset_name,
    data_folder=data_folder,
    split="train",
    batch_size=batch_size,
    nb_steps=nb_steps,
    max_time=1.0,
    spatial_bin=spatial_bin,
    shuffle=True,
    workers=8,
)

def compute_firing_rate(spike_sequences, bin_size):
    """Compute firing rates from spike sequences."""
    nb_channels = spike_sequences.shape[1]
    firing_rates = []
    
    for ch in range(nb_channels):
        # Compute firing rate as spike counts per bin
        spike_counts, _ = np.histogram(spike_sequences[:, ch, :], bins=np.arange(0, nb_steps + bin_size, bin_size))
        firing_rate = spike_counts / bin_size  # Convert to rate (e.g., spikes per bin)
        firing_rates.append(firing_rate)
    
    return np.array(firing_rates)

def analyze_firing_rates(train_loader, nb_steps, bin_size):
    """Analyze and plot histograms of the firing rates in the input sequences from the train_loader."""
    for step, (x, _, y) in enumerate(train_loader):
        input_sequences = x.numpy()  # Assuming shape (batch_size, nb_channels, nb_steps)
        firing_rates = compute_firing_rate(input_sequences, bin_size)

        # Create a histogram of firing rates
        plt.figure(figsize=(10, 6))
        for i in range(firing_rates.shape[0]):
            plt.hist(firing_rates[i], bins=1, alpha=0.5, label=f'Channel {i + 1}', density=True)

        plt.xlabel('Firing Rate (spikes/bin)')
        plt.ylabel('Density')
        plt.title('Histogram of Firing Rates of Input Sequences')
        plt.legend()
        plt.savefig(f"../../plots/SHD/FreqAnalyse/SHD_FiringRates_Histogram.png")
        plt.show()
        print('Histogram saved and displayed.')
        
        break  # Process only the first batch for now

# Example bin size for firing rate calculation
bin_size = 5  # Adjust this value as needed
analyze_firing_rates(train_loader, nb_steps, bin_size)
