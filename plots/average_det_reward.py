import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from datetime import datetime
import numpy as np
import os
from scipy import stats

# Base path
base_path = "logs/Hopper-v5/exp-4/rkl/"

# Create output directory for the plot if it doesn't exist
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Function to parse datetime from folder name
def parse_datetime(folder_name):
    # Extract just the filename from the full path
    filename = folder_name.split('/')[-1]
    # Split by underscore and take the first 4 parts (year, month, day, time)
    date_parts = filename.split('_')[:6]
    # Combine the parts and parse
    datetime_str = f"{date_parts[0]}_{date_parts[1]}_{date_parts[2]}_{date_parts[3]}_{date_parts[4]}_{date_parts[5]}"
    return datetime.strptime(datetime_str, '%Y_%m_%d_%H_%M_%S')

# Function to extract q value from folder name
def extract_q(folder_name):
    match = re.search(r'q(\d+)_', folder_name)
    return int(match.group(1)) if match else None

# Get all folders after 2024-11-10 14-30
target_date = datetime.strptime('2024_11_10_14_30_00', '%Y_%m_%d_%H_%M_%S')
folders = glob.glob(f'{base_path}2024_*_seed*')

# Dictionary to store results for each q value
q_results = {}

for folder in folders:
    folder_date = parse_datetime(folder)
    if folder_date >= target_date:
        q = extract_q(folder)
        if q is not None:
            try:
                # Read progress.csv from the folder
                df = pd.read_csv(f'{folder}/progress.csv')
                if q not in q_results:
                    q_results[q] = []
                q_results[q].append(df['Real Det Return'].values)
            except Exception as e:
                print(f"Error reading file in folder: {folder}")
                print(f"Error: {e}")

# Create and save the plot
plt.figure(figsize=(12, 8))
for q in sorted(q_results.keys()):  # Sort q values for consistent colors
    # Pad or truncate series to same length
    min_length = min(len(series) for series in q_results[q])
    aligned_series = [series[:min_length] for series in q_results[q]]
    
    # Convert to numpy array for easier computation
    data = np.array(aligned_series)
    
    # Calculate mean and confidence interval
    mean = np.mean(data, axis=0)
    stderr = stats.sem(data, axis=0)
    conf_int = stderr * stats.t.ppf((1 + 0.95) / 2, data.shape[0] - 1)
    
    episodes = np.arange(min_length) * 5000
    
    # Plot mean line
    plt.plot(episodes, mean, label=f'num_of_nns={q}')
    # Plot confidence interval
    plt.fill_between(episodes, 
                    mean - conf_int, 
                    mean + conf_int, 
                    alpha=0.2)

plt.xlabel('Episodes')
plt.ylabel('Average Real Det Return')
plt.title('Average Real Det Return by num of NNs')
plt.legend()
plt.grid(True)

# Save the plot with high DPI for quality
plot_path = os.path.join(plot_dir, 'average_real_det_return.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Print some statistics
print("\nNumber of runs for each q value:")
for q in sorted(q_results.keys()):
    print(f"q={q}: {len(q_results[q])} runs")

print(f"\nPlot saved as: {plot_path}")