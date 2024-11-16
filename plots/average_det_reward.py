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

# Add parameter for confidence interval
SHOW_CONFIDENCE_INTERVAL = False

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

# Function to extract clip value from folder name
def extract_clip(folder_name):
    match = re.search(r'qstd(\d+\.\d+)', folder_name)
    if match:
        return float(match.group(1))

    return None

folders = glob.glob(f'{base_path}2024_*_seed*')

# Dictionary to store results for each q and clip value combination
q_clip_results = {}

for folder in folders:
    folder_date = parse_datetime(folder)

    q = extract_q(folder)
    clip = extract_clip(folder)
    print(f"q={q}, clip={clip}")
    if q is not None and clip is not None:
        print(f"Reading folder: {folder}")
        try:
            df = pd.read_csv(f'{folder}/progress.csv')
            key = (q, clip)
            if key not in q_clip_results:
                q_clip_results[key] = []
            q_clip_results[key].append(df['Real Det Return'].values)
        except Exception as e:
            print(f"Error reading file in folder: {folder}")
            print(f"Error: {e}")

# Create and save the plot
plt.figure(figsize=(12, 8))

# Define a set of distinct colors using tableau colors (more distinguishable than rainbow)
distinct_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # yellow-green
    '#17becf',  # cyan
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
]

# Create distinct visual combinations for each line
line_styles = ['-', '--', ':', '-.']

# Calculate how many unique combinations we need
num_combinations = len(q_clip_results)

# Create style mapping using colors first, then line styles if needed
style_mapping = {}
for idx, combo in enumerate(sorted(q_clip_results.keys())):
    color_idx = idx % len(distinct_colors)
    style_idx = idx // len(distinct_colors)
    style = line_styles[style_idx] if idx >= len(distinct_colors) else '-'
    style_mapping[combo] = (distinct_colors[color_idx], style)

for idx, ((q, clip), series_list) in enumerate(sorted(q_clip_results.items())):
    # Pad or truncate series to same length
    min_length = min(len(series) for series in series_list)
    aligned_series = [series[:min_length] for series in series_list]
    
    data = np.array(aligned_series)
    mean = np.mean(data, axis=0)
    
    episodes = np.arange(min_length) * 5000
    
    # Get the color and line style for this combination
    color, line_style = style_mapping[(q, clip)]
    
    label = f'num_of_nns={q}, ' + ('no_clipping' if clip == 'no_clipping' else f'clip={clip}')
    plt.plot(episodes, mean, 
             label=label,
             linestyle=line_style,
             color=color)
    
    if SHOW_CONFIDENCE_INTERVAL:
        stderr = stats.sem(data, axis=0)
        conf_int = stderr * stats.t.ppf((1 + 0.95) / 2, data.shape[0] - 1)
        plt.fill_between(episodes, 
                        mean - conf_int, 
                        mean + conf_int, 
                        alpha=0.2,
                        color=color)

plt.xlabel('Episodes')
plt.ylabel('Average Real Det Return')
plt.title('Average Real Det Return by num of NNs and clip value')
plt.legend()
plt.grid(True)

# Save the plot with high DPI for quality
plot_path = os.path.join(plot_dir, 'average_real_det_return.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Update statistics printing
print("\nNumber of runs for each q and clip value:")
for (q, clip) in sorted(q_clip_results.keys()):
    print(f"q={q}, clip={clip}: {len(q_clip_results[(q, clip)])} runs")

print(f"\nPlot saved as: {plot_path}")