import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from datetime import datetime
import numpy as np
import os
from scipy import stats
import argparse

# Base path
BASE_PATH = "logs/Ant-v5/exp-16/rkl/"

# Create output directory for the plot if it doesn't exist
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Define a set of distinct colors using tableau colors (more distinguishable than rainbow)
DISTINCT_COLORS = [
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
LINE_STYLES = ['-', '--', ':', '-.']

def parse_datetime(folder_name):
    filename = folder_name.split('/')[-1]
    date_parts = filename.split('_')[:6]
    datetime_str = f"{date_parts[0]}_{date_parts[1]}_{date_parts[2]}_{date_parts[3]}_{date_parts[4]}_{date_parts[5]}"
    return datetime.strptime(datetime_str, '%Y_%m_%d_%H_%M_%S')

def extract_q(folder_name):
    match = re.search(r'q(\d+)_', folder_name)
    return int(match.group(1)) if match else None

def extract_clip(folder_name):
    match = re.search(r'qstd(\d+\.\d+)', folder_name)
    return float(match.group(1)) if match else None

def load_data(base_path):
    folders = glob.glob(f'{base_path}2024_*_seed*')
    q_clip_results = {}

    for folder in folders:
        q = extract_q(folder)
        clip = extract_clip(folder)
        print(f"q: {q}, clip: {clip}")
        if q is not None and clip is not None:
            try:
                df = pd.read_csv(f'{folder}/progress.csv')
                print("q:", q, "clip:", clip, "len:", len(df))
                key = (q, clip)
                if key not in q_clip_results:
                    q_clip_results[key] = []
                q_clip_results[key].append(df['Real Det Return'].values)
            except Exception as e:
                print(f"Error reading file in folder: {folder}")
                print(f"Error: {e}")

    return q_clip_results

def plot_data(q_clip_results, show_confidence_interval, q_filter=None):
    plt.figure(figsize=(12, 8))
    filtered_results = {k: v for k, v in q_clip_results.items() if q_filter is None or k[0] in q_filter}
    style_mapping = create_style_mapping(filtered_results)

    for idx, ((q, clip), series_list) in enumerate(sorted(filtered_results.items())):
        min_length = min(len(series) for series in series_list)
        aligned_series = [series[:min_length] for series in series_list]
        data = np.array(aligned_series)
        mean = np.mean(data, axis=0)
        episodes = np.arange(min_length) * 5000
        color, line_style = style_mapping[(q, clip)]
        label = f'num_of_nns={q}, ' + ('no_clipping' if q == 1.0 else f'clip={clip}')
        plt.plot(episodes, mean, label=label, linestyle=line_style, color=color)

        if show_confidence_interval:
            stderr = stats.sem(data, axis=0)
            conf_int = stderr * stats.t.ppf((1 + 0.95) / 2, data.shape[0] - 1)
            plt.fill_between(episodes, mean - conf_int, mean + conf_int, alpha=0.2, color=color)

    plt.xlabel('Episodes')
    plt.ylabel('Average Real Det Return')
    plt.title('Average Real Det Return by num of NNs and clip value')
    plt.legend()
    plt.grid(True)

    # Generate plot filename based on parameters
    q_values_str = '_'.join(map(str, sorted(q_filter))) if q_filter else 'all_q'
    conf_int_str = 'with_conf_int' if show_confidence_interval else 'without_conf_int'
    plot_filename = f'average_real_det_return_{q_values_str}_{conf_int_str}.png'
    plot_path = os.path.join(PLOT_DIR, plot_filename)

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved as: {plot_path}")

def create_style_mapping(q_clip_results):
    style_mapping = {}
    unique_combos = sorted(q_clip_results.keys())
    num_combos = len(unique_combos)
    for idx, combo in enumerate(unique_combos):
        color_idx = idx % len(DISTINCT_COLORS)
        style_idx = idx // len(DISTINCT_COLORS)
        style = LINE_STYLES[style_idx % len(LINE_STYLES)]
        style_mapping[combo] = (DISTINCT_COLORS[color_idx], style)
    return style_mapping

def main():
    parser = argparse.ArgumentParser(description='Plot Average Real Det Return')
    parser.add_argument('--show_confidence_interval', action='store_true', help='Show confidence intervals in the plot')
    parser.add_argument('--q', type=int, nargs='+', help='Specify which q values to plot')
    args = parser.parse_args()

    q_clip_results = load_data(BASE_PATH)
    plot_data(q_clip_results, args.show_confidence_interval, args.q)

    print("\nNumber of runs for each q and clip value:")
    for (q, clip) in sorted(q_clip_results.keys()):
        print(f"q={q}, clip={clip}: {len(q_clip_results[(q, clip)])} runs")

if __name__ == "__main__":
    main()