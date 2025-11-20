import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import re
# Function to calculate Root Mean Square Error
def calculate_rmse(errors):
    return np.sqrt(np.mean(errors ** 2))

# Font settings for plots
font = {
    'size': 14,
    'weight': 'bold'
}
font1 = {
    'size': 12,
    'weight': 'bold'
}
legend_font = {
    'size': 12,
    'weight': 'bold'
}

# Color scheme for different methods
color = {
    0: [128/255, 116/255, 200/255],  # Purple
    1: [168/255, 203/255, 223/255],  # Light blue
    2: [227/255,  98/255,  93/255],  # Red
    3: [240/255, 194/255, 132/255],  # Light orange
    4: [153/255,  34/255,  36/255],  # Dark red
    5: [120/255, 149/255, 193/255]   # Blue-gray
}

# Extract method name from file path
def extract_label_from_path(path):
    # Get filename
    filename = os.path.basename(path)
    
    # Use regex to extract label
    match = re.search(r'([^_/]+)_enu_errors', filename)
    if match:
        label = match.group(1)
    else:
        # Default label if regex doesn't match
        label = filename.split('_')[0]
    
    # Convert to uppercase and return
    return label.upper()

# Visualize ENU errors in subplots
def visualize_enu_errors(file_paths):
    plt.figure(figsize=(14, 12))
    rmse_data = []
    max_limits = {'north': 0, 'east': 0, 'up': 0}

    for idx, file_path in enumerate(file_paths):
        data = pd.read_csv(file_path)
        time = data["time"]
        time = time - time.min()  # Start time from 0
        east = data["E"]
        north = data["N"]
        up = data["U"]

        # Calculate RMSE values
        rmse_n = calculate_rmse(north)
        rmse_e = calculate_rmse(east)
        rmse_u = calculate_rmse(up)
        rmse_2d = np.sqrt(rmse_n ** 2 + rmse_e ** 2)
        rmse_3d = np.sqrt(rmse_n ** 2 + rmse_e ** 2 + rmse_u ** 2)

        rmse_data.append([rmse_n, rmse_e, rmse_u, rmse_2d, rmse_3d])

        label = extract_label_from_path(file_path)

        # Update max limits for symmetric axes
        max_limits['north'] = max(max_limits['north'], np.max(np.abs(north)))
        max_limits['east'] = max(max_limits['east'], np.max(np.abs(east)))
        max_limits['up'] = max(max_limits['up'], np.max(np.abs(up)))

        # Plot North error
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(time, north, label=f"{label}", color=color[idx % len(color)], linewidth=3.5)
        plt.title("North Error", fontdict=font)
        plt.ylabel("North (m)", fontdict=font)
        plt.grid(True, linestyle='-', color='k', alpha=0.4, linewidth=1.5)

        # Plot East error
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(time, east, label=f"{label}", color=color[idx % len(color)], linewidth=3.5)
        plt.title("East Error", fontdict=font)
        plt.ylabel("East (m)", fontdict=font)
        plt.grid(True, linestyle='-', color='k', alpha=0.4, linewidth=1.5)

        # Plot Up error
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(time, up, label=f"{label}", color=color[idx % len(color)], linewidth=3.5)
        plt.title("Up Error", fontdict=font)
        plt.ylabel("Up (m)", fontdict=font)
        plt.grid(True, linestyle='-', color='k', alpha=0.4, linewidth=1.5)

    # Set symmetric y-axis limits
    ax1.set_ylim(-np.ceil(max_limits['north']), np.ceil(max_limits['north']))
    ax2.set_ylim(-np.ceil(max_limits['east']), np.ceil(max_limits['east']))
    ax3.set_ylim(-np.ceil(max_limits['up']), np.ceil(max_limits['up']))

    # Plot RMSE bar chart
    ax4 = plt.subplot(4, 1, 4)
    rmse_df = pd.DataFrame(rmse_data, columns=["North RMSE", "East RMSE", "Up RMSE", "2D RMSE", "3D RMSE"])
    rmse_df.index = [extract_label_from_path(path) for path in file_paths]

    bar_width = 0.15
    x = np.arange(len(rmse_df))  # X positions

    # Draw grid first
    ax4.grid(axis='y', linestyle='-', color='k', alpha=0.4, linewidth=1.5, zorder=0)

    # Draw bar charts for each category
    for i, column in enumerate(rmse_df.columns):
        ax4.bar(x + i * bar_width, rmse_df[column], width=bar_width,
                edgecolor='black', color=color[i], label=column, zorder=3)

    ax4.set_xticks(x + 2 * bar_width)
    ax4.set_xticklabels(rmse_df.index, rotation=0, fontsize=12, fontweight='bold')
    ax4.set_ylabel("RMSE (m)", fontdict=font)
    ax4.set_title("RMSE Statistics", fontdict=font)
    ax4.legend(loc='upper right', prop=legend_font, frameon=True)

    # Show value labels on bars
    for bars in ax4.containers:
        ax4.bar_label(bars, fmt='%.2f', label_type='edge', fontsize=11, fontweight='bold', padding=6)

    ylim = ax4.get_ylim()
    ax4.set_ylim(ylim[0], ylim[1] * 1.15)

    # Bold border for the entire plot
    for spine in ax4.spines.values():
        spine.set_linewidth(1.5)

    # Adjust x-axis density and tick labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Time (s)", fontdict=font)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.ticklabel_format(style='plain', axis='x')
        ax.set_xlim(time.min(), time.max())
        
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
    
    for ax in [ax1, ax2, ax3, ax4]:
        # Set y-axis properties
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')
        ax.tick_params(axis='y', labelsize=12, width=2, length=6)
        ax.yaxis.set_tick_params(which='both', direction='in', labelsize=14, width=2)
        ax.yaxis.set_label_coords(-0.03, 0.5)
        
        # Set x-axis properties
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')
        ax.tick_params(axis='x', labelsize=12, width=2, length=6)
        ax.xaxis.set_tick_params(which='both', direction='in', labelsize=14, width=2)

    plt.subplot(4, 1, 1)
    plt.legend(loc='upper right', prop=legend_font, edgecolor='black', ncol=3, frameon=True)

    ax4.legend(loc='upper right', prop=legend_font, edgecolor='black', ncol=3, frameon=True)
    
    plt.tight_layout()
    plt.show()

# Alternative visualization with split plots
def visualize_enu_errors_split(file_paths):
    # First figure: ENU error curves
    plt.figure(figsize=(20, 6))
    max_limits = {'north': 0, 'east': 0, 'up': 0}

    for idx, file_path in enumerate(file_paths):
        data = pd.read_csv(file_path)
        time = data["time"]
        time = time - time.min()  # Start time from 0
        east = data["E"]
        north = data["N"]
        up = data["U"]

        label = extract_label_from_path(file_path)

        max_limits['north'] = max(max_limits['north'], np.max(np.abs(north)))
        max_limits['east'] = max(max_limits['east'], np.max(np.abs(east)))
        max_limits['up'] = max(max_limits['up'], np.max(np.abs(up)))

        # Plot North error
        ax1 = plt.subplot(1, 3, 1)
        ax1.plot(time, north, label=f"{label}", color=color[idx % len(color)], linewidth=2.5)
        plt.title("North Error", fontdict=font)
        plt.ylabel("North (m)", fontdict=font1)
        plt.grid(True, linestyle='-', color='k', alpha=0.4, linewidth=1)

        # Plot East error
        ax2 = plt.subplot(1, 3, 2)
        ax2.plot(time, east, label=f"{label}", color=color[idx % len(color)], linewidth=2.5)
        plt.title("East Error", fontdict=font)
        plt.ylabel("East (m)", fontdict=font1)
        plt.grid(True, linestyle='-', color='k', alpha=0.4, linewidth=1)

        # Plot Up error
        ax3 = plt.subplot(1, 3, 3)
        ax3.plot(time, up, label=f"{label}", color=color[idx % len(color)], linewidth=2.5)
        plt.title("Up Error", fontdict=font)
        plt.ylabel("Up (m)", fontdict=font1)
        plt.grid(True, linestyle='-', color='k', alpha=0.4, linewidth=1)

    # Set symmetric y-axis limits
    ax1.set_ylim(-np.ceil(max_limits['north']), np.ceil(max_limits['north']))
    ax2.set_ylim(-np.ceil(max_limits['east']), np.ceil(max_limits['east']))
    ax3.set_ylim(-np.ceil(max_limits['up']), np.ceil(max_limits['up']))

    # Adjust x-axis density and tick labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Time (s)", fontdict=font)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.ticklabel_format(style='plain', axis='x')
        ax.set_xlim(time.min(), time.max())
        
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
    
        # Set axis properties
    for ax in [ax1, ax2, ax3]:
        # Set y-axis properties
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')
        ax.tick_params(axis='y', labelsize=12, width=2, length=6)
        ax.yaxis.set_tick_params(which='both', direction='in', labelsize=14, width=2)
        ax.yaxis.set_label_coords(-0.03, 0.5)
        
        # Set x-axis properties
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')
        ax.tick_params(axis='x', labelsize=12, width=2, length=6)
        ax.xaxis.set_tick_params(which='both', direction='in', labelsize=14, width=2)

    plt.subplot(1, 3, 1)
    plt.legend(loc='upper right', prop=legend_font, edgecolor='black', ncol=3, frameon=True)

    plt.tight_layout()
    plt.show()

    # Second figure: RMSE bar chart
    plt.figure(figsize=(10, 6))
    rmse_data = []
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        east = data["E"]
        north = data["N"]
        up = data["U"]

        # Calculate RMSE values
        rmse_n = calculate_rmse(north)
        rmse_e = calculate_rmse(east)
        rmse_u = calculate_rmse(up)
        rmse_2d = np.sqrt(rmse_n ** 2 + rmse_e ** 2)
        rmse_3d = np.sqrt(rmse_n ** 2 + rmse_e ** 2 + rmse_u ** 2)

        rmse_data.append([rmse_n, rmse_e, rmse_u, rmse_2d, rmse_3d])

    rmse_df = pd.DataFrame(rmse_data, columns=["North RMSE", "East RMSE", "Up RMSE", "2D RMSE", "3D RMSE"])
    rmse_df.index = [extract_label_from_path(path) for path in file_paths]

    bar_width = 0.15
    x = np.arange(len(rmse_df))  # X positions

    # Draw grid first
    plt.gca().grid(axis='y', linestyle='-', color='k', alpha=0.4, linewidth=1.5, zorder=0)

    # Draw bar charts for each category
    for i, column in enumerate(rmse_df.columns):
        plt.bar(x + i * bar_width, rmse_df[column], width=bar_width,
                edgecolor='black', color=color[i], label=column, zorder=3)

    plt.xticks(x + 2 * bar_width, rmse_df.index, rotation=0, fontsize=12, fontweight='bold')
    plt.ylabel("RMSE (m)", fontdict=font)
    plt.title("RMSE Statistics", fontdict=font)
    plt.legend(loc='upper right', prop=legend_font, frameon=True)

    # Show value labels on bars
    for bars in plt.gca().containers:
        plt.bar_label(bars, fmt='%.2f', label_type='edge', fontsize=11, fontweight='bold', padding=6)

    ylim = plt.gca().get_ylim()
    plt.gca().set_ylim(ylim[0], ylim[1] * 1.15)

    plt.tight_layout()
    plt.show()

# File paths for different methods
file_path1 = "baseline_results/cssrlib/LF-GNSS-A/cssrlib_enu_errors.csv"
file_path2 = "baseline_results/rtklib_gogps/LF-GNSS-A/gogps_enu_errors.csv"
file_path3 = "baseline_results/rtklib_gogps/LF-GNSS-A/rtklib_enu_errors.csv"
file_path4 = "baseline_results/tdl-gnss/LF-GNSS-A/tdl-gnss_enu_errors.csv"
file_path5 = "lf-gnss_results/LF-GNSS-A/lf-gnss_enu_errors_example.csv"

file_paths = [file_path1, file_path2, file_path3, file_path4, file_path5]

visualize_enu_errors(file_paths)