import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from collections import deque
import argparse
from datetime import datetime

 # TODO 
 # CSV handiling / conversion! - Done



def smooth_values(values, window_size, window_step):
    """
    Smooths a list of values.

    Parameters:
    - values: The series of values to smooth.
    - window_size: How many values to aggregate into each new data point.
                   Increase this to get larger averages.
    - window_step: The offset between the aggregating windows.
                   Increase to get fewer output points. Should be less or
                   equal to window_size. 

    Returns:
    - smoothed_vals: The smoothed list of values. It will have a length of 
                     approximately len(values)/window_step.    
    - smoothed_inds: The list of original indexes corresponding to the
                     smoothed values.
    """
    window = deque()
    window_sum = 0.0

    smoothed_vals = []
    smoothed_inds = []

    for (i, point) in enumerate(values):
        window.append(point)
        window_sum += point

        if len(window) == window_size:
            smoothed_vals.append(window_sum/window_size)
            smoothed_inds.append(i + window_size//2)
            for _ in range(window_step):
                window_sum -= window.popleft()

    return smoothed_vals, smoothed_inds


# TODO - Split into separate make_plot and process_files functions

def process_files(log_file, rank_file, plot_path, window_size):
    """Function to process the input files using pandas."""
    
    # Load metrics log file
    if os.path.exists(log_file):
        data_df = pd.read_csv(log_file)
        print(f"Loaded metrics log file: {log_file}")
    else:
        print(f"Error: Metrics log file not found: {log_file}")
        return
    
    # Load rank error file
    if os.path.exists(rank_file):
        rank_error_df = pd.read_csv(rank_file)
        print(f"Loaded rank error file: {rank_file}")
    else:
        print(f"Error: Rank error file not found: {rank_file}")
        return

    print(data_df.columns)


    # Adding new fields
    data_df['time'] = data_df['tick'] - data_df['tick'].iat[0]
    data_df['pops'] = range(len(data_df))
    data_df['thread_specific_pops'] = data_df.groupby('thread_id').cumcount() + 1

    df_threads = data_df.groupby('thread_id')




    print(f"our data len:{len(data_df)}")
    print(f"rank error len:{len(rank_error_df)}")



    # for viewing new fields 
    data_df.to_csv('new_df.csv', index=False)



    val = int(len(rank_error_df)/window_size)
    smooth_errors, smooth_indexes_er = smooth_values(rank_error_df['rank_error'], val, val)
    smooth_stickiness, smooth_indexes_st = smooth_values(data_df['stickiness'], val, val)
    smooth_iter, smooth_iter_st = smooth_values(data_df['total_iterations'], val, val)

    # Create the plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    # First subplot: Stickiness over iterations
    axs[0].plot(smooth_indexes_st, smooth_stickiness, '-', linewidth=2)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Stickiness')
    axs[0].set_title('Plot of Thread Average Stickiness over Iterations')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # Second subplot: Rank error over iterations
    axs[1].plot(smooth_indexes_er, smooth_errors, '-', linewidth=2, color='red')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Rank Error')
    axs[1].set_title('Plot of Rank Error over Time')
    axs[1].grid(True, linestyle='--', alpha=0.7)




    # Second subplot: Rank error over iterations
    axs[2].plot(smooth_iter_st, smooth_iter, '-', linewidth=2, color='red')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Total Iterations')
    axs[2].set_title('Plot of Total Iterations Error over Iterations')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    axs[2].text(0.8, 0.5, f"window size:{window_size}", 
        transform=axs[2].transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='left')
    

    # Troughput plot
    axs[3].plot(data_df['pops'], data_df['active_threads'], '-', linewidth=2, color='red')
    axs[3].set_xlabel('Iteration')
    axs[3].set_ylabel('Active Threads')
    axs[3].set_title('Plot of Active Threads over Iterations')
    axs[3].grid(True, linestyle='--', alpha=0.7)



    # Date
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

    # Save plot inside env_path/plots/date.png
    plot_dir = os.path.join(plot_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{date}.png")

    print(f"saved plot to:{date}.png")


    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()




def main():
    """Main function to handle argument parsing and execution."""
    parser = argparse.ArgumentParser(description="Process metrics and rank error files.")
    parser.add_argument('-r', type=str, required=True, help="Path to the rank error file.")
    parser.add_argument('-l', type=str, required=True, help="Path to the dynamic logging.")
    parser.add_argument('-p', type=str, required=True, help="Path to the plots.")
    parser.add_argument('-w', type=int, required=True, help="Window size.")
    
    args = parser.parse_args()
    
    # Call processing function with arguments
    process_files(args.r, args.l, args.p, args.w)

if __name__ == "__main__":
    main()




# TODO - thread specific plots, to see outliers
# 
# # Plot iterations over time for each thread in the third subplot (ax[2])
# for thread_id, group in df_threads:
#     # Plot iterations over time for this thread on the third subplot
#     axs[2].plot(group['time'], group['thread_specific_pops'], 
#                label=f'Thread {thread_id}', marker='o', markersize=4, alpha= 0.7)

# # Customize the third subplot
# axs[2].set_xlabel('Time')
# axs[2].set_ylabel('Total Iterations')
# axs[2].set_title('Iterations Over Time Per Thread')
# axs[2].grid(True, alpha=0.3)

# # Add legend to the third subplot (only if there are multiple threads)
# axs[2].legend()

