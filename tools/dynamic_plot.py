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

# FILES

# Use a relative path directly
file_path = '../metrics_log.txt'

rank_error_path = '../metrics.txt'



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

def process_files(env_path, window_size):
    """Function to process the input files using pandas."""

    metrics_log_file = os.path.join(env_path, "metrics_log.txt")
    rank_error_file = os.path.join(env_path, "metrics.txt")
    
    # Load metrics log file
    if os.path.exists(metrics_log_file):
        data_df = pd.read_csv(metrics_log_file)
        print(f"Loaded metrics log file: {metrics_log_file}")
    else:
        print(f"Error: Metrics log file not found: {metrics_log_file}")
        return
    
    # Load rank error file
    if os.path.exists(rank_error_file):
        rank_error_df = pd.read_csv(rank_error_file)
        print(f"Loaded rank error file: {rank_error_file}")
    else:
        print(f"Error: Rank error file not found: {rank_error_file}")
        return
    
    # Example processing
    data_df = pd.read_csv(file_path)

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

    # Create the plots
    fig, axs = plt.subplots(5, 1, figsize=(10, 12))

    # First subplot: Stickiness over iterations
    axs[0].plot(smooth_indexes_st, smooth_stickiness, '-', linewidth=2)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Stickiness')
    axs[0].set_title('Plot of Stickiness over Iterations')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # Second subplot: Rank error over iterations
    axs[1].plot(smooth_indexes_er, smooth_errors, '-', linewidth=2, color='red')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Rank Error')
    axs[1].set_title('Plot of Rank Error over Time')
    axs[1].grid(True, linestyle='--', alpha=0.7)






    # Define the time window size
    time_window = 100000  # Example: Group every 5000 time units

    # Create bins based on time
    data_df['time_bin'] = (data_df['time'] // time_window) * time_window

    # Group by time bins and sum 'pops'
    df_windowed = data_df.groupby(['time_bin'])['pops'].sum().reset_index()

    # Third subplot: Averaged performace per thread over iterations
    axs[2].plot(df_windowed['time_bin'], df_windowed['pops'], '-', linewidth=2, color='red')
    plt.xlabel(f"Time Window ({time_window} units)")
    axs[2].set_ylabel('Total Pops')
    axs[2].set_title('Pops over time')
    axs[2].grid(True, linestyle='--', alpha=0.7)



    # Date
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

    # Save plot inside env_path/plots/date.png
    plot_dir = os.path.join(env_path, "plots")
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
    parser.add_argument('-p', type=str, required=True, help="Path to the testing environment.")
    parser.add_argument('-w', type=int, required=True, help="Window size.")
    
    args = parser.parse_args()
    
    # Call processing function with arguments
    process_files(args.p, args.w)

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

