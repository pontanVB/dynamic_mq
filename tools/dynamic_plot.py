import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from collections import deque

 # TODO 
 # CSV handiling / conversion! - Done

# FILES

# Use a relative path directly
file_path = '../metrics_log.txt'

rank_error_path = '../metrics.txt'


# Read the files
data_df = pd.read_csv(file_path)
rank_error_df = pd.read_csv(rank_error_path)  # Skip first row if needed


data_df = pd.read_csv(file_path)

# Adding new fields
data_df['time'] = data_df['tick'] - data_df['tick'].iat[0]
data_df['pops'] = range(len(data_df))
data_df['thread_specific_pops'] = data_df.groupby('thread_id').cumcount() + 1

df_threads = data_df.groupby('thread_id')



 # for viewing new fields 
data_df.to_csv('new_df.csv', index=False)




print(f"our data len:{len(data_df)}")
print(f"rank error len:{len(rank_error_df)}")


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



val = int(len(rank_error_df['rank_error'])/1000)
smooth_errors, smooth_indexes = smooth_values(rank_error_df['rank_error'], val, val)

# Create the plots
fig, axs = plt.subplots(5, 1, figsize=(10, 12))

# First subplot: Stickiness over iterations
axs[0].plot(data_df['pops'], data_df['stickiness'], '-', linewidth=2)
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Stickiness')
axs[0].set_title('Plot of Stickiness over Time')
axs[0].grid(True, linestyle='--', alpha=0.7)

# Second subplot: Rank error over iterations
axs[1].plot(smooth_indexes, smooth_errors, '-', linewidth=2, color='red')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Rank Error')
axs[1].set_title('Plot of Rank Error over Time')
axs[1].grid(True, linestyle='--', alpha=0.7)


# Plot iterations over time for each thread in the third subplot (ax[2])
for thread_id, group in df_threads:
    # Plot iterations over time for this thread on the third subplot
    axs[2].plot(group['tick'], group['thread_specific_pops'], 
               label=f'Thread {thread_id}', marker='o', markersize=4, alpha= 0.7)

# Customize the third subplot
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Total Iterations')
axs[2].set_title('Iterations Over Time Per Thread')
axs[2].grid(True, alpha=0.3)

# Add legend to the third subplot (only if there are multiple threads)
axs[2].legend()


# Adjust layout and save
plt.tight_layout()
plt.savefig("plot")
plt.close()

# plt.figure()

# fig_2, axs_2 = plt.subplots(3, 1, figsize=(10, 12))

# for i in range(len(df_threads)):
#     axs_2[i].plot(df_threads[i]['pops'], df_threads[i]['total_iterations'], 
#                   label=f'Thread {thread_id}', marker='o', markersize=4, alpha=0.7)
#     axs_2[i].legend()
#     axs_2[i].set_xlabel('Pops')
#     axs_2[i].set_ylabel('Total Iterations')

# plt.savefig("thread_plot")

# plt.close()

