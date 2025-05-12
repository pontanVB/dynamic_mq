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
 # Change to datashader plots?


 # OLD - not using pandas
def smooth_values(values, window_size, window_step):
    """
    Smooths a list of values and calculates the 25th, 50th, and 75th percentiles.
    
    Parameters:
    - values: The series of values to smooth.
    - window_size: How many values to aggregate into each new data point.
    - window_step: The offset between the aggregating windows.
    
    Returns:
    - smoothed_vals: The smoothed list of values (mean).
    - smoothed_mids_25: The smoothed list of 25th percentile values.
    - smoothed_mids_50: The smoothed list of 50th percentile values.
    - smoothed_mids_75: The smoothed list of 75th percentile values.
    - smoothed_inds: The list of original indexes corresponding to the smoothed values.
    """
    window = deque()
    smoothed_vals = []
    smoothed_mids_25 = []
    smoothed_mids_50 = []
    smoothed_mids_75 = []
    smoothed_inds = []

    for i, point in enumerate(values):
        window.append(point)
        
        # When the window reaches the size limit
        if len(window) == window_size:
            # Calculate mean
            smoothed_vals.append(np.mean(window))
            # Calculate percentiles
            smoothed_mids_25.append(np.percentile(window, 25))
            smoothed_mids_50.append(np.percentile(window, 50))  # 50th percentile (median)
            smoothed_mids_75.append(np.percentile(window, 75))
            smoothed_inds.append(i + window_size // 2)

            # Slide the window by window_step
            for _ in range(min(window_step, len(window))):
                window.popleft()

    return smoothed_vals, smoothed_mids_25, smoothed_mids_50, smoothed_mids_75, smoothed_inds

from collections import deque


def smooth_values_pandas(values, window_size, window_step, mids=False):
    series = pd.Series(values)
    rolled = series.rolling(window=window_size)

    smoothed_vals = rolled.mean().iloc[::window_step].dropna().to_numpy()
    smoothed_inds = np.arange(window_size // 2, len(values) - window_size // 2 + 1, window_step)

    if mids:
        smoothed_mids_25 = rolled.quantile(0.25).iloc[::window_step].dropna().to_numpy()
        smoothed_mids_50 = rolled.quantile(0.50).iloc[::window_step].dropna().to_numpy()
        smoothed_mids_75 = rolled.quantile(0.75).iloc[::window_step].dropna().to_numpy()
        return smoothed_vals, smoothed_mids_25, smoothed_mids_50, smoothed_mids_75, smoothed_inds
    else:
        return smoothed_vals, smoothed_inds


def throughput_calc(data_df, window_size, window_step):
    """
    Calculates throughput over non-overlapping or step-wise overlapping windows.

    Parameters:
    - data_df: DataFrame with 'total_iterations' and 'time'.
    - window_size: Number of data points in each window.
    - window_step: Step size to move the window.

    Returns:
    - throughput_vals: List of computed throughput values.
    - time_vals: Corresponding center time index for each throughput value.
    """

    throughput_vals = []
    time_vals = []
    throughput_inds = []
    i = 0

    while i + window_size <= len(data_df):
        times = data_df['time'].iloc[i:i+window_size]

        time_diff = times.iloc[-1] - times.iloc[0]
        iter_sum = window_size

        throughput = iter_sum / time_diff if time_diff != 0 else 0
        throughput_vals.append(throughput)
        time_vals.append(times.iloc[window_size // 2])
        throughput_inds.append(i + window_size//2)

        i += window_step

    return throughput_vals, time_vals, throughput_inds


# Computing the failrate using pandas
def fail_rate_calc(data_df, window_size, window_step):
    # Compute rolling sum of fails
    fails_rolled = data_df['lock_fails'].rolling(window=window_size).sum()

    # Drop NaNs and compute fail rate
    fail_rates = (1 - fails_rolled / (window_size * 2 + fails_rolled)).dropna()

    # Subsample
    smoothed_vals = fail_rates.iloc[::window_step].to_numpy()

    # Compute smoothed indices (center of window)
    smoothed_inds = np.arange(window_size // 2, 
                              window_size // 2 + window_step * len(smoothed_vals), 
                              window_step)

    return smoothed_vals, smoothed_inds

def safe_plot_from_df(ax, df, x_col, y_col, title, xlabel, ylabel, color='blue', smoothing=False, window_size=1, window_step=1, medians=False):
    x_vals = []
    y_vals = []

    if smoothing and y_col in df.columns:
        print("smoothing")

        if medians:
            smooth_vals, smooth_25, smooth_50, smooth_75, smooth_inds = smooth_values_pandas(
                df[y_col], window_size, window_step, medians
            )
            x_vals = smooth_inds

            if len(x_vals) != len(smooth_vals):
                print(f"Difference between smoothed values and indices: {len(x_vals)-len(smooth_vals)}")
                min_length = min(len(x_vals), len(smooth_vals))
                x_vals = x_vals[:min_length]
                smooth_vals = smooth_vals[:min_length]


            ax.plot(x_vals, smooth_vals, '-', linewidth=2, color=color, label='Average')
            ax.plot(x_vals, smooth_25, '--', linewidth=1, color='red', label='25th percentile')
            ax.plot(x_vals, smooth_50, '--', linewidth=1, color='blue', label='50th percentile (Median)')
            ax.plot(x_vals, smooth_75, '--', linewidth=1, color='green', label='75th percentile')

        else:
            smooth_vals, smooth_inds = smooth_values_pandas(
                df[y_col], window_size, window_step, medians
            )
            x_vals = smooth_inds

            if len(x_vals) != len(smooth_vals):
                print(f"Difference between smoothed values and indices: {len(x_vals)-len(smooth_vals)}")
                min_length = min(len(x_vals), len(smooth_vals))
                x_vals = x_vals[:min_length]
                smooth_vals = smooth_vals[:min_length]


            ax.plot(x_vals, smooth_vals, '-', linewidth=2, color=color, label='Average')


    elif {x_col, y_col}.issubset(df.columns):
        x_vals = df[x_col]
        y_vals = df[y_col]
        ax.plot(x_vals, y_vals, '-', linewidth=2, color=color, label=y_col)

    elif not (y_col in df.columns):
        print(f"Column {y_col} is missing from {df.columns}")
        return False
    else:
        print(f"Column {x_col} is missing from {df.columns}")
        return False
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')

    return True



def process_files(log_file, rank_file, plot_name, window_size, window_step):
    """Function to process the input files using pandas."""

    x_val_amount = 0
    plot_amount = 0

    metrics_exist = False
    rank_error_exist = False

    # Load log files
    # Ok to provide a file or not. Not ok to provide invalid file
    if log_file is None:
        print("Dynamic Logs not provided")
        metrics_exist = False
    elif os.path.exists(log_file):
        data_df = pd.read_csv(log_file)
        print(f"Loaded metrics log file: {log_file}")
        metrics_exist = True
    else:
        print(f"Metrics log file not found: {log_file}")
        return

    if rank_file is None:
        rank_error_exist = False
        print("Rank error file not provided")
    elif os.path.exists(rank_file):
        rank_error_df = pd.read_csv(rank_file)
        print(f"Loaded rank error file: {rank_file}")
        rank_error_exist = True
    else:
        print(f"Error: Rank error file not found: {rank_file}")
        return

    # Plot setup

    specific_fields = ['tick', 'active_threads', 'stickiness', 'rank_error','lock_fails', 'delay']

    if metrics_exist:
        x_val_amount = int(len(data_df) / window_size)
        for field in data_df.columns:
            if field in specific_fields:
                plot_amount += 1
    
    if rank_error_exist:
        x_val_amount = int(len(data_df) / window_size)
        plot_amount += 2

    if window_step is None:
        window_step = x_val_amount
            
    fig, axs = plt.subplots(plot_amount, 1, figsize=(10, plot_amount * 4))
    # Make axs always iterable
    if plot_amount == 1:
        axs = [axs]

    plot_index = 0

    # Add derived columns
    if metrics_exist:
        data_df['time'] = (data_df['tick'] - data_df['tick'].iat[0]) / 1e9
        
        data_df['pops'] = range(len(data_df))
        data_df['thread_specific_pops'] = data_df.groupby('thread_id').cumcount() + 1
        print(f"our data len: {len(data_df)}")


        plot_specs = [
            ('stickiness', 'Plot of Thread Average Stickiness over Iterations', 'Stickiness', 'blue', True, True),
            ('active_threads', 'Plot of Active Threads over Iterations', 'Active Threads', 'deepskyblue', False, False),
            #('lock_succes_rate', 'Lock sucess rate over iterations', 'Sucess Rate', 'lime', True, False),
            # Add more tuples as needed
        ]

        for y_col, title, ylabel, color, smoothing, medians in plot_specs:
            if y_col in data_df.columns:
                success = safe_plot_from_df(
                    ax=axs[plot_index],
                    df=data_df,
                    x_col='pops',
                    y_col=y_col,
                    title=title,
                    xlabel='Iteration',
                    ylabel=ylabel,
                    color=color,
                    smoothing=smoothing,
                    window_size=x_val_amount,
                    window_step=window_step,
                    medians=medians
                )
                if success:
                    plot_index += 1


        # Plot 2: Throughput
        avg_iters, avg_iters_times, avg_iters_inds = throughput_calc(data_df, x_val_amount, x_val_amount)
        axs[plot_index].plot(avg_iters_inds, avg_iters, '-', linewidth=2, color='green')
        axs[plot_index].set_title('Throughput Over Iterations')
        axs[plot_index].set_xlabel('Iteration')
        axs[plot_index].set_ylabel('Iterations/s')
        axs[plot_index].grid(True, linestyle='--', alpha=0.7)
        # axs[plot_index].text(0.8, 0.5, f"window size: {window_size}",
        #             transform=axs[plot_index].transAxes,
        #             fontsize=12,
        #             verticalalignment='top',
        #             horizontalalignment='left')
        
        plot_index += 1

        # Plot 3: Fail rate
        rates, rates_inds = fail_rate_calc(data_df, x_val_amount, x_val_amount)
        axs[plot_index].plot(rates_inds, rates, '-', linewidth=2, color='green')
        axs[plot_index].set_title('Sucess rate Over Iterations')
        axs[plot_index].set_xlabel('Iteration')
        axs[plot_index].set_ylabel('Fail rate')
        axs[plot_index].grid(True, linestyle='--', alpha=0.7)
        
        plot_index += 1


    if rank_error_exist:
        print(f"rank error len: {len(rank_error_df)}")

        # Plot 4: Rank Error
        if safe_plot_from_df(
            ax=axs[plot_index],
            df=rank_error_df,
            x_col='pops',  # assuming same indexing; adjust if needed
            y_col='rank_error',
            title='Plot of Rank Error over Iterations',
            xlabel='Iteration',
            ylabel='Rank Error',
            color='red',
            smoothing=True,
            window_size=x_val_amount,
            window_step=window_step,
            medians=True
        ):
            plot_index += 1

        # Plot 4: Delay
        if safe_plot_from_df(
            ax=axs[plot_index],
            df=rank_error_df,
            x_col='pops',  # assuming same indexing; adjust if needed
            y_col='delay',
            title='Delay over Iterations',
            xlabel='Iteration',
            ylabel='Delay',
            color='red',
            smoothing=True,
            window_size=x_val_amount,
            window_step=window_step,
            medians=True
        ):
            plot_index += 1


        # Figure config
        
        # fig.legend(handles='asd', labels= 'asd',loc='outside right upper', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)



        # Date
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        
        # Save plot inside /plots/plt_name-date.png
        os.makedirs("plots", exist_ok=True)

        # Save plot
        plot_dir = os.path.join("plots", f"{plot_name}_{date}.png")

        print(f"saved plot to:{plot_name}_{date}.png")

        # Final adjustments and save
        plt.tight_layout()

        plt.savefig(plot_dir)
        plt.close()





def main():
    """Main function to handle argument parsing and execution."""
    parser = argparse.ArgumentParser(description="Process metrics and rank error files.")
    parser.add_argument('-r', type=str, required=False, help="Path to the rank error file.")
    parser.add_argument('-l', type=str, required=False, help="Path to the dynamic logging.")
    parser.add_argument('-p', type=str, required=True, help="Plot name.")
    parser.add_argument('-w', type=int, required=True, help="Window size.")
    parser.add_argument('-ws', type=int, required=False, help="Window Step.")

    
    args = parser.parse_args()
    
    # Call processing function with arguments
    process_files(args.l, args.r, args.p, args.w, args.ws)

if __name__ == "__main__":
    main()


