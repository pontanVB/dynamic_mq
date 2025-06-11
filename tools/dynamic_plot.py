import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from collections import deque
import argparse
from datetime import datetime
import matplotlib.dates as mdates
from collections import deque


def time_averaged(df: pd.DataFrame, headers, time_sample=1):
    valid_headers = [h for h in headers if h in df.columns]
    if not valid_headers:
        print("None of the provided headers exist in the DataFrame")
        return

    # df = df.set_index('tick')
    # df.index = pd.to_datetime(df.index)

    # Resample once
    resampled = df.resample(f'{time_sample}ms')[valid_headers]

    # Calculate size, percentiles, mean on the resampled groups
    # size = resampled.size().add_suffix('_size')
    mean = resampled.mean().add_suffix('_mean')
    p25     = resampled.quantile(0.25).add_suffix('_p25')
    p50     = resampled.quantile(0.50).add_suffix('_p50')
    p75     = resampled.quantile(0.75).add_suffix('_p75')
    p100    = resampled.quantile(1.00).add_suffix('_p100')

    averaged = pd.concat([p25, p50, p75, p100, mean], axis=1)

    # Relative time in milliseconds
    start_time = averaged.index[0]
    relative_times_ms = (averaged.index - start_time).total_seconds() * 1000

    return averaged, relative_times_ms, valid_headers


def thread_averaged(df: pd.DataFrame, headers, time_sample=1):
    """
    Calculates a mean value per timesample per thread
    """
    valid_headers = [h for h in headers if h in df.columns]
    if not valid_headers:
        print("None of the provided headers exist in the DataFrame")
        return
    print(valid_headers)
    # df = df.set_index('tick')
    # df.index = pd.to_datetime(df.index)

    # Group by thread and then resample per time window
    grouped = df.groupby('thread_id')

    thread_data = {}
    for thread_id, group in grouped:
        resampled = group.resample(f'{time_sample}ms')[valid_headers]
        stats = resampled.mean()  # you can also compute other stats here
        stats['thread_id'] = thread_id
        thread_data[thread_id] = stats

    # Combine all thread DataFrames
    thread_list = pd.concat(thread_data.values())

    # Set time as index and return
    thread_list.index.name = 'time'
    return thread_list



def thread_count_sum(df: pd.DataFrame, time_sample=1):
    grouped = df.groupby('thread_id')
    thread_data = {}

    for thread_id, group in grouped:
        resampled = group.resample(f'{time_sample}ms')['lock_fails']
        lock_fails_sum = resampled.sum()
        elements = resampled.count()
        #elements = elements.where(elements != 0, np.nan)

        success_rate = 2 * elements / (2 * elements + lock_fails_sum)
        throughput = elements * (1000 / time_sample)
        
        # Put relevant fields in a DataFrame and add thread_id column
        throughput = throughput.to_frame(name='throughput')
        elements = elements.to_frame(name='elements')
        success_rate = success_rate.to_frame(name='success_rate')
        stats = pd.concat([elements, success_rate, throughput], axis=1)
        stats['thread_id'] = thread_id

        thread_data[thread_id] = stats

    thread_list = pd.concat(thread_data.values())
    thread_list.index.name = 'time'
    return thread_list




def process_files(log_file, rank_file, plot_name, time_sample=1, time_interval=50):
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
    
    if rank_error_exist and metrics_exist:
        data_df = (pd.concat([data_df, rank_error_df], axis=1))


    # Adding index and sampling
    data_df = data_df.set_index('tick')
    data_df.index = pd.to_datetime(data_df.index)
    resampled_df = data_df.resample(f'{time_sample}ms')
    elements_per_sample = resampled_df.size()
    data_df['success_rate'] = 2 / (data_df['lock_fails'] + 2)
    # data_df['success_rate'] = data_df['success_rate'].where(data_df['success_rate'] > 1e-3, 0.0)


    # Debug file
    # data_df.to_csv('debug_csv.csv')

    
    # Headers to plot
    headers = []

    
    thread_averaged_headers = ['stickiness', 'success_rate']
    # Calculate a dataFrame with per thread average on each timepoint
    thread_averaged_df = thread_averaged(data_df, thread_averaged_headers, time_sample)

    # For each thread at each time, take the resapled lock_fails and sum across threads
    #thread_averaged_df = thread_averaged_df.groupby('time')
    thread_averaged_medians = thread_averaged_df.groupby('time').median()
    thread_averaged_mins = thread_averaged_df.groupby('time').min()
    thread_averaged_max = thread_averaged_df.groupby('time').max()

    # Per/tread contention
    contention_df = thread_count_sum(data_df, time_sample)
    start_time = contention_df.index.min()
    end_time = contention_df.index.max()
    full_time = pd.date_range(start=start_time, end=end_time, freq=f'{time_sample}ms')
    thread_contention_mins = contention_df.groupby('time').min()
    thread_contention_max = contention_df.groupby('time').max()

    # Debug csv
    # contention_df.to_csv('thread_averaged.csv')
    

    # Success rate calc.groupby('time') -- Correct value, the contentnion mean should be the same
    sampled_fails = resampled_df['lock_fails'].sum()
    success_rate = (elements_per_sample * 2) / (elements_per_sample * 2 + sampled_fails)


    # Thread "averaged" plotting
    headers.extend(['sucess_rate', 'stickiness'])

    # Calculate a dataFrame with element averages and percentiles for these headers
    system_headers = ['active_threads', 'rank_error', 'delay']
    averaged_df, times, valid_headers = time_averaged(data_df, system_headers, time_sample)
    headers.extend(valid_headers)


    if len(headers) == 1:
        axs = [axs]


    # Adding Throughput
    throughput = elements_per_sample * (1000 / time_sample)

    # Actual unique threads
    unique_threads_per_sample = data_df.resample(f'{time_sample}ms')['thread_id'].nunique()

    # Adding Operation Delay.
    op_delay = resampled_df['op_delay'].mean()



    


    # Ordering of plots
    headers = ['active_threads', 'success_rate', 'all_threads', 'stickiness', 'troughput', 'rank_error', 'delay']
    #fig, axs = plt.subplots(len(headers), 1, figsize=(10, 4 * len(headers)), sharex=True)
    fig, axs = plt.subplots(5, 2, figsize=(10, 15))  # adjust figsize as needed

    
    # Active threads
    axs[0,0].plot(times, unique_threads_per_sample, '-', linewidth=2, color='blue', label='mean')
    axs[0,0].set_title('Active Threads')
    axs[0,0].set_ylabel('Thread Count')

    # Operation delay (benchmark)
    axs[0,1].plot(times, op_delay, '-', linewidth=2, color='blue', label='mean')
    axs[0,1].set_title('Operation Delay')
    axs[0,1].set_ylabel('Time (ns)')



    # Contention
    axs[1,0].plot(times, success_rate, '-', linewidth=2, color='darkgreen', label='system_mean')
    axs[1,0].plot(times, thread_contention_mins['success_rate'], '--', linewidth=1, color='darkred', label='min')
    axs[1,0].plot(times, thread_contention_max['success_rate'], '--', linewidth=1, color='purple', label='max')
    axs[1,0].set_title('System Contention')
    axs[1,0].set_ylabel('Lock Success Rate')
    axs[1,0].legend()

    for thread_id, group in contention_df.groupby('thread_id'):
        thread_rel_times = (group.index - start_time).total_seconds() * 1000
        axs[1,1].plot(thread_rel_times, group['success_rate'], '-', label=f'Thread {int(thread_id)}', alpha=0.7)

    axs[1,1].set_title('Contention per Thread')
    # axs[1,1].legend()

    # More y ticks if needed
    # for ax in axs[1]:  # second row
    #     ymin, ymax = ax.get_ylim()
    #     ticks = np.linspace(ymin, ymax, 10)
    #     ax.set_yticks(ticks)

    # Stickiness
    axs[2,0].plot(times, thread_averaged_medians['stickiness'], '-', linewidth=2, color='darkgreen', label='median')
    axs[2,0].plot(times, thread_averaged_mins['stickiness']   , '--', linewidth=1, color='red', label='min')
    axs[2,0].plot(times, thread_averaged_max['stickiness']    , '--', linewidth=1, color='purple', label='max')
    axs[2,0].set_title('System Stickiness')
    axs[2,0].set_ylabel('Stickiness')
    axs[2,0].set_yscale('log')
    axs[2,0].legend()

    for thread_id, group in thread_averaged_df.groupby('thread_id'):
        thread_rel_times = (group.index - start_time).total_seconds() * 1000
        axs[2,1].plot(thread_rel_times, group['stickiness'], '-', label=f'Thread {int(thread_id)}', alpha=0.7)

    axs[2,1].set_title('Stickiness per Thread')
    axs[2,1].set_yscale('log')
    # axs[2,1].legend()

    # Throughput
    axs[3,0].plot(times, throughput, '-', linewidth=2, color='darkblue', label='temp')
    axs[3,0].set_title('System Throughput')
    axs[3,0].set_ylabel('Elements / s')


    for thread_id, group in contention_df.groupby('thread_id'):
        throughput_series = group['throughput']
        throughput_full = throughput_series.reindex(full_time)
        throughput_full_filled = throughput_full.fillna(0)

        thread_rel_times = (full_time - start_time).total_seconds() * 1000
        axs[3,1].plot(thread_rel_times, throughput_full_filled, '-', label=f'Thread {int(thread_id)}', alpha=0.7)
        
    axs[3,1].set_title('Throughput per Thread')
    # axs[3,1].legend()

    # Rank error

    axs[4,0].plot(times, averaged_df["rank_error_mean"], '-', linewidth=2, color='orange', label='mean')
    axs[4,0].plot(times, averaged_df["rank_error_p25"], '--', linewidth=1, color='red', label='25%')
    axs[4,0].plot(times, averaged_df["rank_error_p50"], '--', linewidth=1, color='blue', label='50%')
    axs[4,0].plot(times, averaged_df["rank_error_p75"], '--', linewidth=1, color='green', label='75%')
    axs[4,0].plot(times, averaged_df["rank_error_p100"], '--', linewidth=1, color='purple', label='100%')
    axs[4,0].set_title('Rank Error')
    axs[4,0].set_ylabel('Rank Error')
    axs[4,0].set_yscale('log')
    axs[4,0].legend()

    # Delay

    axs[4,1].plot(times, averaged_df["delay_mean"], '-', linewidth=2, color='orange', label='mean')
    axs[4,1].plot(times, averaged_df["delay_p25"], '--', linewidth=1, color='red', label='25%')
    axs[4,1].plot(times, averaged_df["delay_p50"], '--', linewidth=1, color='blue', label='50%')
    axs[4,1].plot(times, averaged_df["delay_p75"], '--', linewidth=1, color='green', label='75%')
    axs[4,1].plot(times, averaged_df["delay_p100"], '--', linewidth=1, color='purple', label='100%')
    axs[4,1].set_title('Delay')
    axs[4,1].set_ylabel('Delay')
    axs[4,1].set_yscale('log')
    axs[4,1].legend()


    # Final adjustments
    for ax_row in axs:
        for ax in ax_row:
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(np.arange(0, times[-1] + 1, time_interval))  # ticks every time_intervals


    
    #axs[-1, 0].set_xticks(np.arange(0, times[-1] + 1, time_interval))  # ticks every time_intervals
    #axs[-1, 1].set_xticks(np.arange(0, times[-1] + 1, time_interval))  # ticks every time_intervals

    axs[-1, 0].set_xlabel(f"Time (ms)({time_sample}ms granularity)")
    axs[-1, 1].set_xlabel(f"Time (ms)({time_sample}ms granularity)")
    plt.tight_layout()



    # Date
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    
    # Save plot inside /plots/plt_name-date.png
    os.makedirs("plots", exist_ok=True)

    # Save plot
    plot_dir = os.path.join("plots", f"{plot_name}_{date}.png")
    plt.savefig(plot_dir)
    print(f"saved plot to:{plot_name}_{date}.png")

    # plt.savefig(plot_dir)
    plt.close()




def main():
    """Main function to handle argument parsing and execution."""
    parser = argparse.ArgumentParser(description="Process metrics and rank error files.")
    parser.add_argument('-r', type=str, required=False, help="Path to the rank error file.")
    parser.add_argument('-l', type=str, required=False, help="Path to the dynamic logging.")
    parser.add_argument('-p', type=str, required=True, help="Plot name.")
    parser.add_argument('-ts', type=int, required=False, help="Time sample (ms).")
    parser.add_argument('-ti', type=int, required=False, help="Time interval (ms).")

    
    args = parser.parse_args()
    
    # Call processing function with arguments
    process_files(args.l, args.r, args.p, args.ts, args.ti)

if __name__ == "__main__":
    main()


