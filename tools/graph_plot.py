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

    full_time_index = pd.date_range(df.index.min(), df.index.max(), freq=f'{time_sample}ms')

    for thread_id, group in grouped:
        resampled = group.resample(f'{time_sample}ms')
        lock_fails_sum = resampled['lock_fails'].sum()
        pushes_sum = resampled['pushes'].sum()
        pops_sum = resampled['pops'].sum()
        elements = pushes_sum + pops_sum 


        success_rate = elements / (elements + lock_fails_sum)
        throughput = elements * (1000 / time_sample)
                
        # Put relevant fields in a DataFrame and add thread_id column
        success_rate = success_rate.to_frame(name='success_rate')
        throughput  = throughput.to_frame(name='throughput')
        pushes_sum  = pushes_sum.to_frame(name='pushes')
        pops_sum    = pops_sum.to_frame(name='pops')
        elements    = elements.to_frame(name='elements')
        stats = pd.concat([elements, success_rate, throughput, pushes_sum, pops_sum], axis=1)
        stats['thread_id'] = thread_id

        thread_data[thread_id] = stats

    thread_list = pd.concat(thread_data.values())
    thread_list.index.name = 'time'
    return thread_list




def process_files(log_file, plot_name):
    """Function to process the input files using pandas."""

    plot_amount = 0


    # Load log files
    # Ok to provide a file or not. Not ok to provide invalid file
    if log_file is None:
        print("Logs not provided")
    elif os.path.exists(log_file):
        data_df = pd.read_csv(log_file)
        print(f"Loaded log file: {log_file}")
    else:
        print(f"Log file not found: {log_file}")
        return



    # Adding index
    data_df = data_df.set_index('tick')
    data_df.index = pd.to_datetime(data_df.index)

    # Granularity calculation
    time_start = data_df.index.min()
    time_end = data_df.index.max()
    duration = time_end - time_start
    duration_ms = duration.total_seconds() * 1000
    granularity  = max(1, np.round(duration_ms / 100))   # 1 % of run, ≥ 1 ms    # granularity = 1
    print(f"{granularity} ms granularity")

    # Sampling and time indexing
    resampled_df = data_df.resample(f'{granularity}ms')
    per_sample_pops = resampled_df['pops'].sum()
    per_sample_pushes = resampled_df['pushes'].sum()
    per_sample_elements = per_sample_pops + per_sample_pushes
    start_time = resampled_df.sum().index[0]
    times = (resampled_df.sum().index - start_time).total_seconds() * 1000

    
    # Calculate a dataFrame with per thread average on each timepoint
    thread_averaged_df = thread_averaged(data_df, ['stickiness', 'success_rate'], granularity)

    # For each thread at each time, take the resapled lock_fails and sum across threads
    #thread_averaged_df = thread_averaged_df.groupby('time')
    thread_averaged_medians = thread_averaged_df.groupby('time').median()
    thread_averaged_mins = thread_averaged_df.groupby('time').min()
    thread_averaged_max = thread_averaged_df.groupby('time').max()

    # Per/tread contention
    contention_df = thread_count_sum(data_df, granularity)
    start_time = contention_df.index.min()
    end_time = contention_df.index.max()
    full_time = pd.date_range(start=start_time, end=end_time, freq=f'{granularity}ms')
    thread_contention_mins = contention_df.groupby('time').min()
    thread_contention_max = contention_df.groupby('time').max()



    # Debug csv
    # contention_df.to_csv('thread_averaged.csv')
    

    # Success rate calc.groupby('time') -- Correct value, the contentnion mean should be the same
    per_sample_fails = resampled_df['lock_fails'].sum()
    success_rate = per_sample_elements / (per_sample_elements + per_sample_fails)




    # Adding Throughput
    throughput = per_sample_elements * (1000 / granularity)
    pushes = per_sample_pushes * 1000 / granularity
    pops = per_sample_pops * 1000 / granularity


    plot_amount = 3
    pos_contention = (0, 0)
    pos_contention_per_thread = (0, 1)
    pos_stickiness = (1, 0)
    pos_stickiness_per_thread = (1, 1)
    pos_throughput = (2,0)
    pos_throughput_per_thread = (2,1)





    # Ordering of plots
    plot_am = len(data_df.columns)
    fig, axs = plt.subplots(plot_amount, 2, figsize=(10, plot_amount * 3))  # adjust figsize as needed
    today = datetime.now().strftime("%Y/%m/%d") 
    fig.suptitle(plot_name, fontsize=16)


    # Contention
    axs[pos_contention].plot(times, success_rate, '-', linewidth=2, color='darkgreen', label='system_mean')
    axs[pos_contention].plot(times, thread_contention_mins['success_rate'], '--', linewidth=1, color='darkred', label='min')
    axs[pos_contention].plot(times, thread_contention_max['success_rate'], '--', linewidth=1, color='purple', label='max')
    axs[pos_contention].set_title('System Contention')
    axs[pos_contention].set_ylabel('Lock Success Rate')
    axs[pos_contention].legend()

    for thread_id, group in contention_df.groupby('thread_id'):
        thread_rel_times = (group.index - start_time).total_seconds() * 1000
        axs[pos_contention_per_thread].plot(thread_rel_times, group['success_rate'], '-', label=f'Thread {int(thread_id)}', alpha=0.7)

    axs[pos_contention_per_thread].set_title('Contention per Thread')
    # axs[1,1].legend()

    # More y ticks if needed
    # for ax in axs[1]:  # second row
    #     ymin, ymax = ax.get_ylim()
    #     ticks = np.linspace(ymin, ymax, 10)
    #     ax.set_yticks(ticks)

    # Stickiness
    axs[pos_stickiness].plot(times, thread_averaged_medians['stickiness'], '-', linewidth=2, color='darkgreen', label='median')
    axs[pos_stickiness].plot(times, thread_averaged_mins['stickiness']   , '--', linewidth=1, color='red', label='min')
    axs[pos_stickiness].plot(times, thread_averaged_max['stickiness']    , '--', linewidth=1, color='purple', label='max')
    axs[pos_stickiness].set_title('System Stickiness')
    axs[pos_stickiness].set_ylabel('Stickiness')
    axs[pos_stickiness].set_yscale('log')
    axs[pos_stickiness].legend()

    for thread_id, group in thread_averaged_df.groupby('thread_id'):
        thread_rel_times = (group.index - start_time).total_seconds() * 1000
        axs[pos_stickiness_per_thread].plot(thread_rel_times, group['stickiness'], '-', label=f'Thread {int(thread_id)}', alpha=0.7)

    axs[pos_stickiness_per_thread].set_title('Stickiness per Thread')
    axs[pos_stickiness_per_thread].set_yscale('log')
    # axs[2,1].legend()

    # Throughput
    axs[pos_throughput].plot(times, throughput, '-', linewidth=2, color='darkblue', label='sum')
    axs[pos_throughput].plot(times, pushes, '--', linewidth=1, color='green', label='pushes')
    axs[pos_throughput].plot(times, pops, '--', linewidth=1, color='red', label='pops')
    axs[pos_throughput].set_title('System Throughput')
    axs[pos_throughput].set_ylabel('Operations / s')
    axs[pos_throughput].legend()

    for thread_id, group in contention_df.groupby('thread_id'):
        throughput_series = group['throughput']
        throughput_full = throughput_series.reindex(full_time)
        throughput_full_filled = throughput_full.fillna(0)

        thread_rel_times = (full_time - start_time).total_seconds() * 1000
        axs[pos_throughput_per_thread].plot(thread_rel_times, throughput_full_filled, '-', label=f'Thread {int(thread_id)}', alpha=0.7)
        
    axs[pos_throughput_per_thread].set_title('Throughput per Thread')
    # axs[3,1].legend()



    # Final adjustments
    maxtime = times.argmax()
    x_ticks = np.linspace(0, times[-1] + 1, 10)
    x_ticks = np.round(x_ticks / 10) * 10
    for ax_row in axs:
        for ax in ax_row:
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(x_ticks)  # ticks every time_intervals
    
    #axs[-1, 0].set_xticks(np.arange(0, times[-1] + 1, time_interval))  # ticks every time_intervals
    #axs[-1, 1].set_xticks(np.arange(0, times[-1] + 1, time_interval))  # ticks every time_intervals

    axs[-1, 0].set_xlabel(f"Time (ms) ({granularity}ms granularity)")
    axs[-1, 1].set_xlabel(f"Time (ms) ({granularity}ms granularity)")
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
    parser.add_argument('-l', type=str, required=False, help="Path to the graph log.", default='metrics_log.txt')
    parser.add_argument('-p', type=str, required=False, help="Plot name.", default='graph plot')
    
    args = parser.parse_args()
    
    # Call processing function with arguments
    process_files(args.l, args.p)

if __name__ == "__main__":
    main()


