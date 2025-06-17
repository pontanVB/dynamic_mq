import json
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime



def process_files(outputs_folder, plot_name):
    data = {}  # {queue: {graph_file: time_ns}}

    for filename in os.listdir(outputs_folder):
        if filename.endswith(".json"):
            print(filename)
            queue_name = filename[:-5]  # strip '.json'
            filepath = os.path.join(outputs_folder, filename)
            with open(filepath, "r") as f:
                d = json.load(f)

            graph_file = os.path.basename(d["settings"]["graph_file"]).removesuffix(".gr")
            time_ns = d["results"]["time_ns"]

            if queue_name not in data:
                data[queue_name] = {}
            data[queue_name][graph_file] = time_ns / 1e6

    # Sorted graph and queue names
    all_graphs = sorted({g for q in data.values() for g in q})
    all_queues = sorted(data.keys())
    n_graphs = len(all_graphs)
    n_queues = len(all_queues)

    # Bar settings
    bar_width = 0.8 / n_queues
    x = np.arange(n_graphs)

    plt.figure(figsize=(12, 6))

    for i, queue in enumerate(all_queues):
        heights = [data[queue].get(graph, 0) for graph in all_graphs]
        x_offset = x + i * bar_width
        plt.bar(x_offset, heights, width=bar_width, label=queue)

    plt.xticks(x + bar_width * (n_queues - 1) / 2, all_graphs, rotation=45, ha='right')
    plt.xlabel("Graph")
    plt.ylabel("Time (ms)")
    plt.title("Execution Time per Graph (per Queue)")
    plt.legend(title="Queue")
    plt.tight_layout()
    plt.grid(axis='y')


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




    return

def main():
    """Main function to handle argument parsing and execution."""
    parser = argparse.ArgumentParser(description="Process metrics and rank error files.")
    parser.add_argument('-o', type=str, required=False, help="Path to output folder.", default='outputs')
    parser.add_argument('-p', type=str, required=False, help="Plot name.", default='graph_comparison')

    
    args = parser.parse_args()
    
    # Call processing function with arguments
    process_files(args.o, args.p)

if __name__ == "__main__":
    main()
