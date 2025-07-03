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
            filepath = os.path.join(outputs_folder, filename)
            with open(filepath, "r") as f:
                d = json.load(f)

            full_name = d["settings"]["pq_name"]
            queue_name = full_name.splitlines()[0].strip()
            graph_file = os.path.basename(d["settings"]["graph_file"]).removesuffix(".gr")
            time_ns = d["results"]["time_ns"]
            if queue_name == "MultiQueue":
                queue_name = "MQ"

            pq_settings = d["settings"].get("pq_settings", {})
            if "mode_name" in pq_settings:
                #queue_name = "MQ"
                queue_mode = pq_settings.get("mode_name", {})
                queue_name += '_' + queue_mode
                print(queue_name)
                stick_params = pq_settings.get("stickiness_parameters", {})
                stick_factor_value = stick_params.get("stick_factor", 0)
                # if stick_factor_value > 1:
                #     queue_name += "Dynamic"

            if queue_name == "k-LSM":
                k_val = str(pq_settings.get("k"))
                print(queue_name + " " + k_val)
                queue_name = queue_name + '_' + k_val
                
            if queue_name not in data:
                data[queue_name] = {}
            
            data[queue_name][graph_file] = time_ns / 1e6


    # -- Cutting outliers --
    # 1. After collecting data into `data`, flatten it into a list of times
    all_times = [t for q in data.values() for t in q.values()]
    if not all_times:
        return

    median = np.median(all_times)
    CAP = median * 10

    # 2. Delete times > CAP 
    for queue in list(data):                    
        for graph in list(data[queue]):         
            if data[queue][graph] > CAP:
                print(f"Removing queue {queue} on {data[queue]} due to being 10x the median")
                del data[queue][graph]          
        if not data[queue]:                     
            del data[queue]
    # -- --

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

    # Sort Legend by ascii code (alphabetically)
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_items = sorted(zip(labels, handles))
    labels, handles = zip(*sorted_items)
    plt.legend(handles, labels)


    plt.legend(title="Queue", loc='center left', bbox_to_anchor=(1, 0.5))
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
