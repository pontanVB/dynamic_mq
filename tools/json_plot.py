import json
import numpy as np
import matplotlib.pyplot as plt
import os


# FILES

json_data = []
data_path = "INSERT OUTPUT PATH"

num_files = len(os.listdir(data_path))

for file in os.listdir(data_path):
    file_path = os.path.join(data_path, file)
    print(file_path)
    with open(file_path, "r") as f:
        json_data.append(json.load(f))



num_iterations = len(json_data[0]["settings"]["thread_intervals"])
num_threads = json_data[0]["settings"]["num_threads"]

# Create a list for each thread. Each maps over the threads in each iteration
thread_fails = np.zeros((num_threads, num_iterations))
iter_data = np.zeros((num_threads, num_iterations))





for file in json_data:
    iter = []
    fails = []
    for thread_id, thread in enumerate(file["results"]["thread_data"]):
        # Store per-thread / iteration executions
        for iteration_id, fail_data in enumerate(thread["fail_data"]):
            iter_data[thread_id][iteration_id] += (fail_data["interval_iterations"])  
            thread_fails[thread_id][iteration_id] += (fail_data["interval_fails"])  
        
        

 # Averages for all files, might need to change if we dont do same iter for all
file_avg_thread_iter = np.zeros((num_threads, num_iterations))
file_avg_thread_fails = np.zeros((num_threads, num_iterations))

for thread_id in range(num_threads):
    for iteration_id in range(num_iterations):
        file_avg_thread_fails[thread_id][iteration_id] = thread_fails[thread_id][iteration_id] / num_files
        file_avg_thread_iter[thread_id][iteration_id] = iter_data[thread_id][iteration_id] / num_files

thread_iter_sum = np.zeros(num_iterations)
thread_fail_avg = np.zeros(num_iterations)

 # Mean for all threads in each iteration (assumingly when only one thread executes we will have ~half the performance of two, with dimininshing returns)
 # IF THERE'S NO ITERATION OR FAILS THEY SHOULDN'T BE INCLUDED IN THE MEAN???
for iter_id in range(num_iterations):
    thread_fail_avg[iter_id] = np.mean(file_avg_thread_fails[:, iter_id])
    thread_iter_sum[iter_id] = np.sum(file_avg_thread_iter[:, iter_id])





# Plotting
plt.figure(figsize=(10, 6))  # Set the figure size

# Plot thread_fail_avg on the y-axis, iter_id on the x-axis
plt.plot(range(num_iterations), thread_fail_avg, label="Thread Fail Average", color='blue', marker='o')

# Plot thread_iter_sum on the y-axis, iter_id on the x-axis
plt.plot(range(num_iterations), thread_iter_sum, label="Thread Iteration Sum", color='red', marker='x')

# Adding titles and labels
plt.title('Thread Fail Average vs Thread Iteration Sum')
plt.xlabel('Iteration, Decreasing thread count by 1 for each')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.grid(True)
plt.savefig("graph/avgs.png")

