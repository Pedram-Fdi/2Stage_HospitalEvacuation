import os
import re
import matplotlib.pyplot as plt

def read_log_file(folder_path, file_name):
    """Read the log file and extract iterations, elapsed times, and global best costs."""
    full_path = os.path.join(folder_path, file_name)
    with open(full_path, 'r') as f:
        text = f.read()
    
    iterations = []
    elapsed_times = []
    best_costs = []
    pattern = r"Iteration:\s*(\d+).*?global_best_cost:\s*([\d\.eE+-]+).*?elapsed_time:\s*([\d\.eE+-]+)"
    
    for line in text.splitlines():
        match = re.search(pattern, line)
        if match:
            iterations.append(int(match.group(1)))
            best_costs.append(float(match.group(2)))
            elapsed_times.append(float(match.group(3)))
    
    return {
        "iteration": iterations,
        "time": elapsed_times,
        "cost": best_costs
    }

def plot_global_best_cost(folder_path, file_name, x_axis='iteration', time_limit=None, iter_limit=None):
    """
    Plot global best cost against either 'iteration' or 'time', with optional limits.
    
    Parameters:
    - folder_path: path to the directory containing the log file
    - file_name: name of the log file
    - x_axis: 'iteration' or 'time' to choose the x-axis
    - time_limit: float, max elapsed time (in seconds) to include
    - iter_limit: int, max iteration count to include
    """
    data = read_log_file(folder_path, file_name)
    
    # Filter based on limits
    filtered_x = []
    filtered_y = []
    for it, t, cost in zip(data["iteration"], data["time"], data["cost"]):
        if iter_limit is not None and it > iter_limit:
            break
        if time_limit is not None and t > time_limit:
            break
        filtered_x.append(it if x_axis == 'iteration' else t)
        filtered_y.append(cost)
    
    if not filtered_x:
        raise ValueError("No data points within given limits.")
    
    xlabel = "Iteration" if x_axis == 'iteration' else "Elapsed Time (s)"
    
    plt.figure()
    plt.plot(filtered_x, filtered_y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel("Global Best Cost")
    plt.title(f"Global Best Cost vs {xlabel}")
    plt.grid(True)
    plt.show()

# --- Example usage ---
folder_path = r"C:\PhD\Thesis\Papers\3rd\Code\Results\2nd\Temp"
file_name   = "ALNStrace_6_10_5_10_3_3_CRP_2Stage_ALNS_250_QMC_42_Q_S_0_0_NS_SOM_Evaluation_False.txt"
#file_name   = "ALNStrace_5_10_5_20_3_1_CRP_2Stage_ALNS_100_RQMC_42_Q_S_1_1_NS_KMPP_Evaluation_False.txt"

# Plot cost vs time, stopping at 5000 seconds
plot_global_best_cost(folder_path, file_name, x_axis='time', time_limit=None)

# Plot cost vs iteration, stopping at iteration 100
#plot_global_best_cost(folder_path, file_name, x_axis='iteration', iter_limit=100)
