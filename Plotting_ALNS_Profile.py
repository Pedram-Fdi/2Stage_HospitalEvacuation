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

def plot_global_best_cost(folder_path, file_names, x_axis='iteration', time_limit=None, iter_limit=None, best_optimal_solution=None):
    """
    Plot global best cost for one or two log files on the same graph.
    Optionally plot a horizontal line for the best optimal solution.

    Parameters:
    - folder_path: directory containing the log files
    - file_names: string or list of strings of log file names
    - x_axis: 'iteration' or 'time' (x-axis)
    - time_limit: max elapsed time to include
    - iter_limit: max iteration count to include
    - best_optimal_solution: float or None, horizontal line value to plot
    """
    
    if isinstance(file_names, str):
        file_names = [file_names]
    if len(file_names) > 2:
        raise ValueError("Currently supports up to two files for comparison.")
    
    plt.figure()
    
    for file_name in file_names:
        data = read_log_file(folder_path, file_name)
        
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
            raise ValueError(f"No data points within given limits for file: {file_name}")
        
        # Determine label based on file name content
        if "S_0_0" in file_name:
            label = "ALNS"
        elif "S_1_1" in file_name:
            label = "DRL-ALNS"
        else:
            label = file_name  # fallback: just show file name
        
        xlabel = "Iteration" if x_axis == 'iteration' else "Elapsed Time (s)"
        plt.plot(filtered_x, filtered_y, marker=None, label=label)
    
    # Add horizontal line for the best optimal solution, if provided
    if best_optimal_solution is not None:
        plt.axhline(y=best_optimal_solution, color='red', linestyle='--', label='Best Optimal Solution (MIP)')
    
    plt.xlabel(xlabel)
    plt.ylabel("Global Best Cost")
    plt.title(f"Global Best Cost vs {xlabel}")
    plt.grid(True)
    plt.legend()
    plt.show()

# --- Example usage ---

folder_path = r"C:\PhD\Thesis\Papers\3rd\Code\Results\Approved-Instances\Temp"
Best_Optimal_Solution_MIP = None
file_name_1 = "ALNStrace_5_20_5_20_3_5_CRP_2Stage_ALNS_350_RQMC_42_Q_S_0_0_NS_NoC_Evaluation_False.txt"
file_name_2 = "ALNStrace_5_20_5_20_3_5_CRP_2Stage_ALNS_350_RQMC_42_Q_S_1_1_NS_NoC_Evaluation_False.txt"

# Plot both files together, x-axis = iteration, up to iteration 1000
#plot_global_best_cost(folder_path, [file_name_1, file_name_2], x_axis='iteration', iter_limit=None, best_optimal_solution=Best_Optimal_Solution_MIP)

# Or plot both files with time axis, no time limit
plot_global_best_cost(folder_path, [file_name_1, file_name_2], x_axis='time', time_limit = None, best_optimal_solution=Best_Optimal_Solution_MIP)
