import matplotlib.pyplot as plt
import os

def plot_facilities(instance, solution=None, only_open_acfs=False, filename="output.png"):
    plt.figure(figsize=(8, 6))

    # Plot disaster areas
    x_d, y_d = zip(*instance.DisasterArea_Position)
    plt.scatter(x_d, y_d, c='red', marker='X', s=100, label='Disaster Area')

    # Plot hospitals
    x_h, y_h = zip(*instance.Hospital_Position)
    plt.scatter(x_h, y_h, c='blue', marker='s', s=80, label='Hospital')

    # Plot ACFs
    acf_positions = []
    if only_open_acfs and solution:
        for i in instance.ACFSet:
            if solution.ACFEstablishment_x_wi[0][i] == 1:
                acf_positions.append(instance.ACF_Position[i])
        label = 'Open ACF'
    else:
        acf_positions = [instance.ACF_Position[i] for i in instance.ACFSet]
        label = 'ACF'

    if acf_positions:
        x_a, y_a = zip(*acf_positions)
        plt.scatter(x_a, y_a, c='green', marker='o', s=100, label=label)

    # Labels and layout
    plt.title("Facility Locations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    output_dir = "UI\Solution_UI"  # Ensure this folder exists
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)

    plt.close()


