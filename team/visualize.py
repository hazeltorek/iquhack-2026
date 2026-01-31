import analysis
import matplotlib.pyplot as plt
from math import log
import numpy as np

data = analysis.data

# All permutations of (is_cpu, is_single)
states = [
    (True, True),   # CPU, single
    (True, False),  # CPU, double
    (False, True),  # GPU, single
    (False, False)  # GPU, double
]
colors = ['blue', 'red', 'green', 'orange']
labels = ['CPU/Single', 'CPU/Double', 'GPU/Single', 'GPU/Double']


# append to x, y
def plot_point(file_name, entry, x, y):
    x.append(entry["n"])
    y.append(log(entry["file_len"])/log(2))


for state, color, label, i in zip(states, colors, labels, range(len(states))):
    x, y = [], []
    for file_name in data:
        entry = data[file_name]
        for result in entry["just_past"]:
            if (result["is_cpu"], result["is_single"]) == state:
                plot_point(file_name, entry, x, y)
    
    plt.scatter(x, y, c=color, label=label, alpha=0.6)
    
    x_arr = np.array(x)
    y_arr = np.array(y)
    coeffs = np.polyfit(x_arr, y_arr, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(min(x), max(x), 100)
    plt.plot(x_line, poly(x_line), c=color, linestyle='--', linewidth=2, alpha=0.8)

print(len(x))

plt.xlabel("Number of Qubits (n)")
plt.ylabel("log2(threshold)")
plt.legend()
plt.title("Threshold vs Number of Qubits")
plt.show()

