import analysis
import matplotlib.pyplot as plt
from math import log
import numpy as np

flattened = analysis.flattened

# All permutations of (is_cpu, is_single)
states = [
    (True, True),   # CPU, single
    (True, False),  # CPU, double
    (False, True),  # GPU, single
    (False, False)  # GPU, double
]
colors = ['blue', 'red', 'green', 'orange']
labels = ['CPU/Single', 'CPU/Double', 'GPU/Single', 'GPU/Double']

for state, color, label in zip(states, colors, labels):
    x, y = [], []
    for record in flattened:
        if (record["is_cpu"], record["is_single"]) == state:
            if record["n_cx"] < 0.01:
                continue
            x.append(log(record["n_cx"]))
            y.append(log(record["threshold"])/log(2))
    
    plt.scatter(x, y, c=color, label=label, alpha=0.6)
    
    x_arr = np.array(x)
    y_arr = np.array(y)
    coeffs = np.polyfit(x_arr, y_arr, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(min(x), max(x), 100)
    plt.plot(x_line, poly(x_line), c=color, linestyle='--', linewidth=2, alpha=0.8)

plt.xlabel("Number of Qubits (n)")
plt.ylabel("log2(file_len)")
plt.legend()
plt.title("File Lines vs Number of Qubits")
plt.show()

