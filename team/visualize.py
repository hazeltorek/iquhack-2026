import analysis
import matplotlib.pyplot as plt
from math import log

data = analysis.data

state = (True, True) # CPU, single
x = [d["file_len"] for d in data.values()]
y = [min([r["threshold"] for r in d["thresholds"] if (r["is_cpu"], r["is_single"]) == state], default=1.0) for d in data.values()]
plt.scatter(x, y)
plt.yscale("log")
plt.show()