import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import math

plt.ion()

data_path = Path("data/hackathon_public.json")
circuits_dir = Path("circuits")

# Load data
with open(data_path) as f:
    data = json.load(f)

circuits = {c["file"]: c for c in data["circuits"]}
results = data["results"]

all_sweeps = []
for result in results:
    if result["status"] != "ok":
        continue
    
    file_name = result["file"]
    threshold_sweep = result.get("threshold_sweep", [])
    
    if not threshold_sweep:
        continue
    
    # Extract
    thresholds = []
    fidelities = []
    for entry in threshold_sweep:
        t = entry.get("threshold")
        f = entry.get("sdk_get_fidelity")
        if t is not None and f is not None and isinstance(f, (int, float)) and not np.isnan(f):
            thresholds.append(t)
            fidelities.append(f)
    
    if len(thresholds) >= 3:
        all_sweeps.append({
            'file': file_name,
            'family': circuits.get(file_name, {}).get('family', 'Unknown'),
            'thresholds': np.array(thresholds),
            'fidelities': np.array(fidelities),
        })

# Sizing
fig, ax = plt.subplots(figsize=(10, 7))

families = sorted(set(s['family'] for s in all_sweeps))
colors = plt.cm.tab20(np.linspace(0, 1, len(families)))
family_colors = {fam: colors[i] for i, fam in enumerate(families)}

# Indv circuit's fidelity curve
for sweep in all_sweeps:
    thresholds = sweep['thresholds']
    fidelities = sweep['fidelities']
    family = sweep['family']
    color = family_colors.get(family, 'gray')
    
    # Sort
    sort_idx = np.argsort(thresholds)
    thresholds = thresholds[sort_idx]
    fidelities = fidelities[sort_idx]
    
    # logspace
    log2_thresholds = np.array([math.log2(t) if t > 0 else 0 for t in thresholds])
    
    try:
        def logistic(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))
        
        p0 = [1.0, 1.0, np.mean(log2_thresholds)]
        
        popt, _ = curve_fit(logistic, log2_thresholds, fidelities, p0=p0, maxfev=10000)
        L, k, x0 = popt
        
        # Plot
        ax.scatter(log2_thresholds, fidelities, alpha=0.3, s=20, color=color, zorder=1)
        
        x_fit = np.linspace(log2_thresholds.min(), log2_thresholds.max(), 200)
        y_fit = logistic(x_fit, L, k, x0)
        y_fit = np.clip(y_fit, 0, 1.1)
        ax.plot(x_fit, y_fit, color=color, alpha=0.4, linewidth=1, zorder=0, label='_nolegend_')
    except:
        # Fail safe
        ax.scatter(log2_thresholds, fidelities, alpha=0.3, s=20, color=color, zorder=1)

# Graphics
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=family_colors[fam], lw=2, label=fam) 
                   for fam in families]
ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

ax.set_xlabel('Threshold (logspace)', fontsize=12)
ax.set_ylabel('Fidelity', fontsize=12)
ax.set_title('Fidelity vs Threshold (Logistic)', 
             fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.1])

# Fidelity targets
ax.axhline(y=0.99, color='r', linestyle='--', linewidth=2, alpha=0.5, label='0.99 target')
ax.axhline(y=0.75, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='0.75 target')

plt.tight_layout()
plt.show(block=False)


try:
    input("\nEnter -> Close")
except EOFError:
    print("\n")