import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features import extract_features_from_file

plt.ion()

# Load data
project_root = Path(__file__).parent.parent.parent
data_path = project_root / "data/hackathon_public.json"
circuits_dir = project_root / "circuits"

with open(data_path) as f:
    data = json.load(f)

circuits = {c["file"]: c for c in data["circuits"]}
results = data["results"]

# Extract fidelity fits and axis_ac for each circuit
circuit_data = []

for result in results:
    if result["status"] != "ok":
        continue
    
    file_name = result["file"]
    threshold_sweep = result.get("threshold_sweep", [])
    
    if not threshold_sweep:
        continue
    
    # Extract thresholds and fidelities
    thresholds = []
    fidelities = []
    for entry in threshold_sweep:
        t = entry.get("threshold")
        f = entry.get("sdk_get_fidelity")
        if t is not None and f is not None and isinstance(f, (int, float)) and not np.isnan(f):
            thresholds.append(t)
            fidelities.append(f)
    
    if len(thresholds) < 3:
        continue
    
    # Sort by threshold
    sort_idx = np.argsort(thresholds)
    thresholds = np.array(thresholds)[sort_idx]
    fidelities = np.array(fidelities)[sort_idx]
    
    # Convert to log2 space
    log2_thresholds = np.array([math.log2(t) if t > 0 else 0 for t in thresholds])
    
    # Fit logistic curve: y = L / (1 + exp(-k*(x - x0)))
    try:
        def logistic(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))
        
        p0 = [1.0, 1.0, np.mean(log2_thresholds)]
        popt, _ = curve_fit(logistic, log2_thresholds, fidelities, p0=p0, maxfev=10000)
        L, k, x0 = popt
        
        # Extract axis_ac from QASM
        qasm_path = circuits_dir / file_name
        if qasm_path.exists():
            features = extract_features_from_file(qasm_path)
            axis_a_volume = features.get('axis_a_volume', 0.0)
            axis_c_packing = features.get('axis_c_packing', 0.0)
            axis_ac = axis_a_volume * axis_c_packing
        else:
            axis_ac = 0.0
        
        circuit_data.append({
            'file': file_name,
            'family': circuits.get(file_name, {}).get('family', 'Unknown'),
            'axis_ac': axis_ac,
            'logistic_L': L,
            'logistic_k': k,
            'logistic_x0': x0,
        })
    except:
        continue

# Sort by axis_ac and select diverse examples
circuit_data_sorted = sorted(circuit_data, key=lambda x: x['axis_ac'])
n_curves = min(8, len(circuit_data_sorted))

# Select evenly spaced circuits
selected_indices = np.linspace(0, len(circuit_data_sorted) - 1, n_curves, dtype=int)
selected_circuits = [circuit_data_sorted[i] for i in selected_indices]

print(f"Selected {len(selected_circuits)} circuits with axis_ac values:")
for c in selected_circuits:
    print(f"  {c['file']}: axis_ac = {c['axis_ac']:.2f}")

# Create single plot
fig, ax = plt.subplots(figsize=(12, 8))

# Color map for axis_ac values
axis_ac_values = [c['axis_ac'] for c in selected_circuits]
norm = plt.Normalize(min(axis_ac_values), max(axis_ac_values))
cmap = plt.cm.viridis

# Plot each circuit's fidelity curve
for i, circuit in enumerate(selected_circuits):
    L = circuit['logistic_L']
    k = circuit['logistic_k']
    x0 = circuit['logistic_x0']
    axis_ac = circuit['axis_ac']
    
    # Reconstruct logistic curve
    def logistic(x):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # Plot curve
    x_curve = np.linspace(0, 8, 200)
    y_curve = logistic(x_curve)
    y_curve = np.clip(y_curve, 0, 1.1)
    
    color = cmap(norm(axis_ac))
    ax.plot(x_curve, y_curve, color=color, alpha=0.6, linewidth=2, 
           label=f"axis_ac={axis_ac:.1f}")
    
    # Mark the point on the curve corresponding to axis_ac
    # Use axis_ac as a threshold value (in log2 space)
    # Since axis_ac is a difficulty metric, map it to log2 threshold space
    # We'll use the threshold where the curve reaches a certain fidelity
    # Or simply use axis_ac as a threshold indicator
    
    # Map axis_ac to log2 threshold space (normalize axis_ac to reasonable range)
    # axis_ac typically ranges from ~0 to ~100+, so we'll scale it
    axis_ac_log2 = np.log2(max(1, axis_ac))  # Convert to log2 space
    
    # Find the fidelity at this threshold
    if 0 <= axis_ac_log2 <= 8:
        fidelity_at_axis_ac = logistic(axis_ac_log2)
            ax.plot(axis_ac_log2, fidelity_at_axis_ac, 'o', 
                   color=color, markersize=10, markeredgecolor='black', 
                   markeredgewidth=1.5, zorder=10)

# Add target lines
ax.axhline(y=0.75, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='0.75 target')
ax.axhline(y=0.99, color='red', linestyle='--', linewidth=2, alpha=0.7, label='0.99 target')

ax.set_xlabel('log2(threshold)', fontsize=14)
ax.set_ylabel('Fidelity', fontsize=14)
ax.set_title('Individual Circuit Fidelity Curves with axis_ac Mapping\n(Points show axis_ac threshold position)', fontsize=15)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

plt.tight_layout()
plt.show(block=False)

print("\nPlot generated. Each point marks where axis_ac maps to on that circuit's fidelity curve.")
try:
    input("\nPress Enter to close...")
except EOFError:
    print("\n")
