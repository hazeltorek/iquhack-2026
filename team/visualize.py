import analysis
import matplotlib.pyplot as plt
from math import log
import numpy as np

flattened = analysis.flattened

# All states to analyze
states = [
    (True, True),   # CPU/Single
    (True, False),  # CPU/Double
    (False, True),  # GPU/Single
    (False, False)  # GPU/Double
]
state_labels = ['CPU/Single', 'CPU/Double', 'GPU/Single', 'GPU/Double']

# Collect all unique families across all states
all_families = set()
for record in flattened:
    all_families.add(record["family"])
all_families = sorted(list(all_families))
family_colors = plt.cm.tab20(np.linspace(0, 1, len(all_families)))
family_to_color = {family: family_colors[i] for i, family in enumerate(all_families)}

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

print("\n" + "="*70)
print("CORRELATION ANALYSIS - n² × threshold vs Runtime")
print("="*70)

for idx, (state, label) in enumerate(zip(states, state_labels)):
    ax = axes[idx]
    
    # Collect data with family information for this state
    x_values = []
    y_values = []
    families_list = []
    
    for record in flattened:
        if (record["is_cpu"], record["is_single"]) == state:
            n2_threshold = log((record["n"] ** 2) * record["threshold"])
            log2_seconds = log(record["seconds"]) / log(2)
            
            x_values.append(n2_threshold)
            y_values.append(log2_seconds)
            families_list.append(record["family"])
    
    if len(x_values) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(label, fontsize=12, fontweight='bold')
        continue
    
    # Convert to numpy arrays
    x = np.array(x_values)
    y = np.array(y_values)
    
    # Calculate correlation
    corr = np.corrcoef(x, y)[0, 1]
    
    # Scatter plot with family colors
    families_in_state = sorted(set(families_list))
    for family in families_in_state:
        mask = [f == family for f in families_list]
        x_fam = x[mask]
        y_fam = y[mask]
        ax.scatter(x_fam, y_fam, c=[family_to_color[family]], alpha=0.7, 
                   edgecolors='k', linewidths=0.5, s=40, label=family)
    
    # Polynomial fit (degree 3)
    coeffs = np.polyfit(x, y, 3)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = coeffs[0] * x_line**3 + coeffs[1] * x_line**2 + coeffs[2] * x_line + coeffs[3]
    ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('log(n³ × threshold)', fontsize=11)
    ax.set_ylabel('log2(seconds)', fontsize=11)
    ax.set_title(f'{label}\nr = {corr:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend only for first subplot to avoid clutter
    if idx == 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    
    # Print stats
    print(f"\n{label}:")
    print(f"  Correlation: r = {corr:.4f}")
    print(f"  Samples: {len(x)}")
    print(f"  Families: {len(families_in_state)}")

plt.tight_layout()
plt.savefig('team/n2_threshold_all_states.png', dpi=150, bbox_inches='tight')

print("\n" + "="*70)
print("Plot saved to: team/n2_threshold_all_states.png")
print("="*70)

plt.show()

