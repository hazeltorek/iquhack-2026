import analysis
import matplotlib.pyplot as plt
from math import log
import numpy as np

flattened = analysis.flattened

# Create single plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

print("\n" + "="*70)
print("THRESHOLD vs SECONDS")
print("="*70)

# Extract threshold and seconds
thresholds = np.array([r["threshold"] for r in flattened])
seconds = np.array([r["seconds"] for r in flattened])

# Calculate correlation
corr = np.corrcoef(thresholds, seconds)[0, 1]

# Scatter plot
ax.scatter(thresholds, seconds, alpha=0.6, s=50, c='blue', edgecolors='k', linewidths=0.5)

# Line of best fit in log-log space
log_thresholds = np.log10(thresholds)
log_seconds = np.log10(seconds)
coeffs = np.polyfit(log_thresholds, log_seconds, 1)
x_fit = np.logspace(np.log10(thresholds.min()), np.log10(thresholds.max()), 100)
y_fit = 10 ** (coeffs[0] * np.log10(x_fit) + coeffs[1])
ax.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.8, label=f'Power law fit')

ax.set_xlabel('Threshold', fontsize=13)
ax.set_ylabel('Seconds', fontsize=13)
ax.set_title(f'Threshold vs Runtime\nCorrelation: {corr:.4f}', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11)

print(f"\nCorrelation: {corr:.4f}")
print(f"Samples: {len(thresholds)}")
print(f"Threshold range: {thresholds.min():.0f} - {thresholds.max():.0f}")
print(f"Runtime range: {seconds.min():.2f}s - {seconds.max():.2f}s")

plt.tight_layout()
plt.savefig('team/threshold_vs_seconds.png', dpi=150, bbox_inches='tight')

print("\n" + "="*70)
print("Plot saved to: team/threshold_vs_seconds.png")
print("="*70)

plt.show()

print(f"\nPlotted {len(file_names)} files")

plt.tight_layout()
plt.savefig('team/log_threshold_vs_fidelity.png', dpi=150, bbox_inches='tight')

print("\n" + "="*70)
print("Plot saved to: team/log_threshold_vs_fidelity.png")
print("="*70)

plt.show()

