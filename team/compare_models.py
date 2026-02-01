import sys
import os
import analysis as analysis
import numpy as np
import matplotlib.pyplot as plt
from math import log
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

flattened = analysis.flattened

# Define test files (same as TimeReg.py)
test_files = {
    'ae_indep_qiskit_130', 'dj_indep_qiskit_30', 'ghz_indep_qiskit_30', 
    'ghz_indep_qiskit_130', 'grover-noancilla_indep_qiskit_11', 
    'grover-v-chain_indep_qiskit_17', 'portfolioqaoa_indep_qiskit_17',
    'portfoliovqe_indep_qiskit_18', 'qft_indep_qiskit_15', 
    'qftentangled_indep_qiskit_30', 'qpeexact_indep_qiskit_30', 
    'wstate_indep_qiskit_130'
}

# Filter for CPU/Single precision (float32) only - GPU not used for prediction
# state = (is_cpu, is_single) = (True, True) = CPU/Float
state = (True, True)

# Collect training and test data
X_train, y_train = [], []
X_test, y_test = [], []
test_file_names = []
test_n_values = []
test_threshold_values = []

for record in flattened:
    if (record["is_cpu"], record["is_single"]) == state:
        x_val = (record["n"] ** 2) * record["threshold"]  # n² × threshold
        y_val = record["seconds"]
        
        if record["file_name"] in test_files:
            X_test.append(x_val)
            y_test.append(y_val)
            test_file_names.append(record["file_name"])
            test_n_values.append(record["n"])
            test_threshold_values.append(record["threshold"])
        else:
            X_train.append(x_val)
            y_train.append(y_val)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("\n" + "="*70)
print("MODEL COMPARISON: n² × threshold Predictor")
print("="*70)
print(f"\n*** CPU/Single Precision (float32) ONLY - GPU data excluded ***")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Test circuits: {len(test_files)}")

# MODEL 1: Power law - seconds = a * (n²×threshold)^b
print("\n" + "-"*70)
print("MODEL 1: POWER LAW")
print("-"*70)
print("Form: seconds = a × (n²×threshold)^b")

try:
    popt_power, _ = curve_fit(lambda x, a, b: a * np.power(x, b), X_train, y_train, 
                               p0=[1.0, 0.5], maxfev=10000)
    a_power, b_power = popt_power
    
    print(f"Fitted: seconds = {a_power:.6f} × (n²×threshold)^{b_power:.6f}")
    
    # Predict on test set
    y_test_pred_power = a_power * np.power(X_test, b_power)
    
    mse_power = mean_squared_error(y_test, y_test_pred_power)
    mae_power = mean_absolute_error(y_test, y_test_pred_power)
    r2_power = r2_score(y_test, y_test_pred_power)
    mape_power = np.mean(np.abs((y_test - y_test_pred_power) / y_test)) * 100
    
    print(f"\nTest Set Performance:")
    print(f"  MSE:  {mse_power:.6f}")
    print(f"  MAE:  {mae_power:.6f}")
    print(f"  R²:   {r2_power:.6f}")
    print(f"  MAPE: {mape_power:.2f}%")
    power_success = True
except Exception as e:
    print(f"Power law fit failed: {e}")
    power_success = False

# MODEL 2: Linear on log-log scale
print("\n" + "-"*70)
print("MODEL 2: LINEAR (log-log scale)")
print("-"*70)
print("Form: log2(seconds) = m × log(n²×threshold) + c")

X_train_log = np.log(X_train)
X_test_log = np.log(X_test)
y_train_log2 = np.log2(y_train)

# Fit linear regression
linear_coeffs = np.polyfit(X_train_log, y_train_log2, 1)
m, c = linear_coeffs

print(f"Fitted: log2(seconds) = {m:.6f} × log(n²×threshold) + {c:.6f}")
print(f"Equivalent power law: seconds = 2^{c:.6f} × (n²×threshold)^{m:.6f}")
print(f"                      seconds ≈ {2**c:.6f} × (n²×threshold)^{m:.6f}")

# Predict on test set
y_test_log2_pred = m * X_test_log + c
y_test_pred_linear = 2 ** y_test_log2_pred

mse_linear = mean_squared_error(y_test, y_test_pred_linear)
mae_linear = mean_absolute_error(y_test, y_test_pred_linear)
r2_linear = r2_score(y_test, y_test_pred_linear)
mape_linear = np.mean(np.abs((y_test - y_test_pred_linear) / y_test)) * 100

print(f"\nTest Set Performance:")
print(f"  MSE:  {mse_linear:.6f}")
print(f"  MAE:  {mae_linear:.6f}")
print(f"  R²:   {r2_linear:.6f}")
print(f"  MAPE: {mape_linear:.2f}%")

# MODEL 3: Polynomial (degree 2) on log-log scale
print("\n" + "-"*70)
print("MODEL 3: POLYNOMIAL (degree 2, log-log scale)")
print("-"*70)
print("Form: log2(seconds) = a×[log(n²×threshold)]² + b×log(n²×threshold) + c")

poly2_coeffs = np.polyfit(X_train_log, y_train_log2, 2)
a2, b2, c2 = poly2_coeffs

print(f"Fitted: log2(seconds) = {a2:.6f}×x² + {b2:.6f}×x + {c2:.6f}")
print(f"        where x = log(n²×threshold)")

# Predict on test set
y_test_log2_pred_poly2 = a2 * X_test_log**2 + b2 * X_test_log + c2
y_test_pred_poly2 = 2 ** y_test_log2_pred_poly2

mse_poly2 = mean_squared_error(y_test, y_test_pred_poly2)
mae_poly2 = mean_absolute_error(y_test, y_test_pred_poly2)
r2_poly2 = r2_score(y_test, y_test_pred_poly2)
mape_poly2 = np.mean(np.abs((y_test - y_test_pred_poly2) / y_test)) * 100

print(f"\nTest Set Performance:")
print(f"  MSE:  {mse_poly2:.6f}")
print(f"  MAE:  {mae_poly2:.6f}")
print(f"  R²:   {r2_poly2:.6f}")
print(f"  MAPE: {mape_poly2:.2f}%")

# Print detailed predictions
print("\n" + "="*70)
print("DETAILED TEST SET PREDICTIONS")
print("="*70)
print(f"{'File':<35} {'n':>3} {'T':>4} {'Actual':>8} {'Poly':>8} {'Linear':>8} {'Poly%Err':>9} {'Lin%Err':>9}")
print("-"*70)

for i in range(len(y_test)):
    actual = y_test[i]
    pred_poly = y_test_pred[i]
    pred_linear = y_test_pred_linear[i]
    err_poly = abs(actual - pred_poly) / actual * 100
    err_linear = abs(actual - pred_linear) / actual * 100
    
    print(f"{test_file_names[i]:<35} {test_n_values[i]:>3} {test_threshold_values[i]:>4} "
          f"{actual:>8.2f} {pred_poly:>8.2f} {pred_linear:>8.2f} {err_poly:>8.1f}% {err_linear:>8.1f}%")

# Create visualization comparing models
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Polynomial model - predicted vs actual
ax1 = axes[0, 0]
ax1.scatter(y_test, y_test_pred, alpha=0.7, edgecolors='k', s=60)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
ax1.set_xlabel('Actual Runtime (s)', fontsize=11)
ax1.set_ylabel('Predicted Runtime (s)', fontsize=11)
ax1.set_title(f'Polynomial Model (degree 3)\nR² = {r2_poly:.4f}, MAPE = {mape_poly:.1f}%', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Linear model - predicted vs actual
ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred_linear, alpha=0.7, edgecolors='k', s=60, color='orange')
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
ax2.set_xlabel('Actual Runtime (s)', fontsize=11)
ax2.set_ylabel('Predicted Runtime (s)', fontsize=11)
ax2.set_title(f'Linear Model\nR² = {r2_linear:.4f}, MAPE = {mape_linear:.1f}%', 
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Error distribution for polynomial model
ax3 = axes[1, 0]
errors_poly = ((y_test_pred - y_test) / y_test) * 100
ax3.hist(errors_poly, bins=15, edgecolor='k', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
ax3.set_xlabel('Percentage Error (%)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title(f'Polynomial Model Error Distribution\nMean = {np.mean(errors_poly):.1f}%, Std = {np.std(errors_poly):.1f}%',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Model comparison
ax4 = axes[1, 1]
x_pos = np.arange(len(test_file_names))
width = 0.35
ax4.bar(x_pos - width/2, errors_poly, width, label='Polynomial', alpha=0.7)
ax4.bar(x_pos + width/2, ((y_test_pred_linear - y_test) / y_test) * 100, width, 
        label='Linear', alpha=0.7, color='orange')
ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax4.set_xlabel('Test Circuit Index', fontsize=11)
ax4.set_ylabel('Percentage Error (%)', fontsize=11)
ax4.set_title('Per-Circuit Error Comparison', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend()

plt.tight_layout()
plt.savefig('team/model_comparison_n2_threshold.png', dpi=150, bbox_inches='tight')

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nPolynomial Model (degree 3):")
print(f"  R² = {r2_poly:.4f}, MAPE = {mape_poly:.2f}%")
print(f"\nLinear Model:")
print(f"  R² = {r2_linear:.4f}, MAPE = {mape_linear:.2f}%")

if r2_poly > r2_linear:
    print(f"\n✓ Polynomial model performs better (ΔR² = +{r2_poly - r2_linear:.4f})")
else:
    print(f"\n✓ Linear model performs better (ΔR² = +{r2_linear - r2_poly:.4f})")

print(f"\nPlot saved to: team/model_comparison_n2_threshold.png")
print("="*70)

plt.show()
