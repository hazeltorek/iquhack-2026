import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_99 import load_data

plt.ion()

# Paths relative to project root
data_path = Path("../data/hackathon_public.json")
circuits_dir = Path("../circuits")

df = load_data(data_path, circuits_dir)

df['axis_ac'] = df['axis_a_volume'] * df['axis_c_packing']

families = sorted(df['family'].unique())
colors = plt.cm.tab20(np.linspace(0, 1, len(families)))
family_colors = {fam: colors[i] for i, fam in enumerate(families)}

# axis_a_volume vs min_threshold
fig1, ax1 = plt.subplots(figsize=(8, 6))
data1 = df[['axis_a_volume', 'min_threshold', 'family']].dropna()
corr1 = data1['axis_a_volume'].corr(data1['min_threshold'])

for family in families:
    family_data = data1[data1['family'] == family]
    if len(family_data) > 0:
        ax1.scatter(family_data['axis_a_volume'], family_data['min_threshold'], 
                   alpha=0.7, s=80, label=family, color=family_colors[family])

if len(data1) > 1:
    z1 = np.polyfit(data1['axis_a_volume'], data1['min_threshold'], 1)
    p1 = np.poly1d(z1)
    x_line1 = np.linspace(data1['axis_a_volume'].min(), data1['axis_a_volume'].max(), 100)
    ax1.plot(x_line1, p1(x_line1), "k--", linewidth=2, alpha=0.8, 
            label=f'Trend (corr={corr1:.3f})')

ax1.set_xlabel('Entanglement Volume', fontsize=12)
ax1.set_ylabel('Minimum Threshold', fontsize=12)
ax1.set_title(f'Etgl Volume vs Min Threshold\nCorrelation: {corr1:.3f}', 
             fontsize=14)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

# axis_c_packing vs min_threshold
fig2, ax2 = plt.subplots(figsize=(8, 6))
data2 = df[['axis_c_packing', 'min_threshold', 'family']].dropna()
corr2 = data2['axis_c_packing'].corr(data2['min_threshold'])

for family in families:
    family_data = data2[data2['family'] == family]
    if len(family_data) > 0:
        ax2.scatter(family_data['axis_c_packing'], family_data['min_threshold'], 
                   alpha=0.7, s=80, label=family, color=family_colors[family])

if len(data2) > 1:
    z2 = np.polyfit(data2['axis_c_packing'], data2['min_threshold'], 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(data2['axis_c_packing'].min(), data2['axis_c_packing'].max(), 100)
    ax2.plot(x_line2, p2(x_line2), "k--", linewidth=2, alpha=0.8, 
            label=f'Trend (corr={corr2:.3f})')

ax2.set_xlabel('Packing', fontsize=12)
ax2.set_ylabel('Minimum Threshold', fontsize=12)
ax2.set_title(f'Packing vs Min Threshold\nCorrelation: {corr2:.3f}', 
             fontsize=14)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

# axis_ac (combined) vs min_threshold
fig3, ax3 = plt.subplots(figsize=(8, 6))
data3 = df[['axis_ac', 'min_threshold', 'family']].dropna()
corr3 = data3['axis_ac'].corr(data3['min_threshold'])

for family in families:
    family_data = data3[data3['family'] == family]
    if len(family_data) > 0:
        ax3.scatter(family_data['axis_ac'], family_data['min_threshold'], 
                   alpha=0.7, s=80, label=family, color=family_colors[family])

if len(data3) > 1:
    z3 = np.polyfit(data3['axis_ac'], data3['min_threshold'], 1)
    p3 = np.poly1d(z3)
    x_line3 = np.linspace(data3['axis_ac'].min(), data3['axis_ac'].max(), 100)
    ax3.plot(x_line3, p3(x_line3), "k--", linewidth=2, alpha=0.8, 
            label=f'Trend (corr={corr3:.3f})')

ax3.set_xlabel('Entanglement Progression', fontsize=12)
ax3.set_ylabel('Minimum Threshold', fontsize=12)
ax3.set_title(f'Etgl Prog vs Min Threshold\nCorrelation: {corr3:.3f}', 
             fontsize=14)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

try:
    input("\nEnter -> Close")
except EOFError:
    print("\n")
