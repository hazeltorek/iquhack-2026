import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.ion()

def plot_scatter(df, x_feat, y_feat):
    
    # Silly useful stuff
    data = df[[x_feat, y_feat]].dropna()
    if len(data) == 0:
        return
    corr = data[x_feat].corr(data[y_feat])
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_feat], data[y_feat], alpha=0.6, s=50)
    if len(data) > 1 and np.std(data[x_feat]) > 1e-10 and np.std(data[y_feat]) > 1e-10:
        try:
            z = np.polyfit(data[x_feat], data[y_feat], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[x_feat].min(), data[x_feat].max(), 100)
            plt.plot(x_line, p(x_line), "r--", linewidth=2, label=f'y = {z[0]:.4f}x + {z[1]:.2f}')
        except:
            pass
    plt.title(f'{x_feat} vs {y_feat}\nCorrelation: {corr:.3f}')
    plt.xlabel(x_feat)
    plt.ylabel(y_feat)
    
    # Settings
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # DON'T TURN THIS OFF IT'S SO ANNOYING, same goes for all others
    plt.show(block=False)

# Iterate for best of each type
def plot_best_correlations(df, corr_thresh, corr_runtime, top_n=5):
    top_thresh = corr_thresh.head(top_n)
    top_runtime = corr_runtime.head(top_n)
    

    has_family = 'family' in df.columns
    if has_family:
        families = sorted(df['family'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
        family_colors = {fam: colors[i] for i, fam in enumerate(families)}
    
    # Threshold plots
    fig1, axes1 = plt.subplots(1, top_n, figsize=(6*top_n, 5))
    try:
        fig1.canvas.manager.set_window_title('Threshold Correlations')
    except:
        pass
    if top_n == 1:
        axes1 = [axes1]
    for idx, (feat, val) in enumerate(top_thresh.items()):
        ax = axes1[idx]
        if has_family:
            data = df[[feat, 'min_threshold', 'family']].dropna()
            for family in families:
                family_data = data[data['family'] == family]
                if len(family_data) > 0:
                    x_vals = family_data[feat].values
                    y_vals = family_data['min_threshold'].values
                    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                    x_vals = x_vals[valid]
                    y_vals = y_vals[valid]
                    if len(x_vals) > 0:
                        ax.scatter(x_vals, y_vals, 
                                 alpha=0.6, s=30, color=family_colors[family], label=family)
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            data_all = df[[feat, 'min_threshold']].dropna()
            if len(data_all) > 1:
                try:
                    z = np.polyfit(data_all[feat], data_all['min_threshold'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(data_all[feat].min(), data_all[feat].max(), 100)
                    ax.plot(x_line, p(x_line), "k--", linewidth=2, alpha=0.5, label='Overall trend')
                except:
                    pass
        else:
            data = df[[feat, 'min_threshold']].dropna()
            if len(data) > 0:
                ax.scatter(data[feat], data['min_threshold'], alpha=0.5, s=30)
                if len(data) > 1 and np.std(data[feat]) > 1e-10 and np.std(data['min_threshold']) > 1e-10:
                    try:
                        z = np.polyfit(data[feat], data['min_threshold'], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(data[feat].min(), data[feat].max(), 100)
                        ax.plot(x_line, p(x_line), "r--", linewidth=1)
                    except:
                        pass
        ax.set_title(f'{feat[:20]}\n{val:.3f}')
        ax.set_xlabel(feat[:15])
        ax.set_ylabel('threshold')
    plt.tight_layout()
    plt.show(block=False)

    # Runtime plots
    fig2, axes2 = plt.subplots(1, top_n, figsize=(6*top_n, 5))
    try:
        fig2.canvas.manager.set_window_title('Runtime Correlations')
    except:
        pass
    if top_n == 1:
        axes2 = [axes2]
    for idx, (feat, val) in enumerate(top_runtime.items()):
        ax = axes2[idx]
        data = df[[feat, 'log_runtime']].dropna()
        if len(data) > 0:
            ax.scatter(data[feat], data['log_runtime'], alpha=0.5, s=30)
            if len(data) > 1 and np.std(data[feat]) > 1e-10 and np.std(data['log_runtime']) > 1e-10:
                try:
                    z = np.polyfit(data[feat], data['log_runtime'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(data[feat].min(), data[feat].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=1)
                except:
                    pass
            ax.set_title(f'{feat[:20]}\n{val:.3f}')
            ax.set_xlabel(feat[:15])
            ax.set_ylabel('log_runtime')
    plt.tight_layout()
    plt.show(block=False)
