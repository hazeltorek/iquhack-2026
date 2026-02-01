
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from visualize import plot_best_correlations
plt.ion()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/hackathon_public.json")
    parser.add_argument("--circuits", type=str, default="circuits")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top correlations to plot")
    parser.add_argument("--fidelity", type=float, default=0.99, choices=[0.75, 0.99], 
                       help="Fidelity target: 0.75 or 0.99 (default: 0.99)")
    args = parser.parse_args()

    # Import fidelity
    if args.fidelity == 0.75:
        from analysis_75 import load_data_075 as load_data, get_correlations
    else:
        from analysis_99 import load_data, get_correlations

    df = load_data(Path(args.data), Path(args.circuits))
    
    # df['axis_ac'] = df['axis_a_volume'] * df['axis_c_packing']
    
    corr_thresh, corr_runtime = get_correlations(df)
    
    print("\nTop 10 threshold correlations of all time:")
    for i, (feat, val) in enumerate(corr_thresh.head(10).items(), 1):
        print(f"{i:2d}. {feat:40s} {val:7.3f}")
    print("\nTop 10 runtime correlations of all time:")
    for i, (feat, val) in enumerate(corr_runtime.head(10).items(), 1):
        print(f"{i:2d}. {feat:40s} {val:7.3f}")
    
    plot_best_correlations(df, corr_thresh, corr_runtime, top_n=args.top_n)
    
    try:
        input("\nEnter -> Close")
    except EOFError:
        print("\n")