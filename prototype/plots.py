import argparse
from pathlib import Path
from analysis import load_data, get_correlations
from visualize import plot_scatter, plot_best_correlations

if __name__ == "__main__":

    # Parser abstraction for mass graphs, it kinda works
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/hackathon_public.json")
    parser.add_argument("--circuits", type=str, default="circuits")
    parser.add_argument("--x", type=str, default="poly_cross_99")
    parser.add_argument("--y", type=str, default="min_threshold")
    args = parser.parse_args()

    df = load_data(Path(args.data), Path(args.circuits))
    corr_thresh, corr_runtime = get_correlations(df)
    
    print("\nTop 10 of all time goated threshold:")
    for i, (feat, val) in enumerate(corr_thresh.head(10).items(), 1):
        print(f"{i:2d}. {feat:40s} {val:7.3f}")
    print("\nTop 10 of all time goated runtime:")
    for i, (feat, val) in enumerate(corr_runtime.head(10).items(), 1):
        print(f"{i:2d}. {feat:40s} {val:7.3f}")

    plot_scatter(df, args.x, args.y)
    plot_best_correlations(df, corr_thresh, corr_runtime, top_n=3)
    
    try:
        input("\nEnter will elim")
    except EOFError:
        print("\nPlots gen")
