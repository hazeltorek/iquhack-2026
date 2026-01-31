import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main(args):
    print(args.tasks)
    print(args.circuits)
    print(args.id_map)
    print(args.out)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type = str, default="data\hackathon_public.json")
    parser.add_argument("--circuits", type = str, default="circuits")
    parser.add_argument("--id-map", type = str, default="data\holdout_public.json")
    parser.add_argument("--out", type = str, default="")
    args = parser.parse_args()
    
    main(args)
                   