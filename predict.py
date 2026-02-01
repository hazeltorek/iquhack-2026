def main(args):
    # print(args.tasks) # data/hackathon_public.json
    # print(args.circuits)
    # print(args.id_map)
    # print(args.out) # predictions

    # Save Threshold

    # Save Forward Wall S




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type = str, default="data\hackathon_public.json")
    parser.add_argument("--circuits", type = str, default="circuits")
    parser.add_argument("--id-map", type = str, default="data\holdout_public.json")
    parser.add_argument("--out", type = str, default="predictions.json")
    parser.add_argument("--out", type = str, default="artifacts")
    parser.add_argument("--out", type = str, default="src")
    args = parser.parse_args()

    main(args)
                   