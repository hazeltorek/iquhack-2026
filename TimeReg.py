import argparse
import json, os, re
import networkx as nx

import copy
import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# pairs of qubits that interact via 2-qubit gates
def get_pairs(qasm):
    # i saw the face of god in a regular expression
    return sorted(list(set([tuple(sorted(list(map(int, p)))) for p in re.findall(r"[a-z]+(?:\((?:-{0,1}(?:\d+\.{0,1}\d+|pi(?:\/\d+){0,1}),{0,1}){1,3}\)){0,1}\sq\[(\d+)\],\s*q\[(\d+)\];", qasm)])))

# build interaction graph and get cool useful statistics out of it
graphs = {f: nx.Graph(get_pairs("\n".join(qasm))) for (f, qasm) in code}
for f in data.keys(): 
    data[f]["degree"] = max([0] + list(map(lambda d: d[1], graphs[(fn := f"{f}.qasm")].degree())))
    data[f]["n_edges"] = graphs[fn].number_of_edges()
    data[f]["centrality"] = max([0] + list(nx.degree_centrality(graphs[fn]).values()))
    data[f]["n_clusters"] = nx.number_connected_components(graphs[fn])

# organized data!
def process_circuit(name):
    a = [circ for circ in circuits if circ["file"] == name][0]
    b = [res for res in results if res["file"] == name]
    qasm = [c for (f, c) in code if f == name][0]
    length = len(qasm)

    # graph metrics
    G = nx.Graph(get_pairs("\n".join(qasm)))
    
    just_past = []
    for test in b:
        run_fid = None
        for run in sorted(test['threshold_sweep'], key=lambda x: x["threshold"]):
            fid = run["sdk_get_fidelity"]
            if isinstance(fid, float) and fid > 0.99:
                run_fid = run
                break
        if run_fid is not None:
            just_past.append({
                "threshold": run_fid["threshold"],
                "is_cpu": test["backend"] == "CPU",
                "is_single": test["precision"] == "single",
                "backend": test["backend"],
                "precision": test["precision"],
                "seconds": run_fid['run_wall_s']
            })

    # organize results by the given predictors (cpu/gpu, single/double)
    pred = {(r["backend"] == "CPU", r["precision"] == "single"): b for r in b}
    return {
        "family": a["family"], 
        "n": a["n_qubits"], 
        "file_len": length, 
        "results": pred,
        "just_past": just_past,
        "max_deg": max([0] + list(map(lambda d: d[1], G.degree())))
    }

def make_flattened():
    ret = []
    for file_name in data:
        entry = data[file_name]

        text = "".join([c for (f, c) in code if f == file_name + ".qasm"][0])
        
        n_meas = len(re.findall(r"\bmeasure\b", text))
        n_cx = len(re.findall(r"\bcx\b", text))
        n_cz = len(re.findall(r"\bcz\b", text))
        n_1q = len(re.findall(r"\b(h|x|y|z|s|sdg|t|tdg|rx|ry|rz|u1|u2|u3)\b", text))
        
        for result in entry["just_past"]:
            ret.append({
                "file_name": file_name,
                "n": entry["n"],
                "file_len": entry["file_len"],
                "lines": text.count("\n"),
                "family": entry["family"],
                "is_cpu": result["is_cpu"],
                "is_single": result["is_single"],
                "threshold": result["threshold"],
                "seconds": result["seconds"],
                "n_meas": n_meas,
                "n_cx": n_cx,
                "n_cz": n_cz,
                "n_1q": n_1q
            })
    return ret


class NonLinearModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),        # Prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)
 
def main(args):
    # print(args.tasks) # data/hackathon_public.json
    # print(args.circuits)
    # print(args.id_map)
    # print(args.out) # circuits

    global circuits, results, code, data

    # read all the circuits into an array
    code = [(f, open(p).readlines()) for f in os.listdir(args.circuits) if os.path.isfile(p := f"{args.circuits}/{f}")]

    # read the data for each circuit into the array
    with open(args.tasks) as f: 
        results = (j := json.load(f))["results"]
        circuits = j["circuits"]

    # organize the data. key is circuit, other stuff is named entries under that with results as tuple keys
    data = {re.match(r"(.+).qasm", f)[1]: process_circuit(f) for (f, _) in code}
    flattened = make_flattened()


    X = [] # Input Feature Data 
    y = [] # Threshold Values

    for val in flattened:

        

        features = [
            # val["n"],
            # val["file_len"], 
            # val["lines"],
            # int(val["is_cpu"]),
            # int(val["is_single"]),
            # val["threshold"],
            # val["n_meas"],
            # val["n_cx"],
            # val["n_cz"],
            # val["n_1q"],


            
        ]

        X.append(features)
        y.append(val["seconds"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_log = np.log2(y) 
    feature_size = len(X[0])

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    model = NonLinearModel(feature_size)
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


    # Create the model
    for epoch in range(1000):
        # Forward pass: Compute predicted y by passing 
        # x to the model
        pred_y = model(X_train_tensor)

        # Compute and print loss
        loss = criterion(pred_y, y_train_tensor)

        # Zero gradients, perform a backward pass, 
        # and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))


    # Add evaluation here:
    model.eval()
    with torch.no_grad():
        # Get predictions on test set
        y_pred_log = model(X_test_tensor)
        
        # Convert back from log space to original seconds
        y_pred = 2 ** np.array(y_pred_log)              
        y_test_original = 2 ** np.array(y_test_tensor)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred)
        mae = mean_absolute_error(y_test_original, y_pred)
        
        print(f'\nTest Results:')
        print(f'MSE: {mse:.2f}')
        print(f'MAE: {mae:.2f}')
        
        # Show sample predictions
        print(f'\nSample Predictions vs Actual:')
        for i in range(min(10, len(y_pred))):
            actual = y_test_original[i][0]
            predicted = y_pred[i][0]
            error_pct = abs(predicted - actual) / actual * 100
            print(f'Expected: {actual:.2f}s, Predicted: {predicted:.2f}s, Error: {error_pct:.1f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type = str, default=r"data\hackathon_public.json")
    parser.add_argument("--circuits", type = str, default="circuits")
    parser.add_argument("--id-map", type = str, default=r"data\holdout_public.json")
    parser.add_argument("--out", type = str, default="predictions.json")
    args = parser.parse_args()

    main(args)
                   