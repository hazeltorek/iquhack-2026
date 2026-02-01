import argparse
import json, os, re
import networkx as nx
from typing import Dict

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

from sklearn.linear_model import LinearRegression

# pairs of qubits that interact via 2-qubit gates
def get_pairs(qasm):
    # i saw the face of god in a regular expression
    return sorted(list(set([tuple(sorted(list(map(int, p)))) for p in re.findall(r"[a-z]+(?:\((?:-{0,1}(?:\d+\.{0,1}\d+|pi(?:\/\d+){0,1}),{0,1}){1,3}\)){0,1}\sq\[(\d+)\],\s*q\[(\d+)\];", qasm)])))

# organized data!
def process_circuit(name):
    a = [circ for circ in circuits if circ["file"] == name][0]
    b = [res for res in results if res["file"] == name]
    qasm = [c for (f, c) in code if f == name][0]
    qasm_text = "\n".join(qasm)

    depth_proxy_dict = extract_depth_proxy(qasm_text)
    circuit_depth_dict = extract_circuit_depth(qasm_text, a["n_qubits"])

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

    # fuck shit mcballs
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
    if len(degrees) > 1:
        degree_variance = sum((d - avg_degree) ** 2 for d in degrees) / len(degrees)
    else:
        degree_variance = 0.0

    # organize results by the given predictors (cpu/gpu, single/double)
    pred = {(r["backend"] == "CPU", r["precision"] == "single"): b for r in b}
    return {
        "family": a["family"], 
        "n": a["n_qubits"], 
        "file_len": length, 
        "results": pred,
        "just_past": just_past,
        "max_deg": max([0] + list(map(lambda d: d[1], G.degree()))),
        "ent_var": degree_variance,
        "depth_proxy": depth_proxy_dict["depth_proxy"],  # Extract the value
        "circuit_depth": circuit_depth_dict["circuit_depth"],  # Extract the value
        "n_barriers": circuit_depth_dict["n_barriers"]
    }

def extract_circuit_depth(qasm_text: str, n_qubits: int) -> Dict[str, float]:
    """Extract true circuit depth using time-evolution analysis (QASMBench method)."""
    # Parse gates
    lines = [ln.strip() for ln in qasm_text.splitlines() 
             if ln.strip() and not ln.strip().startswith("//") 
             and not ln.strip().startswith("OPENQASM")
             and not ln.strip().startswith("include")
             and not ln.strip().startswith("qreg")
             and not ln.strip().startswith("creg")]
    
    qubit_timeline = {q: [] for q in range(n_qubits)}
    
    qubit_pattern = re.compile(r"q\[(\d+)\]")
    
    current_time = 0
    active_qubits = set()
    
    for line in lines:
        if line.startswith("barrier"):
            # Barrier forces a new time step
            if active_qubits:
                current_time += 1
                active_qubits = set()
            continue
        
        if line.startswith("measure"):
            # Measurements happen at the end, don't affect depth
            continue
        
        # Extract qubits involved in this gate
        qubits_in_gate = [int(m) for m in qubit_pattern.findall(line)]
        
        if qubits_in_gate:
            if active_qubits.isdisjoint(set(qubits_in_gate)):
                active_qubits.update(qubits_in_gate)
            else:
                current_time += 1
                active_qubits = set(qubits_in_gate)
    
    # Final time step
    if active_qubits:
        current_time += 1
    
    circuit_depth = current_time
    
    n_barriers = len(re.findall(r"\bbarrier\b", qasm_text))
    
    return {
        "circuit_depth": float(circuit_depth),
        "n_barriers": float(n_barriers),
        "depth_proxy": float(len([ln for ln in lines if not ln.startswith("measure")])),
    }


def extract_depth_proxy(qasm_text: str) -> Dict[str, float]:
    """Extract crude depth proxy by counting barriers or estimating layers."""
    # Count barriers (often used to separate layers)
    n_barriers = len(re.findall(r"\bbarrier\b", qasm_text))
    
    # Simple heuristic this is very approximate but can help
    lines = [ln.strip() for ln in qasm_text.splitlines() 
             if ln.strip() and not ln.strip().startswith("//")]
    
    #Big brother recommended
    gate_lines = [ln for ln in lines 
                  if not ln.startswith("measure") and not ln.startswith("barrier")]
    depth_proxy = len(gate_lines)
    
    return {
        "n_barriers": float(n_barriers),
        "depth_proxy": float(depth_proxy),
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
                "max_deg": entry["max_deg"],
                "ent_var": entry["ent_var"],
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
                "n_1q": n_1q,
                "depth_proxy": entry["depth_proxy"],
                "circuit_depth": entry["circuit_depth"],
                "n_barriers": entry["n_barriers"] 
                
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
            # val["max_deg"],
            # # val["file_len"], 
            # val["lines"],
            val["threshold"],
            # val["n_meas"],
            int(val["is_cpu"]),
            int(val["is_single"]),
            # val["n_cx"],
            # val["n_cz"],
            # val["n_1q"],   
            # val["ent_var"],
            val["depth_proxy"],
            #val["circuit_depth"],
            #val["n_barriers"]    
        ]
 
        X.append(features)
        y.append(val["seconds"])


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_log = np.log2(y)

    feature_size = len(X[0])

    
    # Define test files 
    test_files = {
        'ae_indep_qiskit_130', 'dj_indep_qiskit_30', 'ghz_indep_qiskit_30', 
        'ghz_indep_qiskit_130', 'grover-noancilla_indep_qiskit_11', 
        'grover-v-chain_indep_qiskit_17', 'portfolioqaoa_indep_qiskit_17',
        'portfoliovqe_indep_qiskit_18', 'qft_indep_qiskit_15', 
        'qftentangled_indep_qiskit_30', 'qpeexact_indep_qiskit_30', 
        'wstate_indep_qiskit_130'
    }

    # Split data based on file names
    X_train, X_test, y_train, y_test = [], [], [], []

    for i, val in enumerate(flattened):
        if val["file_name"] in test_files:
            X_test.append(X_scaled[i])
            y_test.append(y_log[i])
        else:
            X_train.append(X_scaled[i])
            y_train.append(y_log[i])

    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    model = NonLinearModel(feature_size)
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # model = LinearRegression().fit(X_train, y_train) 
    # y_pred = model.predict(X_test)

    # # Convert back from log space to original seconds
    # actual_original = 2 ** y_test 
    # predicted_original = 2 ** y_pred    

    # for i in range(len(actual_original)):
    #     actual = actual_original[i]
    #     predicted = predicted_original[i]
    #     error_pct = abs(predicted - actual) / actual * 100
    #     print(f'Expected: {actual:.2f}s, Predicted: {predicted:.2f}s, Error: {error_pct:.1f}%')
        

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
        for i in range(len(y_pred)):
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
                   