import argparse
import json, os, re
import networkx as nx
from typing import Dict

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
            if isinstance(fid, float) and fid > 0.75:
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
            nn.Linear(128, 64),
            nn.ReLU(),
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
            val["n"],
            val["threshold"],
            # # val["max_deg"],
            # # # val["file_len"], 
            # #val["lines"],
            # #val["threshold"],
            # # val["n_meas"],
            int(val["is_cpu"]),
            int(val["is_single"]),
            # # val["n_cx"],
            # # val["n_cz"],
            # #val["n_1q"],   
            val["depth_proxy"],
            # #val["circuit_depth"],
            # #val["n_barriers"]    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

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
        

        y_pred_log_clamped = torch.clamp(y_pred_log, min=-5, max=5)

        # Convert back from log space to original seconds
        y_pred = 2 ** np.array(y_pred_log_clamped)              
        y_test_original = 2 ** np.array(y_test_tensor)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred)
        mae = mean_absolute_error(y_test_original, y_pred)
        
        print(f'\nTest Results:')
        print(f'MSE: {mse:.2f}')
        print(f'MAE: {mae:.2f}')
        
        # Plot prediction errors
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Predicted vs Actual scatter plot
        ax1 = axes[0, 0]
        ax1.scatter(y_test_original, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
        min_val = min(y_test_original.min(), y_pred.min())
        max_val = max(y_test_original.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        ax1.set_xlabel('Actual Runtime (seconds)', fontsize=11)
        ax1.set_ylabel('Predicted Runtime (seconds)', fontsize=11)
        ax1.set_title('Predicted vs Actual Runtime', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error distribution histogram
        ax2 = axes[0, 1]
        errors = (y_pred - y_test_original).flatten()
        ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
        ax2.set_xlabel('Prediction Error (seconds)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Percentage error
        ax3 = axes[1, 0]
        percent_errors = np.abs((y_pred - y_test_original) / y_test_original * 100).flatten()
        ax3.scatter(range(len(percent_errors)), percent_errors, alpha=0.6, edgecolors='k', linewidths=0.5)
        ax3.axhline(y=percent_errors.mean(), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {percent_errors.mean():.1f}%')
        ax3.set_xlabel('Test Sample Index', fontsize=11)
        ax3.set_ylabel('Absolute Percentage Error (%)', fontsize=11)
        ax3.set_title('Percentage Error per Sample', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Error vs Actual value
        ax4 = axes[1, 1]
        ax4.scatter(y_test_original, errors, alpha=0.6, edgecolors='k', linewidths=0.5)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero error')
        ax4.set_xlabel('Actual Runtime (seconds)', fontsize=11)
        ax4.set_ylabel('Prediction Error (seconds)', fontsize=11)
        ax4.set_title('Error vs Actual Runtime', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f'\nError Statistics:')
        print(f'Mean Absolute Error: {mae:.2f} seconds')
        print(f'Mean Percentage Error: {percent_errors.mean():.2f}%')
        print(f'Median Percentage Error: {np.median(percent_errors):.2f}%')
        print(f'Max Percentage Error: {percent_errors.max():.2f}%')
        
        # Plot error vs threshold
        test_thresholds = []
        test_errors = []
        test_percent_errors = []
        
        for i, val in enumerate(flattened):
            if val["file_name"] in test_files:
                idx = len(test_thresholds)
                if idx < len(y_pred):
                    test_thresholds.append(val["threshold"])
                    test_errors.append(errors[idx])
                    test_percent_errors.append(percent_errors[idx])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Error vs Threshold (log scale)
        ax1 = axes[0]
        scatter = ax1.scatter(test_thresholds, test_errors, 
                             c=test_percent_errors, cmap='coolwarm', 
                             alpha=0.7, edgecolors='k', linewidths=0.5, s=60)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Prediction Error (seconds)', fontsize=12)
        ax1.set_title('Prediction Error vs Threshold', fontsize=13, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='% Error')
        
        # Percentage Error vs Threshold
        ax2 = axes[1]
        ax2.scatter(test_thresholds, test_percent_errors, 
                   alpha=0.7, edgecolors='k', linewidths=0.5, s=60)
        ax2.axhline(y=percent_errors.mean(), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {percent_errors.mean():.1f}%')
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel('Absolute Percentage Error (%)', fontsize=12)
        ax2.set_title('Percentage Error vs Threshold', fontsize=13, fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Show sample predictions
        print(f'\nSample Predictions vs Actual:')
        for i in range(len(y_pred)):
            actual = y_test_original[i][0]
            predicted = y_pred[i][0]
            error_pct = abs(predicted - actual) / actual * 100
            print(f'Expected: {actual:.2f}s, Predicted: {predicted:.2f}s, Error: {error_pct:.1f}%, File: {flattened[i]["file_name"]}')
    
    # Plot expected time vs threshold for a sample circuit
    print(f'\n\nPlotting Expected Time vs Threshold...')
    
    # Pick a representative circuit from the test set
    sample_circuit = None
    for val in flattened:
        if val["file_name"] in test_files:
            sample_circuit = val
            break
    
    if sample_circuit:
        # Generate threshold values where log2(threshold) ranges from 1 to 8
        # This means threshold ranges from 2^1=2 to 2^8=256
        threshold_range = np.logspace(1, 8, 100, base=2)
        
        predictions_by_config = {}
        colors_map = {
            (True, True): 'blue',    # CPU/Single
            (True, False): 'red',    # CPU/Double
            (False, True): 'green',  # GPU/Single
            (False, False): 'orange' # GPU/Double
        }
        labels_map = {
            (True, True): 'CPU/Single',
            (True, False): 'CPU/Double',
            (False, True): 'GPU/Single',
            (False, False): 'GPU/Double'
        }
        
        # Generate predictions for each configuration
        for is_cpu in [True, False]:
            for is_single in [True, False]:
                config = (is_cpu, is_single)
                predictions = []
                
                for threshold in threshold_range:
                    # Build feature vector (must match training features)
                    features = [
                        threshold,
                        int(is_cpu),
                        int(is_single),
                        sample_circuit["depth_proxy"]
                    ]
                    
                    # Scale features using the same scaler
                    features_scaled = scaler.transform([features])
                    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                    
                    # Predict
                    with torch.no_grad():
                        pred_log = model(features_tensor)
                        pred_seconds = 2 ** pred_log.item()
                        predictions.append(pred_seconds)
                
                predictions_by_config[config] = predictions
        
        # Plot
        plt.figure(figsize=(12, 8))
        for config, preds in predictions_by_config.items():
            plt.plot(threshold_range, preds, 
                    color=colors_map[config], 
                    label=labels_map[config],
                    linewidth=2, alpha=0.8)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Expected Runtime (seconds)', fontsize=12)
        plt.title(f'Expected Runtime vs Threshold\nCircuit: {sample_circuit["file_name"]} (depth_proxy={sample_circuit["depth_proxy"]})', 
                 fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f'Plotted predictions for circuit: {sample_circuit["file_name"]}')


    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
        'feature_size': feature_size,
        'epoch': epoch,
        'loss': loss.item(),
    }, 'artifacts\model_forward_wall_s.pth')
    print("Model checkpoint saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type = str, default=r"data\hackathon_public.json")
    parser.add_argument("--circuits", type = str, default="circuits")
    parser.add_argument("--id-map", type = str, default=r"data\holdout_public.json")
    parser.add_argument("--out", type = str, default="predictions.json")
    args = parser.parse_args()

    main(args)
                   