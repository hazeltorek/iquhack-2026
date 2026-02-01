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
from sklearn.preprocessing import StandardScaler


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