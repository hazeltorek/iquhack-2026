# Simple test to see QASM

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import networkx as nx


# Stolen
THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
THRESHOLD_TO_INDEX = {t: i for i, t in enumerate(THRESHOLD_LADDER)}


def extract_basic_features(qasm_text: str) -> Dict[str, float]:
    # Reg my beloved
    n_meas = len(re.findall(r"\bmeasure\b", qasm_text))
    n_cx = len(re.findall(r"\bcx\b", qasm_text))
    n_cz = len(re.findall(r"\bcz\b", qasm_text))
    
    # 1-qubit gates
    n_1q = len(re.findall(
        r"\b(h|x|y|z|s|sdg|t|tdg|rx|ry|rz|u1|u2|u3)\b", 
        qasm_text
    ))
    
    # File stats
    lines = qasm_text.splitlines()
    n_lines = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("//"))
    n_total_gates = n_cx + n_cz + n_1q + n_meas
    
    return {
        "n_meas": float(n_meas),
        "n_cx": float(n_cx),
        "n_cz": float(n_cz),
        "n_1q": float(n_1q),
        "n_total_gates": float(n_total_gates),
        "n_lines": float(n_lines),
        "n_2q_gates": float(n_cx + n_cz),
    }


def extract_qubit_pairs(qasm_text: str) -> List[Tuple[int, int]]:
    # 2-qubit gates:
    pattern = r"[a-z]+(?:\((?:-{0,1}(?:\d+\.{0,1}\d+|pi(?:\/\d+){0,1}),{0,1}){1,3}\)){0,1}\sq\[(\d+)\],\s*q\[(\d+)\];"
    matches = re.findall(pattern, qasm_text)
    
    # Tuples 
    pairs = set()
    for q1_str, q2_str in matches:
        q1, q2 = int(q1_str), int(q2_str)
        pairs.add((min(q1, q2), max(q1, q2)))
    
    return sorted(list(pairs))


def extract_graph_features(qasm_text: str) -> Dict[str, float]:
    pairs = extract_qubit_pairs(qasm_text)
    
    if not pairs:
        return {
            "n_edges": 0.0,
            "max_degree": 0.0,
            "avg_degree": 0.0,
            "n_connected_components": 0.0,
            "max_centrality": 0.0,
            "avg_qubit_distance": 0.0,
            "entanglement_variance": 0.0,
        }
    
    # Build
    G = nx.Graph(pairs)
    
    # Graphics
    n_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    max_degree = max(degrees) if degrees else 0.0
    avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
    n_components = nx.number_connected_components(G)
    
    # Centrality
    centrality = nx.degree_centrality(G)
    max_centrality = max(centrality.values()) if centrality else 0.0
    
    # Average qubit distance in 2-qubit gates
    distances = [abs(q1 - q2) for q1, q2 in pairs]
    avg_distance = sum(distances) / len(distances) if distances else 0.0
    
    # Entanglement Variance QASMBench metri thingy
    if len(degrees) > 1:
        degree_variance = sum((d - avg_degree) ** 2 for d in degrees) / len(degrees)
    else:
        degree_variance = 0.0
    
    return {
        "n_edges": float(n_edges),
        "max_degree": float(max_degree),
        "avg_degree": float(avg_degree),
        "n_connected_components": float(n_components),
        "max_centrality": float(max_centrality),
        "avg_qubit_distance": float(avg_distance),
        "entanglement_variance": float(degree_variance),
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


def extract_all_features(qasm_text: str, n_qubits: Optional[int] = None) -> Dict[str, float]:
    features = {}
    
    # Features
    features.update(extract_basic_features(qasm_text))
    
    features.update(extract_graph_features(qasm_text))
    
    # Add n_qubits if provided
    if n_qubits is not None:
        features["n_qubits"] = float(n_qubits)
    else:
        # Try to extract from QASM
        match = re.search(r"qreg\s+q\[(\d+)\];", qasm_text)
        if match:
            features["n_qubits"] = float(match.group(1))
        else:
            features["n_qubits"] = 0.0
    
    if features["n_qubits"] > 0:
        depth_features = extract_circuit_depth(qasm_text, int(features["n_qubits"]))
        features.update(depth_features)
    else:
        features.update(extract_depth_proxy(qasm_text))
        features["circuit_depth"] = 0.0
    
    if features["n_qubits"] > 0:
        features["gates_per_qubit"] = features["n_total_gates"] / features["n_qubits"]
        features["density"] = features["n_edges"] / (features["n_qubits"] * (features["n_qubits"] - 1) / 2) if features["n_qubits"] > 1 else 0.0
        
        # Occupancy of gate slots
        # Gate density = total_gates / (n_qubits * circuit_depth)
        if features.get("circuit_depth", 0) > 0:
            features["gate_density"] = features["n_total_gates"] / (features["n_qubits"] * features["circuit_depth"])
        else:
            features["gate_density"] = 0.0
        
        # Measurement Density QASMBench metric
        if features["n_meas"] > 0 and features.get("circuit_depth", 0) > 0:
            features["measurement_density"] = features["n_meas"] / (features["n_qubits"] * features["circuit_depth"])
        else:
            features["measurement_density"] = 0.0
    else:
        features["gates_per_qubit"] = 0.0
        features["density"] = 0.0
        features["gate_density"] = 0.0
        features["measurement_density"] = 0.0
    
    return features


def extract_features_from_file(qasm_path: Path, n_qubits: Optional[int] = None) -> Dict[str, float]:
    qasm_text = qasm_path.read_text(encoding="utf-8")
    return extract_all_features(qasm_text, n_qubits)


def get_feature_names() -> List[str]:
    # Get keys
    dummy_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[5];\nh q[0];\ncx q[0],q[1];"
    features = extract_all_features(dummy_qasm, n_qubits=5)
    return sorted(features.keys())


def encode_context(backend: str, precision: str) -> Dict[str, float]:
    return {
        "is_cpu": 1.0 if backend == "CPU" else 0.0,
        "is_gpu": 1.0 if backend == "GPU" else 0.0,
        "is_single": 1.0 if precision == "single" else 0.0,
        "is_double": 1.0 if precision == "double" else 0.0,
    }


def encode_family(family: str, all_families: Optional[List[str]] = None) -> Dict[str, float]:
    if all_families is None:
        # This shit suks
        all_families = [
            "GHZ", "QFT", "Deutsch_Jozsa", "Grover", "QAOA", "VQE",
            "Amplitude_Estimation", "QPE_Exact", "QNN", "W-State",
            "TwoLocalRandom", "GraphState", "CutBell", "Ground_State",
            "PortfolioQAOA", "PortfolioVQE", "PricingCall", "Shor"
        ]
    
    encoding = {}
    for fam in all_families:
        encoding[f"family_{fam}"] = 1.0 if family == fam else 0.0
    
    return encoding

