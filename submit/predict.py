"""
Submission predict.py for iQuHACK 2026 Circuit Fingerprint Challenge
Combines threshold prediction (binning model) with runtime prediction (neural network)
"""
import argparse
import json
import pickle
import numpy as np
from pathlib import Path
import sys
import re

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from features import extract_features_from_file

THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]


# ============================================================================
# THRESHOLD PREDICTION (from prototype/predict_final_binning.py)
# ============================================================================

def load_threshold_artifacts(artifacts_dir):
    """Load trained threshold prediction model"""
    artifacts_path = Path(artifacts_dir)
    
    model_file = artifacts_path / 'entanglement_model.pkl'
    scaler_file = artifacts_path / 'entanglement_scaler.pkl'
    metadata_file = artifacts_path / 'entanglement_metadata.json'
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return model, scaler, metadata


def compute_aggressiveness(predicted_value, difficulty_features):
    """Compute how aggressive to be in binning down threshold"""
    if predicted_value < 2:
        base_aggressiveness = 0.1
    elif predicted_value < 4:
        base_aggressiveness = 0.2
    elif predicted_value < 8:
        base_aggressiveness = 0.4
    elif predicted_value < 16:
        base_aggressiveness = 0.7
    else:
        base_aggressiveness = 1.0
    
    entanglement_stress = difficulty_features.get('entanglement_stress', 0.0)
    axis_a_volume = difficulty_features.get('axis_a_volume', 0.0)
    axis_c_packing = difficulty_features.get('axis_c_packing', 0.0)
    circuit_depth = difficulty_features.get('circuit_depth', 0.0)
    n_qubits = difficulty_features.get('n_qubits', 0.0)
    
    stress_norm = np.clip(np.log1p(entanglement_stress + 1e-10) / 8.0, 0, 1)
    volume_norm = np.clip(np.log1p(axis_a_volume + 1e-10) / 6.0, 0, 1)
    packing_norm = np.clip(np.log1p(axis_c_packing + 1e-10) / 6.0, 0, 1)
    depth_norm = np.clip(np.log1p(circuit_depth + 1e-10) / 8.0, 0, 1)
    qubits_norm = np.clip(n_qubits / 130.0, 0, 1)
    
    feature_boost = max(stress_norm, volume_norm, packing_norm, depth_norm, qubits_norm)
    aggressiveness = base_aggressiveness + (feature_boost * 0.3)
    return np.clip(aggressiveness, 0, 1.5)


def aggressive_bin(predicted_value, difficulty_features, ladder=THRESHOLD_LADDER):
    """Bin predicted threshold to ladder with aggressive downward adjustment"""
    aggressiveness = compute_aggressiveness(predicted_value, difficulty_features)
    
    target_rung_idx = 0
    for i, rung in enumerate(ladder):
        if predicted_value <= rung:
            target_rung_idx = i
            break
    else:
        target_rung_idx = len(ladder) - 1
    
    if target_rung_idx < len(ladder) - 1:
        current_rung = ladder[target_rung_idx]
        next_rung = ladder[target_rung_idx + 1]
        gap = next_rung - current_rung
        
        shift_factor = aggressiveness * (1.0 + np.log1p(gap) / 8.0)
        adjusted_threshold = current_rung - (shift_factor * gap * 0.7)
        
        if predicted_value > adjusted_threshold:
            return next_rung
        else:
            if aggressiveness > 0.8 and (predicted_value > current_rung * 0.7):
                return next_rung
            return current_rung
    else:
        return ladder[-1]


def predict_threshold(qasm_path, model, scaler, metadata, n_qubits=None):
    """Predict minimum threshold for a circuit"""
    features = extract_features_from_file(qasm_path, n_qubits)
    
    # Compute entanglement_stress
    axis_a_volume = features.get('axis_a_volume', 0.0)
    axis_c_packing = features.get('axis_c_packing', 0.0)
    entanglement_stress = axis_a_volume * axis_c_packing
    features['entanglement_stress'] = entanglement_stress
    
    # Get feature columns
    feature_cols = metadata['features']
    X = np.array([[features.get(f, 0.0) for f in feature_cols]])
    X_scaled = scaler.transform(X)
    
    predicted_threshold_continuous = model.predict(X_scaled)[0]
    
    # Get difficulty features for binning
    difficulty_feature_names = metadata.get('difficulty_features', 
        ['entanglement_stress', 'axis_a_volume', 'axis_c_packing', 'circuit_depth', 'n_qubits'])
    difficulty_features = {f: features.get(f, 0.0) for f in difficulty_feature_names}
    
    # Apply aggressive binning
    predicted_threshold = aggressive_bin(predicted_threshold_continuous, difficulty_features)
    
    return int(predicted_threshold)


# ============================================================================
# RUNTIME PREDICTION (from TimeReg.py)
# ============================================================================

class NonLinearModel(nn.Module):
    """Neural network for runtime prediction"""
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


def get_pairs(qasm):
    """Extract pairs of qubits that interact via 2-qubit gates"""
    return sorted(list(set([tuple(sorted(list(map(int, p)))) for p in re.findall(
        r"[a-z]+(?:\((?:-{0,1}(?:\d+\.{0,1}\d+|pi(?:\/\d+){0,1}),{0,1}){1,3}\)){0,1}\sq\[(\d+)\],\s*q\[(\d+)\];", 
        qasm)])))


def extract_runtime_features(qasm_path, is_cpu=True, is_single=True, threshold=1):
    """Extract features for runtime prediction"""
    with open(qasm_path, 'r') as f:
        qasm_text = f.read()
    
    # Extract n_qubits from QASM
    qreg_match = re.search(r"qreg\s+q\[(\d+)\]", qasm_text)
    n = int(qreg_match.group(1)) if qreg_match else 0
    
    # Gate counts
    n_meas = len(re.findall(r"\bmeasure\b", qasm_text))
    n_cx = len(re.findall(r"\bcx\b", qasm_text))
    n_cz = len(re.findall(r"\bcz\b", qasm_text))
    n_1q = len(re.findall(r"\b(h|x|y|z|s|sdg|t|tdg|rx|ry|rz|u1|u2|u3)\b", qasm_text))
    
    # Depth proxy
    lines = [ln.strip() for ln in qasm_text.splitlines() 
             if ln.strip() and not ln.strip().startswith("//")]
    gate_lines = [ln for ln in lines 
                  if not ln.startswith("measure") and not ln.startswith("barrier")]
    depth_proxy = len(gate_lines)
    
    # Build feature vector (12 features matching TimeReg.py)
    features = [
        n,
        threshold,
        n_meas,
        int(is_cpu),
        int(is_single),
        depth_proxy,
        n_cx + n_cz,  # Total 2-qubit gates
        n_1q,
        n * threshold,  # n × threshold
        (n ** 2) * threshold,  # n² × threshold
        n * depth_proxy,  # n × depth
        depth_proxy * threshold,  # depth × threshold
    ]
    
    return features


def load_runtime_model(artifacts_dir):
    """Load trained runtime prediction model"""
    model_path = Path(artifacts_dir) / 'model_weights.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Runtime model not found: {model_path}")
    
    # Model expects 12 features
    model = NonLinearModel(input_size=12)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model


def predict_runtime(qasm_path, model, is_cpu=True, is_single=True, threshold=1):
    """Predict runtime for a circuit"""
    # Extract features
    features = extract_runtime_features(qasm_path, is_cpu, is_single, threshold)
    
    # Convert to tensor
    X = torch.tensor([features], dtype=torch.float32)
    
    # Predict (model outputs log2(seconds))
    with torch.no_grad():
        y_pred_log = model(X)
        
    # Clamp log value to prevent overflow (2^9 ≈ 512s, well above our 400s cap)
    y_pred_log_clamped = np.clip(y_pred_log.numpy()[0][0], -10, 9)
    
    # Convert from log space to seconds
    y_pred = 2 ** y_pred_log_clamped
    
    # Cap at 400 seconds
    y_pred = min(y_pred, 400.0)
    
    # Ensure non-negative
    y_pred = max(y_pred, 0.0)
    
    return float(y_pred)


# ============================================================================
# MAIN PREDICTION PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Predict threshold and runtime for holdout circuits')
    parser.add_argument('--tasks', type=str, required=True, help='Path to holdout tasks JSON')
    parser.add_argument('--circuits', type=str, required=True, help='Directory containing QASM files')
    parser.add_argument('--id-map', type=str, required=True, help='JSON mapping task id to QASM filename')
    parser.add_argument('--out', type=str, required=True, help='Output path for predictions JSON')
    parser.add_argument('--artifacts', type=str, default='artifacts', help='Directory containing trained artifacts')
    
    args = parser.parse_args()
    
    # Load models
    print("Loading threshold prediction model...")
    threshold_model, threshold_scaler, threshold_metadata = load_threshold_artifacts(args.artifacts)
    
    print("Loading runtime prediction model...")
    runtime_model = load_runtime_model(args.artifacts)
    
    # Load tasks and id map
    with open(args.tasks) as f:
        tasks_data = json.load(f)
        tasks = tasks_data.get('tasks', tasks_data)  # Handle both formats
    
    with open(args.id_map) as f:
        id_map_data = json.load(f)
    
    id_to_file = {entry['id']: entry['qasm_file'] for entry in id_map_data.get('entries', [])}
    
    circuits_dir = Path(args.circuits)
    predictions = []
    
    print(f"\nProcessing {len(tasks)} tasks...")
    
    for task in tasks:
        task_id = task['id']
        processor = task.get('processor', 'CPU')
        precision = task.get('precision', 'single')
        
        is_cpu = (processor == 'CPU')
        is_single = (precision == 'single')
        
        if task_id not in id_to_file:
            print(f"Warning: No QASM file mapping for task {task_id}")
            continue
        
        qasm_file = id_to_file[task_id]
        qasm_path = circuits_dir / qasm_file
        
        if not qasm_path.exists():
            print(f"Warning: QASM file not found: {qasm_path}")
            continue
        
        try:
            # Predict minimum threshold
            predicted_threshold = predict_threshold(
                qasm_path, threshold_model, threshold_scaler, threshold_metadata
            )
            
            # Predict runtime using predicted threshold
            predicted_runtime = predict_runtime(
                qasm_path, runtime_model, is_cpu, is_single, predicted_threshold
            )
            
            predictions.append({
                'id': task_id,
                'predicted_threshold_min': predicted_threshold,
                'predicted_forward_wall_s': predicted_runtime,
            })
            
            print(f"  {task_id}: threshold={predicted_threshold}, runtime={predicted_runtime:.2f}s")
            
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Write output
    output = {
        'schema': 'iqhack_predictions_v1',
        'predictions': predictions
    }
    
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[SUCCESS] Generated {len(predictions)} predictions")
    print(f"[SUCCESS] Written to: {args.out}")


if __name__ == "__main__":
    main()
