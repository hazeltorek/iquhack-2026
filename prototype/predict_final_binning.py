"""
Prediction script for final aggressive binning model
"""
import argparse
import json
import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from features import extract_features_from_file

THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def load_artifacts(artifacts_dir="artifacts"):
    artifacts_path = Path(artifacts_dir)
    
    model_file = artifacts_path / 'entanglement_model.pkl'
    scaler_file = artifacts_path / 'entanglement_scaler.pkl'
    metadata_file = artifacts_path / 'entanglement_metadata.json'
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}. Run train_final_binning.py first.")
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
    
    return float(predicted_threshold_continuous), int(predicted_threshold)

def main():
    parser = argparse.ArgumentParser(description='Predict threshold with aggressive binning')
    parser.add_argument('--tasks', type=str, required=True, help='Path to holdout tasks JSON')
    parser.add_argument('--circuits', type=str, required=True, help='Directory containing QASM files')
    parser.add_argument('--id-map', type=str, required=True, help='JSON mapping task id to QASM filename')
    parser.add_argument('--out', type=str, required=True, help='Output path for predictions JSON')
    parser.add_argument('--artifacts', type=str, default='artifacts', help='Directory containing trained artifacts')
    
    args = parser.parse_args()
    
    # Load artifacts
    model, scaler, metadata = load_artifacts(args.artifacts)
    
    # Load tasks and id map
    with open(args.tasks) as f:
        tasks = json.load(f)
    
    with open(args.id_map) as f:
        id_map_data = json.load(f)
    
    id_to_file = {entry['id']: entry['qasm_file'] for entry in id_map_data.get('entries', [])}
    
    circuits_dir = Path(args.circuits)
    predictions = []
    
    for task in tasks.get('tasks', []):
        task_id = task['id']
        
        if task_id not in id_to_file:
            print(f"Warning: No QASM file mapping for task {task_id}")
            continue
        
        qasm_file = id_to_file[task_id]
        qasm_path = circuits_dir / qasm_file
        
        if not qasm_path.exists():
            print(f"Warning: QASM file not found: {qasm_path}")
            continue
        
        try:
            predicted_threshold_continuous, predicted_threshold = predict_threshold(
                qasm_path, model, scaler, metadata
            )
            
            predictions.append({
                'id': task_id,
                'predicted_threshold_min': predicted_threshold,
            })
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            continue
    
    output = {
        'schema': 'iqhack_predictions_v1',
        'predictions': predictions
    }
    
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Written to: {args.out}")

if __name__ == "__main__":
    main()
