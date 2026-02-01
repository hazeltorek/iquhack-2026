import argparse
import json
import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from features import extract_features_from_file

def load_artifacts(artifacts_dir="artifacts"):
    artifacts_path = Path(artifacts_dir)
    
    model_file = artifacts_path / 'entanglement_model.pkl'
    scaler_file = artifacts_path / 'entanglement_scaler.pkl'
    metadata_file = artifacts_path / 'entanglement_metadata.json'
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}. Run train_entanglement_difficulty.py first.")
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

def extract_all_features(qasm_path, n_qubits=None):
    features = extract_features_from_file(qasm_path, n_qubits)
    
    # Compute entanglement_stress
    axis_a_volume = features.get('axis_a_volume', 0.0)
    axis_c_packing = features.get('axis_c_packing', 0.0)
    entanglement_stress = axis_a_volume * axis_c_packing
    features['entanglement_stress'] = entanglement_stress
    
    # Extras
    gate_density = features.get('gate_density', 0.0)
    n_edges = features.get('n_edges', 0.0)
    entanglement_variance = features.get('entanglement_variance', 0.0)
    
    features['gate_density_x_n_edges'] = gate_density * n_edges
    features['gate_density_div_entanglement_variance'] = gate_density / (entanglement_variance + 1e-10)
    
    return features

def predict_threshold(qasm_path, model, scaler, metadata, n_qubits=None):
    # Extract all
    features = extract_all_features(qasm_path, n_qubits)
    feature_cols = metadata['features']
    
    # Vector
    X = np.array([[features.get(f, 0.0) for f in feature_cols]])
    X_scaled = scaler.transform(X)
    
    predicted_threshold_continuous = model.predict(X_scaled)[0]
    
    # Binning 
    THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    safety_margin = metadata.get('safety_margin', 0.0)
    
    adjusted = predicted_threshold_continuous + safety_margin
    for threshold in THRESHOLD_LADDER:
        if adjusted <= threshold:
            predicted_threshold = threshold
            break
    else:
        predicted_threshold = THRESHOLD_LADDER[-1]
    
    return float(predicted_threshold_continuous), int(predicted_threshold)

def main():
    parser = argparse.ArgumentParser(description='Predict threshold difficulty for holdout circuits')
    parser.add_argument('--tasks', type=str, required=True, help='Path to holdout tasks JSON')
    parser.add_argument('--circuits', type=str, required=True, help='Directory containing QASM files')
    parser.add_argument('--id-map', type=str, required=True, help='JSON mapping task id to QASM filename')
    parser.add_argument('--out', type=str, required=True, help='Output path for predictions JSON')
    parser.add_argument('--artifacts', type=str, default='artifacts', help='Directory containing trained artifacts')
    
    args = parser.parse_args()
    
    # Load
    model, scaler, metadata = load_artifacts(args.artifacts)
    with open(args.tasks) as f:
        tasks = json.load(f)
    
    with open(args.id_map) as f:
        id_map_data = json.load(f)
    
    # id -> qasm_file
    id_to_file = {entry['id']: entry['qasm_file'] for entry in id_map_data.get('entries', [])}
    
    circuits_dir = Path(args.circuits)
    predictions = []
    
    for task in tasks.get('tasks', []):
        task_id = task['id']
        
        if task_id not in id_to_file:
            print(f"No QASM file mapping for task {task_id}")
            continue
        
        qasm_file = id_to_file[task_id]
        qasm_path = circuits_dir / qasm_file
        
        if not qasm_path.exists():
            print(f"QASM file not found: {qasm_path}")
            continue
        
        # Predict
        try:
            predicted_threshold_continuous, predicted_threshold = predict_threshold(
                qasm_path, model, scaler, metadata
            )
            
            predictions.append({
                'id': task_id,
                'predicted_threshold_min': predicted_threshold,
                'predicted_threshold_continuous': predicted_threshold_continuous,
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

if __name__ == "__main__":
    main()
