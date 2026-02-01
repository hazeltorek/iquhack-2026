import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from analysis_75 import load_data_075

THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]

TEST_CIRCUITS = {
    'ae_indep_qiskit_130.qasm',
    'dj_indep_qiskit_30.qasm',
    'ghz_indep_qiskit_30.qasm',
    'ghz_indep_qiskit_130.qasm',
    'grover-noancilla_indep_qiskit_11.qasm',
    'grover-v-chain_indep_qiskit_17.qasm',
    'portfolioqaoa_indep_qiskit_17.qasm',
    'portfoliovqe_indep_qiskit_18.qasm',
    'qft_indep_qiskit_15.qasm',
    'qftentangled_indep_qiskit_30.qasm',
    'qpeexact_indep_qiskit_30.qasm',
    'wstate_indep_qiskit_130.qasm',
}

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

def train_final_binning():
    
    df = load_data_075(Path("../data/hackathon_public.json"), Path("../circuits"))
    df['entanglement_stress'] = df['axis_a_volume'] * df['axis_c_packing']
    
    train_mask = ~df['file_name'].isin(TEST_CIRCUITS)
    test_mask = df['file_name'].isin(TEST_CIRCUITS)
    
    train_data = df[train_mask].copy()
    test_data = df[test_mask].copy()
    
    from analysis_75 import get_correlations
    corr_thresh, _ = get_correlations(df)
    top_features = corr_thresh.head(20).index.tolist()
    
    feature_cols = [f for f in top_features if f in train_data.columns]
    
    print(f"Using {len(feature_cols)} features")
    print()
    
    X_train = train_data[feature_cols].fillna(0.0).values
    y_train = train_data['min_threshold'].values
    
    X_test = test_data[feature_cols].fillna(0.0).values
    y_test = test_data['min_threshold'].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=3, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    difficulty_feature_names = ['entanglement_stress', 'axis_a_volume', 'axis_c_packing', 
                                'circuit_depth', 'n_qubits']
    available_difficulty_features = [f for f in difficulty_feature_names if f in train_data.columns]
    
    y_pred_train_binned = []
    y_pred_test_binned = []
    
    for i in range(len(train_data)):
        difficulty_features = {f: train_data.iloc[i][f] for f in available_difficulty_features}
        binned = aggressive_bin(y_pred_train[i], difficulty_features)
        y_pred_train_binned.append(binned)
    
    for i in range(len(test_data)):
        difficulty_features = {f: test_data.iloc[i][f] for f in available_difficulty_features}
        binned = aggressive_bin(y_pred_test[i], difficulty_features)
        y_pred_test_binned.append(binned)
    
    y_pred_train_binned = np.array(y_pred_train_binned)
    y_pred_test_binned = np.array(y_pred_test_binned)
    
    train_corr = np.corrcoef(y_pred_train_binned, y_train)[0, 1]
    test_corr = np.corrcoef(y_pred_test_binned, y_test)[0, 1]
    train_under = (y_pred_train_binned < y_train).sum()
    test_under = (y_pred_test_binned < y_test).sum()
    
    print(f"Train correlation: {train_corr:.3f}, Underestimation: {train_under}/{len(y_train)} ({100*train_under/len(y_train):.1f}%)")
    print(f"Test correlation:  {test_corr:.3f}, Underestimation: {test_under}/{len(y_test)} ({100*test_under/len(y_test):.1f}%)")
    print()
    
    # Save artifacts (use root artifacts directory)
    output_path = Path("../artifacts")
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / 'entanglement_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(output_path / 'entanglement_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    metadata = {
        'model_type': 'random_forest_aggressive_binning',
        'features': feature_cols,
        'difficulty_features': available_difficulty_features,
        'train_correlation': float(train_corr),
        'test_correlation': float(test_corr),
        'train_underestimation_rate': float(train_under / len(y_train)),
        'test_underestimation_rate': float(test_under / len(y_test)),
        'fidelity_target': 0.75,
        'used_recommended_split': True,
        'binning_method': 'feature_based_aggressive',
    }
    
    with open(output_path / 'entanglement_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved artifacts to {output_path}/")
    print("  - entanglement_model.pkl")
    print("  - entanglement_scaler.pkl")
    print("  - entanglement_metadata.json")
    print()
    
    return model, scaler, feature_cols, available_difficulty_features, metadata

if __name__ == "__main__":
    train_final_binning()
