import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
from analysis_75 import load_data_075

def train_entanglement_difficulty():
    # Recommended test split
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
    
    df = load_data_075(Path("data/hackathon_public.json"), Path("circuits"))
    df['entanglement_stress'] = df['axis_a_volume'] * df['axis_c_packing']

    train_mask = ~df['file_name'].isin(TEST_CIRCUITS)
    test_mask = df['file_name'].isin(TEST_CIRCUITS)
    
    train_data = df[train_mask].copy()
    test_data = df[test_mask].copy()
    
    feature_cols = [
        'entanglement_stress',
        'n_barriers',
        'n_cz',
        'gate_density_x_n_edges',
        'gate_density_div_entanglement_variance',
    ]
    
    # Filtering
    feature_cols = [f for f in feature_cols if f in train_data.columns]
    
    print(f"Features (all QASM-derived): {feature_cols}")
    print()
    
    X_train = train_data[feature_cols].values
    y_train = train_data['min_threshold'].values
    
    X_test = test_data[feature_cols].values
    y_test = test_data['min_threshold'].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Predictions/results
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_corr = np.corrcoef(y_pred_train, y_train)[0, 1]
    test_corr = np.corrcoef(y_pred_test, y_test)[0, 1]
    
    # Binning
    THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    train_errors = train_data['min_threshold'].values - y_pred_train
    underestimation_errors = train_errors[train_errors < 0]
    if len(underestimation_errors) > 0:
        safety_margin = np.percentile(np.abs(underestimation_errors), 75)
    else:
        safety_margin = 0.0
    
    def map_to_ladder(predicted, safety_margin=0.0):
        adjusted = predicted + safety_margin
        for threshold in THRESHOLD_LADDER:
            if adjusted <= threshold:
                return threshold
        return THRESHOLD_LADDER[-1]
    
    y_pred_test_conservative = np.array([map_to_ladder(p, safety_margin) for p in y_pred_test])
    
    test_corr_cons = np.corrcoef(y_pred_test_conservative, y_test)[0, 1]
    test_under = (y_pred_test_conservative < y_test).sum()
    
    output_path = Path("../artifacts")
    output_path.mkdir(exist_ok=True)
    
    model_name = 'entanglement_model'
    scaler_name = 'entanglement_scaler'
    metadata_name = 'entanglement_metadata'
    
    with open(output_path / f'{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(output_path / f'{scaler_name}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    metadata = {
        'model_type': 'ridge',
        'features': feature_cols,
        'train_correlation': float(train_corr),
        'test_correlation': float(test_corr),
        'test_correlation_conservative': float(test_corr_cons),
        'safety_margin': float(safety_margin),
        'test_underestimation_rate': float(test_under / len(y_test)),
        'fidelity_target': 0.75,
        'used_recommended_split': True,
    }
    
    with open(output_path / f'{metadata_name}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model, scaler, metadata

if __name__ == "__main__":
    train_entanglement_difficulty()
