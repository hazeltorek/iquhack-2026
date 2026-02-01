import json
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from analysis_75 import load_data_075, create_interactions

plt.ion()

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

def test_model():
    model, scaler, metadata = load_artifacts()
    print(f"  Model type: {metadata['model_type']}")
    print(f"  Features: {metadata['features']}")
    print(f"  Fidelity target: {metadata['fidelity_target']}")
    print()
    
    df = load_data_075(Path("data/hackathon_public.json"), Path("circuits"))
    
    # Create entanglement_stress (if not already created by load_data_075)
    if 'entanglement_stress' not in df.columns:
        df['entanglement_stress'] = df['axis_a_volume'] * df['axis_c_packing']
    
    # Ensure interaction features exist (they should be created by load_data_075, but double-check)
    # The interaction features should already be in df from load_data_075, but let's verify
    required_interactions = ['gate_density_x_n_edges', 'gate_density_div_entanglement_variance']
    missing_interactions = [f for f in required_interactions if f not in df.columns]
    
    if missing_interactions:
        print(f"Warning: Missing interaction features: {missing_interactions}")
        print("Creating interaction features...")
        base_features = [c for c in df.columns if c not in ['min_threshold', 'forward_runtime', 'log_runtime', 'family', 'file_name', 'axis_ac', 'entanglement_stress']]
        interactions_df = create_interactions(df, base_features)
        df = pd.concat([df, interactions_df], axis=1)
    
    train_mask = ~df['file_name'].isin(TEST_CIRCUITS)
    test_mask = df['file_name'].isin(TEST_CIRCUITS)
    
    train_data = df[train_mask].copy()
    test_data = df[test_mask].copy()
    
    train_circuits = train_data['file_name'].nunique()
    test_circuits = test_data['file_name'].nunique()
    train_samples = len(train_data)
    test_samples_before = len(test_data)
    
    feature_cols = metadata['features']
    feature_cols = [f for f in feature_cols if f in test_data.columns]
    
    # Check for NaN values and handle them
    test_data_features = test_data[feature_cols + ['min_threshold']].copy()
    
    # Check which features have NaN
    nan_counts = test_data_features[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"Warning: NaN values found in features:")
        for feat, count in nan_counts.items():
            if count > 0:
                print(f"  {feat}: {count} NaN values")
        print()
        
        # Fill NaN with 0 (conservative approach - these are interaction features)
        print("Filling NaN values with 0...")
        test_data_features[feature_cols] = test_data_features[feature_cols].fillna(0.0)
    
    # Check which rows still have NaN (should be none after fillna)
    nan_mask = test_data_features[feature_cols].isna().any(axis=1)
    if nan_mask.sum() > 0:
        print(f"Warning: {nan_mask.sum()} test samples still have NaN values after filling")
        print("Dropping remaining rows with NaN values...")
        test_data_features = test_data_features[~nan_mask]
    
    # Extract features and targets after NaN handling
    X_test = test_data_features[feature_cols].values
    y_test = test_data_features['min_threshold'].values
    
    test_samples_after = len(test_data_features)
    
    # Check for any remaining NaN or inf values
    if np.isnan(X_test).any() or np.isinf(X_test).any():
        print("Error: Still have NaN or Inf values after cleaning")
        print(f"NaN count: {np.isnan(X_test).sum()}")
        print(f"Inf count: {np.isinf(X_test).sum()}")
        # Fill with 0 as fallback
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred_test = model.predict(X_test_scaled)
    
    # Compute correlation
    test_corr = np.corrcoef(y_pred_test, y_test)[0, 1]

    print(f"Test correlation: {test_corr:.3f}")
    print()
    
    # Additional metrics
    mae = np.mean(np.abs(y_pred_test - y_test))
    rmse = np.sqrt(np.mean((y_pred_test - y_test)**2))
    
    print("Additional metrics:")
    print(f"  MAE (Mean Absolute Error): {mae:.2f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.2f}")
    print()
    
    # Binning
    THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    safety_margin = metadata.get('safety_margin', 0.0)
    
    def map_to_ladder(predicted, safety_margin=0.0):
        adjusted = predicted + safety_margin
        for threshold in THRESHOLD_LADDER:
            if adjusted <= threshold:
                return threshold
        return THRESHOLD_LADDER[-1]
    
    y_pred_test_conservative = np.array([map_to_ladder(p, safety_margin) for p in y_pred_test])
    
    test_corr_cons = np.corrcoef(y_pred_test_conservative, y_test)[0, 1]
    test_under = (y_pred_test_conservative < y_test).sum()
    
    print(f"  Test correlation: {test_corr_cons:.3f}")
    print(f"  Underestimation: {test_under} / {len(y_test)} ({100*test_under/len(y_test):.1f}%)")
    print()
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use test_data_features (after NaN cleaning) for family info
    if 'family' in test_data_features.columns:
        families = sorted(test_data_features['family'].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(families)))
        family_colors = {fam: colors[i] for i, fam in enumerate(families)}
        
        for family in families:
            family_mask = test_data_features['family'] == family
            if family_mask.sum() > 0:
                ax.scatter(y_test[family_mask], y_pred_test[family_mask],
                          alpha=0.6, s=50, label=family, color=family_colors[family])
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax.scatter(y_test, y_pred_test, alpha=0.6, s=50)
    
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label=f'Perfect (corr={test_corr:.3f})')
    
    ax.set_xlabel('Actual min_threshold', fontsize=12)
    ax.set_ylabel('Predicted min_threshold', fontsize=12)
    ax.set_title(f'Model Test Results\nCorrelation: {test_corr:.3f} | Split: {train_circuits}/{test_circuits} circuits', 
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)
    
    try:
        input("\nEnter -> Close")
    except EOFError:
        print("\n")
    
    return test_corr, train_circuits, test_circuits

if __name__ == "__main__":
    test_model()
