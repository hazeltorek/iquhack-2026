import json
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.polynomial import Polynomial
import sys
from features import extract_features_from_file, encode_context

def extract_polynomial_features(threshold_sweep, degree=5):
    thresholds = []
    fidelities = []
    for entry in threshold_sweep:
        t = entry.get("threshold")
        f = entry.get("sdk_get_fidelity")
        if t is not None and f is not None and isinstance(f, (int, float)) and not np.isnan(f):
            thresholds.append(t)
            fidelities.append(f)
    
    # if len(thresholds) < 3:
    #     return {f"poly_coef_{i}": 0.0 for i in range(degree + 1)} | {
    #         "poly_slope_start": 0.0, "poly_slope_mid": 0.0, "poly_slope_end": 0.0,
    #         "poly_curvature": 0.0, "poly_intercept": 0.0, "poly_max_fidelity": 0.0,
    #         "poly_min_fidelity": 0.0, "poly_fidelity_range": 0.0, "poly_cross_99": 0.0
    #     }

    # failsafe
    if len(thresholds) < 3:
        return _empty_poly_features(degree)

    thresholds = np.array(thresholds)
    fidelities = np.array(fidelities)
    log_thresholds = np.log1p(thresholds)
    
    try:
        poly = Polynomial.fit(log_thresholds, fidelities, degree)
        coefs = poly.convert().coef
        
        features = {}
        
        # Polynomial coe recurse
        for i in range(min(len(coefs), degree + 1)):
            features[f"poly_coef_{i}"] = float(coefs[i])
        for i in range(len(coefs), degree + 1):
            features[f"poly_coef_{i}"] = 0.0
        
        # Not really necessary
        min_thresh = min(thresholds)
        max_thresh = max(thresholds)
        mid_thresh = (min_thresh + max_thresh) / 2
        log_min = np.log1p(min_thresh)
        log_max = np.log1p(max_thresh)
        log_mid = np.log1p(mid_thresh)
        
        features["poly_slope_start"] = float(poly.deriv(1)(log_min))
        features["poly_slope_mid"] = float(poly.deriv(1)(log_mid))
        features["poly_slope_end"] = float(poly.deriv(1)(log_max))
        features["poly_curvature"] = float(poly.deriv(2)(log_mid))
        features["poly_intercept"] = float(poly(log_min))
        
        # Plotter vals
        features["poly_max_fidelity"] = float(np.max(fidelities))
        features["poly_min_fidelity"] = float(np.min(fidelities))
        features["poly_fidelity_range"] = float(np.max(fidelities) - np.min(fidelities))
        
        # IMPORTANT <-- the actual fidelity thresh params
        cross_99_threshold = None
        for t_val in np.linspace(log_min, log_max, 1000):
            if poly(t_val) >= 0.99:
                cross_99_threshold = np.expm1(t_val)
                break
        features["poly_cross_99"] = float(cross_99_threshold) if cross_99_threshold else float(max_thresh)
        
        return features
    except:
        return _empty_poly_features(degree)

def _empty_poly_features(degree):
    return {f"poly_coef_{i}": 0.0 for i in range(degree + 1)} | {
        "poly_slope_start": 0.0, "poly_slope_mid": 0.0, "poly_slope_end": 0.0,
        "poly_curvature": 0.0, "poly_intercept": 0.0, "poly_max_fidelity": 0.0,
        "poly_min_fidelity": 0.0, "poly_fidelity_range": 0.0, "poly_cross_99": 0.0
    }

def create_interactions(df, base_features):
    interactions = {}
    important = ['n_qubits', 'n_2q_gates', 'circuit_depth', 'gate_density', 
                 'entanglement_variance', 'n_edges', 'max_degree', 'gates_per_qubit']
    important = [f for f in important if f in base_features]
    
    for i, f1 in enumerate(important):
        for f2 in important[i+1:]:
            if f1 in df.columns and f2 in df.columns:
                interactions[f"{f1}_x_{f2}"] = df[f1] * df[f2]
                interactions[f"{f1}_div_{f2}"] = df[f1] / (df[f2] + 1e-10)
    
    return pd.DataFrame(interactions)

def load_data(data_path, circuits_dir):
    with open(data_path) as f:
        data = json.load(f)
    circuits = {c["file"]: c for c in data["circuits"]}
    results = data["results"]
    
    rows = []
    for result in results:
        if result["status"] != "ok":
            continue
        
        file_name = result["file"]
        backend = result["backend"]
        precision = result["precision"]
        circuit_info = circuits.get(file_name)
        if not circuit_info:
            continue
        
        qasm_path = circuits_dir / file_name
        if not qasm_path.exists():
            continue
        
        # Pull Circuit structure features
        try:
            features = extract_features_from_file(qasm_path, n_qubits=circuit_info.get("n_qubits"))
        except:
            continue
      
        # Feel free to remove
        context = encode_context(backend, precision)
        features.update(context)
        
        # Extract polynomial features from threshold_sweep
        threshold_sweep = result.get("threshold_sweep", [])
        poly_features = extract_polynomial_features(threshold_sweep, degree=5)
        features.update(poly_features)
        
        # Extract target variables
        min_threshold = None
        for sweep_entry in sorted(threshold_sweep, key=lambda x: x.get("threshold", 999)):
            threshold = sweep_entry.get("threshold")
            if threshold is None:
                continue
            fidelity = sweep_entry.get("sdk_get_fidelity")
            if fidelity is not None and isinstance(fidelity, (int, float)) and fidelity >= 0.99:
                min_threshold = threshold
                break
        if min_threshold is None:
            thresholds = [s.get("threshold") for s in threshold_sweep if s.get("threshold") is not None]
            if thresholds:
                min_threshold = max(thresholds)
            else:
                continue
        
        forward = result.get("forward")
        if forward is None:
            continue
        forward_runtime = forward.get("run_wall_s")
        if forward_runtime is None or not isinstance(forward_runtime, (int, float)):
            continue
        
        features["min_threshold"] = min_threshold
        features["forward_runtime"] = forward_runtime
        features["log_runtime"] = np.log1p(forward_runtime)
        features["family"] = circuit_info.get("family", "Unknown")
        features["file_name"] = file_name
        
        rows.append(features)
    
    df = pd.DataFrame(rows)
    
    # Create interaction features
    base_features = [c for c in df.columns if c not in ['min_threshold', 'forward_runtime', 'log_runtime', 'family', 'file_name']]
    interactions_df = create_interactions(df, base_features)
    df = pd.concat([df, interactions_df], axis=1)
    
    return df

def get_correlations(df):
    exclude = ['min_threshold', 'forward_runtime', 'log_runtime', 'is_cpu', 'is_gpu', 'is_single', 'is_double']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    
    corr_thresh = {}
    corr_runtime = {}
    
    for feat in numeric_cols:
        if feat in df.columns:
            thresh_val = df[feat].corr(df['min_threshold'])
            runtime_val = df[feat].corr(df['log_runtime'])
            if not np.isnan(thresh_val):
                corr_thresh[feat] = thresh_val
            if not np.isnan(runtime_val):
                corr_runtime[feat] = runtime_val
    
    return pd.Series(corr_thresh).sort_values(ascending=False), pd.Series(corr_runtime).sort_values(ascending=False)
