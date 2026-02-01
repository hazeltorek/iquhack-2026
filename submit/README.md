# Circuit Fingerprint Challenge Submission

## Team Information
Team: ket-in-4k

## Approach Summary

Our solution combines two specialized models:

### 1. Threshold Prediction (Binning Model)
- **Model Type**: Scikit-learn regression with aggressive binning strategy
- **Features Extracted**:
  - Circuit topology: entanglement stress, graph metrics (degree, centrality)
  - Structural features: circuit depth, gate counts (1Q, 2Q, measurements)
  - Derived metrics: axis_a_volume (qubit×depth), axis_c_packing (edges/volume)
- **Strategy**: 
  - Train regression model to predict continuous threshold value
  - Apply aggressive binning algorithm that:
    - Bins to THRESHOLD_LADDER [1, 2, 4, 8, 16, 32, 64, 128, 256]
    - Adjusts aggressiveness based on circuit complexity metrics
    - Prefers lower thresholds for simpler circuits (reduced simulation cost)
    - More conservative for high-entanglement circuits

### 2. Runtime Prediction (Neural Network)
- **Model Type**: PyTorch feed-forward neural network (128→64→1)
- **Architecture**: 
  - Input: 12 features
  - Hidden layers with ReLU activation
  - Output: log₂(runtime in seconds)
- **Features Used**:
  - Base: n_qubits, threshold, n_measurements, backend (CPU/GPU), precision (single/double)
  - Circuit metrics: depth_proxy, 2Q gates (CX+CZ), 1Q gates
  - Interaction terms: n×threshold, n²×threshold, n×depth, depth×threshold
- **Training**:
  - 1000 epochs with Adam optimizer (lr=0.001)
  - Log-scale prediction for handling wide runtime range (0.01s - 400s)
  - Predictions capped at 400s maximum

### 3. Integration
The pipeline:
1. Extract features from QASM file
2. Predict minimum threshold using binning model
3. Use predicted threshold as input to runtime model
4. Output both predictions per task

## Validation Approach

- **Threshold Model**: Validated on training circuits achieving 99% fidelity
- **Runtime Model**: 
  - Train/test split on 44 circuits from hackathon_public.json
  - 12 holdout test circuits never seen during training
  - Feature engineering validated through correlation analysis
  - Model weights saved and loaded to ensure reproducibility

## Known Limitations

1. **Threshold Prediction**:
   - Aggressive binning may occasionally predict below true minimum (scores 0)
   - Trade-off: prefer lower thresholds to minimize false positives above minimum

2. **Runtime Prediction**:
   - Trained primarily on CPU/single-precision data
   - GPU predictions use same model with backend flag
   - Extrapolation to very large circuits (>130 qubits) not validated
   - Cap at 400s may underestimate extremely long-running circuits

3. **Generalization**:
   - Models trained on specific circuit families from public dataset
   - Performance on novel circuit patterns may vary
   - Limited validation on double-precision workloads

## File Structure

```
submit/
├── predict.py              # Main submission script
├── requirements.txt        # Python dependencies
├── src/
│   └── features.py         # Feature extraction utilities
└── artifacts/
    ├── entanglement_model.pkl      # Threshold prediction model
    ├── entanglement_scaler.pkl     # Feature scaler for threshold
    ├── entanglement_metadata.json  # Model metadata
    └── model_weights.pth           # Runtime prediction neural network
```

## Run Command

```bash
python predict.py \
  --tasks data/holdout_public.json \
  --circuits circuits/ \
  --id-map id_map.json \
  --out predictions.json \
  --artifacts artifacts
```

## Dependencies

- Python 3.8+
- PyTorch 1.12+
- scikit-learn 1.1+
- numpy, scipy, networkx
- See requirements.txt for complete list
