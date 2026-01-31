import argparse
import json, os, re
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# organized data!
def process_circuit(name):
    a = [circ for circ in circuits if circ["file"] == name][0]
    b = [res for res in results if res["file"] == name]
    length = [len(c) for (f, c) in code if f == name][0]
    
    just_past = []
    for test in b:
        run_fid = None
        for run in sorted(test['threshold_sweep'], key=lambda x: x["threshold"]):
            fid = run["sdk_get_fidelity"]
            if isinstance(fid, float) and fid > 0.99:
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

    # organize results by the given predictors (cpu/gpu, single/double)
    pred = {(r["backend"] == "CPU", r["precision"] == "single"): b for r in b}
    return {
        "family": a["family"], 
        "n": a["n_qubits"], 
        "file_len": length, 
        "results": pred,
        "just_past": just_past,
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
                "n_1q": n_1q
            })
    return ret


# Simple Logistic Regression Model
class MulticlassLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
 
    def forward(self, x):
        out = self.linear(x)
        return out
 
def main(args):
    # print(args.tasks) # data/hackathon_public.json
    # print(args.circuits)
    # print(args.id_map)
    # print(args.out) # circuits

    global circuits, results, code, data

    # read all the circuits into an array
    code = [(f, open(p).readlines()) for f in os.listdir(args.circuits) if os.path.isfile(p := f"{args.circuits}/{f}")]

    # read the data for each circuit into the array
    with open(args.tasks) as f: 
        results = (j := json.load(f))["results"]
        circuits = j["circuits"]

    # organize the data. key is circuit, other stuff is named entries under that with results as tuple keys
    data = {re.match(r"(.+).qasm", f)[1]: process_circuit(f) for (f, _) in code}
    flattened = make_flattened()


    

    # print(flattened[:5])

    X = [] # Input Feature Data 
    y = [] # Threshold Values

    
    for val in flattened:
        features = [
            val["n"],
            val["file_len"], 
            val["lines"],
            int(val["is_cpu"]),
            int(val["is_single"]),
            val["seconds"],
            val["n_meas"],
            val["n_cx"],
            val["n_cz"],
            val["n_1q"]
        ]

        X.append(features)
        y.append(val["threshold"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    threshold_classes = [1, 2, 4, 8, 16, 32, 64, 128, 256] # Output 
    feature_size = len(X[0])                               # Feature length
    y_mapped = [threshold_classes.index(threshold) for threshold in y]

    # Use y_mapped instead of y in train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_mapped, test_size=0.3, random_state=42)

    

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create a DataLoader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=feature_size, shuffle=True)

    
    model = MulticlassLogisticRegression(feature_size, len(threshold_classes))

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train Model
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Test Model
    new_X = torch.tensor(np.random.randn(5, feature_size), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(new_X)
        _, predicted = torch.max(outputs, 1)
        #print('Predicted classes:', predicted)

    # Model Accuracy
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print('Accuracy:', accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type = str, default="data\hackathon_public.json")
    parser.add_argument("--circuits", type = str, default="circuits")
    parser.add_argument("--id-map", type = str, default="data\holdout_public.json")
    parser.add_argument("--out", type = str, default="predictions.json")
    args = parser.parse_args()

    main(args)
                   