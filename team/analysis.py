import json, os, re
import matplotlib.pyplot as plt

os.chdir("..")

# read all the circuits into an array
code = [(f, open(p).readlines()) for f in os.listdir("circuits") if os.path.isfile(p := f"circuits/{f}")]

# read the data for each circuit into the array
with open("data/hackathon_public.json") as f: 
    results = (j := json.load(f))["results"]
    circuits = j["circuits"]

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
        for result in entry["just_past"]:
            ret.append((
                file_name,
                entry["n"],
                entry["file_len"],
                result["is_cpu"],
                result["is_single"],
                result["threshold"],
                result["seconds"]
            ))
    return ret

# organize the data. key is circuit, other stuff is named entries under that with results as tuple keys
data = {re.match(r"(.+).qasm", f)[1]: process_circuit(f) for (f, _) in code}
flattened = make_flattened()

print(flattened[:5])