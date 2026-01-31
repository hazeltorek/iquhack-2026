import json, os, re
import matplotlib.pyplot as plt

os.chdir("..")
cats = {(True, True): {}, (True, False): {}, (False, True): {}, (False, False): {}}

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
    
    for test in b:
        del test['forward']
        run_fid = None
        for run in test['threshold_sweep']:
            fid = run["sdk_get_fidelity"]
            if isinstance(fid, float) and fid > 0.99:
                run_fid = run
                break
        if run_fid is not None:
            cats[(test["backend"] == "CPU", test["precision"] == "single")][length] = run_fid['threshold']


    # organize results by the given predictors (cpu/gpu, single/double)
    pred = {(r["backend"] == "CPU", r["precision"] == "single"): b for r in b}
    return {
        "family": a["family"], "n": a["n_qubits"], 
        "file_len": length, 
        "results": pred
    }

# organize the data. key is circuit, other stuff is named entries under that with results as tuple keys
data = {re.match(r"(.+).qasm", f)[1]: process_circuit(f) for (f, _) in code}

