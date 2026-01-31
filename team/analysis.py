import json, os, re
import matplotlib.pyplot as plt
import networkx as nx

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

# organize the data. key is circuit, other stuff is named entries under that with results as tuple keys
data = {re.match(r"(.+).qasm", f)[1]: process_circuit(f) for (f, _) in code}

# pairs of qubits that interact via 2-qubit gates
def get_pairs(qasm):
    # i saw the face of god in a regular expression
    return sorted(list(set([tuple(sorted(list(map(int, p)))) for p in re.findall(r"[a-z]+(?:\((?:-{0,1}(?:\d+\.{0,1}\d+|pi(?:\/\d+){0,1}),{0,1}){1,3}\)){0,1}\sq\[(\d+)\],\s*q\[(\d+)\];", qasm)])))

# build interaction graph and get cool useful statistics out of it
graphs = {f: nx.Graph(get_pairs("\n".join(qasm))) for (f, qasm) in code}
for f in data.keys(): 
    data[f]["degree"] = max([0] + list(map(lambda d: d[1], graphs[(fn := f"{f}.qasm")].degree())))
    data[f]["n_edges"] = graphs[fn].number_of_edges()
    data[f]["centrality"] = max([0] + list(nx.degree_centrality(graphs[fn]).values()))
    data[f]["n_clusters"] = nx.number_connected_components(graphs[fn])

print(data[code[2][0][:-5]]["degree"])
print(data[code[2][0][:-5]]["n_edges"])
print(data[code[2][0][:-5]]["centrality"])
print(data[code[2][0][:-5]]["n_clusters"])
