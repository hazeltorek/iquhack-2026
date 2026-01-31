import re
from pathlib import Path

text = Path("../circuits/ae_indep_qiskit_20.qasm").read_text(encoding="utf-8")

n_lines = sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("//"))
n_meas  = len(re.findall(r"\bmeasure\b", text))
n_cx    = len(re.findall(r"\bcx\b", text))
n_cz    = len(re.findall(r"\bcz\b", text))
n_1q    = len(re.findall(r"\b(h|x|y|z|s|sdg|t|tdg|rx|ry|rz|u1|u2|u3)\b", text))

print(n_lines, n_meas, n_cx, n_cz, n_1q)