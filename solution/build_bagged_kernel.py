"""Write (and verify locally) the Kaggle notebook for the one pre-registered public batch (plan §16, P-09): the bagged-E30
ranking (1000 bootstrap isotonic fits, seed 0, as in edge_models2.py E39) with the 5 / 7 / 9 / 11 most uncertain rows of A
flipped. Files R05.csv, R07.csv, R09.csv, R11.csv; the notebook checks each against outputs/edge_models2/E39_hedge*.csv.

    python build_bagged_kernel.py --exec-dir DIR_WITH_../data
    kaggle kernels push -p kaggle_bagged_kernel
"""
import argparse
import hashlib
import importlib.util
import json
import os

import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KERNEL_ID = "hosen42/kacchi-aloo-bagged-e30-batch"
SIZES = (5, 7, 9, 11)

_spec = importlib.util.spec_from_file_location("bfk", os.path.join(HERE, "build_final_kernel.py"))
bfk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bfk)

BATCH = r'''EXPECTED = {expected}

rng = np.random.default_rng(0)
P39 = np.zeros(len(x_te))
for _ in range(1000):                                   # bagged E30: bootstrap the training rows, refit, average
    ii = rng.integers(len(r_tr), size=len(r_tr))
    P39 += isotonic_edges(r_tr[ii], y_tr[ii], x_te)
P39 /= 1000

A = band(x_te, LO, HI)
order = np.lexsort((np.minimum(abs(x_te - LO), abs(x_te - HI)), np.abs(P39 - 0.5)))
rows = []
for m in {sizes}:
    v = A.copy()
    v[order[:m]] ^= 1
    name = f"R{{m:02d}}"
    sub = pd.DataFrame({{"wedding_id": te.wedding_id, "went_back_for_seconds": v.astype(int)}})
    sub.to_csv(f"{{name}}.csv", index=False)
    fp = hashlib.md5("".join(f"{{i}}{{p}}" for i, p in zip(sub.wedding_id, sub.went_back_for_seconds)).encode()).hexdigest()
    rows.append(dict(file=f"{{name}}.csv", rule=f"A + flip the {{m}} most uncertain bagged-E30 rows", rows_vs_A=int((v != A).sum()),
                     flipped=" ".join(te.wedding_id[order[:m]]), matches_plan=fp == EXPECTED[name]))
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print("all tickets match the pre-registered files:", bool(summary.matches_plan.all()))'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "kaggle_bagged_kernel"))
    ap.add_argument("--exec-dir", default=None)
    a = ap.parse_args()
    expected = {f"R{m:02d}": bfk.fingerprint(os.path.join(ROOT, f"outputs/edge_models2/E39_hedge{m}.csv")) for m in SIZES}
    head = ("# Kacchi Aloo: one pre-registered batch from bagged E30 (R05, R07, R09, R11)\n\n"
            "Bagged E30 averages 1000 bootstrap refits of the edge-wise isotonic model. Its ranking of uncertain test "
            "rows was frozen, and the four hedge sizes 5 / 7 / 9 / 11 were fixed, before any of these files was "
            "submitted; the batch is submitted once and not tuned on public scores. Each ticket is the sharp band A with "
            "its m most uncertain rows flipped. No row is labelled by hand.")
    nb = new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                "language_info": {"name": "python"}})
    nb.cells = [new_markdown_cell(head), new_code_cell(bfk.bsn.LOAD), new_code_cell(bfk.bsn.MIDCUT),
                new_code_cell(bfk.bsn.B_PROBS), new_code_cell(bfk.EDGE_MODELS),
                new_code_cell(BATCH.format(expected=json.dumps(expected, indent=1), sizes=SIZES))]
    if a.exec_dir:
        os.makedirs(a.exec_dir, exist_ok=True)
        run = nbformat.reads(nbformat.writes(nb), as_version=4)
        NotebookClient(run, timeout=1800, kernel_name="python3", resources={"metadata": {"path": a.exec_dir}}).execute()
        text = "".join(o.get("text", "") for o in run.cells[-1].outputs)
        print(text)
        assert text.rstrip().endswith("True"), "notebook does not reproduce the pre-registered files"
        nb = run
    os.makedirs(a.dir, exist_ok=True)
    nbformat.write(nb, os.path.join(a.dir, "kacchi-aloo-bagged-e30-batch.ipynb"))
    meta = {"id": KERNEL_ID, "title": "Kacchi Aloo Bagged E30 Batch", "code_file": "kacchi-aloo-bagged-e30-batch.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [], "competition_sources": ["the-great-kacchi-aloo-mystery"],
            "kernel_sources": [], "model_sources": [], "docker_image": bfk.DOCKER}
    with open(os.path.join(a.dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", a.dir)


if __name__ == "__main__":
    main()
