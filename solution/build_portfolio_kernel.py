"""Write (and verify locally) the Kaggle notebook that regenerates the E23 portfolio tickets P01..P22 from the data.

Reads ../outputs/portfolio25/specs.json + P??.csv written by portfolio25.py --write. Each ticket is a rule:
band(lo, hi) or window(side, start, width) on the sharp band's uncertainty ranking; the notebook prints, per file,
whether it matches the optimizer's file.

    python build_portfolio_kernel.py [--exec-dir DIR_WITH_../data]   # then: kaggle kernels push -p kaggle_portfolio_kernel
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
KERNEL_ID = "hosen42/kacchi-aloo-portfolio"
DOCKER = "gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a"

_spec = importlib.util.spec_from_file_location("bsn", os.path.join(ROOT, "research/scripts/build_submission_notebooks.py"))
bsn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsn)

TICKETS = r'''SPECS = {specs}
EXPECTED = {expected}

A = band(te.apg.values, LO, HI)
order = np.lexsort((np.minimum(abs(te.apg.values - LO), abs(te.apg.values - HI)), np.abs(P - 0.5)))
low = te.apg.values < 1.5
side_order = {{"both": order, "lower": order[low[order]], "upper": order[~low[order]]}}


def ticket(spec):
    if spec["family"] == "band":
        return band(te.apg.values, spec["lo"], spec["hi"])
    v = A.copy()
    v[side_order[spec["side"]][spec["start"]:spec["start"] + spec["width"]]] ^= 1
    return v


def describe(spec):
    if spec["family"] == "band":
        return f"band {{spec['lo']}} <= aloo/guest <= {{spec['hi']}}"
    edge = {{"both": "both edges", "lower": "lower edge", "upper": "upper edge"}}[spec["side"]]
    return f"A + flip uncertainty ranks {{spec['start'] + 1}}-{{spec['start'] + spec['width']}} ({{edge}})"


rows = []
for k, spec in enumerate(SPECS, 1):
    v = ticket(spec)
    name = f"P{{k:02d}}.csv"
    sub = pd.DataFrame({{"wedding_id": te.wedding_id, "went_back_for_seconds": v.astype(int)}})
    sub.to_csv(name, index=False)
    fp = hashlib.md5("".join(f"{{i}}{{p}}" for i, p in zip(sub.wedding_id, sub.went_back_for_seconds)).encode()).hexdigest()
    rows.append(dict(file=name, rule=describe(spec), rows_vs_A=int((v != A).sum()), ones=int(v.sum()), matches_plan=fp == EXPECTED[name]))
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print("all tickets match the optimizer's files:", bool(summary.matches_plan.all()))'''


def fingerprint(path):
    s = pd.read_csv(path)
    return hashlib.md5("".join(f"{i}{p}" for i, p in zip(s.wedding_id, s.went_back_for_seconds.astype(int))).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "kaggle_portfolio_kernel"))
    ap.add_argument("--exec-dir", default=None, help="run the notebook here (needs ../data) to verify before pushing")
    a = ap.parse_args()
    pdir = os.path.join(ROOT, "outputs/portfolio25")
    specs = json.load(open(os.path.join(pdir, "specs.json")))
    expected = {f"P{k:02d}.csv": fingerprint(os.path.join(pdir, f"P{k:02d}.csv")) for k in range(1, len(specs) + 1)}
    head = ("# Kacchi Aloo: final-selection portfolio (P01–P%02d)\n\n"
            "This competition lets a team select up to 25 final submissions, and the private leaderboard counts the "
            "best of them. Every team uses nearly the same potato-per-guest band, so the private ranking is decided "
            "by a few near-edge weddings whose labels are noisy. These tickets cover different plausible outcomes for "
            "those weddings. Each is a rule computed from the data:\n\n"
            "* **band(lo, hi)**: predict 1 when lo ≤ aloo/guest ≤ hi (shifted edges);\n"
            "* **window**: the sharp band A with a block of its most uncertain rows flipped. Uncertainty ranks come "
            "from the band's out-of-fold error rate at each distance from the edge (the same ranking that produced B).\n\n"
            "Which rules to include was chosen by a simulation that maximises the chance of finishing first and in "
            "the top 5 (`solution/portfolio25.py`, E23). No test row is labelled by hand.\n\n"
            "Run on Kaggle with the competition data attached; it writes `P01.csv` … one file per ticket." % len(specs))
    nb = new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                "language_info": {"name": "python"}})
    nb.cells = [new_markdown_cell(head), new_code_cell(bsn.LOAD), new_code_cell(bsn.MIDCUT), new_code_cell(bsn.B_PROBS),
                new_code_cell(TICKETS.format(specs=json.dumps(specs, indent=1), expected=json.dumps(expected, indent=1)))]
    if a.exec_dir:
        os.makedirs(a.exec_dir, exist_ok=True)
        run = nbformat.reads(nbformat.writes(nb), as_version=4)
        NotebookClient(run, timeout=900, kernel_name="python3", resources={"metadata": {"path": a.exec_dir}}).execute()
        text = "".join(o.get("text", "") for o in run.cells[-1].outputs)
        print(text)
        assert text.rstrip().endswith("True"), "notebook does not reproduce the optimizer's tickets"
        nb = run
    os.makedirs(a.dir, exist_ok=True)
    nbformat.write(nb, os.path.join(a.dir, "kacchi-aloo-portfolio.ipynb"))
    meta = {"id": KERNEL_ID, "title": "Kacchi Aloo Portfolio", "code_file": "kacchi-aloo-portfolio.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": [], "competition_sources": ["the-great-kacchi-aloo-mystery"], "kernel_sources": [],
            "model_sources": [], "docker_image": DOCKER}
    with open(os.path.join(a.dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", a.dir)


if __name__ == "__main__":
    main()
