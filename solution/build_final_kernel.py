"""Write (and verify locally) the Kaggle notebook that regenerates the E38 tickets from the data, one file per ticket.

Reads ../outputs/portfolio_final/selection.json and <name>.csv written by portfolio_final.py. Tickets that are new
rules get files Q01.csv, Q02.csv ...; each is band(lo, hi), window(ranking, side, start, width) on the E16 / E30 / E31
uncertainty ranking, or the E34 / E36 majority vector. The notebook prints whether every file matches the optimizer's.

    python build_final_kernel.py --names N0567,N0610 --exec-dir DIR_WITH_../data
    kaggle kernels push -p kaggle_final_kernel
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
KERNEL_ID = "hosen42/kacchi-aloo-final-tickets"
DOCKER = "gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a"

_spec = importlib.util.spec_from_file_location("bsn", os.path.join(ROOT, "research/scripts/build_submission_notebooks.py"))
bsn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsn)

EDGE_MODELS = r'''from scipy.special import betaln
from sklearn.isotonic import IsotonicRegression

r_tr, y_tr, x_te = tr.apg.values, tr.y.values.astype(float), te.apg.values


def plateaus(r, y, lo_win, hi_win):
    mid = (r > lo_win[1]) & (r < hi_win[0])
    out = (r < lo_win[0]) | (r > hi_win[1])
    jef = lambda m: (y[m].sum() + 0.5) / (m.sum() + 1.0)
    return jef(mid), jef(out)


def isotonic_edges(r, y, x, lo_win=(0.85, 1.15), hi_win=(1.80, 2.20)):
    """E30: increasing isotonic fit on the lower-edge window, decreasing on the upper, plateau rates elsewhere."""
    p_mid, p_out = plateaus(r, y, lo_win, hi_win)
    wl = (r >= lo_win[0]) & (r <= lo_win[1]); wh = (r >= hi_win[0]) & (r <= hi_win[1])
    iso_l = IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds="clip").fit(r[wl], y[wl])
    iso_h = IsotonicRegression(increasing=False, y_min=0, y_max=1, out_of_bounds="clip").fit(r[wh], y[wh])
    p = np.where((x > lo_win[1]) & (x < hi_win[0]), p_mid, p_out)
    p = np.where((x >= lo_win[0]) & (x <= lo_win[1]), iso_l.predict(np.clip(x, *lo_win)), p)
    return np.where((x >= hi_win[0]) & (x <= hi_win[1]), iso_h.predict(np.clip(x, *hi_win)), p)


def changepoint_edge(r, y, win, x, a0=1.0, b0=1.0):
    """E31, one edge: cut uniform on the window, a constant rate on each side (Beta prior), cut integrated out."""
    m = (r >= win[0]) & (r <= win[1])
    rr, yy = r[m], y[m]
    u = np.unique(np.r_[win[0], rr, win[1]])
    lo_k, hi_k = u[:-1], u[1:]
    above = rr[None, :] >= hi_k[:, None]
    n_a, k_a = above.sum(1), (above * yy[None, :]).sum(1)
    n_b, k_b = len(rr) - n_a, yy.sum() - k_a
    ll = (betaln(a0 + k_a, b0 + n_a - k_a) + betaln(a0 + k_b, b0 + n_b - k_b) - 2 * betaln(a0, b0)
          + np.log(np.maximum(hi_k - lo_k, 1e-12)))
    post = np.exp(ll - ll.max()); post /= post.sum()
    th_a, th_b = (a0 + k_a) / (a0 + b0 + n_a), (a0 + k_b) / (a0 + b0 + n_b)
    q = np.clip((x[None, :] - lo_k[:, None]) / (hi_k - lo_k)[:, None], 0, 1)
    return post @ (q * th_a[:, None] + (1 - q) * th_b[:, None])


def changepoint_edges(r, y, x, lo_win=(0.85, 1.15), hi_win=(1.80, 2.25)):
    p_mid, p_out = plateaus(r, y, lo_win, hi_win)
    p = np.where((x > lo_win[1]) & (x < hi_win[0]), p_mid, p_out)
    il = (x >= lo_win[0]) & (x <= lo_win[1]); ih = (x >= hi_win[0]) & (x <= hi_win[1])
    p[il] = changepoint_edge(r, y, lo_win, x[il])
    p[ih] = changepoint_edge(r, y, hi_win, x[ih])
    return p


P30 = isotonic_edges(r_tr, y_tr, x_te)
P31 = changepoint_edges(r_tr, y_tr, x_te)
RANKING_P = {"E16": P, "E30": P30, "E31": P31}
print("rows with 0.25 < P < 0.75:", {k: int(((v > 0.25) & (v < 0.75)).sum()) for k, v in RANKING_P.items()})'''

TICKETS = r'''SPECS = {specs}
EXPECTED = {expected}

A = band(x_te, LO, HI)
d_edge = np.minimum(abs(x_te - LO), abs(x_te - HI))
low = x_te < 1.5


def ticket(spec):
    if spec["family"] == "band":
        return band(x_te, spec["lo"], spec["hi"])
    order = np.lexsort((d_edge, np.abs(RANKING_P[spec["ranking"]] - 0.5)))
    o = {{"both": order, "lower": order[low[order]], "upper": order[~low[order]]}}[spec["side"]]
    v = A.copy()
    v[o[spec["start"]:spec["start"] + spec["width"]]] ^= 1
    return v


def describe(spec):
    if spec["family"] == "band":
        return f"band {{spec['lo']}} <= aloo/guest <= {{spec['hi']}}"
    edge = {{"both": "both edges", "lower": "lower edge", "upper": "upper edge"}}[spec["side"]]
    return f"A + flip {{spec['ranking']}} uncertainty ranks {{spec['start'] + 1}}-{{spec['start'] + spec['width']}} ({{edge}})"


rows = []
for name, spec in SPECS.items():
    v = ticket(spec)
    sub = pd.DataFrame({{"wedding_id": te.wedding_id, "went_back_for_seconds": v.astype(int)}})
    sub.to_csv(f"{{name}}.csv", index=False)
    fp = hashlib.md5("".join(f"{{i}}{{p}}" for i, p in zip(sub.wedding_id, sub.went_back_for_seconds)).encode()).hexdigest()
    rows.append(dict(file=f"{{name}}.csv", rule=describe(spec), rows_vs_A=int((v != A).sum()), ones=int(v.sum()),
                     matches_plan=fp == EXPECTED[name]))
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print("all tickets match the optimizer's files:", bool(summary.matches_plan.all()))'''


def fingerprint(path):
    s = pd.read_csv(path)
    return hashlib.md5("".join(f"{i}{p}" for i, p in zip(s.wedding_id, s.went_back_for_seconds.astype(int))).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "kaggle_final_kernel"))
    ap.add_argument("--exec-dir", default=None, help="run the notebook here (needs ../data) to verify before pushing")
    ap.add_argument("--names", default=None, help="comma-separated optimizer tickets to include, in order (e.g. N0567,N0610)")
    ap.add_argument("--pdir", default=os.path.join(ROOT, "outputs/portfolio_final"))
    ap.add_argument("--key", default="new", help="selection.json list to read: new or extend")
    ap.add_argument("--prefix", default="Q")
    ap.add_argument("--kernel-id", default=KERNEL_ID)
    a = ap.parse_args()
    pdir = a.pdir
    sel = json.load(open(os.path.join(pdir, "selection.json")))
    by_name = {t["name"]: t for t in sel[a.key]}
    new = [by_name[n] for n in a.names.split(",")] if a.names else sel[a.key]
    specs = {f"{a.prefix}{k:02d}": t["spec"] for k, t in enumerate(new, 1)}
    assert all(s["family"] in ("band", "window") for s in specs.values()), "E34/E36 tickets need their own cells"
    expected = {f"{a.prefix}{k:02d}": fingerprint(os.path.join(pdir, f"{t['name']}.csv")) for k, t in enumerate(new, 1)}
    head = (f"# Kacchi Aloo: tickets {a.prefix}01–{a.prefix}{len(specs):02d}\n\n"
            "This competition lets a team select up to 25 final submissions and the private leaderboard counts the "
            "best of them. These tickets were chosen in order of marginal value by E38 (`solution/portfolio_final.py`), "
            "which weighs the final selection over three label models of the noisy band edges (B's out-of-fold distance model E16, a "
            "parametric band, and edge-wise isotonic regression E30) by how well each explains our known public "
            "scores. Each ticket is a rule computed from the training data:\n\n"
            "* **band(lo, hi)**: predict 1 when lo ≤ aloo/guest ≤ hi;\n"
            "* **window**: the sharp band A with a block of rows flipped, taken from the uncertainty ranking of E16 "
            "(out-of-fold error by distance), E30 (isotonic edges) or E31 (Bayesian change-point).\n\n"
            "No test row is labelled by hand. Run on Kaggle with the competition data attached; it writes one CSV "
            "per ticket.")
    nb = new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                "language_info": {"name": "python"}})
    nb.cells = [new_markdown_cell(head), new_code_cell(bsn.LOAD), new_code_cell(bsn.MIDCUT), new_code_cell(bsn.B_PROBS),
                new_code_cell(EDGE_MODELS),
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
    nbformat.write(nb, os.path.join(a.dir, "kacchi-aloo-final-tickets.ipynb"))
    meta = {"id": a.kernel_id, "title": " ".join(w.capitalize() for w in a.kernel_id.split("/")[1].split("-")), "code_file": "kacchi-aloo-final-tickets.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [], "competition_sources": ["the-great-kacchi-aloo-mystery"],
            "kernel_sources": [], "model_sources": [], "docker_image": DOCKER}
    with open(os.path.join(a.dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(pdir, f"{a.prefix.lower()}_names.json"), "w") as f:
        json.dump({f"{a.prefix}{k:02d}": t["name"] for k, t in enumerate(new, 1)}, f, indent=1)
    print("wrote", a.dir)


if __name__ == "__main__":
    main()
