"""Build a self-contained notebook per Kaggle submission, execute each, check that it reproduces the submitted file
exactly (prediction fingerprint), then write them to submission_notebooks/ ranked by public score (01 = best, ties by
submission time) as <rank>_<score>_<submission>_<rule>.ipynb. Of submissions with the same score and the same method
(band rule / flipped rows) only the earliest is kept (--all keeps every one). Nothing else is left in the folder.

Scores come from outputs/kaggle_submissions.csv; refresh it first:
    kaggle competitions submissions the-great-kacchi-aloo-mystery --csv --page-size 100 > outputs/kaggle_submissions.csv
    python research/scripts/build_submission_notebooks.py
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "submission_notebooks")

LOAD = r'''import glob, hashlib, os, unicodedata
import numpy as np, pandas as pd

found = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True)) + [p for p in ("../data/train.csv", "data/train.csv") if os.path.exists(p)]
root = os.path.dirname(found[0])
tr_raw = pd.read_csv(f"{root}/train.csv", dtype=str)
te_raw = pd.read_csv(f"{root}/test.csv", dtype=str)


def to_number(v):
    """Float from a cell; any Unicode decimal digit (e.g. Bangla) becomes ASCII."""
    if pd.isna(v):
        return np.nan
    s = "".join(str(unicodedata.decimal(c)) if unicodedata.decimal(c, None) is not None else c
                for c in str(v).strip().replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return np.nan


def add_apg(d):
    """aloo per guest; the few aloo counts typed x10 (ratio > 3) are divided by 10."""
    guests, aloo = d.guests.map(to_number), d.aloo_count.map(to_number)
    aloo = np.where(aloo / guests > 3.0, aloo / 10, aloo)
    return d.assign(apg=aloo / guests)


tr = add_apg(tr_raw).dropna(subset=["apg"]).reset_index(drop=True)   # 24 train rows have no aloo_count
tr["y"] = tr.went_back_for_seconds.astype(int)
te = add_apg(te_raw)
print(f"data: {root} | train {len(tr)} rows with aloo_count | test {len(te)} rows")'''

MIDCUT = r'''def midcut_cuts(r, y):
    """Each edge sits at the midpoint between neighbouring training ratios that maximises accuracy on its side of
    1.5 (median of tied optima)."""
    def best_cut(x, yy, side):
        xs = np.sort(np.unique(x))
        c = (xs[:-1] + xs[1:]) / 2
        accs = np.array([((x >= t) == yy).mean() if side == "lo" else ((x <= t) == yy).mean() for t in c])
        b = np.flatnonzero(accs == accs.max())
        return c[b[len(b) // 2]]
    m = r < 1.5
    return best_cut(r[m], y[m], "lo"), best_cut(r[~m], y[~m], "hi")


def band(x, lo, hi):
    return ((x >= lo) & (x <= hi)).astype(int)'''

A_MODEL = r'''LO, HI = midcut_cuts(tr.apg.values, tr.y.values)
print(f"cut-points {LO:.5f} <= aloo/guest <= {HI:.5f}; train accuracy {(band(tr.apg.values, LO, HI) == tr.y.values).mean():.4f}")
pred = band(te.apg.values, LO, HI)'''

B_PROBS = r'''from sklearn.model_selection import StratifiedKFold


def sharp_band_probabilities(tr, te, bins=(0.0, 0.01, 0.035, 0.1, 0.4, np.inf), repeats=10, seed=0):
    """P(y=1) per test row = the sharp band's out-of-fold error rate at that distance from the nearest edge.
    Test rows inside the training gap around an edge get P from a uniform prior on where the cut lies."""
    r, y = tr.apg.values, tr.y.values
    lo, hi = midcut_cuts(r, y)
    d_oof, e_oof = [], []
    for k in range(repeats):
        for t, v in StratifiedKFold(5, shuffle=True, random_state=seed + k).split(tr, y):
            l2, h2 = midcut_cuts(r[t], y[t])
            d_oof.append(np.minimum(abs(r[v] - l2), abs(r[v] - h2)))
            e_oof.append(band(r[v], l2, h2) != y[v])
    d_oof, e_oof = np.concatenate(d_oof), np.concatenate(e_oof)
    idx = np.digitize(d_oof, bins[1:-1])
    err = np.array([(e_oof[idx == b].sum() + 0.5) / ((idx == b).sum() + 1.0) for b in range(len(bins) - 1)])
    below_lo, above_lo = r[r < lo].max(), r[(r >= lo) & (r < 1.5)].min()
    below_hi, above_hi = r[(r <= hi) & (r > 1.5)].max(), r[r > hi].min()
    x = te.apg.values
    d = np.minimum(abs(x - lo), abs(x - hi))
    e = err[np.digitize(d, bins[1:-1])]
    P = np.where(band(x, lo, hi) == 1, 1 - e, e)
    q_lo = np.clip((x - below_lo) / (above_lo - below_lo), 0, 1)
    q_hi = np.clip((above_hi - x) / (above_hi - below_hi), 0, 1)
    g_lo, g_hi = (x > below_lo) & (x < above_lo), (x > below_hi) & (x < above_hi)
    P = np.where(g_lo, q_lo * (1 - err[0]) + (1 - q_lo) * err[0], P)
    P = np.where(g_hi, q_hi * (1 - err[0]) + (1 - q_hi) * err[0], P)
    return P, (lo, hi), err


P, (LO, HI), err = sharp_band_probabilities(tr, te)
print(f"cut-points {LO:.5f} / {HI:.5f}; out-of-fold error rate by distance to the edge "
      f"(0-0.01, 0.01-0.035, 0.035-0.1, 0.1-0.4, >0.4): {np.round(err, 3)}")'''

B_HEDGE = r'''def hedge_pair(p, tiebreak, n_public=181, sims=20000, seed=0, max_flip=40):
    """A = p > 0.5. B = A with its m most uncertain rows flipped (|p - 0.5| ascending, ties by distance to the
    nearest edge). m maximises E[max(private correct of A, of B)] over simulated labels y_i ~ Bernoulli(p_i)
    and random 181-row public splits."""
    rng = np.random.default_rng(seed)
    n = len(p)
    A = (p > 0.5).astype(int)
    order = np.lexsort((tiebreak, np.abs(p - 0.5)))
    Y = rng.random((sims, n)) < p[None, :]
    priv = np.ones((sims, n), bool)
    for s in range(sims):
        priv[s, rng.choice(n, n_public, replace=False)] = False
    accA = ((Y == A[None, :]) & priv).sum(1)
    gain = {}
    for m in range(max_flip + 1):
        B = A.copy(); B[order[:m]] = 1 - B[order[:m]]
        gain[m] = float(np.maximum(accA, ((Y == B[None, :]) & priv).sum(1)).mean() - accA.mean())
    m_best = max(gain, key=gain.get)
    B = A.copy(); B[order[:m_best]] = 1 - B[order[:m_best]]
    return A, B, m_best, gain


d_edge = np.minimum(abs(te.apg.values - LO), abs(te.apg.values - HI))
A, pred, m, gain = hedge_pair(P, d_edge)
assert (A == band(te.apg.values, LO, HI)).all(), "A must equal the sharp band"
print(f"m = {m} rows flipped; expected best-of-two private gain over A alone: {gain[m]:+.3f} rows")
print(te.assign(P=P.round(3), A=A, B=pred).loc[A != pred, ["wedding_id", "guests", "aloo_count", "apg", "P", "A", "B"]]
      .sort_values("apg").to_string(index=False))'''

C_MODEL = r'''LO, HI = 1.0, 1.975   # variant C: chosen by the final-selection simulation (solution/win_prob.py) as B's partner
inside = lambda x: ((x > LO) & (x < HI)).astype(int)
print(f"band {LO} < aloo/guest < {HI}; train accuracy {(inside(tr.apg.values) == tr.y.values).mean():.4f}")
pred = inside(te.apg.values)'''

WRITE = r'''sub = pd.DataFrame({{"wedding_id": te.wedding_id, "went_back_for_seconds": np.asarray(pred).astype(int)}})
sub.to_csv("submission.csv", index=False)
fp = hashlib.md5("".join(f"{{i}}{{p}}" for i, p in zip(sub.wedding_id, sub.went_back_for_seconds)).encode()).hexdigest()
print(f"wrote submission.csv: {{sub.went_back_for_seconds.sum()}} ones / {{len(sub)}} rows")
print(f"identical to the submitted file {submitted!r}: {{fp == {expected!r}}}")'''

RUN_NOTE = ("**Run it:** on Kaggle, upload this notebook and add the competition data (*Add Input → Competitions → "
            "The Great Kacchi Aloo Mystery*), then *Save & Run All* and submit `submission.csv`. Locally, run it from "
            "this folder (it reads `../data/train.csv` and `../data/test.csv`). The last cell checks the output "
            "against the file that was actually submitted.")

TICKET_BAND = r'''LO_T, HI_T = {lo!r}, {hi!r}   # ticket {name}: band chosen by the portfolio optimizer (E23)
print(f"band {{LO_T}} <= aloo/guest <= {{HI_T}}; train accuracy {{(band(tr.apg.values, LO_T, HI_T) == tr.y.values).mean():.4f}}")
pred = band(te.apg.values, LO_T, HI_T)'''

TICKET_WINDOW = r'''SIDE, START, WIDTH = {side!r}, {start}, {width}   # ticket {name}: flip uncertainty ranks {r1}-{r2} ({edge})
A = band(te.apg.values, LO, HI)
order = np.lexsort((np.minimum(abs(te.apg.values - LO), abs(te.apg.values - HI)), np.abs(P - 0.5)))
low = te.apg.values < 1.5
o = {{"both": order, "lower": order[low[order]], "upper": order[~low[order]]}}[SIDE]
pred = A.copy()
pred[o[START:START + WIDTH]] ^= 1
print(te.assign(P=P.round(3), A=A, ticket=pred).loc[A != pred, ["wedding_id", "guests", "aloo_count", "apg", "P", "A", "ticket"]]
      .sort_values("apg").to_string(index=False))'''

TICKET_RANKED = r'''SIDE, START, WIDTH, RANKING = {side!r}, {start}, {width}, {ranking!r}   # ticket {name}: flip {ranking} ranks {r1}-{r2} ({edge})
A = band(te.apg.values, LO, HI)
order = np.lexsort((np.minimum(abs(te.apg.values - LO), abs(te.apg.values - HI)), np.abs(RANKING_P[RANKING] - 0.5)))
low = te.apg.values < 1.5
o = {{"both": order, "lower": order[low[order]], "upper": order[~low[order]]}}[SIDE]
pred = A.copy()
pred[o[START:START + WIDTH]] ^= 1
print(te.assign(P=RANKING_P[RANKING].round(3), A=A, ticket=pred).loc[A != pred, ["wedding_id", "guests", "aloo_count", "apg", "P", "A", "ticket"]]
      .sort_values("apg").to_string(index=False))'''

SUBS = os.path.join(ROOT, "outputs/kaggle_submissions.csv")
PORT = os.path.join(ROOT, "outputs/portfolio25")
EDGE = {"both": "both edges", "lower": "lower edge", "upper": "upper edge"}

BASE = [  # (label, Kaggle file name, time filter, local reference file, rule, method, extra text, cells)
    ("A_sharp_band", "sub_A_primary_sharpband.csv", None, "outputs/sub_A_primary_sharpband.csv",
     "0.97068 ≤ aloo/guest ≤ 2.00983", "band",
     "Guests go back for seconds when the potatoes per guest fall inside a band. Each edge is placed at the midpoint "
     "between the two neighbouring training ratios that maximises training accuracy on its side. 10×5 CV accuracy "
     "0.954; every other column, soft edges and boosting scored the same or worse.", [MIDCUT, A_MODEL]),
    ("B_hedge", "sub_B_hedge.csv", None, "outputs/sub_B_hedge.csv", "A with its 9 most uncertain rows flipped", "flip",
     "B is A with the m most uncertain test rows flipped; m is chosen by Monte Carlo to maximise the expected best-of-two "
     "private score, with each row's probability taken from the sharp band's out-of-fold error rate at that distance "
     "from the edge. The rows are picked by this algorithm, not by hand.", [MIDCUT, B_PROBS, B_HEDGE]),
    ("C_band_1.000-1.975", "sub_C_band_1p00_1p975.csv", None, "outputs/sub_C_band_1p00_1p975.csv",
     "1.000 < aloo/guest < 1.975", "band",
     "C shifts both edges (lower to 1.0, upper to 1.975). Every leaderboard team uses nearly the same band, so C is a "
     "ticket other teams do not hold (E21/E22, `solution/win_prob.py`, `solution/win_prob2.py`). The same predictions "
     "were submitted again at 17:45 UTC from notebook "
     "[kacchi-aloo-band-submission](https://www.kaggle.com/code/hosen42/kacchi-aloo-band-submission) v2.", [C_MODEL]),
]


def fingerprint(path):
    s = pd.read_csv(path)
    return hashlib.md5("".join(f"{i}{p}" for i, p in zip(s.wedding_id, s.went_back_for_seconds.astype(int))).encode()).hexdigest()


def lookup(subs, file_name, at=None):
    """(public score or None, 'YYYY-MM-DD HH:MM' or None) of the earliest submission of file_name."""
    m = subs[subs.fileName == file_name]
    if at:
        m = m[m.date.dt.strftime("%H:%M") == at]
    if m.empty:
        return None, None
    r = m.sort_values("date").iloc[0]
    return (None if pd.isna(r.publicScore) else float(r.publicScore)), r.date.strftime("%Y-%m-%d %H:%M")


def score_tag(score):
    return "pending" if score is None else f"{score:.5f}"


def info_table(file_name, when, score, rule):
    sc = "not scored yet" if score is None else f"{score:.5f} ({round(score * 181)}/181)"
    return (f"| | |\n|---|---|\n| Submitted file | `{file_name}` |\n| Submitted (UTC) | {when or 'not yet'} |\n"
            f"| Public score | {sc} |\n| Rule | {rule} |")


def run(nb, label, ref):
    """Execute in OUT and check the submission.csv it writes against the submitted file."""
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": OUT}}).execute()
    assert fingerprint(os.path.join(OUT, "submission.csv")) == fingerprint(os.path.join(ROOT, ref)), f"{label} != {ref}"
    return nb


def new_nb(title, table, text, cells, ref):
    nb = new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                "language_info": {"name": "python"}})
    nb.cells = [new_markdown_cell(f"{title}\n\n{table}\n\n{text}\n\n{RUN_NOTE}"), new_code_cell(LOAD)]
    nb.cells += [new_code_cell(c) for c in cells]
    nb.cells.append(new_code_cell(WRITE.format(submitted=os.path.basename(ref), expected=fingerprint(os.path.join(ROOT, ref)))))
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="keep submissions that share score and method with an earlier one")
    args = ap.parse_args()
    subs = pd.read_csv(SUBS, parse_dates=["date"])
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):  # everything here is generated by this script
        if f.endswith((".ipynb", ".md", ".csv")):
            os.remove(os.path.join(OUT, f))
    entries = []  # dict(label, method, score, when, nb)
    for label, kfile, at, ref, rule, method, text, cells in BASE:
        score, when = lookup(subs, kfile, at)
        nb = new_nb(f"# Submission {label.split('_')[0]}: {rule}", info_table(kfile, when, score, rule), text, cells, ref)
        entries.append(dict(label=label, method=method, score=score, when=when, nb=run(nb, label, ref)))

    # the 10:53 submission came from the theory notebook (needs statsmodels + matplotlib); its output equals A
    score, when = lookup(subs, "submission.csv", "10:53")
    nb = nbformat.read(os.path.join(ROOT, "solution/kaggle_theory_kernel/aloo-theory.ipynb"), as_version=4)
    entries.append(dict(label="A_theory_notebook", method="band", score=score, when=when,
                        nb=run(nb, "A_theory_notebook", "outputs/sub_A_primary_sharpband.csv")))

    # E23 portfolio tickets, one notebook each (produced together by Kaggle notebook kacchi-aloo-portfolio v1);
    # P21/P22 were replaced by Q01/Q02 (E38) before being submitted, so only submitted P tickets are kept
    specs = json.load(open(os.path.join(PORT, "specs.json")))
    for k, sp in enumerate(specs, 1):
        tk = f"P{k:02d}"
        score, when = lookup(subs, f"{tk}.csv")
        if when is None:
            continue
        ref = f"outputs/portfolio25/{tk}.csv"
        if sp["family"] == "band":
            rule, method = f"{sp['lo']} ≤ aloo/guest ≤ {sp['hi']}", "band"
            label = f"{tk}_band_{sp['lo']}-{sp['hi']}"
            cells = [MIDCUT, TICKET_BAND.format(lo=sp["lo"], hi=sp["hi"], name=tk)]
        else:
            r1, r2 = sp["start"] + 1, sp["start"] + sp["width"]
            rule, method = f"A with uncertainty ranks {r1}–{r2} flipped ({EDGE[sp['side']]})", "flip"
            label = f"{tk}_flip_{sp['side']}_r{r1}-{r2}"
            cells = [MIDCUT, B_PROBS, TICKET_WINDOW.format(side=sp["side"], start=sp["start"], width=sp["width"], name=tk,
                                                           r1=r1, r2=r2, edge=EDGE[sp["side"]])]
        text = ("One of the 22 tickets of the final-selection portfolio (E23, `solution/portfolio25.py`): the competition "
                "lets a team select up to 25 final submissions and the private leaderboard counts the best of them. "
                f"Submitted as `{tk}.csv` from the Kaggle notebook "
                "[kacchi-aloo-portfolio](https://www.kaggle.com/code/hosen42/kacchi-aloo-portfolio) v1 "
                "(source: `solution/kaggle_portfolio_kernel/`), which writes all 22 tickets.")
        nb = new_nb(f"# Submission {tk}: {rule}", info_table(f"{tk}.csv", when, score, rule), text, cells, ref)
        entries.append(dict(label=label, method=method, score=score, when=when, nb=run(nb, tk, ref)))

    # E38 tickets Q01.. (Kaggle notebook kacchi-aloo-final-tickets v2): windows on the E16 / E30 / E31 rankings or bands
    fdir = os.path.join(ROOT, "outputs/portfolio_final")
    q_names = json.load(open(os.path.join(fdir, "q_names.json")))
    q_specs = {t["name"]: t["spec"] for t in json.load(open(os.path.join(fdir, "selection.json")))["new"]}
    _s = importlib.util.spec_from_file_location("bfk", os.path.join(ROOT, "solution/build_final_kernel.py"))
    bfk = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(bfk)
    for tk, nname in q_names.items():
        sp = q_specs[nname]
        score, when = lookup(subs, f"{tk}.csv")
        ref = f"outputs/portfolio_final/{nname}.csv"
        if sp["family"] == "band":
            rule, method = f"{sp['lo']} ≤ aloo/guest ≤ {sp['hi']}", "band"
            label = f"{tk}_band_{sp['lo']}-{sp['hi']}"
            cells = [MIDCUT, TICKET_BAND.format(lo=sp["lo"], hi=sp["hi"], name=tk)]
        else:
            r1, r2 = sp["start"] + 1, sp["start"] + sp["width"]
            rule, method = f"A with {sp['ranking']} uncertainty ranks {r1}–{r2} flipped ({EDGE[sp['side']]})", "flip"
            label = f"{tk}_flip_{sp['ranking']}_{sp['side']}_r{r1}-{r2}"
            cells = [MIDCUT, B_PROBS, bfk.EDGE_MODELS,
                     TICKET_RANKED.format(side=sp["side"], start=sp["start"], width=sp["width"], ranking=sp["ranking"],
                                          name=tk, r1=r1, r2=r2, edge=EDGE[sp["side"]])]
        text = ("Added by E38 (`solution/portfolio_final.py`), which re-weighs the 25 final selections over three label "
                "models of the noisy band edges by how well each explains the known public scores; it filled the two "
                f"slots left after P01–P20. Submitted as `{tk}.csv` from the Kaggle notebook "
                "[kacchi-aloo-final-tickets](https://www.kaggle.com/code/hosen42/kacchi-aloo-final-tickets) v2 "
                "(source: `solution/kaggle_final_kernel/`).")
        nb = new_nb(f"# Submission {tk}: {rule}", info_table(f"{tk}.csv", when, score, rule), text, cells, ref)
        entries.append(dict(label=label, method=method, score=score, when=when, nb=run(nb, tk, ref)))
    os.remove(os.path.join(OUT, "submission.csv"))

    # rank: best public score first, ties by submission time; unscored last
    entries.sort(key=lambda e: (e["score"] is None, -(e["score"] or 0), e["when"] or "9999"))
    seen, rank = set(), 0
    for e in entries:
        key = (e["score"], e["method"])
        if e["score"] is not None and key in seen and not args.all:
            print(f"   skipped {e['label']} ({score_tag(e['score'])}, {e['method']}): same score and method as an earlier one")
            continue
        seen.add(key)
        rank += 1
        name = f"{rank:02d}_{score_tag(e['score'])}_{e['label']}.ipynb"
        nbformat.write(e["nb"], os.path.join(OUT, name))
        print(name)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
