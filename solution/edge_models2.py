"""E39-E41: the remaining E30-family ideas, under the same frozen protocol as edge_models.py (10x5 folds, seed 2026).

  E39  bagged E30: isotonic edges refit on bootstrap samples of the training fold, probabilities averaged
  E40  E30 window ensemble: isotonic edges averaged over lower windows {0.85-1.15, 0.88-1.12, 0.90-1.10} x
       upper windows {1.80-2.20, 1.85-2.15, 1.90-2.10}
  E41  OOF-weighted consensus of E16 / E30 / E31: simplex weights (step 0.1) chosen on inner OOF edge log-loss
       inside each outer training fold, so the outer score is honest

Test outputs: MAP vectors, and a batch of hedges fixed before any public score is seen: the bagged-E30 ranking with
the 5 / 7 / 9 / 11 most uncertain rows flipped (Q01 is the single-fit E30 ranking with 7).

    python edge_models2.py --data ../data --out ../outputs
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import edge_models as em  # noqa: E402
import kacchi_pipeline as kp  # noqa: E402

LO_WINS = [(0.85, 1.15), (0.88, 1.12), (0.90, 1.10)]
HI_WINS = [(1.80, 2.20), (1.85, 2.15), (1.90, 2.10)]
GRID = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=3) if abs(sum(w) - 1) < 1e-9]


def iso(r, y, x, lo_win=(0.85, 1.15), hi_win=(1.80, 2.20)):
    return em.m_E30(pd.DataFrame({"apg": r, "y": y}), pd.DataFrame({"apg": x}), lo_win=lo_win, hi_win=hi_win)[0]


def m_E39(t, *aps, B=200, seed=0):
    rng = np.random.default_rng(seed)
    r, y = t.apg.values, t.y.values.astype(float)
    xs = [a.apg.values for a in aps]
    acc = [np.zeros(len(x)) for x in xs]
    for _ in range(B):
        ii = rng.integers(len(r), size=len(r))
        for k, x in enumerate(xs):
            acc[k] += iso(r[ii], y[ii], x)
    return [a / B for a in acc]


def m_E40(t, *aps):
    r, y = t.apg.values, t.y.values.astype(float)
    return [np.mean([iso(r, y, a.apg.values, lw, hw) for lw in LO_WINS for hw in HI_WINS], 0) for a in aps]


def edge_logloss(p, y, d):
    m = d < 0.1
    pc = np.clip(p[m], 0.005, 0.995)
    return float(-(y[m] * np.log(pc) + (1 - y[m]) * np.log(1 - pc)).mean())


def m_E41(t, *aps, repeats=2):
    """Weights from inner OOF predictions of E16/E30/E31 on the training fold, then refit all three on the fold."""
    t = t.reset_index(drop=True)
    r, y = t.apg.values, t.y.values.astype(float)
    F = kp.make_folds(t, repeats, seed=77)
    oof = np.zeros((3, repeats, len(t)))
    for rep in range(repeats):
        for f in range(5):
            v = F[:, rep] == f
            for k, fn in enumerate((em.m_E16, em.m_E30, em.m_E31)):
                oof[k, rep, v] = fn(t[~v], t[v])[0]
    lo, hi = em.fast_midcut(r, y)
    d = np.minimum(abs(r - lo), abs(r - hi))
    w = min(GRID, key=lambda w: np.mean([edge_logloss(np.tensordot(w, oof[:, rep], 1), y, d) for rep in range(repeats)]))
    m_E41.weights.append(w)
    full = [[fn(t, a)[0] for fn in (em.m_E16, em.m_E30, em.m_E31)] for a in aps]
    return [np.tensordot(w, np.array(p), 1) for p in full]


m_E41.weights = []

MODELS = {"A (midcut band)": em.m_A, "E16 B's distance model": em.m_E16, "E30 isotonic edges": em.m_E30,
          "E39 bagged E30": m_E39, "E40 E30 window ensemble": m_E40, "E41 OOF-weighted E16/E30/E31": m_E41}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../outputs")
    args, _ = ap.parse_known_args()
    t0 = time.time()
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    te = kp.clean(pd.read_csv(f"{args.data}/test.csv", dtype=str), False)
    r, y, X = tr.apg.values, tr.y.values.astype(float), te.apg.values
    kp.midcut_rafiur = em.fast_midcut
    lo_A, hi_A = em.fast_midcut(r, y)
    F = kp.make_folds(tr, 10)
    d_ref = np.minimum(abs(r - lo_A), abs(r - hi_A))
    rows, folds = [], {}
    for name, fn in MODELS.items():
        oof, accs = kp.run_cv(fn, tr, F)
        folds[name] = accs
        m = d_ref < 0.1
        pc = np.clip(oof[m], 0.005, 0.995)
        rows.append(dict(model=name, cv_acc=accs.mean(), edge_acc=((oof[m] > 0.5) == (y[m, None] == 1)).mean(),
                         edge_logloss=float(-(y[m, None] * np.log(pc) + (1 - y[m, None]) * np.log(1 - pc)).mean()),
                         edge_brier=float(((oof[m] - y[m, None]) ** 2).mean()), size_stress=kp.size_stress(fn, tr)))
        print(f"  {name:30s} cv {accs.mean():.4f}   [{time.time() - t0:.0f}s]", flush=True)
    n_tr, n_te = len(tr) * 4 / 5, len(tr) / 5
    for row in rows:
        dm, _, p = kp.corrected_ttest(folds[row["model"]], folds["A (midcut band)"], n_tr, n_te)
        row.update(delta_vs_A=dm, p_corrected=p)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    W = np.array(m_E41.weights[:50])
    print(f"\nE41 weights over outer folds (E16, E30, E31): mean {W.mean(0).round(2)}, most common "
          f"{pd.Series([tuple(np.round(w, 1)) for w in W]).value_counts().index[0]}")

    rd = lambda f: pd.read_csv(f"{args.out}/{f}").went_back_for_seconds.values.astype(int)
    A = rd("sub_A_primary_sharpband.csv")
    Q01, P02 = rd("portfolio_final/N0567.csv"), rd("portfolio25/P02.csv")
    tests = {"E39": m_E39(tr, te, B=1000)[0], "E40": m_E40(tr, te)[0], "E41": m_E41(tr, te)[0]}
    print(f"full-data E41 weights (E16, E30, E31): {np.round(m_E41.weights[-1], 1)}")
    od = f"{args.out}/edge_models2"
    os.makedirs(od, exist_ok=True)
    d_edge = np.minimum(abs(X - lo_A), abs(X - hi_A))
    cands = {f"{k}_MAP": (p > 0.5).astype(int) for k, p in tests.items()}
    order = np.lexsort((d_edge, np.abs(tests["E39"] - 0.5)))
    for m in (5, 7, 9, 11):
        v = A.copy(); v[order[:m]] ^= 1
        cands[f"E39_hedge{m}"] = v
    trows = []
    for name, v in cands.items():
        trows.append(dict(candidate=name, ones=int(v.sum()), ham_A=int((v != A).sum()), ham_Q01=int((v != Q01).sum()),
                          ham_P02=int((v != P02).sum()),
                          flips_vs_A=" ".join(f"{te.wedding_id[i]}:{A[i]}>{v[i]}" for i in np.where(v != A)[0])))
        pd.DataFrame({"wedding_id": te.wedding_id, "went_back_for_seconds": v}).to_csv(f"{od}/{name}.csv", index=False)
    print(pd.DataFrame(trows).to_string(index=False, max_colwidth=110))
    print("E39 bagged probabilities of the 11 most uncertain rows: "
          + " ".join(f"{te.wedding_id[i]}={tests['E39'][i]:.3f}" for i in order[:11]))
    pd.DataFrame(rows).to_csv(f"{od}/cv_metrics.csv", index=False)
    pd.DataFrame({"wedding_id": te.wedding_id, "apg": X, **tests}).to_csv(f"{od}/test_probabilities.csv", index=False)
    print(f"wrote {od}   done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
