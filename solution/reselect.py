"""E42: re-select the 25 finals from every submitted ticket, aiming at private rank 1 and 2, with updated competitors.

By 2026-10-09 the public LB shows Rafiur Rahman at 1.00000 (41 entries) and FOYSAL at 0.99447 (35 entries): public
scores that no model reaches (the band's out-of-fold ceiling is ~95%), i.e. public labels learned from the leaderboard.
That says nothing about their private rows, so each is modelled as before (near-edge variants of the shared band
consistent with their last model-level public score: Rafiur 173, FOYSAL 175), but with 25 tickets each (the most
they can select). Scenarios and evidence weights follow E38 (E16 OOF / parametric band / E30 isotonic).

Candidates are only already-submitted, distinct vectors: A, B, C, P01-P20, Q01, Q02, R05-R11, S01-S19 (no new
submission is needed). Starting from the current 25, swap a selected ticket for an unselected one while the
training objective 0.5 P(rank 1) + 0.5 P(rank <= 2) improves by > 0.2 points; report on held-out scenarios, also with
the old competitor model for comparison.

    python reselect.py --data ../data --out ../outputs
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import edge_models as em  # noqa: E402
import kacchi_pipeline as kp  # noqa: E402
import portfolio2 as p2  # noqa: E402
import portfolio25  # noqa: E402
from portfolio25 import competitor_scores, p_rank, thresholds  # noqa: E402
from portfolio_robust import _init, _work  # noqa: E402
from win_prob import build_pool, correct  # noqa: E402

GENS = ["oof", "param", "iso"]
OLD_TEAMS = list(portfolio25.TEAMS)
NEW_TEAMS = [("Rafiur Rahman", 173, 40.25, 25), ("FOYSAL", 175, 27.38, 25), ("Dadhichi Sarker", 175, 39.74, 15),
             ("রোদশী", 174, 31.86, 16), ("Str1k3rFl0", 174, 24.01, 6), ("Raiyana Rufaida", 174, 31.77, 4),
             ("amar_nam_adnan", 173, 18.21, 4), ("Tasnimul Ferdous", 173, 17.29, 4), ("Md. Najmul Hasan Shihab", 173, 19.50, 5),
             ("el mencho", 173, 25.13, 4), ("sky_is_green", 173, 39.79, 3), ("Adib Rahman", 173, 40.66, 3),
             ("Md Abdul Al Hasib", 172, 39.66, 3)]
CURRENT = ["A", "B", "C"] + [f"P{k:02d}" for k in range(1, 21)] + ["Q01", "Q02"]


def submitted_tickets(out):
    """name -> (vector, entry time in hours since 2026-10-06 00:00 UTC) for every distinct submitted ticket."""
    subs = pd.read_csv(f"{out}/kaggle_submissions.csv", parse_dates=["date"])
    t0 = pd.Timestamp("2026-10-06")
    when = lambda f: (subs[subs.fileName == f].date.min() - t0).total_seconds() / 3600
    rd = lambda f: pd.read_csv(f"{out}/{f}").went_back_for_seconds.values.astype(int)
    tk = {"A": (rd("sub_A_primary_sharpband.csv"), when("sub_A_primary_sharpband.csv")),
          "B": (rd("sub_B_hedge.csv"), when("sub_B_hedge.csv")),
          "C": (rd("sub_C_band_1p00_1p975.csv"), when("sub_C_band_1p00_1p975.csv"))}
    for k in range(1, 21):
        tk[f"P{k:02d}"] = (rd(f"portfolio25/P{k:02d}.csv"), when(f"P{k:02d}.csv"))
    tk["Q01"] = (rd("portfolio_final/N0567.csv"), when("Q01.csv"))
    tk["Q02"] = (rd("portfolio_final/N0610.csv"), when("Q02.csv"))
    for m in (5, 7, 9, 11):
        tk[f"R{m:02d}"] = (rd(f"edge_models2/E39_hedge{m}.csv"), when(f"R{m:02d}.csv"))
    for s, n in json.load(open(f"{out}/portfolio_final_check/s_names.json")).items():
        tk[s] = (rd(f"portfolio_final_check/{n}.csv"), when(f"{s}.csv"))
    seen = {}
    for name, (v, _) in tk.items():
        assert v.tobytes() not in seen, f"{name} duplicates {seen.get(v.tobytes())}"
        seen[v.tobytes()] = name
    return tk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../outputs")
    ap.add_argument("--s", type=int, default=6000, help="accepted scenarios per generator and half")
    ap.add_argument("--pin", default="", help="comma-separated tickets that may not be swapped out (e.g. A)")
    args, _ = ap.parse_known_args()
    rng = np.random.default_rng(2033)
    t0 = time.time()
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    te = kp.clean(pd.read_csv(f"{args.data}/test.csv", dtype=str), False)
    x, y, X = tr.apg.values, tr.y.values.astype(float), te.apg.values
    kp.midcut_rafiur = em.fast_midcut
    P16, info = kp.sharp_band_probabilities(tr, te, repeats=10)
    lo, hi = info["cuts"]
    A = kp.band(X, lo, hi, closed=True).astype(int)
    E0 = kp.band(X, *kp.accopt_foysal(x, y)).astype(int)
    P30 = em.m_E30(tr, te)[0]
    fits = {f: p2.fit(x, y, f) for f in ("laplace", "logistic")}
    aic = {f: 6 + 2 * v[1] for f, v in fits.items()}
    wl = 1 / (1 + np.exp(-(aic["logistic"] - aic["laplace"]) / 2))
    Pb = []
    for _ in range(200):
        f = "laplace" if rng.random() < wl else "logistic"
        ii = rng.integers(len(x), size=len(x))
        Pb.append(p2.p_band(X, p2.fit(x[ii], y[ii], f, th0=fits[f][0])[0], f))
    Pb = np.array(Pb)

    tk = submitted_tickets(args.out)
    assert (tk["A"][0] == A).all()
    names = list(tk)
    V = np.array([tk[n][0] for n in names]); T = np.array([tk[n][1] for n in names])
    B, C = tk["B"][0], tk["C"][0]
    print(f"{len(names)} distinct submitted tickets   [{time.time() - t0:.0f}s]", flush=True)

    comps = [P16[None], Pb, P30[None]]
    starts = np.cumsum([0] + [len(c) for c in comps[:-1]]); sizes = [len(c) for c in comps]
    got = {g: [[], []] for g in range(len(GENS))}
    raw = np.zeros(len(GENS)); seed = 12000
    with Pool(12, initializer=_init, initargs=(np.vstack(comps), starts, sizes, np.array([A, B, C]), [172, 173, 173], 83)) as pool:
        while True:
            need = [g for g in range(len(GENS)) if sum(len(v) for v in got[g][0]) < 2 * args.s]
            if not need:
                break
            for gi, Yk, Pk, r in pool.map(_work, [(seed + i, need[i % len(need)]) for i in range(12)]):
                got[gi][0].append(Yk); got[gi][1].append(Pk); raw[gi] += r
            seed += 12
    acc = np.array([sum(len(v) for v in got[g][0]) / raw[g] for g in range(len(GENS))])
    evid = acc / acc.sum()
    Ys, Ps, gen, half = [], [], [], []
    for g in range(len(GENS)):
        Ys.append(np.concatenate(got[g][0])[: 2 * args.s]); Ps.append(np.concatenate(got[g][1])[: 2 * args.s])
        gen += [g] * 2 * args.s; half += [0] * args.s + [1] * args.s
    Y, priv, gen, half = np.concatenate(Ys), np.concatenate(Ps), np.array(gen), np.array(half)
    w = evid[gen] / args.s
    print("evidence weights: " + ", ".join(f"{g} {v:.3f}" for g, v in zip(GENS, evid)) + f"   [{time.time() - t0:.0f}s]", flush=True)

    P_mix = 0.4 * P16 + 0.6 * Pb.mean(0)
    cpool, _ = build_pool(A, E0, X, P_mix, lo, hi, rng)
    near = cpool[np.minimum((cpool != A).sum(1), (cpool != E0).sum(1)) <= 4]
    npub, npriv = correct(near, Y, ~priv), correct(near, Y, priv)
    models = {}
    for label, teams in (("new", NEW_TEAMS), ("old", OLD_TEAMS)):
        portfolio25.TEAMS[:] = teams
        models[label] = (competitor_scores(npub, npriv, rng), np.array([t for _, _, t, _ in teams]))
    portfolio25.TEAMS[:] = OLD_TEAMS
    Q = correct(V, Y, priv)
    TH = {(label, h, R): thresholds(models[label][0][:, half == h], models[label][1], R)
          for label in models for h in (0, 1) for R in (1, 2, 3, 5)}

    def hits(idx, h, label):
        q = Q[np.array(idx)][:, half == h]
        s = q.max(0)
        t = np.where(q == s[None], T[np.array(idx)][:, None], np.inf).min(0)
        return {R: p_rank(s, t, *TH[(label, h, R)]) for R in (1, 2, 3, 5)}

    def objective(idx, h=0, label="new"):
        hh = hits(idx, h, label)
        return float(((0.5 * hh[1] + 0.5 * hh[2]) * w[half == h]).sum())

    port = [names.index(n) for n in CURRENT]
    swaps = []
    while True:
        base = objective(port)
        best = (0.002, None, None)
        for i in port:
            if names[i] in args.pin.split(","):
                continue
            for j in range(len(names)):
                if j in port:
                    continue
                cand = [k for k in port if k != i] + [j]
                gain = objective(cand) - base
                if gain > best[0]:
                    best = (gain, i, j)
        if best[1] is None:
            break
        port = [k for k in port if k != best[1]] + [best[2]]
        swaps.append((names[best[1]], names[best[2]], round(best[0], 4)))
        print(f"  swap {names[best[1]]} -> {names[best[2]]}: +{best[0]:.4f} (train)", flush=True)

    rows = []
    for pname, idx in (("current 25", [names.index(n) for n in CURRENT]), ("re-selected 25", port)):
        for label in ("new", "old"):
            hh = hits(idx, 1, label)
            rows.append(dict(portfolio=pname, competitors=label,
                             **{f"P_top{R}": float((hh[R] * w[half == 1]).sum()) for R in (1, 2, 3, 5)}))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    final = sorted((names[k] for k in port), key=names.index)
    print("\nre-selected 25: " + " ".join(final))
    od = f"{args.out}/reselect" + (f"_pin_{args.pin.replace(',', '_')}" if args.pin else "")
    os.makedirs(od, exist_ok=True)
    res.to_csv(f"{od}/comparison.csv", index=False)
    with open(f"{od}/selection.json", "w") as f:
        json.dump(dict(final=final, swaps=swaps, evidence=dict(zip(GENS, map(float, evid)))), f, indent=1)
    print(f"wrote {od}   done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
