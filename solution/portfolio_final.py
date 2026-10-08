"""E38: minimal-change improvement of the 25 final selections under an evidence-weighted label model.

E37 showed the current 25 is strong under the generators it was built on (E16 OOF, parametric band) and weaker under
E30 isotonic / E31 change-point labels. Here the generators are weighted by their evidence, i.e. how likely each
makes our known public facts (acceptance rate of the conditioning A=172, B=173, C=173, 83 ones; E31 had ~20x less
evidence than E16 and ~2% weight, so it is left out). Starting from the 23 submitted tickets (A, B, C, P01-P20):
  1. fill the 2 open slots (P21/P22 are not submitted yet) with the best tickets from P21, P22 and new candidates;
  2. then swap a submitted ticket for a new candidate while the training objective improves by > 0.3 points.
Objective: 0.5 P(private rank 1) + 0.5 P(private top 5), evidence-weighted; reported on held-out scenarios.

    python portfolio_final.py --data ../data --out ../outputs [--max-swaps 6]
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
from portfolio25 import TEAMS, competitor_scores, p_rank, thresholds  # noqa: E402
from portfolio_robust import T_A, T_B, T_C, T_NEW, T_P, _init, _work, windows  # noqa: E402
from win_prob import build_pool, correct  # noqa: E402

GENS = ["oof", "param", "iso"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../outputs")
    ap.add_argument("--s", type=int, default=6000, help="accepted scenarios per generator and half")
    ap.add_argument("--max-swaps", type=int, default=6)
    ap.add_argument("--start-final", action="store_true",
                    help="start from the submitted final 25 (A, B, C, P01-P20, Q01, Q02) and only search swaps")
    ap.add_argument("--extend", type=int, default=0, help="after the final portfolio, greedily rank this many further tickets")
    ap.add_argument("--extra", default=None, help="directory of extra candidate CSVs (e.g. ../outputs/edge_models2)")
    args, _ = ap.parse_known_args()
    rng = np.random.default_rng(2032)
    t0 = time.time()
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    te = kp.clean(pd.read_csv(f"{args.data}/test.csv", dtype=str), False)
    x, y, X = tr.apg.values, tr.y.values.astype(float), te.apg.values
    kp.midcut_rafiur = em.fast_midcut
    P16, info = kp.sharp_band_probabilities(tr, te, repeats=10)
    lo, hi = info["cuts"]
    A = kp.band(X, lo, hi, closed=True).astype(int)
    rd = lambda f: pd.read_csv(f"{args.out}/{f}").went_back_for_seconds.values.astype(int)
    B, C = rd("sub_B_hedge.csv"), rd("sub_C_band_1p00_1p975.csv")
    PT = [rd(f"portfolio25/P{k:02d}.csv") for k in range(1, 23)]
    E0 = kp.band(X, *kp.accopt_foysal(x, y)).astype(int)
    P30, P31 = em.m_E30(tr, te)[0], em.m_E31(tr, te)[0]
    E34 = (em.m_E34(tr, te)[0] > 0.5).astype(int)
    E36 = (em.m_E36(tr, te)[0] > 0.5).astype(int)
    fits = {f: p2.fit(x, y, f) for f in ("laplace", "logistic")}
    aic = {f: 6 + 2 * v[1] for f, v in fits.items()}
    wl = 1 / (1 + np.exp(-(aic["logistic"] - aic["laplace"]) / 2))
    Pb = []
    for _ in range(200):
        f = "laplace" if rng.random() < wl else "logistic"
        ii = rng.integers(len(x), size=len(x))
        Pb.append(p2.p_band(X, p2.fit(x[ii], y[ii], f, th0=fits[f][0])[0], f))
    Pb = np.array(Pb)
    comps = [P16[None], Pb, P30[None]]
    starts = np.cumsum([0] + [len(c) for c in comps[:-1]]); sizes = [len(c) for c in comps]

    got = {g: [[], []] for g in range(len(GENS))}
    raw = np.zeros(len(GENS)); seed = 9000
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
    print("evidence weights: " + ", ".join(f"{g} {w:.3f}" for g, w in zip(GENS, evid)) + f"   [{time.time() - t0:.0f}s]", flush=True)
    Ys, Ps, gen, half = [], [], [], []
    for g in range(len(GENS)):
        Ys.append(np.concatenate(got[g][0])[: 2 * args.s]); Ps.append(np.concatenate(got[g][1])[: 2 * args.s])
        gen += [g] * 2 * args.s; half += [0] * args.s + [1] * args.s
    Y, priv, gen, half = np.concatenate(Ys), np.concatenate(Ps), np.array(gen), np.array(half)
    w = evid[gen] / args.s           # per-scenario weight; each half sums to 1

    P_mix = 0.4 * P16 + 0.6 * Pb.mean(0)
    cpool, _ = build_pool(A, E0, X, P_mix, lo, hi, rng)
    near = cpool[np.minimum((cpool != A).sum(1), (cpool != E0).sum(1)) <= 4]
    CS = competitor_scores(correct(near, Y, ~priv), correct(near, Y, priv), rng)
    ct = np.array([t for _, _, t, _ in TEAMS])

    names, vecs, times, specs = ["A", "B", "C"], [A, B, C], [T_A, T_B, T_C], [None] * 3
    for k, v in enumerate(PT, 1):
        names.append(f"P{k:02d}"); vecs.append(v); times.append(T_P if k <= 20 else T_NEW); specs.append(None)
    d_edge = np.minimum(abs(X - lo), abs(X - hi))
    cands = windows(A, P16, d_edge, X, "E16") + windows(A, P30, d_edge, X, "E30") + windows(A, P31, d_edge, X, "E31")
    xs = np.unique(X); mids = (xs[:-1] + xs[1:]) / 2
    for cl in mids[(mids > 0.94) & (mids < 1.05)]:
        for ch in mids[(mids > 1.94) & (mids < 2.07)]:
            l5, h5 = round(float(cl), 5), round(float(ch), 5)
            cands.append((dict(family="band", lo=l5, hi=h5), ((X >= l5) & (X <= h5)).astype(int)))
    cands += [(dict(family="E34"), E34), (dict(family="E36"), E36)]
    if args.extra:
        for f in sorted(os.listdir(args.extra)):
            if f.endswith(".csv") and f != "test_probabilities.csv" and f != "cv_metrics.csv":
                cands.append((dict(family="extra", file=f), pd.read_csv(f"{args.extra}/{f}").went_back_for_seconds.values.astype(int)))
    seen = {v.tobytes() for v in vecs}
    for sp, v in cands:
        if v.tobytes() not in seen:
            seen.add(v.tobytes())
            names.append(f"N{len(names) - 24:04d}"); vecs.append(v); times.append(T_NEW); specs.append(sp)
    V, T = np.array(vecs), np.array(times)
    Q = correct(V, Y, priv)
    print(f"tickets {len(V)}   [{time.time() - t0:.0f}s]", flush=True)

    th = {(h, R): thresholds(CS[:, half == h], ct, R) for h in (0, 1) for R in (1, 3, 5)}

    def state(idx, h):
        q = Q[np.array(idx)][:, half == h]
        s = q.max(0)
        return s, np.where(q == s[None], T[np.array(idx)][:, None], np.inf).min(0)

    def objective_add(s, t, h, cand):
        """Weighted objective of (portfolio state s, t) + each ticket in cand, vectorised over cand."""
        qc = Q[cand][:, half == h]
        ns = np.maximum(s[None], qc)
        nt = np.where(qc > s[None], T[cand][:, None], np.where(qc == s[None], np.minimum(t[None], T[cand][:, None]), t[None]))
        ww = w[half == h]
        return sum(0.5 * (p_rank(ns, nt, th[(h, R)][0][None], th[(h, R)][1][None]) * ww[None]).sum(1) for R in (1, 5))

    def report(idx, h=1):
        s, t = state(idx, h)
        out = {}
        for R in (1, 3, 5):
            hit = p_rank(s, t, *th[(h, R)])
            out[f"P_top{R}"] = float((hit * w[half == h]).sum())
            for g, gname in enumerate(GENS):
                m = gen[half == h] == g
                out[f"P_top{R}_{gname}"] = float(hit[m].mean())
        return out

    fixed = list(range(23))
    port = fixed.copy()
    pool_idx = np.array([j for j in range(len(V)) if j >= 23])
    if args.start_final:                                    # Q01/Q02 are already submitted
        for q in ("N0567", "N0610"):
            qv = pd.read_csv(f"{args.out}/portfolio_final/{q}.csv").went_back_for_seconds.values.astype(int)
            port.append(int(np.where((V == qv).all(1))[0][0]))
    for _ in range(0 if args.start_final else 2):           # 1. fill the two open slots
        s, t = state(port, 0)
        obj = objective_add(s, t, 0, pool_idx)
        obj[np.isin(pool_idx, port)] = -1
        port.append(int(pool_idx[obj.argmax()]))
    fill = port.copy()
    swaps = []
    for _ in range(args.max_swaps):                         # 2. swaps
        s_all, t_all = state(port, 0)
        base = (objective_add(s_all, t_all, 0, np.array([port[0]])))[0]
        best = (0.003, None, None)
        for i in port:
            rest = [k for k in port if k != i]
            s, t = state(rest, 0)
            cand = np.array([j for j in pool_idx if j not in port])
            obj = objective_add(s, t, 0, cand)
            if obj.max() - base > best[0]:
                best = (obj.max() - base, i, int(cand[obj.argmax()]))
        if best[1] is None:
            break
        port = [k for k in port if k != best[1]] + [best[2]]
        swaps.append((names[best[1]], names[best[2]], best[0]))
        print(f"  swap out {names[best[1]]} for {names[best[2]]}: +{best[0]:.4f} (train)", flush=True)

    extend = []
    for _ in range(args.extend):                            # 3. further tickets in order of marginal value
        s, t = state(port + extend, 0)
        cand = np.array([j for j in pool_idx if j not in port + extend])
        obj = objective_add(s, t, 0, cand)
        extend.append(int(cand[obj.argmax()]))
        print(f"  extend {names[extend[-1]]}: objective {obj.max():.4f}", flush=True)

    rows = []
    for pname, idx in (("A+B+C", [0, 1, 2]), ("current 25 (A,B,C,P01-P22)", list(range(25))),
                       ("23 submitted + 2 best", fill), ("after swaps", port)):
        rows.append(dict(portfolio=pname, **report(idx)))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))
    new = [j for j in port if j >= 25 or j in (23, 24)]
    print("\nfinal selection: " + " ".join(names[j] for j in sorted(port, key=lambda j: (j >= 25, j))))
    for j in new:
        v = V[j]
        print(f"  {names[j]}: {specs[j] or 'portfolio ticket'} ({int((v != A).sum())} rows vs A): "
              + " ".join(f"{te.wedding_id[i]}:{A[i]}>{v[i]}" for i in np.where(v != A)[0]))
    od = f"{args.out}/portfolio_final" + ("_check" if args.start_final else "")
    os.makedirs(od, exist_ok=True)
    res.to_csv(f"{od}/comparison.csv", index=False)
    with open(f"{od}/selection.json", "w") as f:
        json.dump(dict(evidence=dict(zip(GENS, map(float, evid))), selection=[names[j] for j in port],
                       new=[dict(name=names[j], spec=specs[j]) for j in new], swaps=swaps,
                       extend=[dict(name=names[j], spec=specs[j]) for j in extend]), f, indent=1)
    for j in new + extend:
        pd.DataFrame({"wedding_id": te.wedding_id, "went_back_for_seconds": V[j]}).to_csv(f"{od}/{names[j]}.csv", index=False)
    print(f"wrote {od}   done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
