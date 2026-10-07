"""E37: is the 25-ticket final selection robust to the label model, and would a re-chosen 25 do better?

Scenario generators, each conditioned on our known public facts (A=172, B=173, C=173, 83 public ones) as in E22/E23:
  oof   : E16, B's distance model (rows independent)
  param : bootstrap of the log-space band MLE from portfolio2 (rows share the edges)
  iso   : E30 isotonic edges (rows independent)
  cp    : E31 Bayesian change-point posterior draws (rows share the cuts and side rates)
  robust: the four pooled with equal weight
Portfolios compared per generator on held-out scenarios: A+B+C, the current 25 (A, B, C, P01-P22) and a 25 re-chosen
greedily on the robust pool's training scenarios from the submitted tickets plus new rule-based candidates (uncertainty
windows on the E16 / E30 / E31 rankings, shifted bands, the E34 and E36 vectors). Competitors as in E23.

    python portfolio_robust.py --data ../data --out ../outputs
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
from win_prob import build_pool, correct  # noqa: E402

T_A, T_B, T_C = 34.48, 34.49, 40.59
T_P = 42.25      # P01-P20 entered 2026-10-07 18:14-18:17 UTC
T_NEW = 48.2     # anything entered after the 2026-10-08 00:00 UTC quota reset, P21 and P22 included
GENS = ["oof", "param", "iso", "cp"]
N_TEST, N_PUB = 400, 181

_G = {}


def _init(Pall, starts, sizes, V, n, ones):
    _G.update(Pall=Pall.astype(np.float32), starts=np.array(starts), sizes=np.array(sizes), V=V.astype(np.float32),
              n=np.array(n), ones=ones)


def _work(task, n=50000, rounds=20):
    seed, gi = task
    rng = np.random.default_rng(seed)
    g = _G
    outY, outP, raw = [], [], 0
    for _ in range(rounds):
        P = g["Pall"][g["starts"][gi] + (rng.random(n) * g["sizes"][gi]).astype(int)]
        Y = rng.random((n, N_TEST), dtype=np.float32) < P
        pub = np.zeros((n, N_TEST), bool)
        np.put_along_axis(pub, np.argpartition(rng.random((n, N_TEST), dtype=np.float32), N_PUB, 1)[:, :N_PUB], True, 1)
        keep = (Y & pub).sum(1) == g["ones"]
        Yk, pk = Y[keep], pub[keep]
        Yf, pf = Yk.astype(np.float32), pk.astype(np.float32)
        pc = (g["V"] @ (Yf * pf).T + (1 - g["V"]) @ ((1 - Yf) * pf).T).round()
        k2 = (pc == g["n"][:, None]).all(0)
        outY.append(Yk[k2]); outP.append(~pk[k2]); raw += n
    return gi, np.concatenate(outY), np.concatenate(outP), raw


def windows(A, ranking_p, d_edge, X, tag):
    order = np.lexsort((d_edge, np.abs(ranking_p - 0.5)))
    low = X < 1.5
    out = []
    for side, o in (("both", order), ("lower", order[low[order]]), ("upper", order[~low[order]])):
        for s0 in range(26):
            for w in range(1, 9):
                v = A.copy(); v[o[s0:s0 + w]] ^= 1
                out.append((dict(family="window", ranking=tag, side=side, start=s0, width=w), v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../outputs")
    ap.add_argument("--s", type=int, default=5000, help="accepted scenarios per generator and half")
    args, _ = ap.parse_known_args()
    rng = np.random.default_rng(2031)
    t0 = time.time()
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    te = kp.clean(pd.read_csv(f"{args.data}/test.csv", dtype=str), False)
    x, y, X = tr.apg.values, tr.y.values.astype(float), te.apg.values
    kp.midcut_rafiur = em.fast_midcut  # identical cuts, faster
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
    Pcp = em.changepoint_draws(x, y, X, 2000, rng)
    comps = [P16[None], Pb, P30[None], Pcp]
    starts = np.cumsum([0] + [len(c) for c in comps[:-1]]); sizes = [len(c) for c in comps]
    Pall = np.vstack(comps)
    print(f"generators ready   [{time.time() - t0:.0f}s]", flush=True)

    got = {g: [[], []] for g in range(len(GENS))}
    raw = np.zeros(len(GENS))
    seed = 5000
    with Pool(12, initializer=_init, initargs=(Pall, starts, sizes, np.array([A, B, C]), [172, 173, 173], 83)) as pool:
        while True:
            need = [g for g in range(len(GENS)) if sum(len(v) for v in got[g][0]) < 2 * args.s]
            if not need:
                break
            tasks = [(seed + i, need[i % len(need)]) for i in range(12)]
            seed += 12
            for gi, Yk, Pk, r in pool.map(_work, tasks):
                got[gi][0].append(Yk); got[gi][1].append(Pk); raw[gi] += r
            print("  accepted " + ", ".join(f"{GENS[g]} {sum(len(v) for v in got[g][0])}" for g in range(len(GENS)))
                  + f"   [{time.time() - t0:.0f}s]", flush=True)
    Ys, Ps, gen, half = [], [], [], []
    for g in range(len(GENS)):
        Yg, Pg = np.concatenate(got[g][0])[: 2 * args.s], np.concatenate(got[g][1])[: 2 * args.s]
        Ys.append(Yg); Ps.append(Pg); gen += [g] * len(Yg); half += [0] * args.s + [1] * args.s
        print(f"  {GENS[g]}: acceptance {sum(len(v) for v in got[g][0]) / raw[g]:.6f}")
    Y, priv, gen, half = np.concatenate(Ys), np.concatenate(Ps), np.array(gen), np.array(half)

    P_mix = 0.4 * P16 + 0.6 * Pb.mean(0)
    cpool, _ = build_pool(A, E0, X, P_mix, lo, hi, rng)
    near = cpool[np.minimum((cpool != A).sum(1), (cpool != E0).sum(1)) <= 4]
    CS = competitor_scores(correct(near, Y, ~priv), correct(near, Y, priv), rng)
    ct = np.array([t for _, _, t, _ in TEAMS])
    print(f"competitors ready   [{time.time() - t0:.0f}s]", flush=True)

    # ---- tickets: submitted ones first, then new rule-based candidates (deduplicated)
    names, vecs, times, specs = ["A", "B", "C"], [A, B, C], [T_A, T_B, T_C], [None, None, None]
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
    seen = {v.tobytes() for v in vecs}
    for sp, v in cands:
        if v.tobytes() not in seen:
            seen.add(v.tobytes())
            names.append(f"N{len(names) - 24:04d}"); vecs.append(v); times.append(T_NEW); specs.append(sp)
    V, T = np.array(vecs), np.array(times)
    Q = correct(V, Y, priv)
    print(f"tickets {len(V)} (25 existing + {len(V) - 25} new)   [{time.time() - t0:.0f}s]", flush=True)

    def stats(idx, m):
        q = Q[np.array(idx)][:, m]
        s = q.max(0)
        t = np.where(q == s[None], T[np.array(idx)][:, None], np.inf).min(0)
        return {R: p_rank(s, t, *thresholds(CS[:, m], ct, R)).mean() for R in (1, 3, 5)}, s.mean()

    trm = half == 0
    th = {R: thresholds(CS[:, trm], ct, R) for R in (1, 5)}
    cur_s, cur_t, chosen = np.full(trm.sum(), -1, np.int16), np.full(trm.sum(), np.inf), []
    Qtr = Q[:, trm]
    for _ in range(25):
        ns = np.maximum(cur_s[None], Qtr)
        nt = np.where(Qtr > cur_s[None], T[:, None], np.where(Qtr == cur_s[None], np.minimum(cur_t[None], T[:, None]), cur_t[None]))
        obj = sum(0.5 * p_rank(ns, nt, th[R][0][None], th[R][1][None]).mean(1) for R in (1, 5))
        obj[chosen] = -1
        j = int(obj.argmax()); chosen.append(j)
        cur_t = nt[j]; cur_s = ns[j]

    ports = {"A+B+C": [0, 1, 2], "current 25": list(range(25)), "re-chosen 25": chosen}
    rows = []
    for pname, idx in ports.items():
        for gname in GENS + ["robust"]:
            m = (half == 1) & ((gen == GENS.index(gname)) if gname != "robust" else True)
            pr, es = stats(idx, m)
            rows.append(dict(portfolio=pname, generator=gname, P_top1=pr[1], P_top3=pr[3], P_top5=pr[5], E_best_private=es))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.pivot(index="portfolio", columns="generator", values="P_top1").round(4).to_string())
    print(res.pivot(index="portfolio", columns="generator", values="P_top5").round(4).to_string())

    kept = [names[j] for j in chosen if j < 25]
    dropped = [names[j] for j in range(25) if j not in chosen]
    added = [(names[j], specs[j], int((V[j] != A).sum())) for j in chosen if j >= 25]
    print(f"\nre-chosen 25 keeps {len(kept)} submitted: {' '.join(kept)}")
    print(f"drops: {' '.join(dropped) or '-'}")
    for n_, sp, d in added:
        print(f"adds {n_}: {sp} ({d} rows vs A): " + " ".join(f"{te.wedding_id[i]}:{A[i]}>{V[names.index(n_)][i]}" for i in np.where(V[names.index(n_)] != A)[0]))
    od = f"{args.out}/portfolio_robust"
    os.makedirs(od, exist_ok=True)
    res.to_csv(f"{od}/robustness.csv", index=False)
    with open(f"{od}/rechosen.json", "w") as f:
        json.dump(dict(kept=kept, dropped=dropped, added=[dict(name=n_, spec=sp) for n_, sp, _ in added]), f, indent=1)
    for n_, _, _ in added:
        pd.DataFrame({"wedding_id": te.wedding_id, "went_back_for_seconds": V[names.index(n_)]}).to_csv(f"{od}/{n_}.csv", index=False)
    print(f"wrote {od}   done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
