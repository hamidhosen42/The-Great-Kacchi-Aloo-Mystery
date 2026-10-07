"""E20: best-of-K final-selection portfolio under EDGE UNCERTAINTY (extends E19, portfolio.py).

E19 drew test labels independently from the out-of-fold sharp-band probabilities. Under that model the rows near one
edge are independent, but in reality they share the same unknown edge: if the true lower edge is 1.0 (the generator
agent's [1,2] story) then W1119, W1157 and W1179 are all 0 together. A portfolio that knows this can cover such
joint scenarios with band-shift vectors instead of scattered single flips.

Scenario model (mixture, all fitted on train only):
  * 'oof'   : independent y_i ~ Bernoulli(P_oof_i) from kp.sharp_band_probabilities (E16);
  * 'param' : draw (L, H, b) from a bootstrap of the log-space band MLE  P(y=1|x) = F((ln H-ln x)/b) - F((ln L-ln x)/b)
              with F = Laplace or logistic CDF (family picked by Akaike weight), then y_i ~ Bernoulli(P(x_i)).
  * optional --condition: keep only scenarios consistent with the already-known public facts
    (A scored 172/181, B scored 173/181, the all-zeros benchmark scored 98/181 -> 83 ones in the public set).
Candidates: random uncertainty-weighted flips of A, all single/pair flips among the top 20 uncertain rows, and
closed bands with shifted cuts on a grid around both edges. Greedy max of E[max_k private correct] on TRAINING
scenarios, reported on independent EVALUATION scenarios (and per component, for robustness).
No labels or leaderboard probing are used to build vectors (Rule 4b); only scores of genuine earlier submissions.

    python portfolio2.py --data ../data --out ../outputs --k 25 [--condition] [--write]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, ".")
import kacchi_pipeline as kp  # noqa: E402

N_TEST, N_PUB = 400, 181


def cdf(z, fam):
    if fam == "laplace":
        return np.where(z < 0, 0.5 * np.exp(np.minimum(z, 0)), 1 - 0.5 * np.exp(-np.maximum(z, 0)))
    return 1 / (1 + np.exp(-z))


def p_band(x, th, fam):
    lL, lH, lb = th
    b, lx = np.exp(lb), np.log(x)
    return np.clip(cdf((lH - lx) / b, fam) - cdf((lL - lx) / b, fam), 1e-6, 1 - 1e-6)


def fit(x, y, fam, th0=(np.log(0.985), np.log(2.0), np.log(0.04))):
    nll = lambda th: -(y * np.log(p_band(x, th, fam)) + (1 - y) * np.log(1 - p_band(x, th, fam))).sum()
    r = minimize(nll, th0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-6, maxiter=2000))
    return r.x, r.fun


def draw_scenarios(S, P_oof, X_te, boot, w_oof, rng, cond=None, A=None, B=None, chunk=40000, max_raw=40_000_000):
    """Return Y (S,400) bool, priv (S,400) bool, comp (S,) 0=oof 1=param. With cond, rejection-sample on public facts."""
    Ys, Ps, Cs, got, raw = [], [], [], 0, 0
    fams, thetas = boot
    Pb = np.array([p_band(X_te, th, f) for f, th in zip(fams, thetas)])        # (n_boot, 400)
    while got < S:
        n = chunk
        comp = rng.random(n) >= w_oof
        P = np.where(comp[:, None], Pb[rng.integers(len(Pb), size=n)], P_oof[None, :])
        Y = rng.random((n, N_TEST)) < P
        pub = np.argsort(rng.random((n, N_TEST)), 1)[:, :N_PUB]
        priv = np.ones((n, N_TEST), bool)
        np.put_along_axis(priv, pub, False, 1)
        keep = np.ones(n, bool)
        if cond is not None:
            keep &= ((Y == A[None, :]) & ~priv).sum(1) == cond["A"]
            keep &= ((Y == B[None, :]) & ~priv).sum(1) == cond["B"]
            keep &= (Y & ~priv).sum(1) == cond["ones"]
        Ys.append(Y[keep]); Ps.append(priv[keep]); Cs.append(comp[keep]); got += keep.sum(); raw += n
        if raw >= max_raw:
            break
    Y, priv, comp = np.concatenate(Ys)[:S], np.concatenate(Ps)[:S], np.concatenate(Cs)[:S]
    return Y, priv, comp, got / raw


def priv_correct(V, Y, priv):
    """(n_vec, S) private-correct counts via two matmuls."""
    V = V.astype(np.float32)
    PY, PN = (priv & Y).astype(np.float32), (priv & ~Y).astype(np.float32)
    return (V @ PY.T + (1 - V) @ PN.T).round().astype(np.int16)


def greedy(pool, pool_tr, seed_idx, K):
    chosen = list(seed_idx)
    cur = pool_tr[chosen].max(0)
    while len(chosen) < K:
        j = int(np.argmax(np.maximum(pool_tr, cur[None, :]).mean(1)))
        chosen.append(j); cur = np.maximum(cur, pool_tr[j])
    return chosen


def report(name, V, Y, priv, comp):
    pc = priv_correct(V, Y, priv)
    best = np.maximum.accumulate(pc, 0)
    out = dict(portfolio=name, E_acc_A=pc[0].mean() / 219, E_max_acc_K=best[-1].mean() / 219,
               P_ge_213=(best[-1] >= 213).mean(), P_ge_213_A=(pc[0] >= 213).mean())
    for c, nm in ((0, "oof"), (1, "param")):
        m = comp == c
        if m.any():
            out[f"E_max_acc_{nm}"] = best[-1][m].mean() / 219
            out[f"P_ge_213_{nm}"] = (best[-1][m] >= 213).mean()
    return out, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../outputs")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--u", type=int, default=40)
    ap.add_argument("--pool", type=int, default=6000)
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--w_oof", type=float, default=0.4, help="mixture weight of the OOF-empirical component")
    ap.add_argument("--s_train", type=int, default=6000)
    ap.add_argument("--s_eval", type=int, default=20000)
    ap.add_argument("--condition", action="store_true", help="condition scenarios on A=172, B=173, 83 public ones")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--seed_b", default="../outputs/sub_B_hedge.csv")
    args, _ = ap.parse_known_args()
    rng = np.random.default_rng(2027)
    t0 = time.time()
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    te = kp.clean(pd.read_csv(f"{args.data}/test.csv", dtype=str), False)
    x, y, X = tr.apg.values, tr.y.values.astype(float), te.apg.values
    P_oof, info = kp.sharp_band_probabilities(tr, te, repeats=10)
    lo, hi = info["cuts"]
    A = kp.band(X, lo, hi, closed=True).astype(int)
    B = pd.read_csv(args.seed_b).went_back_for_seconds.values.astype(int)

    # ---- parametric log-space band fits + bootstrap (train only)
    fits = {f: fit(x, y, f) for f in ("laplace", "logistic")}
    aic = {f: 2 * 3 + 2 * v[1] for f, v in fits.items()}
    amin = min(aic.values())
    w = {f: np.exp(-(a - amin) / 2) for f, a in aic.items()}
    w = {f: v / sum(w.values()) for f, v in w.items()}
    for f, (th, nl) in fits.items():
        print(f"MLE {f:8s} L={np.exp(th[0]):.4f} H={np.exp(th[1]):.4f} b={np.exp(th[2]):.4f}  AIC={aic[f]:.1f}  Akaike w={w[f]:.2f}")
    fams, thetas = [], []
    for i in range(args.n_boot):
        f = "laplace" if rng.random() < w["laplace"] else "logistic"
        ii = rng.integers(len(x), size=len(x))
        th, _ = fit(x[ii], y[ii], f, th0=fits[f][0])
        fams.append(f); thetas.append(th)
    TH = np.exp(np.array(thetas))
    print(f"bootstrap ({args.n_boot}): L 5-95% {np.percentile(TH[:, 0], [5, 50, 95]).round(4)}  H {np.percentile(TH[:, 1], [5, 50, 95]).round(4)}"
          f"  b {np.percentile(TH[:, 2], [5, 50, 95]).round(4)}   [{time.time() - t0:.0f}s]")

    cond = dict(A=172, B=173, ones=83) if args.condition else None
    Ytr, Ptr, Ctr, acc_rate = draw_scenarios(args.s_train, P_oof, X, (fams, thetas), args.w_oof, rng, cond, A, B)
    Yev, Pev, Cev, _ = draw_scenarios(args.s_eval, P_oof, X, (fams, thetas), args.w_oof, rng, cond, A, B)
    print(f"scenarios train {len(Ytr)} eval {len(Yev)}; acceptance {acc_rate:.4f}; param share {Ctr.mean():.2f}   [{time.time() - t0:.0f}s]")

    # ---- candidate pool
    P_mix = args.w_oof * P_oof + (1 - args.w_oof) * np.mean([p_band(X, th, f) for f, th in zip(fams, thetas)], 0)
    d = np.minimum(abs(X - lo), abs(X - hi))
    unc = np.lexsort((d, np.abs(P_mix - 0.5)))[: args.u]
    q = np.where(A[unc] == 1, 1 - P_mix[unc], P_mix[unc])
    pool = [A, B]
    for _ in range(args.pool):
        f = rng.random(len(unc)) < np.clip(q * rng.uniform(0.3, 1.6), 0, 0.95)
        v = A.copy(); v[unc[f]] ^= 1; pool.append(v)
    for i in range(len(unc)):
        v = A.copy(); v[unc[i]] ^= 1; pool.append(v)
        for j in range(i + 1, min(len(unc), 20)):
            w2 = v.copy(); w2[unc[j]] ^= 1; pool.append(w2)
    xs = np.sort(X)
    cut_lo = [(a + b) / 2 for a, b in zip(xs[:-1], xs[1:]) if 0.93 < a < 1.04]
    cut_hi = [(a + b) / 2 for a, b in zip(xs[:-1], xs[1:]) if 1.95 < a < 2.06]
    for cl in cut_lo + [lo]:
        for ch in cut_hi + [hi]:
            pool.append(kp.band(X, cl, ch, closed=True).astype(int))
    pool = np.unique(np.array(pool), axis=0)
    iA = int(np.where((pool == A).all(1))[0][0]); iB = int(np.where((pool == B).all(1))[0][0])
    pool_tr = priv_correct(pool, Ytr, Ptr)
    print(f"pool {len(pool)} (band-shift grid {len(cut_lo) + 1}x{len(cut_hi) + 1})   [{time.time() - t0:.0f}s]")

    # ---- portfolios: new (mixture scenarios) vs E19-style (OOF-only scenarios), both seeded with A and B
    ch_new = greedy(pool, pool_tr, [iA, iB], args.k)
    m_oof = Ctr == 0
    ch_old = greedy(pool, pool_tr[:, m_oof], [iA, iB], args.k)
    rows = []
    for nm, ch in (("A only", [iA]), ("A+B", [iA, iB]), ("E19-style (oof scenarios)", ch_old), ("E20 mixture", ch_new)):
        r, best = report(nm, pool[ch], Yev, Pev, Cev)
        rows.append(r)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))

    V = pool[ch_new]
    _, best = report("E20", V, Yev, Pev, Cev)
    summ = pd.DataFrame(dict(k=np.arange(1, len(V) + 1), E_max_private_acc=best.mean(1) / 219, P_ge_213=(best >= 213).mean(1),
                             rows_vs_A=[int((v != A).sum()) for v in V], ones=[int(v.sum()) for v in V],
                             is_band=[bool(any((kp.band(X, cl, ch, closed=True).astype(int) == v).all() for cl in cut_lo + [lo] for ch in cut_hi + [hi])) for v in V]))
    summ["flips_vs_A"] = [" ".join(f"{te.wedding_id[i]}:{A[i]}>{v[i]}" for i in np.where(v != A)[0]) for v in V]
    print(summ.round(4).to_string(index=False, max_colwidth=120))
    if args.write:
        od = f"{args.out}/portfolio"
        os.makedirs(od, exist_ok=True)
        for i, v in enumerate(V):
            pd.DataFrame({"wedding_id": te.wedding_id, "went_back_for_seconds": v}).to_csv(f"{od}/p{i + 1:02d}.csv", index=False)
        summ.to_csv(f"{od}/portfolio_summary.csv", index=False)
        res.to_csv(f"{od}/portfolio_comparison.csv", index=False)
        print("wrote", len(V), "files to", od)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
