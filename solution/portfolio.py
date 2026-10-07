"""E19: final-selection portfolio when up to K submissions count (Kaggle scores the BEST selected one on private).

Greedy construction under the out-of-fold sharp-band probability model (E16):
  * scenarios: y_i ~ Bernoulli(P_i) for every test row, plus a random 181/219 public/private split;
  * candidate vectors: A with a random subset of the U most-uncertain rows flipped (flip prob ~ uncertainty);
  * step k adds the candidate that maximises E[max over chosen vectors of private correct], estimated on a
    TRAINING set of scenarios and reported on an independent EVALUATION set (no optimistic bias).
No labels or leaderboard feedback are used; the construction is a fixed algorithm (Rule 4b compliant).

    python portfolio.py --data ../data --out ../outputs --k 25
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import kacchi_pipeline as kp  # noqa: E402


def scenarios(P, n_pub, S, rng):
    Y = rng.random((S, len(P))) < P[None, :]
    priv = np.ones((S, len(P)), bool)
    for s in range(S):
        priv[s, rng.choice(len(P), n_pub, replace=False)] = False
    return Y, priv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../outputs")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--u", type=int, default=40, help="number of most-uncertain rows eligible for flipping")
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--s_train", type=int, default=6000)
    ap.add_argument("--s_eval", type=int, default=20000)
    ap.add_argument("--write", action="store_true", help="write the K portfolio CSVs")
    ap.add_argument("--seed_b", default="../outputs/sub_B_hedge.csv", help="already-submitted hedge to keep in the portfolio")
    args, _ = ap.parse_known_args()
    rng = np.random.default_rng(2026)
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    te = kp.clean(pd.read_csv(f"{args.data}/test.csv", dtype=str), False)
    P, info = kp.sharp_band_probabilities(tr, te, repeats=10)
    lo, hi = info["cuts"]
    A = kp.band(te.apg.values, lo, hi, closed=True).astype(int)
    d = np.minimum(abs(te.apg.values - lo), abs(te.apg.values - hi))
    unc = np.lexsort((d, np.abs(P - 0.5)))[: args.u]            # most uncertain rows (ties: closer to an edge)
    # P(flip is right) for each eligible row = P(y != A)
    q = np.where(A[unc] == 1, 1 - P[unc], P[unc])

    Ytr, Ptr = scenarios(P, 181, args.s_train, rng)
    Yev, Pev = scenarios(P, 181, args.s_eval, rng)

    def priv_correct(v, Y, priv):
        return ((Y == v[None, :]) & priv).sum(1)

    # candidate pool: flip each eligible row independently with prob ~ its chance of being wrong in A (scaled)
    pool = []
    for _ in range(args.pool):
        scale = rng.uniform(0.3, 1.6)
        f = rng.random(len(unc)) < np.clip(q * scale, 0, 0.95)
        v = A.copy(); v[unc[f]] = 1 - v[unc[f]]
        pool.append(v)
    for i in range(len(unc)):                                   # all single flips and pairs among the top 20
        v = A.copy(); v[unc[i]] = 1 - v[unc[i]]; pool.append(v)
        for j in range(i + 1, min(len(unc), 20)):
            w = v.copy(); w[unc[j]] = 1 - w[unc[j]]; pool.append(w)
    pool = np.unique(np.array(pool), axis=0)
    pool_tr = np.array([priv_correct(v, Ytr, Ptr) for v in pool])     # (n_pool, S_train)

    seeds = [A]
    if args.seed_b and os.path.exists(args.seed_b):
        seeds.append(pd.read_csv(args.seed_b).went_back_for_seconds.values.astype(int))
    chosen, cur = list(seeds), np.max([priv_correct(v, Ytr, Ptr) for v in seeds], axis=0)
    rows = [dict(k=i + 1, E_max_private_train=np.max([priv_correct(v, Ytr, Ptr) for v in seeds[: i + 1]], axis=0).mean()) for i in range(len(seeds))]
    for k in range(len(seeds) + 1, args.k + 1):
        gains = np.maximum(pool_tr, cur[None, :]).mean(1)
        j = int(np.argmax(gains))
        chosen.append(pool[j]); cur = np.maximum(cur, pool_tr[j])
        rows.append(dict(k=k, E_max_private_train=cur.mean()))
    # honest evaluation on fresh scenarios
    ev = np.array([priv_correct(v, Yev, Pev) for v in chosen])
    best = np.maximum.accumulate(ev, axis=0)
    out = pd.DataFrame(rows)
    out["E_max_private_eval"] = best.mean(1)
    out["E_max_private_acc"] = out.E_max_private_eval / 219
    out["P_max_ge_0.97 (>=213/219)"] = (best >= 213).mean(1)
    out["rows_flipped_vs_A"] = [int((v != A).sum()) for v in chosen]
    pd.set_option("display.width", 200)
    print(f"eligible uncertain rows: {len(unc)}; pool size {len(pool)}")
    print(out.round(4).to_string(index=False))
    if args.write:
        os.makedirs(f"{args.out}/portfolio", exist_ok=True)
        for i, v in enumerate(chosen):
            pd.DataFrame({"wedding_id": te.wedding_id, "went_back_for_seconds": v}).to_csv(f"{args.out}/portfolio/p{i + 1:02d}.csv", index=False)
        out.to_csv(f"{args.out}/portfolio/portfolio_summary.csv", index=False)
        print("wrote", len(chosen), "files to", f"{args.out}/portfolio")


if __name__ == "__main__":
    main()
