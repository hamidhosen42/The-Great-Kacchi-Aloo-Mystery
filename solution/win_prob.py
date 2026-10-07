"""E21: choose the two final selections that maximise P(private rank 1), not E[best-of-two accuracy].

Scenarios (test labels + public/private split) come from portfolio2.py's mixture generator, conditioned on our known
public facts (A=172, B=173, 83 public ones). In each scenario every leaderboard team is modelled as a near-edge variant
of the shared sharp band (A / E0 with a few flips or a shifted cut) whose public score equals the team's real public
score; their private score is then read off. Private ties go to the earlier submission (Kaggle FR 7b): our A/B were
entered 2026-10-07 10:28 UTC, a new file would be entered after every current team.
Every candidate vector is built by algorithm (flips of uncertain rows / cut shifts); no labels are hand-chosen (FR 4b).

    python win_prob.py --data ../data --out ../outputs [--write]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import kacchi_pipeline as kp  # noqa: E402
import portfolio2 as p2  # noqa: E402

# (team, public correct out of 181, submission time UTC as hours since 2026-10-06 00:00) from the 2026-10-07 16:10 UTC LB
TEAMS = [("FOYSAL", 175, 27.4), ("Dadhichi Sarker", 175, 39.7), ("রোদশী", 174, 31.9), ("Str1k3rFl0", 174, 24.0),
         ("Raiyana Rufaida", 174, 31.8), ("amar_nam_adnan", 173, 18.2), ("Tasnimul Ferdous", 173, 17.3),
         ("Md. Najmul Hasan Shihab", 173, 19.5), ("el mencho", 173, 25.1), ("Rafiur Rahman", 173, 40.3),
         ("sky_is_green", 173, 39.8)]
T_OURS_OLD, T_OURS_NEW = 34.5, 99.0


def build_pool(A, E0, X, P_mix, lo, hi, rng, n_rand=4000, u=30):
    d = np.minimum(abs(X - lo), abs(X - hi))
    unc = np.lexsort((d, np.abs(P_mix - 0.5)))[:u]
    q = np.where(A[unc] == 1, 1 - P_mix[unc], P_mix[unc])
    pool = []
    for base in (A, E0):
        pool.append(base)
        for i in range(len(unc)):
            v = base.copy(); v[unc[i]] ^= 1; pool.append(v)
            for j in range(i + 1, min(len(unc), 20)):
                w = v.copy(); w[unc[j]] ^= 1; pool.append(w)
                for k in range(j + 1, min(len(unc), 12)):
                    z = w.copy(); z[unc[k]] ^= 1; pool.append(z)
    for _ in range(n_rand):
        f = rng.random(len(unc)) < np.clip(q * rng.uniform(0.3, 1.6), 0, 0.95)
        v = A.copy(); v[unc[f]] ^= 1; pool.append(v)
    xs = np.sort(X)
    cl = [(a + b) / 2 for a, b in zip(xs[:-1], xs[1:]) if 0.93 < a < 1.04]
    ch = [(a + b) / 2 for a, b in zip(xs[:-1], xs[1:]) if 1.95 < a < 2.06]
    for c1 in cl:
        for c2 in ch:
            pool.append(kp.band(X, c1, c2, closed=True).astype(int))
    return np.unique(np.array(pool), axis=0), unc


def correct(V, Y, mask):
    V = V.astype(np.float32)
    return (V @ (mask & Y).astype(np.float32).T + (1 - V) @ (mask & ~Y).astype(np.float32).T).round().astype(np.int16)


def competitor_best(pool_pub, pool_priv, comp_idx, rng):
    """(S,) best competitor private score and its submission time, sampling each team's 2 finals per scenario."""
    S = pool_pub.shape[1]
    best = np.full(S, -1, np.int16); t_best = np.full(S, np.inf)
    pub, priv = pool_pub[comp_idx], pool_priv[comp_idx]
    for _, T, tt in TEAMS:
        for need in (T, T):  # two finals, both consistent with the team's public score
            ok = pub == need
            ok[:, ~ok.any(0)] = np.abs(pub[:, ~ok.any(0)] - need) == np.abs(pub[:, ~ok.any(0)] - need).min(0)
            r = rng.random(ok.shape) * ok
            pick = r.argmax(0)
            sc = priv[pick, np.arange(S)]
            better = (sc > best) | ((sc == best) & (tt < t_best))
            best = np.where(better, sc, best); t_best = np.where(better, tt, t_best)
    return best, t_best


def win(ours_a, ours_b, t_a, t_b, cb, ct):
    s = np.maximum(ours_a, ours_b)
    t = np.where(ours_a > ours_b, t_a, np.where(ours_b > ours_a, t_b, min(t_a, t_b)))
    return (s > cb) | ((s == cb) & (t < ct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../outputs")
    ap.add_argument("--s", type=int, default=8000)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--w_oof", type=float, default=0.4)
    ap.add_argument("--write", action="store_true")
    args, _ = ap.parse_known_args()
    rng = np.random.default_rng(2028)
    t0 = time.time()
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    te = kp.clean(pd.read_csv(f"{args.data}/test.csv", dtype=str), False)
    x, y, X = tr.apg.values, tr.y.values.astype(float), te.apg.values
    P_oof, info = kp.sharp_band_probabilities(tr, te, repeats=10)
    lo, hi = info["cuts"]
    A = kp.band(X, lo, hi, closed=True).astype(int)
    B = pd.read_csv(f"{args.out}/sub_B_hedge.csv").went_back_for_seconds.values.astype(int)
    E0 = kp.band(X, *kp.accopt_foysal(x, y)).astype(int)
    print(f"A vs E0 differ on {(A != E0).sum()} rows; A vs B on {(A != B).sum()}")

    fits = {f: p2.fit(x, y, f) for f in ("laplace", "logistic")}
    aic = {f: 6 + 2 * v[1] for f, v in fits.items()}
    wl = 1 / (1 + np.exp(-(aic["logistic"] - aic["laplace"]) / 2))
    fams, thetas = [], []
    for _ in range(args.n_boot):
        f = "laplace" if rng.random() < wl else "logistic"
        ii = rng.integers(len(x), size=len(x))
        fams.append(f); thetas.append(p2.fit(x[ii], y[ii], f, th0=fits[f][0])[0])
    cond = dict(A=172, B=173, ones=83)
    Y, priv, comp, acc = p2.draw_scenarios(2 * args.s, P_oof, X, (fams, thetas), args.w_oof, rng, cond, A, B)
    S = len(Y) // 2
    print(f"scenarios {len(Y)} (acceptance {acc:.4f})   [{time.time() - t0:.0f}s]")

    P_mix = args.w_oof * P_oof + (1 - args.w_oof) * np.mean([p2.p_band(X, th, f) for f, th in zip(fams, thetas)], 0)
    pool, unc = build_pool(A, E0, X, P_mix, lo, hi, rng)
    for v in (A, B, E0):
        if not (pool == v).all(1).any():
            pool = np.vstack([pool, v])
    iA = int(np.where((pool == A).all(1))[0][0]); iB = int(np.where((pool == B).all(1))[0][0])
    pool_pub, pool_priv = correct(pool, Y, ~priv), correct(pool, Y, priv)
    near = np.minimum((pool != A).sum(1), (pool != E0).sum(1)) <= 4
    comp_idx = np.where(near)[0]
    cb, ct = competitor_best(pool_pub, pool_priv, comp_idx, rng)
    print(f"pool {len(pool)}, competitor neighbourhood {len(comp_idx)}; competitor-best private mean {cb.mean():.2f}/219"
          f"   [{time.time() - t0:.0f}s]")

    tr_, ev_ = slice(0, S), slice(S, 2 * S)
    pa, pb = pool_priv[iA], pool_priv[iB]

    def p_win(i, j, ti, tj, sl):
        return win(pool_priv[i][sl], pool_priv[j][sl], ti, tj, cb[sl], ct[sl]).mean()

    rows = [("A + B (old timestamps)", iA, iB, T_OURS_OLD, T_OURS_OLD)]
    # best partner for A (A keeps its 10:28 timestamp; partner is new)
    wA = np.array([win(pa[tr_], pool_priv[k][tr_], T_OURS_OLD, T_OURS_NEW, cb[tr_], ct[tr_]).mean() for k in range(len(pool))])
    wB = np.array([win(pb[tr_], pool_priv[k][tr_], T_OURS_OLD, T_OURS_NEW, cb[tr_], ct[tr_]).mean() for k in range(len(pool))])
    kA, kB = int(wA.argmax()), int(wB.argmax())
    rows += [("A + best new partner", iA, kA, T_OURS_OLD, T_OURS_NEW), ("B + best new partner", iB, kB, T_OURS_OLD, T_OURS_NEW)]
    # best pair of two new files (greedy: best single, then best partner)
    w1 = np.array([win(pool_priv[k][tr_], pool_priv[k][tr_], T_OURS_NEW, T_OURS_NEW, cb[tr_], ct[tr_]).mean() for k in range(len(pool))])
    k1 = int(w1.argmax())
    w2 = np.array([win(pool_priv[k1][tr_], pool_priv[k][tr_], T_OURS_NEW, T_OURS_NEW, cb[tr_], ct[tr_]).mean() for k in range(len(pool))])
    rows.append(("two new files", k1, int(w2.argmax()), T_OURS_NEW, T_OURS_NEW))

    out = []
    for nm, i, j, ti, tj in rows:
        s = np.maximum(pool_priv[i][ev_], pool_priv[j][ev_])
        out.append(dict(pair=nm, P_rank1=p_win(i, j, ti, tj, ev_), P_rank1_oof=win(pool_priv[i][ev_][comp[ev_] == 0], pool_priv[j][ev_][comp[ev_] == 0], ti, tj, cb[ev_][comp[ev_] == 0], ct[ev_][comp[ev_] == 0]).mean(),
                        E_best_private=s.mean() / 219, flips_1=" ".join(f"{te.wedding_id[r]}:{A[r]}>{pool[i][r]}" for r in np.where(pool[i] != A)[0]),
                        flips_2=" ".join(f"{te.wedding_id[r]}:{A[r]}>{pool[j][r]}" for r in np.where(pool[j] != A)[0])))
    res = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False, max_colwidth=110))
    if args.write:
        od = f"{args.out}/winprob"
        os.makedirs(od, exist_ok=True)
        res.to_csv(f"{od}/pairs.csv", index=False)
        for nm, i, j, _, _ in rows[1:]:
            tag = nm.split()[0].lower()
            for k, idx in (("1", i), ("2", j)):
                pd.DataFrame({"wedding_id": te.wedding_id, "went_back_for_seconds": pool[idx]}).to_csv(f"{od}/{tag}_{k}.csv", index=False)
        print("wrote", od)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
