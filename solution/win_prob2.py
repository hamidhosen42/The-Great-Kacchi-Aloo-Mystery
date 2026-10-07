"""E22: final-pair choice for P(private rank 1) and P(private top 5), conditioned on ALL our public scores.

Extends win_prob.py (E21):
  * scenarios are conditioned on every genuine public score we hold (A=172, B=173, C=173, 83 public ones),
    drawn with a parallel rejection sampler (argpartition split, 12 workers);
  * every leaderboard team gets its own private score (2 finals, each a near-edge variant of the shared band whose
    public score equals the team's), so full ranks are computed; ties go to the earlier submission (FR 7b);
  * pairs are searched separately for P(rank 1) and P(rank <= 5), on training scenarios, and reported on held-out
    scenarios, per generator component (oof / param) for robustness.
All candidate vectors are algorithmic (cut shifts / flips of the most uncertain rows); nothing is hand-labelled (FR 4b).

    python win_prob2.py --data ../data --out ../outputs [--s 8000] [--write]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import kacchi_pipeline as kp  # noqa: E402
import portfolio2 as p2  # noqa: E402
from win_prob import build_pool, correct  # noqa: E402

N_TEST, N_PUB = 400, 181
# (team, public correct /181, time of best entry, hours since 2026-10-06 00:00 UTC) — LB of 2026-10-07 16:10 UTC
TEAMS = [("FOYSAL", 175, 27.38), ("Dadhichi Sarker", 175, 39.74), ("রোদশী", 174, 31.86), ("Str1k3rFl0", 174, 24.01),
         ("Raiyana Rufaida", 174, 31.77), ("amar_nam_adnan", 173, 18.21), ("Tasnimul Ferdous", 173, 17.29),
         ("Md. Najmul Hasan Shihab", 173, 19.50), ("el mencho", 173, 25.13), ("Rafiur Rahman", 173, 40.25),
         ("sky_is_green", 173, 39.79), ("Md Abdul Al Hasib", 172, 39.66)]
T_A, T_B, T_C, T_NEW = 34.48, 34.49, 40.59, 42.0

_G = {}


def _init(P_oof, Pb, w_oof, facts_V, facts_n, ones):
    _G.update(P_oof=P_oof.astype(np.float32), Pb=Pb.astype(np.float32), w=w_oof, V=facts_V.astype(np.float32),
              n=np.array(facts_n), ones=ones)


def _work(seed, n=50000, rounds=40):
    rng = np.random.default_rng(seed)
    g = _G
    outY, outP, outC, raw = [], [], [], 0
    for _ in range(rounds):
        comp = rng.random(n) >= g["w"]
        P = np.where(comp[:, None], g["Pb"][rng.integers(len(g["Pb"]), size=n)], g["P_oof"][None, :])
        Y = rng.random((n, N_TEST), dtype=np.float32) < P
        pub = np.zeros((n, N_TEST), bool)
        np.put_along_axis(pub, np.argpartition(rng.random((n, N_TEST), dtype=np.float32), N_PUB, 1)[:, :N_PUB], True, 1)
        keep = (Y & pub).sum(1) == g["ones"]
        Yk, pk = Y[keep], pub[keep]
        Yf, pf = Yk.astype(np.float32), pk.astype(np.float32)
        pc = (g["V"] @ (Yf * pf).T + (1 - g["V"]) @ ((1 - Yf) * pf).T).round()
        k2 = (pc == g["n"][:, None]).all(0)
        outY.append(Yk[k2]); outP.append(~pk[k2]); outC.append(comp[keep][k2]); raw += n
    return np.concatenate(outY), np.concatenate(outP), np.concatenate(outC), raw


def rank_of(s, t, CS, ct):
    """1-based private rank of our score s (S,) entered at time t (S,) vs team scores CS (n_team,S) at times ct."""
    return 1 + ((CS > s[None]) | ((CS == s[None]) & (ct[:, None] < t[None]))).sum(0)


def pair_rank(si, sj, ti, tj, CS, ct):
    s = np.maximum(si, sj)
    t = np.where(si > sj, ti, np.where(sj > si, tj, min(ti, tj)))
    return rank_of(s, np.broadcast_to(t, s.shape).astype(float), CS, ct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../outputs")
    ap.add_argument("--s", type=int, default=8000, help="scenarios per half (train / eval)")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--w_oof", type=float, default=0.4)
    ap.add_argument("--write", action="store_true")
    args, _ = ap.parse_known_args()
    rng = np.random.default_rng(2029)
    t0 = time.time()
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    te = kp.clean(pd.read_csv(f"{args.data}/test.csv", dtype=str), False)
    x, y, X = tr.apg.values, tr.y.values.astype(float), te.apg.values
    P_oof, info = kp.sharp_band_probabilities(tr, te, repeats=10)
    lo, hi = info["cuts"]
    A = kp.band(X, lo, hi, closed=True).astype(int)
    rd = lambda f: pd.read_csv(f"{args.out}/{f}").went_back_for_seconds.values.astype(int)
    B, C = rd("sub_B_hedge.csv"), rd("sub_C_band_1p00_1p975.csv")
    E0 = kp.band(X, *kp.accopt_foysal(x, y)).astype(int)
    assert (A == rd("sub_A_primary_sharpband.csv")).all()

    fits = {f: p2.fit(x, y, f) for f in ("laplace", "logistic")}
    aic = {f: 6 + 2 * v[1] for f, v in fits.items()}
    wl = 1 / (1 + np.exp(-(aic["logistic"] - aic["laplace"]) / 2))
    fams, thetas = [], []
    for _ in range(args.n_boot):
        f = "laplace" if rng.random() < wl else "logistic"
        ii = rng.integers(len(x), size=len(x))
        fams.append(f); thetas.append(p2.fit(x[ii], y[ii], f, th0=fits[f][0])[0])
    Pb = np.array([p2.p_band(X, th, f) for f, th in zip(fams, thetas)])
    TH = np.exp(np.array(thetas))
    print(f"bootstrap L {np.percentile(TH[:, 0], [5, 50, 95]).round(4)} H {np.percentile(TH[:, 1], [5, 50, 95]).round(4)}")

    facts_V, facts_n = np.array([A, B, C]), [172, 173, 173]
    Ys, Ps, Cs, raw, seed = [], [], [], 0, 0
    with Pool(12, initializer=_init, initargs=(P_oof, Pb, args.w_oof, facts_V, facts_n, 83)) as pool_:
        while sum(len(v) for v in Ys) < 2 * args.s:
            for Yk, Pk, Ck, r in pool_.map(_work, range(seed, seed + 12)):
                Ys.append(Yk); Ps.append(Pk); Cs.append(Ck); raw += r
            seed += 12
            print(f"  accepted {sum(len(v) for v in Ys)} / raw {raw:,}   [{time.time() - t0:.0f}s]", flush=True)
    Y, priv, comp = np.concatenate(Ys)[: 2 * args.s], np.concatenate(Ps)[: 2 * args.s], np.concatenate(Cs)[: 2 * args.s]
    S = len(Y) // 2
    print(f"scenarios {len(Y)}; acceptance {len(np.concatenate(Ys)) / raw:.5f}; param share {comp.mean():.2f}")

    P_mix = args.w_oof * P_oof + (1 - args.w_oof) * Pb.mean(0)
    pool, _ = build_pool(A, E0, X, P_mix, lo, hi, rng)
    for v in (A, B, C, E0):
        if not (pool == v).all(1).any():
            pool = np.vstack([pool, v])
    idx = lambda v: int(np.where((pool == v).all(1))[0][0])
    iA, iB, iC = idx(A), idx(B), idx(C)
    pub_c, priv_c = correct(pool, Y, ~priv), correct(pool, Y, priv)
    assert (pub_c[iA] == 172).all() and (pub_c[iC] == 173).all()

    # ---- competitors: per team, 2 finals sampled from the near-edge neighbourhood matching its public score
    near = np.where(np.minimum((pool != A).sum(1), (pool != E0).sum(1)) <= 4)[0]
    npub, npriv = pub_c[near], priv_c[near]
    CS, ct = [], []
    for _, T, tt in TEAMS:
        best = np.full(len(Y), -1, np.int16)
        for _ in range(2):
            dist = np.abs(npub - T)
            ok = dist == dist.min(0)
            pick = (rng.random(ok.shape) * ok).argmax(0)
            best = np.maximum(best, npriv[pick, np.arange(len(Y))])
        CS.append(best); ct.append(tt)
    CS, ct = np.array(CS), np.array(ct)
    print(f"pool {len(pool)}, neighbourhood {len(near)}; team mean private: "
          + ", ".join(f"{n.split()[0]} {m:.1f}" for (n, _, _), m in zip(TEAMS, CS.mean(1))) + f"   [{time.time() - t0:.0f}s]")

    trs, evs = slice(0, S), slice(S, 2 * S)
    existing = {"A": (iA, T_A), "B": (iB, T_B), "C": (iC, T_C)}

    def score_all(base_i, base_t, sl, objective):
        """objective value of (base, k) for every k in pool, partner k entered at T_NEW."""
        sb, CSs = priv_c[base_i][sl], CS[:, sl]
        out = np.empty(len(pool))
        for k in range(len(pool)):
            r = pair_rank(sb, priv_c[k][sl], base_t, T_NEW, CSs, ct)
            out[k] = (r <= objective).mean()
        return out

    cands = [(f"{a}+{b}", existing[a], existing[b]) for a, b in (("A", "B"), ("A", "C"), ("B", "C"))]
    for obj in (1, 5):
        for nm, (bi, bt) in existing.items():
            w = score_all(bi, bt, trs, obj)
            cands.append((f"{nm}+new(top{obj})", (bi, bt), (int(w.argmax()), T_NEW)))
        w1 = np.array([(rank_of(priv_c[k][trs], np.full(S, T_NEW), CS[:, trs], ct) <= obj).mean() for k in range(len(pool))])
        k1 = int(w1.argmax())
        w2 = score_all(k1, T_NEW, trs, obj)
        cands.append((f"new+new(top{obj})", (k1, T_NEW), (int(w2.argmax()), T_NEW)))

    rows = []
    for nm, (i, ti), (j, tj) in cands:
        r = pair_rank(priv_c[i][evs], priv_c[j][evs], ti, tj, CS[:, evs], ct)
        ce = comp[evs]
        rows.append(dict(pair=nm, P_top1=(r == 1).mean(), P_top3=(r <= 3).mean(), P_top5=(r <= 5).mean(),
                         P_top5_oof=(r[ce == 0] <= 5).mean(), P_top5_param=(r[ce == 1] <= 5).mean(),
                         P_top1_oof=(r[ce == 0] == 1).mean(), P_top1_param=(r[ce == 1] == 1).mean(),
                         E_rank=r.mean(), E_priv=np.maximum(priv_c[i][evs], priv_c[j][evs]).mean(),
                         new_i=i if ti == T_NEW else -1, new_j=j if tj == T_NEW else -1))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.drop(columns=["new_i", "new_j"]).round(4).to_string(index=False))
    for r in rows:
        for key in ("new_i", "new_j"):
            k = r[key]
            if k >= 0:
                v = pool[k]
                ones = X[v == 1]
                is_band = ((v == 1) == ((X >= ones.min()) & (X <= ones.max()))).all()
                print(f"{r['pair']:18s} {key}: band={is_band} [{ones.min():.4f}, {ones.max():.4f}] vs A: "
                      + " ".join(f"{te.wedding_id[q]}:{A[q]}>{v[q]}" for q in np.where(v != A)[0]))
    if args.write:
        od = f"{args.out}/winprob2"
        os.makedirs(od, exist_ok=True)
        res.to_csv(f"{od}/pairs.csv", index=False)
        for r in rows:
            for key in ("new_i", "new_j"):
                if r[key] >= 0:
                    nm = r["pair"].replace("+", "_").replace("(", "_").replace(")", "")
                    pd.DataFrame({"wedding_id": te.wedding_id, "went_back_for_seconds": pool[r[key]]}).to_csv(f"{od}/{nm}_{key}.csv", index=False)
        print("wrote", od)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
