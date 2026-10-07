"""E23: the competition allows up to 25 final selections, so add 22 algorithmic tickets to A, B, C that maximise
P(private rank 1) and P(private top 5) of the best selected submission.

Kaggle scores a team by its best selected submission on the private LB. Scenarios and competitor modelling follow
win_prob2.py (E22), except that every team now holds one ticket per entry (Kaggle auto-selects by public score when a
team selects nothing). Candidate tickets come from two rule families that a notebook regenerates from the data alone:
  * band(lo, hi): lo <= aloo/guest <= hi, cuts on a grid between neighbouring test ratios near each edge;
  * window(side, start, width): A with uncertainty ranks start..start+width-1 flipped. Ranks come from the sharp
    band's out-of-fold probabilities (|P - 0.5| ascending, ties by distance to the edge; B = window(both, 0, 9));
    side = both edges / lower edge only / upper edge only.
Greedy selection on training scenarios, reported on held-out scenarios. Nothing is hand-labelled (FR 4b).

    python portfolio25.py --data ../data --out ../outputs [--write]
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
import kacchi_pipeline as kp  # noqa: E402
import portfolio2 as p2  # noqa: E402
from win_prob import build_pool, correct  # noqa: E402
from win_prob2 import _init, _work  # noqa: E402

N_NEW = 22
# (team, best public correct /181, earliest LB time seen in hours since 2026-10-06 00:00 UTC, entries) — LB 2026-10-07 18:05 UTC
TEAMS = [("FOYSAL", 175, 27.38, 10), ("Dadhichi Sarker", 175, 39.74, 9), ("রোদশী", 174, 31.86, 8), ("Str1k3rFl0", 174, 24.01, 6),
         ("Raiyana Rufaida", 174, 31.77, 4), ("amar_nam_adnan", 173, 18.21, 4), ("Tasnimul Ferdous", 173, 17.29, 4),
         ("Md. Najmul Hasan Shihab", 173, 19.50, 5), ("el mencho", 173, 25.13, 4), ("Rafiur Rahman", 173, 40.25, 4),
         ("sky_is_green", 173, 39.79, 3), ("Adib Rahman", 173, 40.66, 3), ("Md Abdul Al Hasib", 172, 39.66, 3)]
T_A, T_B, T_C, T_NEW = 34.48, 34.49, 40.59, 42.5


def competitor_scores(npub, npriv, rng, mult=1):
    """(n_team, S) best private score per team: one ticket matching its best public score, the rest within 3 rows."""
    S = npub.shape[1]
    CS = []
    for _, T, _, n in TEAMS:
        best = np.full(S, -1, np.int16)
        for k in range(n * mult):
            ok = (npub == T) if k == 0 else (npub <= T) & (npub >= T - 3)
            none = ~ok.any(0)
            if none.any():
                dist = np.abs(npub[:, none] - T)
                ok[:, none] = dist == dist.min(0)
            pick = (rng.random(ok.shape) * ok).argmax(0)
            best = np.maximum(best, npriv[pick, np.arange(S)])
        CS.append(best)
    return np.array(CS)


def thresholds(CS, ct, R):
    """Score and time of the R-th best team per scenario (order: score desc, time asc)."""
    order = np.lexsort((np.broadcast_to(ct[:, None], CS.shape), -CS), axis=0)
    return np.take_along_axis(CS, order, 0)[R - 1], ct[order][R - 1]


def p_rank(s, t, cR, tR):
    return (s > cR) | ((s == cR) & (t < tR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../outputs")
    ap.add_argument("--s", type=int, default=8000, help="scenarios per half (train / eval)")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--w_oof", type=float, default=0.4)
    ap.add_argument("--write", action="store_true")
    args, _ = ap.parse_known_args()
    rng = np.random.default_rng(2030)
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

    # ---- scenarios (identical generator and conditioning to E22)
    fits = {f: p2.fit(x, y, f) for f in ("laplace", "logistic")}
    aic = {f: 6 + 2 * v[1] for f, v in fits.items()}
    wl = 1 / (1 + np.exp(-(aic["logistic"] - aic["laplace"]) / 2))
    fams, thetas = [], []
    for _ in range(args.n_boot):
        f = "laplace" if rng.random() < wl else "logistic"
        ii = rng.integers(len(x), size=len(x))
        fams.append(f); thetas.append(p2.fit(x[ii], y[ii], f, th0=fits[f][0])[0])
    Pb = np.array([p2.p_band(X, th, f) for f, th in zip(fams, thetas)])
    Ys, Ps, Cs, raw, seed = [], [], [], 0, 1000
    with Pool(12, initializer=_init, initargs=(P_oof, Pb, args.w_oof, np.array([A, B, C]), [172, 173, 173], 83)) as pool_:
        while sum(len(v) for v in Ys) < 2 * args.s:
            for Yk, Pk, Ck, r in pool_.map(_work, range(seed, seed + 12)):
                Ys.append(Yk); Ps.append(Pk); Cs.append(Ck); raw += r
            seed += 12
    Y, priv, comp = np.concatenate(Ys)[: 2 * args.s], np.concatenate(Ps)[: 2 * args.s], np.concatenate(Cs)[: 2 * args.s]
    S = len(Y) // 2
    trs, evs = slice(0, S), slice(S, 2 * S)
    print(f"scenarios {len(Y)}; param share {comp.mean():.2f}   [{time.time() - t0:.0f}s]", flush=True)

    # ---- competitors: one ticket per entry, near-edge variants of A / E0 consistent with their public scores
    P_mix = args.w_oof * P_oof + (1 - args.w_oof) * Pb.mean(0)
    cpool, _ = build_pool(A, E0, X, P_mix, lo, hi, rng)
    near = cpool[np.minimum((cpool != A).sum(1), (cpool != E0).sum(1)) <= 4]
    npub, npriv = correct(near, Y, ~priv), correct(near, Y, priv)
    ct = np.array([t for _, _, t, _ in TEAMS])
    CS = competitor_scores(npub, npriv, rng)
    CS2 = competitor_scores(npub, npriv, rng, mult=2)
    print(f"competitor neighbourhood {len(near)}; best-team private mean {CS.max(0).mean():.2f}/219   [{time.time() - t0:.0f}s]", flush=True)

    # ---- our candidate tickets
    d_edge = np.minimum(abs(X - lo), abs(X - hi))
    order = np.lexsort((d_edge, np.abs(P_oof - 0.5)))
    low = X < 1.5
    side_order = {"both": order, "lower": order[low[order]], "upper": order[~low[order]]}
    specs, vecs = [], []
    for side, o in side_order.items():
        for s0 in range(0, 26):
            for w in range(1, 9):
                if s0 + w > len(o):
                    continue
                v = A.copy(); v[o[s0:s0 + w]] ^= 1
                specs.append(dict(family="window", side=side, start=s0, width=w)); vecs.append(v)
    xs = np.unique(X)
    mids = (xs[:-1] + xs[1:]) / 2
    for cl in mids[(mids > 0.94) & (mids < 1.05)]:
        for ch in mids[(mids > 1.94) & (mids < 2.07)]:
            l5, h5 = round(float(cl), 5), round(float(ch), 5)
            v = ((X >= l5) & (X <= h5)).astype(int)
            assert (v == ((X >= cl) & (X <= ch))).all()
            specs.append(dict(family="band", lo=l5, hi=h5)); vecs.append(v)
    vecs = np.array(vecs)
    _, first = np.unique(vecs, axis=0, return_index=True)
    keep = [i for i in sorted(first) if not any((vecs[i] == z).all() for z in (A, B, C))]
    specs, vecs = [specs[i] for i in keep], vecs[keep]
    Q = correct(vecs, Y, priv)
    base = correct(np.array([A, B, C]), Y, priv)
    print(f"candidate tickets {len(vecs)}   [{time.time() - t0:.0f}s]", flush=True)

    def start_state(sl):
        s = base[:, sl].max(0)
        t = np.where(base[0, sl] == s, T_A, np.where(base[1, sl] == s, T_B, T_C))
        return s, t

    def evaluate(chosen, sl, CSx):
        s, t = start_state(sl)
        out = []
        th = {R: thresholds(CSx[:, sl], ct, R) for R in (1, 3, 5)}
        for k in range(len(chosen) + 1):
            if k:
                q = Q[chosen[k - 1], sl]
                t = np.where(q > s, T_NEW, t); s = np.maximum(s, q)
            out.append(dict(tickets=3 + k, P_top1=p_rank(s, t, *th[1]).mean(), P_top3=p_rank(s, t, *th[3]).mean(),
                            P_top5=p_rank(s, t, *th[5]).mean(), E_best_private=s.mean()))
        return pd.DataFrame(out)

    def greedy(weights):
        s, t = start_state(trs)
        th = {R: thresholds(CS[:, trs], ct, R) for R in weights}
        chosen = []
        for _ in range(N_NEW):
            qs = Q[:, trs]
            ns = np.maximum(s[None], qs); nt = np.where(qs > s[None], T_NEW, t[None])
            obj = sum(w * p_rank(ns, nt, th[R][0][None], th[R][1][None]).mean(1) for R, w in weights.items())
            obj[chosen] = -1
            j = int(obj.argmax())
            chosen.append(j)
            t = np.where(Q[j, trs] > s, T_NEW, t); s = np.maximum(s, Q[j, trs])
        return chosen

    results = {}
    for name, w in (("top1", {1: 1.0}), ("top5", {5: 1.0}), ("mix", {1: 0.5, 5: 0.5})):
        ch = greedy(w)
        ev = evaluate(ch, evs, CS)
        ev2 = evaluate(ch, evs, CS2).iloc[-1]
        oof, par = comp[evs] == 0, comp[evs] == 1
        results[name] = ch
        print(f"\n== objective {name}: held-out, by number of selected tickets")
        print(ev.iloc[[0, 2, 5, 10, 15, 20, 22]].round(4).to_string(index=False))
        print(f"   competitors with 2x tickets: P_top1 {ev2.P_top1:.4f}  P_top5 {ev2.P_top5:.4f}")
    pick = "mix"
    ch = results[pick]
    rows = []
    for r, j in enumerate(ch, 1):
        v = vecs[j]
        rows.append(dict(ticket=f"P{r:02d}", **specs[j], rows_vs_A=int((v != A).sum()), ones=int(v.sum()),
                         flips=" ".join(f"{te.wedding_id[i]}:{A[i]}>{v[i]}" for i in np.where(v != A)[0])))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"\n== portfolio ({pick})")
    print(tab.to_string(index=False, max_colwidth=90))
    if args.write:
        od = f"{args.out}/portfolio25"
        os.makedirs(od, exist_ok=True)
        for r, j in enumerate(ch, 1):
            pd.DataFrame({"wedding_id": te.wedding_id, "went_back_for_seconds": vecs[j]}).to_csv(f"{od}/P{r:02d}.csv", index=False)
        with open(f"{od}/specs.json", "w") as f:
            json.dump([{k: v for k, v in specs[j].items()} for j in ch], f, indent=1)
        tab.to_csv(f"{od}/portfolio.csv", index=False)
        evaluate(ch, evs, CS).to_csv(f"{od}/curve.csv", index=False)
        print("wrote", od)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
