"""E29-E36: independent edge-probability models for the sharp band, under the frozen validation protocol.

Each model sees the training data only (never public scores). Same 10x5 folds as the experiment ledger
(kp.make_folds, seed 2026). Reported per model: OOF accuracy, edge accuracy / log-loss / Brier by distance to A's
cuts, size-stress accuracy, paired comparison with A (corrected t-test, discordant rows), and on the test set the
decision vector, Hamming distances to A / B / C / P02 and the number of uncertain rows.

  E16  B's model: sharp band + OOF error rate by distance to the nearest edge (reference)
  E29  bootstrap hard band: P = share of bootstrap midcut bands that contain the row
  E30  edge-specific isotonic regression (increasing on 0.85-1.15, decreasing on 1.80-2.20)
  E31  Bayesian change-point per edge (uniform cut prior, Beta-Binomial sides, cut integrated out)
  E32  jackknife hard band: P = share of leave-one-out midcut bands that contain the row
  E34  four-sided OOF error rates (lower/upper edge x inside/outside x distance bucket)
  E35  consensus: mean P of E29, E30, E31, E32, E34
  E33  nested 1-SE band: fixed-band grid, inner CV, simplest band (closest to 1, 2) within 1 SE of the best
  E36  P02 structural test: best training band on lo in [0.98, 1.02], hi in [1.98, 2.04]

    python edge_models.py --data ../data --out ../outputs
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import betaln
from scipy.stats import binomtest
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
import kacchi_pipeline as kp  # noqa: E402

BINS = (0.0, 0.01, 0.035, 0.1, 0.4, np.inf)


def fast_midcut(r, y):
    """Same cuts as kp.midcut_rafiur (midpoint cut with the most correct rows per side, median of ties), via prefix sums."""
    def best(x, yy, side):
        xs = np.unique(x)
        c = (xs[:-1] + xs[1:]) / 2
        idx = np.searchsorted(xs, x)
        c1 = np.cumsum(np.bincount(idx, weights=yy, minlength=len(xs)))
        c0 = np.cumsum(np.bincount(idx, weights=1 - yy, minlength=len(xs)))
        ok = (c1[-1] - c1[:-1]) + c0[:-1] if side == "lo" else c1[:-1] + (c0[-1] - c0[:-1])
        b = np.flatnonzero(ok == ok.max())
        return c[b[len(b) // 2]]
    m = r < 1.5
    return best(r[m], y[m], "lo"), best(r[~m], y[~m], "hi")


def hard_band(x, lo, hi):
    return ((x >= lo) & (x <= hi)).astype(float)


# ----------------------------------------------------------------------------------------------- models: fn(t, *frames) -> [P]
def m_A(t, *aps):
    lo, hi = fast_midcut(t.apg.values, t.y.values.astype(float))
    return [hard_band(a.apg.values, lo, hi) for a in aps]


def m_E16(t, *aps):
    return [kp.sharp_band_probabilities(t.reset_index(drop=True), a, repeats=10)[0] for a in aps]


def m_E29(t, *aps, B=300, seed=0):
    rng = np.random.default_rng(seed)
    r, y = t.apg.values, t.y.values.astype(float)
    cuts = [fast_midcut(r[ii], y[ii]) for ii in (rng.integers(len(r), size=len(r)) for _ in range(B))]
    return [np.mean([hard_band(a.apg.values, lo, hi) for lo, hi in cuts], 0) for a in aps]


def m_E32(t, *aps):
    r, y = t.apg.values, t.y.values.astype(float)
    keep = np.ones(len(r), bool)
    cuts = {}
    for i in range(len(r)):
        keep[i] = False
        c = fast_midcut(r[keep], y[keep])
        cuts[c] = cuts.get(c, 0) + 1
        keep[i] = True
    n = sum(cuts.values())
    return [sum(w * hard_band(a.apg.values, lo, hi) for (lo, hi), w in cuts.items()) / n for a in aps]


def plateaus(r, y, lo_win, hi_win):
    mid = (r > lo_win[1]) & (r < hi_win[0])
    out = (r < lo_win[0]) | (r > hi_win[1])
    jef = lambda m: (y[m].sum() + 0.5) / (m.sum() + 1.0)
    return jef(mid), jef(out)


def m_E30(t, *aps, lo_win=(0.85, 1.15), hi_win=(1.80, 2.20)):
    r, y = t.apg.values, t.y.values.astype(float)
    p_mid, p_out = plateaus(r, y, lo_win, hi_win)
    wl = (r >= lo_win[0]) & (r <= lo_win[1]); wh = (r >= hi_win[0]) & (r <= hi_win[1])
    iso_l = IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds="clip").fit(r[wl], y[wl])
    iso_h = IsotonicRegression(increasing=False, y_min=0, y_max=1, out_of_bounds="clip").fit(r[wh], y[wh])
    res = []
    for a in aps:
        x = a.apg.values
        p = np.where((x > lo_win[1]) & (x < hi_win[0]), p_mid, p_out)
        p = np.where((x >= lo_win[0]) & (x <= lo_win[1]), iso_l.predict(np.clip(x, *lo_win)), p)
        p = np.where((x >= hi_win[0]) & (x <= hi_win[1]), iso_h.predict(np.clip(x, *hi_win)), p)
        res.append(p)
    return res


def changepoint_edge(r, y, win, x, a0=1.0, b0=1.0):
    """P(y=1) at x for one edge: cut uniform on win, constant rate on each side (Beta(a0,b0)), cut integrated out."""
    m = (r >= win[0]) & (r <= win[1])
    rr, yy = r[m], y[m]
    u = np.unique(np.r_[win[0], rr, win[1]])
    lo_k, hi_k = u[:-1], u[1:]                                     # cut lies in (lo_k, hi_k)
    above = rr[None, :] >= hi_k[:, None]                           # rows above the cut for each interval
    n_a, k_a = above.sum(1), (above * yy[None, :]).sum(1)
    n_b, k_b = len(rr) - n_a, yy.sum() - k_a
    ll = (betaln(a0 + k_a, b0 + n_a - k_a) + betaln(a0 + k_b, b0 + n_b - k_b) - 2 * betaln(a0, b0)
          + np.log(np.maximum(hi_k - lo_k, 1e-12)))
    post = np.exp(ll - ll.max()); post /= post.sum()
    th_a, th_b = (a0 + k_a) / (a0 + b0 + n_a), (a0 + k_b) / (a0 + b0 + n_b)
    q = np.clip((x[None, :] - lo_k[:, None]) / (hi_k - lo_k)[:, None], 0, 1)   # P(x above the cut | interval)
    p_above = q * th_a[:, None] + (1 - q) * th_b[:, None]   # rates belong to sides, so this serves both edges
    return post @ p_above, post, (lo_k + hi_k) / 2


def changepoint_draws(r, y, x, n_draws, rng, lo_win=(0.85, 1.15), hi_win=(1.80, 2.25), a0=1.0, b0=1.0):
    """(n_draws, len(x)) label probabilities drawn from the E31 posterior: per edge a cut (interval ~ posterior, uniform
    inside it) and Beta-posterior side rates; Beta-posterior plateau rates away from the edges."""
    mid, out = (r > lo_win[1]) & (r < hi_win[0]), (r < lo_win[0]) | (r > hi_win[1])
    P = np.where((x > lo_win[1]) & (x < hi_win[0]),
                 rng.beta(a0 + y[mid].sum(), b0 + (1 - y[mid]).sum(), size=(n_draws, 1)),
                 rng.beta(a0 + y[out].sum(), b0 + (1 - y[out]).sum(), size=(n_draws, 1)))
    for win in (lo_win, hi_win):
        m = (r >= win[0]) & (r <= win[1])
        rr, yy = r[m], y[m]
        u = np.unique(np.r_[win[0], rr, win[1]])
        lo_k, hi_k = u[:-1], u[1:]
        above = rr[None, :] >= hi_k[:, None]
        n_a, k_a = above.sum(1), (above * yy[None, :]).sum(1)
        n_b, k_b = len(rr) - n_a, yy.sum() - k_a
        ll = (betaln(a0 + k_a, b0 + n_a - k_a) + betaln(a0 + k_b, b0 + n_b - k_b) + np.log(np.maximum(hi_k - lo_k, 1e-12)))
        post = np.exp(ll - ll.max()); post /= post.sum()
        k = rng.choice(len(post), size=n_draws, p=post)
        cut = rng.uniform(lo_k[k], hi_k[k])
        th_a, th_b = rng.beta(a0 + k_a[k], b0 + n_a[k] - k_a[k]), rng.beta(a0 + k_b[k], b0 + n_b[k] - k_b[k])
        w = (x >= win[0]) & (x <= win[1])
        P[:, w] = np.where(x[None, w] >= cut[:, None], th_a[:, None], th_b[:, None])
    return P


def m_E31(t, *aps, lo_win=(0.85, 1.15), hi_win=(1.80, 2.25)):
    r, y = t.apg.values, t.y.values.astype(float)
    p_mid, p_out = plateaus(r, y, lo_win, hi_win)
    res = []
    for a in aps:
        x = a.apg.values
        p = np.where((x > lo_win[1]) & (x < hi_win[0]), p_mid, p_out)
        il = (x >= lo_win[0]) & (x <= lo_win[1]); ih = (x >= hi_win[0]) & (x <= hi_win[1])
        if il.any():
            p[il] = changepoint_edge(r, y, lo_win, x[il])[0]
        if ih.any():
            p[ih] = changepoint_edge(r, y, hi_win, x[ih])[0]
        res.append(p)
    return res


def four_sided_probabilities(t, a, repeats=10, seed=0):
    """E34: like kp.sharp_band_probabilities, but the OOF error rate is estimated per (edge, inside/outside, distance)."""
    t = t.reset_index(drop=True)
    r, y = t.apg.values, t.y.values.astype(float)
    lo, hi = fast_midcut(r, y)
    nb = len(BINS) - 1
    k_err, n_err = np.zeros((2, 2, nb)), np.zeros((2, 2, nb))
    for k in range(repeats):
        for tt, v in StratifiedKFold(5, shuffle=True, random_state=seed + k).split(t, y):
            l2, h2 = fast_midcut(r[tt], y[tt])
            rv = r[v]
            side = (rv >= 1.5).astype(int)
            inside = hard_band(rv, l2, h2).astype(int)
            d = np.minimum(abs(rv - l2), abs(rv - h2))
            b = np.digitize(d, BINS[1:-1])
            np.add.at(n_err, (side, inside, b), 1)
            np.add.at(k_err, (side, inside, b), inside != y[v])
    err = (k_err + 0.5) / (n_err + 1.0)
    x = a.apg.values
    side = (x >= 1.5).astype(int)
    inside = hard_band(x, lo, hi).astype(int)
    d = np.minimum(abs(x - lo), abs(x - hi))
    e = err[side, inside, np.digitize(d, BINS[1:-1])]
    P = np.where(inside == 1, 1 - e, e)
    below_lo, above_lo = r[r < lo].max(), r[(r >= lo) & (r < 1.5)].min()
    below_hi, above_hi = r[(r <= hi) & (r > 1.5)].max(), r[r > hi].min()
    q_lo = np.clip((x - below_lo) / (above_lo - below_lo), 0, 1)
    q_hi = np.clip((above_hi - x) / (above_hi - below_hi), 0, 1)
    g_lo, g_hi = (x > below_lo) & (x < above_lo), (x > below_hi) & (x < above_hi)
    P = np.where(g_lo, q_lo * (1 - err[0, 1, 0]) + (1 - q_lo) * err[0, 0, 0], P)
    P = np.where(g_hi, q_hi * (1 - err[1, 1, 0]) + (1 - q_hi) * err[1, 0, 0], P)
    return P, err, n_err


def m_E34(t, *aps):
    return [four_sided_probabilities(t, a)[0] for a in aps]


def m_E35(t, *aps):
    parts = [m(t, *aps) for m in (m_E29, m_E30, m_E31, m_E32, m_E34)]
    return [np.mean([p[i] for p in parts], 0) for i in range(len(aps))]


def band_grid(r, lo_rng, hi_rng):
    xs = np.unique(r)
    mids = (xs[:-1] + xs[1:]) / 2
    return mids[(mids >= lo_rng[0]) & (mids <= lo_rng[1])], mids[(mids >= hi_rng[0]) & (mids <= hi_rng[1])]


def grid_correct(r, y, L, H):
    """(n_lo, n_hi) correct counts of every fixed band [L_i, H_j] (separable: lo only acts below 1.5, hi above)."""
    lw, up = r < 1.5, r >= 1.5
    cl = ((r[lw][None, :] >= L[:, None]) == (y[lw][None, :] == 1)).sum(1)
    ch = ((r[up][None, :] <= H[:, None]) == (y[up][None, :] == 1)).sum(1)
    return cl[:, None] + ch[None, :]


def select_1se(t, lo_rng=(0.93, 1.05), hi_rng=(1.95, 2.07), seed=0):
    """E33: inner 5x2 CV accuracy of every fixed band; simplest (closest to 1, 2) within 1 SE of the best."""
    r, y = t.apg.values, t.y.values.astype(float)
    L, H = band_grid(r, lo_rng, hi_rng)
    accs = []
    for k in range(2):
        for _, v in StratifiedKFold(5, shuffle=True, random_state=seed + k).split(t, y):
            accs.append(grid_correct(r[v], y[v], L, H) / len(v))
    accs = np.array(accs)
    mean, se = accs.mean(0), accs.std(0, ddof=1) / np.sqrt(len(accs))
    i, j = np.unravel_index(mean.argmax(), mean.shape)
    ok = mean >= mean[i, j] - se[i, j]
    dist = np.abs(L[:, None] - 1.0) + np.abs(H[None, :] - 2.0)
    i2, j2 = np.unravel_index(np.where(ok, dist, np.inf).argmin(), dist.shape)
    return L[i2], H[j2]


def select_best(t, lo_rng=(0.98, 1.02), hi_rng=(1.98, 2.04)):
    """E36: band with the most correct training rows on the restricted grid (median of ties per side)."""
    r, y = t.apg.values, t.y.values.astype(float)
    L, H = band_grid(r, lo_rng, hi_rng)
    lw, up = r < 1.5, r >= 1.5
    cl = ((r[lw][None, :] >= L[:, None]) == (y[lw][None, :] == 1)).sum(1)
    ch = ((r[up][None, :] <= H[:, None]) == (y[up][None, :] == 1)).sum(1)
    bl, bh = np.flatnonzero(cl == cl.max()), np.flatnonzero(ch == ch.max())
    return L[bl[len(bl) // 2]], H[bh[len(bh) // 2]]


SELECTED = {"E33": [], "E36": []}


def m_E33(t, *aps):
    lo, hi = select_1se(t)
    SELECTED["E33"].append((lo, hi))
    return [hard_band(a.apg.values, lo, hi) for a in aps]


def m_E36(t, *aps):
    lo, hi = select_best(t)
    SELECTED["E36"].append((lo, hi))
    return [hard_band(a.apg.values, lo, hi) for a in aps]


MODELS = {"A (midcut band)": m_A, "E16 B's distance model": m_E16, "E29 bootstrap band": m_E29,
          "E30 isotonic edges": m_E30, "E31 Bayes change-point": m_E31, "E32 jackknife band": m_E32,
          "E34 four-sided": m_E34, "E35 consensus": m_E35, "E33 nested 1-SE band": m_E33, "E36 P02 test band": m_E36}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../outputs")
    ap.add_argument("--repeats", type=int, default=10)
    args, _ = ap.parse_known_args()
    t0 = time.time()
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    te = kp.clean(pd.read_csv(f"{args.data}/test.csv", dtype=str), False)
    r, y = tr.apg.values, tr.y.values.astype(float)

    # fast_midcut must reproduce the pipeline's cuts exactly, then the pipeline uses it too (speed only)
    rng = np.random.default_rng(1)
    for ii in [np.arange(len(r))] + [rng.integers(len(r), size=len(r)) for _ in range(200)]:
        assert np.allclose(fast_midcut(r[ii], y[ii]), kp.midcut_rafiur(r[ii], y[ii]), rtol=0, atol=0)
    kp.midcut_rafiur = fast_midcut
    lo_A, hi_A = fast_midcut(r, y)

    F = kp.make_folds(tr, args.repeats)
    d_ref = np.minimum(abs(r - lo_A), abs(r - hi_A))
    rows, oofs, folds = [], {}, {}
    for name, fn in MODELS.items():
        oof, accs = kp.run_cv(fn, tr, F)
        oofs[name], folds[name] = oof, accs
        pred = oof > 0.5
        row = dict(model=name, cv_acc=accs.mean(), cv_sd=accs.std(ddof=1))
        for thr in (0.01, 0.035, 0.1):
            m = d_ref < thr
            row[f"edge_acc<{thr}"] = (pred[m] == (y[m, None] == 1)).mean()
        m = d_ref < 0.1
        pc = np.clip(oof[m], 0.005, 0.995)
        row["edge_logloss"] = float(-(y[m, None] * np.log(pc) + (1 - y[m, None]) * np.log(1 - pc)).mean())
        row["edge_brier"] = float(((oof[m] - y[m, None]) ** 2).mean())
        row["size_stress"] = kp.size_stress(fn, tr)
        rows.append(row)
        print(f"  {name:24s} cv {row['cv_acc']:.4f}   [{time.time() - t0:.0f}s]", flush=True)

    base = "A (midcut band)"
    n_tr, n_te = len(tr) * 4 / 5, len(tr) / 5
    for row in rows:
        nm = row["model"]
        dm, se, p = kp.corrected_ttest(folds[nm], folds[base], n_tr, n_te)
        right, a_right = (oofs[nm] > 0.5) == (y[:, None] == 1), (oofs[base] > 0.5) == (y[:, None] == 1)
        b, c = (right & ~a_right).sum(0).mean(), (~right & a_right).sum(0).mean()
        b0, c0 = int((right[:, 0] & ~a_right[:, 0]).sum()), int((~right[:, 0] & a_right[:, 0]).sum())
        row.update(delta_vs_A=dm, p_corrected=p, rows_gained=b, rows_lost=c,
                   mcnemar_p_rep0=binomtest(b0, b0 + c0).pvalue if b0 + c0 else 1.0)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))

    for k in ("E33", "E36"):
        sel = np.array(SELECTED[k][: 5 * args.repeats])
        print(f"\n{k}: selected cuts over {len(sel)} outer folds: lo median {np.median(sel[:, 0]):.5f} "
              f"[{np.percentile(sel[:, 0], 5):.5f}, {np.percentile(sel[:, 0], 95):.5f}], hi median {np.median(sel[:, 1]):.5f} "
              f"[{np.percentile(sel[:, 1], 5):.5f}, {np.percentile(sel[:, 1], 95):.5f}]; "
              f"share lo > 1.00: {(sel[:, 0] > 1.0).mean():.2f}, share hi > 2.01: {(sel[:, 1] > 2.01).mean():.2f}")

    # ---- full-data fits and test predictions
    rd = lambda f: pd.read_csv(f"{args.out}/{f}").went_back_for_seconds.values.astype(int)
    A, B, C = rd("sub_A_primary_sharpband.csv"), rd("sub_B_hedge.csv"), rd("sub_C_band_1p00_1p975.csv")
    P02 = rd("portfolio25/P02.csv")
    tests, trows = {}, []
    for name, fn in MODELS.items():
        if name.startswith("E29"):
            p = m_E29(tr, te, B=5000)[0]
        elif name.startswith("E35"):
            p = np.mean([tests[k] for k in tests if k[:3] in ("E29", "E30", "E31", "E32", "E34")], 0)
        else:
            p = fn(tr, te)[0]
        tests[name] = p
        v = (p > 0.5).astype(int)
        trows.append(dict(model=name, ones=int(v.sum()), uncertain_25_75=int(((p > 0.25) & (p < 0.75)).sum()),
                          ham_A=int((v != A).sum()), ham_B=int((v != B).sum()), ham_C=int((v != C).sum()),
                          ham_P02=int((v != P02).sum()),
                          flips_vs_A=" ".join(f"{te.wedding_id[i]}:{A[i]}>{v[i]}" for i in np.where(v != A)[0])))
    print("\n== test set")
    print(pd.DataFrame(trows).to_string(index=False, max_colwidth=120))
    print(f"\nfull-data E33 cuts {select_1se(tr)}, E36 cuts {select_best(tr)}")
    _, l_post, l_mid = changepoint_edge(r, y, (0.85, 1.15), np.array([1.0]))
    _, h_post, h_mid = changepoint_edge(r, y, (1.80, 2.25), np.array([2.0]))
    for nm, post, mid in (("lower", l_post, l_mid), ("upper", h_post, h_mid)):
        cdf = np.cumsum(post)
        q = [mid[np.searchsorted(cdf, s)] for s in (0.025, 0.25, 0.5, 0.75, 0.975)]
        print(f"E31 {nm} cut posterior quantiles 2.5/25/50/75/97.5%: {np.round(q, 4)}")
    rng = np.random.default_rng(0)
    bc = np.array([fast_midcut(r[ii], y[ii]) for ii in (rng.integers(len(r), size=len(r)) for _ in range(5000))])
    for k, nm in enumerate(("lower", "upper")):
        print(f"E29 bootstrap {nm} cut quantiles 2.5/25/50/75/97.5%: {np.round(np.percentile(bc[:, k], [2.5, 25, 50, 75, 97.5]), 4)}")

    od = f"{args.out}/edge_models"
    os.makedirs(od, exist_ok=True)
    res.to_csv(f"{od}/cv_metrics.csv", index=False)
    pd.DataFrame(trows).to_csv(f"{od}/test_summary.csv", index=False)
    pd.DataFrame({"wedding_id": te.wedding_id, "apg": te.apg, **{k: v for k, v in tests.items()}}).to_csv(f"{od}/test_probabilities.csv", index=False)
    with open(f"{od}/selected_cuts.json", "w") as f:
        json.dump({k: [list(map(float, c)) for c in v] for k, v in SELECTED.items()}, f)
    print(f"wrote {od}   done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
