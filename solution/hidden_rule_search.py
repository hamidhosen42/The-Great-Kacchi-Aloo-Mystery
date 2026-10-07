"""E18: search for a hidden rule behind the edge errors.

Hypothesis family: each edge is a sharp cut on a *modified* ratio
    r = apg * exp(beta * z)        (z = a standardised column: the cut shifts multiplicatively with z)
or on an alternative ratio (aloo / mutton, aloo / (guests - k*col), ...).
The two edges are searched separately (lower: apg < 1.5, upper: apg >= 1.5).

Step 1: in-sample best errors per (edge, covariate, beta) vs the plain cut (optimistic, for ranking).
Step 2: honest 10x5 repeated CV of the top candidates, where beta and the cuts are re-fitted inside every fold.

    python hidden_rule_search.py --data ../data
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
import kacchi_pipeline as kp  # noqa: E402

BETAS = np.round(np.arange(-0.20, 0.2001, 0.01), 3)


def best_cut_errors(r, y, side):
    """Min errors of a single cut on r: side='lo' predicts 1 above the cut, 'hi' predicts 1 below it."""
    o = np.argsort(r); ys = y[o]
    if side == "lo":   # errors = positives below + negatives above
        below_pos = np.r_[0, np.cumsum(ys)]
        above_neg = np.r_[np.cumsum((1 - ys)[::-1])[::-1], 0]
        e = below_pos + above_neg
    else:              # errors = negatives below + positives above
        below_neg = np.r_[0, np.cumsum(1 - ys)]
        above_pos = np.r_[np.cumsum(ys[::-1])[::-1], 0]
        e = below_neg + above_pos
    k = int(np.argmin(e)); rs = r[o]
    cut = (rs[k - 1] + rs[k]) / 2 if 0 < k < len(rs) else (rs[0] - 1e-9 if k == 0 else rs[-1] + 1e-9)
    return int(e[k]), cut


def covariates(d):
    Z = {}
    for c in ["mutton_pg", "borhani_pg", "fairy_pg", "guests"]:
        Z[f"log_{c}"] = np.log(d[c].values)
    for c in ["dhol_players", "aunties_asking_when_marriage", "drone"]:
        Z[c] = d[c].values.astype(float)
    for e in ["Gaye Holud", "Biye", "Bou-bhat"]:
        Z[f"event={e}"] = (d.event == e).values.astype(float)
    for c in sorted(d.city.unique()):
        Z[f"city={c}"] = (d.city == c).values.astype(float)
    Z["id_num"] = d.id_num.values.astype(float)
    Z["aloo_mod10"] = (d.aloo.values % 10).astype(float)
    Z["guests_mod10"] = (d.guests.values % 10).astype(float)
    return Z


def standardise(z, ref):
    s = ref.std()
    return (z - ref.mean()) / (s if s > 0 else 1.0)


def fit_edge(apg, y, z, side):
    """Best (beta, cut) for one edge on rows of that edge; returns (errors, beta, cut)."""
    best = None
    for b in BETAS:
        e, cut = best_cut_errors(apg * np.exp(b * z), y, side)
        if best is None or e < best[0] or (e == best[0] and abs(b) < abs(best[1])):
            best = (e, b, cut)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--top", type=int, default=6)
    args, _ = ap.parse_known_args()
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    Zall = covariates(tr)
    out = []
    for side, mask in [("lo", tr.apg.values < 1.5), ("hi", tr.apg.values >= 1.5)]:
        apg, y = tr.apg.values[mask], tr.y.values[mask]
        e0, c0 = best_cut_errors(apg, y, side)
        print(f"[{side}] plain cut {c0:.5f}: {e0} errors on {mask.sum()} rows")
        for name, zf in Zall.items():
            z = standardise(zf[mask], zf)
            e, b, cut = fit_edge(apg, y, z, side)
            out.append(dict(edge=side, covariate=name, errors=e, plain=e0, gain=e0 - e, beta=b, cut=cut))
        # alternative ratios for this edge
        for name, r in {"aloo/mutton_kg": tr.aloo.values / tr.mutton_kg.values, "aloo/borhani": tr.aloo.values / tr.borhani_glasses.values,
                        "aloo/(guests-dhol)": tr.aloo.values / (tr.guests.values - tr.dhol_players.values),
                        "aloo/(guests-aunties)": tr.aloo.values / (tr.guests.values - tr.aunties_asking_when_marriage.values),
                        "(aloo-aunties)/guests": (tr.aloo.values - tr.aunties_asking_when_marriage.values) / tr.guests.values,
                        "(aloo+dhol)/guests": (tr.aloo.values + tr.dhol_players.values) / tr.guests.values}.items():
            e, cut = best_cut_errors(r[mask], y, side)
            out.append(dict(edge=side, covariate="RATIO " + name, errors=e, plain=e0, gain=e0 - e, beta=np.nan, cut=cut))
    R = pd.DataFrame(out).sort_values(["gain"], ascending=False)
    pd.set_option("display.width", 200)
    print("\nIn-sample (optimistic) top candidates:")
    print(R.head(20).to_string(index=False))

    # ---------------- honest CV of the top candidates (beta + cuts re-fitted per fold)
    F = kp.make_folds(tr, args.repeats)
    n_tr, n_te = len(tr) * 4 / 5, len(tr) / 5

    def predictor(cov_lo, cov_hi):
        def fn(t, a):
            pred_lo, pred_hi = np.ones(len(a), bool), np.ones(len(a), bool)
            for side, cov in (("lo", cov_lo), ("hi", cov_hi)):
                mt = t.apg.values < 1.5 if side == "lo" else t.apg.values >= 1.5
                zt_full = covariates(t)[cov] if cov else np.zeros(len(t))
                za_full = covariates(a)[cov] if cov else np.zeros(len(a))
                zt, za = standardise(zt_full, zt_full), standardise(za_full, zt_full)
                if cov:
                    _, b, cut = fit_edge(t.apg.values[mt], t.y.values[mt], zt[mt], side)
                else:
                    b, cut = 0.0, best_cut_errors(t.apg.values[mt], t.y.values[mt], side)[1]
                r = a.apg.values * np.exp(b * za)
                if side == "lo":
                    pred_lo = r > cut
                else:
                    pred_hi = r < cut
            return (pred_lo & pred_hi).astype(float)
        return lambda t, *aps: [fn(t, a) for a in aps]

    _, acc0 = kp.run_cv(predictor(None, None), tr, F)
    print(f"\nCV plain sharp band: {acc0.mean():.4f} ± {acc0.std():.4f}")
    cands = R[~R.covariate.str.startswith("RATIO")].groupby("edge").head(args.top)
    rows = []
    for _, c in cands.iterrows():
        cov_lo, cov_hi = (c.covariate, None) if c.edge == "lo" else (None, c.covariate)
        _, acc = kp.run_cv(predictor(cov_lo, cov_hi), tr, F)
        d, se, p = kp.corrected_ttest(acc, acc0, n_tr, n_te)
        rows.append(dict(edge=c.edge, covariate=c.covariate, insample_gain=c.gain, beta_full=c.beta, cv=acc.mean(), delta=d, p=p))
    C = pd.DataFrame(rows).sort_values("p")
    C["bonferroni_p"] = np.minimum(C.p * len(Zall) * 2, 1)
    print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
