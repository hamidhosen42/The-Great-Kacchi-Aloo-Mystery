"""Approximate Bayesian Computation over the band DGP, using only numbers published in notebook logs.

DGP family:  apg ~ U(0.4028, 2.70);  y ~ Bernoulli( s(k*(g(apg)-g(lo))) * s(k*(g(hi)-g(apg))) )
             g = identity (form 'lin') or log (form 'log');  s = logistic.
Prior:       form ~ {lin, log};  lo ~ U(0.95, 1.03);  hi ~ U(1.96, 2.05);  k ~ logU(10, 120).
Observed summaries (776 clean train rows, from Kaggle execution logs of public notebooks):
  * positive counts in 10 edge bins (0.1 wide) of the binned table,
  * errors of the fixed (1,2) rule = 43, errors of the accuracy-optimal band = 33,
  * accuracy-optimal cut-points (plateau centre, 0.0025 grid) = (0.9700, 2.0087).
Output: ABC posterior (nearest 0.5% by standardised distance), posterior-predictive Bayes cut-points,
and expected PRIVATE accuracy (219 rows) of candidate rules averaged over the posterior.
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import expit

LO_G = np.arange(0.85, 1.15, 0.0025)
HI_G = np.arange(1.85, 2.20, 0.0025)
EDGE_BINS = [(0.9, 1.0), (1.0, 1.1), (1.1, 1.2), (1.6, 1.7), (1.7, 1.8), (1.8, 1.9), (1.9, 2.0), (2.0, 2.1), (2.1, 2.2), (2.2, 2.3)]
OBS_BIN_POS = np.array([12, 25, 36, 23, 38, 34, 18, 11, 1, 2], float)
OBS_BIN_N = np.array([37, 32, 37, 24, 39, 37, 23, 37, 35, 35], float)
OBS = dict(err12=43, erropt=33, alo=0.9700, ahi=2.0087)


def p_of(x, form, lo, hi, k):
    if form == "log":
        z = np.log(x)
        return expit(k * (z - np.log(lo))) * expit(k * (np.log(hi) - z))
    return expit(k * (x - lo)) * expit(k * (hi - x))


def accopt(x, y):
    m = x < 1.5
    xl, yl = x[m], y[m]
    xh, yh = x[~m], y[~m]
    e_lo = ((xl[None, :] < LO_G[:, None]) & (yl[None, :] == 1)).sum(1) + ((xl[None, :] > LO_G[:, None]) & (yl[None, :] == 0)).sum(1)
    e_hi = ((xh[None, :] > HI_G[:, None]) & (yh[None, :] == 1)).sum(1) + ((xh[None, :] < HI_G[:, None]) & (yh[None, :] == 0)).sum(1)
    return LO_G[e_lo == e_lo.min()].mean(), HI_G[e_hi == e_hi.min()].mean(), int(e_lo.min() + e_hi.min())


def one(seed):
    rng = np.random.default_rng(seed)
    form = "log" if rng.random() < 0.5 else "lin"
    lo = rng.uniform(0.95, 1.03); hi = rng.uniform(1.96, 2.05); k = float(np.exp(rng.uniform(np.log(10), np.log(120))))
    x = rng.uniform(0.4028, 2.70, 776)
    y = rng.binomial(1, p_of(x, form, lo, hi, k))
    rates = []
    for a, b in EDGE_BINS:
        mm = (x > a) & (x <= b)
        rates.append(y[mm].mean() if mm.any() else 0.0)
    err12 = int((((x > 1) & (x < 2)).astype(int) != y).sum())
    alo, ahi, erropt = accopt(x, y)
    return [0 if form == "lin" else 1, lo, hi, k, err12, erropt, alo, ahi] + rates


def private_eval(args):
    """Expected private accuracy of candidate rules under one posterior draw (analytic in y, MC in x)."""
    form, lo, hi, k, seed, rules = args
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.4028, 2.70, (200, 219))
    p = p_of(x, "log" if form == 1 else "lin", lo, hi, k)
    out = {}
    for name, (a, b) in rules.items():
        pred = (x > a) & (x < b)
        out[name] = float(np.where(pred, p, 1 - p).mean())
    out["bayes"] = float(np.maximum(p, 1 - p).mean())
    return out


RULES = {}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    OUT = sys.argv[1]
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    t0 = time.time()
    with Pool(12) as pool:
        S = np.array(pool.map(one, range(NS), chunksize=500))
    print(f"{NS} simulations in {time.time() - t0:.0f}s")
    form, lo, hi, k = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    summ = S[:, 4:]
    obs = np.array([OBS["err12"], OBS["erropt"], OBS["alo"], OBS["ahi"]] + list(OBS_BIN_POS / OBS_BIN_N))
    sd = summ.std(0)
    d = np.sqrt((((summ - obs) / sd) ** 2).sum(1))
    keep = d <= np.quantile(d, 0.005)
    print(f"accepted {keep.sum()} (0.5%), distance cut {np.quantile(d, 0.005):.3f}")
    post = dict(p_log_form=float(form[keep].mean()),
                lo_mean=float(lo[keep].mean()), lo_q=np.quantile(lo[keep], [.025, .25, .5, .75, .975]).round(4).tolist(),
                hi_mean=float(hi[keep].mean()), hi_q=np.quantile(hi[keep], [.025, .25, .5, .75, .975]).round(4).tolist(),
                k_median=float(np.median(k[keep])), k_q=np.quantile(k[keep], [.025, .5, .975]).round(1).tolist())
    print("posterior:", json.dumps(post, indent=1))
    # how well do accepted sims reproduce the two error counts?
    print("accepted sims: err(1,2) mean %.1f | err(accopt) mean %.1f | accopt lo %.4f hi %.4f" %
          tuple(summ[keep][:, :4].mean(0)))
    # posterior predictive P(y=1|x) and the Bayes cut-points it implies
    xs = np.arange(0.85, 2.20, 0.0005)
    PP = np.mean([p_of(xs, "log" if f == 1 else "lin", a, b, kk) for f, a, b, kk in zip(form[keep], lo[keep], hi[keep], k[keep])], 0)
    pos = xs[PP > 0.5]
    bayes_lo, bayes_hi = float(pos.min()), float(pos.max())
    print(f"posterior-predictive Bayes cut-points: ({bayes_lo:.4f}, {bayes_hi:.4f})")
    for q in [0.97, 0.98, 0.99, 1.0, 1.01, 1.98, 1.99, 2.0, 2.01, 2.02]:
        print(f"   P(y=1 | apg={q:.2f}) = {PP[np.argmin(abs(xs - q))]:.3f}")
    RULES.update({"fixed_1_2": (1.0, 2.0), "accopt_0.970_2.0087": (0.970, 2.0087), "logmle_0.9911_1.9912": (0.9911, 1.9912),
                  "rafiur_0.971_2.010": (0.971, 2.010), "posterior_bayes": (bayes_lo, bayes_hi)})
    idx = np.flatnonzero(keep)
    with Pool(12) as pool:
        ev = pool.map(private_eval, [(form[i], lo[i], hi[i], k[i], int(i), RULES) for i in idx])
    names = list(ev[0].keys())
    M = np.array([[e[n] for n in names] for e in ev])
    print("\nExpected PRIVATE accuracy (219 rows), averaged over the ABC posterior:")
    res = {}
    for j, n in enumerate(names):
        res[n] = dict(mean=float(M[:, j].mean()), rows_vs_fixed=float((M[:, j] - M[:, names.index('fixed_1_2')]).mean() * 219))
        print(f"   {n:24s} {M[:, j].mean():.4f}   Δ vs fixed(1,2): {res[n]['rows_vs_fixed']:+.2f} rows")
    json.dump(dict(n_sims=NS, accepted=int(keep.sum()), posterior=post, bayes_cutpoints=[bayes_lo, bayes_hi],
                   pp_curve={f"{q:.2f}": float(PP[np.argmin(abs(xs - q))]) for q in np.arange(0.90, 1.105, 0.01).tolist() + np.arange(1.90, 2.105, 0.01).tolist()},
                   private_expectation=res),
              open(f"{OUT}/abc_summary.json", "w"), indent=1)
    print("wrote abc_summary.json")
