"""ABC model comparison: logistic-edge vs heavy-tailed 'noisy ratio' (Student-t) DGPs, linear vs log space.

y = 1{ lo < g^-1( g(apg) + sigma * T_nu ) < hi }  =>  P(y=1|apg) = F_nu((g(apg)-g(lo))/sigma) - F_nu((g(apg)-g(hi))/sigma)
Logistic family kept for reference (k = 1/sigma). Same summaries/observations as abc.py.
"""
import json, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t
from abc_band import accopt, EDGE_BINS, OBS_BIN_POS, OBS_BIN_N, OBS  # noqa  (local abc.py)

FAMS = ["lin_logit", "log_logit", "lin_t", "log_t"]


def p_of(x, fam, lo, hi, s, nu):
    g = np.log if fam.startswith("log") else (lambda v: v)
    a, b = (g(x) - g(lo)) / s, (g(x) - g(hi)) / s
    if fam.endswith("logit"):
        return expit(a) * expit(-b)
    return np.clip(student_t.cdf(a, nu) - student_t.cdf(b, nu), 0, 1)


def draw(rng):
    fam = FAMS[rng.integers(4)]
    lo = rng.uniform(0.95, 1.03); hi = rng.uniform(1.96, 2.05)
    s = float(np.exp(rng.uniform(np.log(0.004), np.log(0.1))))
    nu = float(np.exp(rng.uniform(np.log(0.3), np.log(50))))
    return fam, lo, hi, s, nu


def one(seed):
    rng = np.random.default_rng(seed)
    fam, lo, hi, s, nu = draw(rng)
    x = rng.uniform(0.4028, 2.70, 776)
    y = rng.binomial(1, p_of(x, fam, lo, hi, s, nu))
    rates = []
    for a, b in EDGE_BINS:
        m = (x > a) & (x <= b)
        rates.append(y[m].mean() if m.any() else 0.0)
    far = int(y[(x <= 0.9) | (x > 2.3)].sum() + (1 - y[(x > 1.2) & (x <= 1.6)]).sum())  # obs: 0 flips far from edges
    err12 = int((((x > 1) & (x < 2)).astype(int) != y).sum())
    alo, ahi, erropt = accopt(x, y)
    return [FAMS.index(fam), lo, hi, s, nu, err12, erropt, alo, ahi, far] + rates


def private_eval(args):
    fam, lo, hi, s, nu, seed, rules = args
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.4028, 2.70, (100, 219))
    p = p_of(x, fam, lo, hi, s, nu)
    out = {n: float(np.where((x > a) & (x < b), p, 1 - p).mean()) for n, (a, b) in rules.items()}
    out["bayes"] = float(np.maximum(p, 1 - p).mean())
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    OUT, NS = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    with Pool(12) as pool:
        S = np.array(pool.map(one, range(10**7, 10**7 + NS), chunksize=500))
    print(f"{NS} sims in {time.time()-t0:.0f}s")
    fam, lo, hi, s, nu = S[:, 0].astype(int), S[:, 1], S[:, 2], S[:, 3], S[:, 4]
    summ = S[:, 5:]
    obs = np.array([OBS["err12"], OBS["erropt"], OBS["alo"], OBS["ahi"], 0] + list(OBS_BIN_POS / OBS_BIN_N))
    sd = summ.std(0)
    d = np.sqrt((((summ - obs) / sd) ** 2).sum(1))
    keep = d <= np.quantile(d, 0.0025)
    print(f"accepted {keep.sum()} (0.25%), cut {np.quantile(d, .0025):.3f}")
    fam_post = {FAMS[i]: float((fam[keep] == i).mean()) for i in range(4)}
    print("posterior family probabilities (equal priors):", {k: round(v, 3) for k, v in fam_post.items()})
    print("accepted: err12 %.1f  erropt %.1f  alo %.4f  ahi %.4f  far-flips %.2f" % tuple(summ[keep][:, :5].mean(0)))
    q = lambda v: np.quantile(v[keep], [.025, .25, .5, .75, .975]).round(4).tolist()
    post = dict(family=fam_post, lo=q(lo), hi=q(hi), sigma=q(s), nu=q(nu))
    print("lo  quantiles", post["lo"]); print("hi  quantiles", post["hi"]); print("sigma", post["sigma"]); print("nu", post["nu"])
    xs = np.arange(0.85, 2.20, 0.0005)
    idx = np.flatnonzero(keep)
    PP = np.mean([p_of(xs, FAMS[fam[i]], lo[i], hi[i], s[i], nu[i]) for i in idx], 0)
    pos = xs[PP > 0.5]; bl, bh = float(pos.min()), float(pos.max())
    print(f"posterior-predictive Bayes cut-points: ({bl:.4f}, {bh:.4f})")
    pp = {f"{v:.3f}": round(float(PP[np.argmin(abs(xs - v))]), 3) for v in [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 1.0, 1.01, 1.02, 1.03, 1.97, 1.98, 1.99, 2.0, 2.005, 2.01, 2.02, 2.03]}
    print("P(y=1|apg):", pp)
    rules = {"fixed_1_2": (1.0, 2.0), "accopt_0.970_2.0087": (0.970, 2.0087), "logmle_0.9911_1.9912": (0.9911, 1.9912),
             "rafiur_0.971_2.010": (0.971, 2.010), "posterior_bayes": (bl, bh)}
    with Pool(12) as pool:
        ev = pool.map(private_eval, [(FAMS[fam[i]], lo[i], hi[i], s[i], nu[i], int(i), rules) for i in idx])
    names = list(ev[0]); M = np.array([[e[n] for n in names] for e in ev]); j0 = names.index("fixed_1_2")
    res = {}
    print("Expected PRIVATE accuracy (219 rows) over posterior:")
    for j, n in enumerate(names):
        res[n] = dict(mean=float(M[:, j].mean()), rows_vs_fixed=float((M[:, j] - M[:, j0]).mean() * 219))
        print(f"   {n:24s} {M[:, j].mean():.4f}  Δ vs fixed(1,2) {res[n]['rows_vs_fixed']:+.2f} rows")
    json.dump(dict(n_sims=NS, accepted=int(keep.sum()), posterior=post, bayes_cutpoints=[bl, bh], pp=pp, private=res),
              open(f"{OUT}/abc2_summary.json", "w"), indent=1)
