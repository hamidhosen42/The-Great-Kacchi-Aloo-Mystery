"""Simulation study for The Great Kacchi Aloo Mystery (runs WITHOUT competition data).

Inputs are only numbers printed in the public notebooks' Kaggle execution logs:
  * the 0.1-wide binned table of P(seconds | aloo-per-guest) with bin counts (776 clean train rows),
  * apg range [0.4028, 2.70] in train+test, 400 test rows, public LB = 181 rows.

Steps
  1. Fit candidate data-generating processes (DGPs) to the binned table by binned maximum likelihood.
  2. Monte-Carlo: generate train (776) / test (400 = 181 public + 219 private) from a DGP and compare
     threshold procedures used by public notebooks + alternatives, measured on the PRIVATE part.
  3. CV selection-bias test: how often does 5-fold CV prefer the accuracy-optimal band over the fixed
     (1,2) band by >= the gap actually observed in the public notebooks (+0.0112), when (1,2) is true?
  4. Final-selection hedge: E[max(private acc A, private acc B)] when B flips the m most uncertain rows.
  5. Leaderboard noise: distribution of public / private accuracy of the Bayes rule.
"""
import json
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8")
OUT = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rng = np.random.default_rng(20261007)

# ---------------------------------------------------------------- 1. binned table from notebook logs
edges = np.round(np.arange(0.4, 2.81, 0.1), 2)
means = np.array([0, 0, 0, 0, 0, .32, .78, .97, 1, 1, 1, 1, .96, .97, .92, .78, .30, .03, .06, 0, 0, 0, 0, 0])
counts = np.array([37, 33, 32, 36, 39, 37, 32, 37, 34, 38, 31, 25, 24, 39, 37, 23, 37, 35, 35, 30, 31, 36, 37, 1])
pos = np.round(means * counts).astype(int)
assert counts.sum() == 776 and pos.sum() == 328, (counts.sum(), pos.sum())
lo_e, hi_e = edges[:-1], edges[1:]
lo_e = lo_e.copy(); lo_e[0] = 0.4028; hi_e = hi_e.copy(); hi_e[-1] = 2.70  # observed support


def p_band(x, lo, hi, k_lo, k_hi, form):
    if form == "lin_logit":
        return expit(k_lo * (x - lo)) * expit(k_hi * (hi - x))
    if form == "log_logit":
        z = np.log(x)
        return expit(k_lo * (z - np.log(lo))) * expit(k_hi * (np.log(hi) - z))
    if form == "lin_probit":  # y = 1{lo < x + eps < hi}, eps ~ N(0, 1/k)
        return np.clip(norm.cdf((x - lo) * k_lo) - norm.cdf((x - hi) * k_hi), 0, 1)
    if form == "log_probit":
        z = np.log(x)
        return np.clip(norm.cdf((z - np.log(lo)) * k_lo) - norm.cdf((z - np.log(hi)) * k_hi), 0, 1)
    raise ValueError(form)


def bin_p(theta, form, sym):
    lo, hi, k1 = theta[:3]
    k2 = k1 if sym else theta[3]
    xs = lo_e[:, None] + (hi_e - lo_e)[:, None] * (np.arange(200)[None, :] + 0.5) / 200
    return p_band(xs, lo, hi, k1, k2, form).mean(1)


def nll_binned(theta, form, sym):
    if theta[2] <= 0 or (not sym and theta[3] <= 0):
        return 1e9
    p = np.clip(bin_p(theta, form, sym), 1e-9, 1 - 1e-9)
    return -(pos * np.log(p) + (counts - pos) * np.log(1 - p)).sum()


fits = {}
print("=== 1. DGP fits to the binned public table (776 rows) ===")
for form in ["lin_logit", "log_logit", "lin_probit", "log_probit"]:
    for sym in [True, False]:
        k0 = {"lin_logit": 20, "log_logit": 25, "lin_probit": 12, "log_probit": 15}[form]
        x0 = [1.0, 2.0, k0] if sym else [1.0, 2.0, k0, k0]
        best = None
        for start_lo in (0.97, 1.0, 1.03):
            x0[0] = start_lo
            r = minimize(nll_binned, x0, args=(form, sym), method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
            if best is None or r.fun < best.fun:
                best = r
        npar = 3 if sym else 4
        name = f"{form}_{'sym' if sym else 'asym'}"
        fits[name] = dict(theta=best.x.tolist(), nll=best.fun, aic=2 * best.fun + 2 * npar, form=form, sym=sym)
        print(f"{name:18s} lo={best.x[0]:.4f} hi={best.x[1]:.4f} k={np.round(best.x[2:], 2).tolist()}  NLL={best.fun:.3f}  AIC={2 * best.fun + 2 * npar:.2f}")

# profile-likelihood 95% CI for lo and hi under the best symmetric linear-logit and log-logit fits
def profile_ci(name, idx):
    f = fits[name]; th = np.array(f["theta"]); base = f["nll"]
    grid = np.arange(th[idx] - 0.08, th[idx] + 0.08, 0.0025)
    ok = []
    for g in grid:
        free = [i for i in range(len(th)) if i != idx]
        def obj(v):
            t = th.copy(); t[free] = v; t[idx] = g
            return nll_binned(t, f["form"], f["sym"])
        r = minimize(obj, th[free], method="Nelder-Mead", options=dict(maxiter=2000))
        if 2 * (r.fun - base) <= 3.84:
            ok.append(g)
    return (min(ok), max(ok)) if ok else (np.nan, np.nan)

ci = {}
for name in ["lin_logit_sym", "log_logit_sym"]:
    ci[name] = dict(lo=profile_ci(name, 0), hi=profile_ci(name, 1))
    print(f"profile 95% CI {name}: lo in [{ci[name]['lo'][0]:.4f}, {ci[name]['lo'][1]:.4f}]  hi in [{ci[name]['hi'][0]:.4f}, {ci[name]['hi'][1]:.4f}]")

# ---------------------------------------------------------------- 2. procedures
LO_G = np.arange(0.85, 1.15, 0.0025)
HI_G = np.arange(1.85, 2.20, 0.0025)


def fit_band_accopt(x, y):
    """FOYSAL-style accuracy-optimal (lo, hi) with ties -> centre of plateau. Separable because lo<1.15<1.85<hi."""
    # errors from lower edge: positives below lo + negatives in [lo, 1.5)
    m_lo = x < 1.5
    e_lo = np.array([((x < g) & m_lo & (y == 1)).sum() + ((x > g) & m_lo & (y == 0)).sum() for g in LO_G])
    m_hi = ~m_lo
    e_hi = np.array([((x > g) & m_hi & (y == 1)).sum() + ((x < g) & m_hi & (y == 0)).sum() for g in HI_G])
    return LO_G[e_lo == e_lo.min()].mean(), HI_G[e_hi == e_hi.min()].mean()


def fit_mle(x, y, form):
    def nll(t):
        if t[2] <= 0:
            return 1e9
        p = np.clip(p_band(x, t[0], t[1], t[2], t[2], form), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    k0 = 20 if form == "lin_logit" else 25
    r = minimize(nll, [1.0, 2.0, k0], method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-5, fatol=1e-6))
    return r.x


def knn_log(xtr, ytr, xte, k=5):
    a, b = np.log(xtr), np.log(xte)
    idx = np.argsort(np.abs(b[:, None] - a[None, :]), axis=1)[:, :k]
    return (ytr[idx].mean(1) > 0.5).astype(int)


def band(x, lo, hi):
    return ((x > lo) & (x < hi)).astype(int)


def simulate_dgp(name, R, with_lgbm=False):
    f = fits[name]; th = f["theta"]; form = f["form"]
    lo, hi, k1 = th[:3]; k2 = k1 if f["sym"] else th[3]
    P = lambda x: p_band(x, lo, hi, k1, k2, form)
    res = {k: [] for k in ["bayes", "fixed_1_2", "accopt_band", "mle_lin", "mle_log", "knn5_log"] + (["lgbm_log"] if with_lgbm else [])}
    pub = {k: [] for k in res}
    cvgap, trgap, thr = [], [], []
    hedge = {m: [] for m in [0, 2, 4, 6, 8, 12, 16, 24]}
    t0 = time.time()
    if with_lgbm:
        import lightgbm as lgb
    for r in range(R):
        xtr = rng.uniform(0.4028, 2.70, 776); ytr = rng.binomial(1, P(xtr))
        xte = rng.uniform(0.4028, 2.70, 400); yte = rng.binomial(1, P(xte))
        is_pub = np.zeros(400, bool); is_pub[rng.choice(400, 181, replace=False)] = True
        preds = {}
        preds["bayes"] = (P(xte) > 0.5).astype(int)
        preds["fixed_1_2"] = band(xte, 1, 2)
        alo, ahi = fit_band_accopt(xtr, ytr); preds["accopt_band"] = band(xte, alo, ahi); thr.append((alo, ahi))
        t = fit_mle(xtr, ytr, "lin_logit"); preds["mle_lin"] = band(xte, t[0], t[1])
        t = fit_mle(xtr, ytr, "log_logit"); preds["mle_log"] = band(xte, t[0], t[1])
        preds["knn5_log"] = knn_log(xtr, ytr, xte)
        if with_lgbm:
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=4, min_child_samples=15,
                                   subsample=0.8, subsample_freq=1, random_state=r, verbose=-1)
            preds["lgbm_log"] = m.fit(np.log(xtr)[:, None], ytr).predict(np.log(xte)[:, None])
        for k, p in preds.items():
            res[k].append((p[~is_pub] == yte[~is_pub]).mean())
            pub[k].append((p[is_pub] == yte[is_pub]).mean())
        # 3. CV gap (accopt band re-fit per fold) vs fixed (1,2); and in-sample error gap
        folds = rng.permutation(776) % 5
        acc_cv = np.mean([(band(xtr[folds == j], *fit_band_accopt(xtr[folds != j], ytr[folds != j])) == ytr[folds == j]).mean() for j in range(5)])
        acc_fixed = (band(xtr, 1, 2) == ytr).mean()
        cvgap.append(acc_cv - acc_fixed)
        trgap.append(int((band(xtr, 1, 2) != ytr).sum() - (band(xtr, alo, ahi) != ytr).sum()))
        # 4. hedge: A = MLE-lin band; B = A with the m test rows of smallest |P_hat-0.5| flipped
        th_hat = fit_mle(xtr, ytr, "lin_logit")
        phat = p_band(xte, th_hat[0], th_hat[1], th_hat[2], th_hat[2], "lin_logit")
        A = band(xte, th_hat[0], th_hat[1]); order = np.argsort(np.abs(phat - 0.5))
        accA = (A[~is_pub] == yte[~is_pub]).mean()
        for m in hedge:
            B = A.copy(); B[order[:m]] = 1 - B[order[:m]]
            hedge[m].append(max(accA, (B[~is_pub] == yte[~is_pub]).mean()))
    print(f"  [{name}] {R} replicates in {time.time() - t0:.0f}s")
    return res, pub, np.array(cvgap), np.array(trgap), np.array(thr), hedge


summary = {"fits": fits, "profile_ci": ci, "R": R, "dgps": {}}
for name in ["lin_logit_sym", "log_logit_sym"]:
    print(f"\n=== 2-5. Monte Carlo under DGP {name} (theta={np.round(fits[name]['theta'], 4).tolist()}) ===")
    res, pub, cvgap, trgap, thr, hedge = simulate_dgp(name, R, with_lgbm=(name == "lin_logit_sym"))
    rows = {}
    base = np.array(res["accopt_band"])
    for k, v in res.items():
        v = np.array(v)
        rows[k] = dict(private_mean=v.mean(), private_sd=v.std(), private_p05=np.quantile(v, .05), private_p95=np.quantile(v, .95),
                       mean_rows_vs_accopt=((v - base) * 219).mean(), p_beats_accopt=(v > base).mean(), p_ties_accopt=(v == base).mean(),
                       public_mean=np.mean(pub[k]), public_sd=np.std(pub[k]))
        print(f"  {k:12s} private {v.mean():.4f} ± {v.std():.4f}  (5-95%: {np.quantile(v, .05):.4f}-{np.quantile(v, .95):.4f})"
              f"  Δrows vs accopt {((v - base) * 219).mean():+.2f}  P(beat)={(v > base).mean():.2f} P(tie)={(v == base).mean():.2f}"
              f"  | public {np.mean(pub[k]):.4f} ± {np.std(pub[k]):.4f}")
    obs_cv_gap, obs_tr_gap = 0.9558 - 0.9446, 10
    p_cv = (cvgap >= obs_cv_gap).mean(); p_tr = (trgap >= obs_tr_gap).mean()
    print(f"  CV gap (accopt-CV minus fixed(1,2)): mean {cvgap.mean():+.4f}, sd {cvgap.std():.4f}; P(gap >= observed {obs_cv_gap:+.4f}) = {p_cv:.3f}")
    print(f"  in-sample error reduction from optimising cut-points: mean {trgap.mean():.1f} rows, P(>= observed 10) = {p_tr:.3f}")
    print(f"  accopt thresholds: lo mean {thr[:, 0].mean():.4f} sd {thr[:, 0].std():.4f} | hi mean {thr[:, 1].mean():.4f} sd {thr[:, 1].std():.4f}")
    hm = {m: float(np.mean(v)) for m, v in hedge.items()}
    print("  hedge E[max(A,B)] private acc by #flipped rows m:", {m: round(v, 4) for m, v in hm.items()},
          f" -> best gain {max(hm.values()) - hm[0]:+.4f} (= {(max(hm.values()) - hm[0]) * 219:+.2f} rows)")
    pb = np.array(pub["bayes"])
    print(f"  Bayes rule public acc: mean {pb.mean():.4f} sd {pb.std():.4f}; P(public >= 175/181) = {(pb >= 175 / 181 - 1e-9).mean():.3f}; P(<= 172/181) = {(pb <= 172 / 181 + 1e-9).mean():.3f}")
    summary["dgps"][name] = dict(theta=fits[name]["theta"], procedures=rows, cv_gap_mean=float(cvgap.mean()), cv_gap_sd=float(cvgap.std()),
                                 p_cv_gap_ge_observed=float(p_cv), tr_gap_mean=float(trgap.mean()), p_tr_gap_ge_observed=float(p_tr),
                                 accopt_lo_mean=float(thr[:, 0].mean()), accopt_lo_sd=float(thr[:, 0].std()),
                                 accopt_hi_mean=float(thr[:, 1].mean()), accopt_hi_sd=float(thr[:, 1].std()),
                                 hedge=hm, bayes_public_mean=float(pb.mean()), bayes_public_sd=float(pb.std()),
                                 p_public_ge_175=float((pb >= 175 / 181 - 1e-9).mean()))

json.dump(summary, open(f"{OUT}/simulation_summary.json", "w"), indent=1, default=float)
print("\nwrote", f"{OUT}/simulation_summary.json")
