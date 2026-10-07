#!/usr/bin/env python
"""The Great Kacchi Aloo Mystery: private-LB-oriented pipeline.

Runs the data audit, canonical repeated CV, the E0-E14 experiment ladder, an extrapolation stress test
and the final-submission hedge, then writes submissions plus an experiments table.

    python kacchi_pipeline.py --data ./data --out ./outputs              # full run (10x5 CV)
    python kacchi_pipeline.py --data ./data --out ./outputs --repeats 3  # quicker
    python kacchi_pipeline.py --data ./data --out ./outputs --public-subs ./public_subs   # also diff vs public files

--data must contain train.csv, test.csv and sample_submission.csv. On Kaggle, --data defaults to the
competition input folder.
"""
import argparse
import glob
import json
import os
import time
import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, ks_2samp, spearmanr
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import lightgbm as lgb
except ImportError:  # LightGBM experiments are skipped if it is missing
    lgb = None

warnings.filterwarnings("ignore")

TARGET = "went_back_for_seconds"
NUM = ["guests", "aloo_count", "mutton_kg", "borhani_glasses", "fairy_lights", "dhol_players", "aunties_asking_when_marriage"]
CAT = ["event", "city", "drone_photographer"]
X10_RATIO = 3.0           # clean aloo/guest never exceeds ~2.7 in the public audits
N_PUBLIC, N_TEST = 181, 400  # public LB size inferred from leaderboard score granularity


# ----------------------------------------------------------------------------------------------- cleaning
def to_number(v):
    """Parse a cell to float, converting any Unicode decimal digit (Bangla, Devanagari, ...) to ASCII."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().replace(",", "")
    s = "".join(str(unicodedata.decimal(c)) if unicodedata.decimal(c, None) is not None else c for c in s)
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean(raw, is_train):
    d = raw.copy()
    for c in NUM:
        d[c + "_nonascii"] = d[c].astype(str).str.contains(r"[^\x00-\x7F]", regex=True) & d[c].notna()
        d[c] = d[c].map(to_number)
    d["drone"] = (d.drone_photographer.astype(str).str.strip().str.lower() == "yes").astype(int)
    d["aloo_missing"] = d.aloo_count.isna().astype(int)
    raw_apg = d.aloo_count / d.guests
    d["apg_raw"] = raw_apg
    d["aloo_x10"] = (raw_apg > X10_RATIO).astype(int)
    d["aloo"] = np.where(d.aloo_x10 == 1, d.aloo_count / 10, d.aloo_count)
    d["apg"] = d.aloo / d.guests
    d["log_apg"] = np.log(d.apg)
    d["mutton_pg"] = d.mutton_kg / d.guests
    d["borhani_pg"] = d.borhani_glasses / d.guests
    d["fairy_pg"] = d.fairy_lights / d.guests
    d["id_num"] = d.wedding_id.astype(str).str.extract(r"(\d+)")[0].astype(float)
    if is_train:
        d["y"] = d[TARGET].astype(int)
    return d


# ----------------------------------------------------------------------------------------------- audit
def psi(a, b, bins=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    q[0], q[-1] = -np.inf, np.inf
    ea = np.histogram(a, q)[0] / len(a) + 1e-6
    eb = np.histogram(b, q)[0] / len(b) + 1e-6
    return float(((eb - ea) * np.log(eb / ea)).sum())


def adversarial_auc(tr, te, cols, seed=0):
    X = pd.concat([tr[cols], te[cols]]).reset_index(drop=True)
    X = pd.get_dummies(X, columns=[c for c in cols if not pd.api.types.is_numeric_dtype(X[c])], drop_first=True).astype(float).fillna(-1)
    z = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    oof = np.zeros(len(z))
    for t, v in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, z):
        m = RandomForestClassifier(300, min_samples_leaf=5, random_state=seed, n_jobs=-1).fit(X.iloc[t], z[t])
        oof[v] = m.predict_proba(X.iloc[v])[:, 1]
    return float(roc_auc_score(z, oof))


def audit(tr_raw, te_raw, tr_all, te):
    A = {"shapes": {"train": list(tr_raw.shape), "test": list(te_raw.shape)}}
    A["missing"] = {"train": tr_raw.isna().sum()[lambda s: s > 0].to_dict(), "test": te_raw.isna().sum()[lambda s: s > 0].to_dict()}
    A["non_ascii_numeric_cells"] = {c: [int(tr_all[c + "_nonascii"].sum()), int(te[c + "_nonascii"].sum())] for c in NUM}
    A["x10_rows"] = {"train": tr_all.loc[tr_all.aloo_x10 == 1, "wedding_id"].tolist(), "test": te.loc[te.aloo_x10 == 1, "wedding_id"].tolist(),
                     "raw_ratios_train": tr_all.loc[tr_all.aloo_x10 == 1, "apg_raw"].round(2).tolist()}
    A["post_fix_apg_range"] = {s: [float(d.apg.min()), float(d.apg.max())] for s, d in [("train", tr_all), ("test", te)]}
    A["per_guest_ranges"] = {c: {"train": [float(tr_all[c].min()), float(tr_all[c].max())], "test": [float(te[c].min()), float(te[c].max())]}
                             for c in ["apg", "mutton_pg", "borhani_pg", "fairy_pg"]}
    A["target_rate_train"] = float(tr_all.y.mean())
    A["target_rate_by_flag"] = {f: tr_all.groupby(f).y.agg(["mean", "count"]).round(3).to_dict() for f in ["aloo_missing", "aloo_x10", "fairy_lights_nonascii"]}
    # duplicates (feature-level, ignoring id and target)
    feat = NUM + CAT
    htr = pd.util.hash_pandas_object(tr_raw[feat].astype(str), index=False)
    hte = pd.util.hash_pandas_object(te_raw[feat].astype(str), index=False)
    A["duplicates"] = {"within_train": int(htr.duplicated().sum()), "within_test": int(hte.duplicated().sum()),
                       "train_test_exact": int(len(set(htr) & set(hte)))}
    key = lambda d: d.guests.astype(int).astype(str) + "_" + d.aloo.round(0).astype("Int64").astype(str)
    A["duplicates"]["train_test_same_guests_and_aloo"] = int(len(set(key(tr_all.dropna(subset=["aloo"]))) & set(key(te))))
    # id structure + target-vs-order leakage test
    A["ids"] = {"train": [float(tr_all.id_num.min()), float(tr_all.id_num.max())], "test": [float(te.id_num.min()), float(te.id_num.max())],
                "overlap": int(len(set(tr_all.id_num) & set(te.id_num)))}
    rho, p = spearmanr(tr_all.id_num, tr_all.y)
    A["leak_target_vs_id"] = {"spearman": float(rho), "p": float(p), "auc_id": float(roc_auc_score(tr_all.y, tr_all.id_num))}
    c = tr_all.dropna(subset=["apg"])
    inband = ((c.apg > 1) & (c.apg < 2)).astype(int)
    resid = (c.y != inband).astype(int)
    rho2, p2 = spearmanr(c.id_num, resid)
    A["leak_band_errors_vs_id"] = {"spearman": float(rho2), "p": float(p2)}
    # distribution shift: raw vs per-guest
    A["shift"] = {}
    for col in ["guests", "aloo", "mutton_kg", "borhani_glasses", "fairy_lights", "dhol_players", "aunties_asking_when_marriage",
                "apg", "mutton_pg", "borhani_pg", "fairy_pg"]:
        a, b = tr_all[col].dropna().values, te[col].dropna().values
        A["shift"][col] = {"ks": float(ks_2samp(a, b).statistic), "ks_p": float(ks_2samp(a, b).pvalue), "psi": psi(a, b),
                           "train_max": float(a.max()), "test_max": float(b.max()), "test_above_train_max": int((b > a.max()).sum())}
    for col in ["event", "city", "drone_photographer"]:
        A["shift"][col] = {"train": tr_all[col].value_counts(normalize=True).round(3).to_dict(), "test": te[col].value_counts(normalize=True).round(3).to_dict()}
    raw_cols = ["guests", "aloo", "mutton_kg", "borhani_glasses", "fairy_lights", "dhol_players", "aunties_asking_when_marriage", "drone", "event", "city"]
    ratio_cols = ["apg", "mutton_pg", "borhani_pg", "fairy_pg", "dhol_players", "aunties_asking_when_marriage", "drone", "event", "city"]
    trc = tr_all.dropna(subset=["apg"])
    A["adversarial_auc"] = {"raw_features": adversarial_auc(trc, te, raw_cols), "per_guest_features": adversarial_auc(trc, te, ratio_cols),
                            "apg_only": adversarial_auc(trc, te, ["apg"])}
    return A


# ----------------------------------------------------------------------------------------------- band helpers
def band(x, lo, hi, closed=False):
    x = np.asarray(x)
    return (((x >= lo) & (x <= hi)) if closed else ((x > lo) & (x < hi))).astype(float)


def accopt_foysal(r, y, lo_g=np.arange(0.85, 1.15, 0.0025), hi_g=np.arange(1.85, 2.2, 0.0025)):
    """Exact re-implementation of the public 'band_rule' fit (accuracy-optimal, ties -> plateau centre)."""
    H = r[None, :] < hi_g[:, None]
    best, arg = -1, []
    for lo in lo_g:
        acc = (((r > lo)[None, :] & H).astype(int) == y).mean(1)
        m = acc.max()
        if m > best + 1e-12:
            best, arg = m, []
        if abs(m - best) < 1e-12:
            arg += [(lo, hi_g[j]) for j in np.where(acc == m)[0]]
    return tuple(np.array(arg).mean(0))


def midcut_rafiur(r, y):
    """Exact re-implementation of the public 'Goldilocks band' fit (midpoint cuts, median of argmax)."""
    def best_cut(x, yy, side):
        xs = np.sort(np.unique(x)); c = (xs[:-1] + xs[1:]) / 2
        accs = np.array([((x >= t) == yy).mean() if side == "lo" else ((x <= t) == yy).mean() for t in c])
        b = np.flatnonzero(accs == accs.max())
        return c[b[len(b) // 2]]
    m = r < 1.5
    return best_cut(r[m], y[m], "lo"), best_cut(r[~m], y[~m], "hi")


def best_band_any(r, y):
    """Accuracy-optimal band on an arbitrary ratio via prefix sums (all cut pairs). Returns (lo, hi, train_acc)."""
    o = np.argsort(r); rs, ys = r[o], y[o]
    n = len(r)
    cpos = np.r_[0, np.cumsum(ys)]; cneg = np.r_[0, np.cumsum(1 - ys)]
    # predict 1 for indices [i, j): errors = pos before i + neg in [i,j) + pos after j
    i = np.arange(n + 1)[:, None]; j = np.arange(n + 1)[None, :]
    err = cpos[i] + (cneg[j] - cneg[i]) + (cpos[n] - cpos[j])
    err = np.where(j >= i, err, n)
    ii, jj = np.unravel_index(np.argmin(err), err.shape)
    cut = lambda k: (rs[k - 1] + rs[k]) / 2 if 0 < k < n else (rs[0] - 1e-9 if k == 0 else rs[-1] + 1e-9)
    return cut(ii), cut(jj), 1 - err[ii, jj] / n


def p_soft(x, lo, hi, k, form):
    if form == "log":
        z = np.log(x)
        return expit(k * (z - np.log(lo))) * expit(k * (np.log(hi) - z))
    return expit(k * (x - lo)) * expit(k * (hi - x))


def fit_soft(x, y, form):
    def nll(t):
        if t[2] <= 0 or t[0] >= t[1]:
            return 1e9
        p = np.clip(p_soft(x, t[0], t[1], t[2], form), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    best = None
    for k0 in (15, 25, 40):
        r = minimize(nll, [1.0, 2.0, k0], method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def bagged_soft(x, y, B=200, seed=0):
    """Bootstrap-averaged soft band over both forms, AIC-weighted: a cheap posterior-predictive P(y=1|apg)."""
    rng = np.random.default_rng(seed)
    fits = {f: fit_soft(x, y, f) for f in ("lin", "log")}
    aic = {f: 2 * v[1] + 6 for f, v in fits.items()}
    m = min(aic.values()); w = {f: np.exp(-(a - m) / 2) for f, a in aic.items()}; s = sum(w.values()); w = {f: v / s for f, v in w.items()}
    params = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        for f in ("lin", "log"):
            params.append((f, w[f], fit_soft(x[i], y[i], f)[0]))
    def predict(xn):
        xn = np.asarray(xn, float)
        tot = sum(wf for _, wf, _ in params)
        return sum(wf * p_soft(xn, t[0], t[1], t[2], f) for f, wf, t in params) / tot
    grid = np.arange(0.85, 2.2, 0.0005); pg = predict(grid); pos = grid[pg > 0.5]
    return predict, dict(weights=w, mle={f: v[0].round(4).tolist() for f, v in fits.items()}, bayes_cuts=[float(pos.min()), float(pos.max())])


def fit_soft_het(z, g, y, g_ref):
    """Log-space soft band whose steepness depends on wedding size: k_i = exp(a + gamma*log(g_i/g_ref)).
    gamma>0 means bigger weddings have sharper edges (noise from miscounting a few potatoes/guests)."""
    lg = np.log(g / g_ref)
    def nll(t):
        lo, hi, a, gam = t
        if lo <= 0 or hi <= lo:
            return 1e9
        k = np.exp(a + gam * lg)
        p = np.clip(expit(k * (z - np.log(lo))) * expit(k * (np.log(hi) - z)), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    best = None
    for a0 in (np.log(20), np.log(30), np.log(45)):
        for g0 in (-0.5, 0.0, 0.5):
            r = minimize(nll, [1.0, 2.0, a0, g0], method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-6, fatol=1e-8))
            if best is None or r.fun < best.fun:
                best = r
    return best.x, best.fun


def p_soft_het(z, g, t, g_ref):
    k = np.exp(t[2] + t[3] * np.log(g / g_ref))
    return expit(k * (z - np.log(t[0]))) * expit(k * (np.log(t[1]) - z))


# ----------------------------------------------------------------------------------------------- models
# Every model: fn(train_df, *apply_dfs) -> list of P(y=1) arrays (hard rules return 0/1).
def m_band_fixed(t, *aps):
    return [band(a.apg, 1.0, 2.0) for a in aps]


def m_band_accopt(t, *aps):
    lo, hi = accopt_foysal(t.apg.values, t.y.values)
    return [band(a.apg, lo, hi) for a in aps]


def m_band_midcut(t, *aps):
    lo, hi = midcut_rafiur(t.apg.values, t.y.values)
    return [band(a.apg, lo, hi, closed=True) for a in aps]


def m_soft(form):
    def f(t, *aps):
        th, _ = fit_soft(t.apg.values, t.y.values, form)
        return [p_soft(a.apg.values, th[0], th[1], th[2], form) for a in aps]
    return f


def m_soft_bagged(B):
    def f(t, *aps):
        pred, _ = bagged_soft(t.apg.values, t.y.values, B=B)
        return [pred(a.apg.values) for a in aps]
    return f


def m_tree_d2(t, *aps):
    m = DecisionTreeClassifier(max_depth=2, random_state=42).fit(t[["apg"]], t.y)
    return [m.predict_proba(a[["apg"]])[:, 1] for a in aps]


def m_logreg_quad(t, *aps):
    Z = lambda d: np.c_[d.log_apg, d.log_apg ** 2]
    m = LogisticRegression(C=1e4, max_iter=5000).fit(Z(t), t.y)
    return [m.predict_proba(Z(a))[:, 1] for a in aps]


def m_knn(k):
    def f(t, *aps):
        m = KNeighborsClassifier(k).fit(t[["log_apg"]], t.y)
        return [m.predict_proba(a[["log_apg"]])[:, 1] for a in aps]
    return f


def m_lgbm_logapg(t, *aps):
    ps = [np.zeros(len(a)) for a in aps]
    for s in range(3):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=4, min_child_samples=15, subsample=0.8,
                               subsample_freq=1, random_state=s, verbose=-1).fit(t[["log_apg"]], t.y)
        for i, a in enumerate(aps):
            ps[i] += m.predict_proba(a[["log_apg"]])[:, 1] / 3
    return ps


RAW = ["guests", "aloo", "mutton_kg", "borhani_glasses", "fairy_lights", "drone", "dhol_players", "aunties_asking_when_marriage"]
RATIO = ["apg", "mutton_pg", "borhani_pg", "fairy_pg", "drone", "dhol_players", "aunties_asking_when_marriage"]


def _design(t, aps, cols, cats=("event", "city")):
    X = pd.concat([t[cols + list(cats)]] + [a[cols + list(cats)] for a in aps], keys=range(len(aps) + 1))
    X = pd.get_dummies(X, columns=list(cats), drop_first=True).astype(float)
    return X.xs(0), [X.xs(i + 1) for i in range(len(aps))]


def m_rf_raw_starter(t, *aps):
    cols = ["guests", "aloo", "mutton_kg", "borhani_glasses", "fairy_lights", "drone", "dhol_players", "aunties_asking_when_marriage"]
    m = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(t[cols], t.y)
    return [m.predict_proba(a[cols])[:, 1] for a in aps]


def m_lgbm(cols):
    def f(t, *aps):
        Xt, Xa = _design(t, aps, cols)
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=10, verbose=-1, random_state=0).fit(Xt, t.y)
        return [m.predict_proba(x)[:, 1] for x in Xa]
    return f


def m_logreg_raw(t, *aps):
    Xt, Xa = _design(t, aps, RAW)
    sc = StandardScaler().fit(Xt)
    m = LogisticRegression(max_iter=5000).fit(sc.transform(Xt), t.y)
    return [m.predict_proba(sc.transform(x))[:, 1] for x in Xa]


def m_pseudo(base_fn, test_df, conf=0.98):
    """Pseudo-label confident test rows with a model fitted on the fold's train part only, then refit."""
    def f(t, *aps):
        p = base_fn(t, test_df)[0]
        sel = (p > conf) | (p < 1 - conf)
        pl = test_df[sel].copy(); pl["y"] = (p[sel] > 0.5).astype(int)
        return base_fn(pd.concat([t, pl], ignore_index=True), *aps)
    return f


def m_blend(fns, how="mean"):
    def f(t, *aps):
        outs = [fn(t, *aps) for fn in fns]
        res = []
        for i in range(len(aps)):
            P = np.vstack([o[i] for o in outs])
            res.append(P.mean(0) if how == "mean" else ((P > 0.5).mean(0) > 0.5).astype(float))
        return res
    return f


def m_stack(fns, inner_folds=5):
    """Nested stacking: meta-model trained on inner-CV OOF of the outer-train part only."""
    def f(t, *aps):
        t = t.reset_index(drop=True)
        oof = np.zeros((len(t), len(fns)))
        for ti, vi in StratifiedKFold(inner_folds, shuffle=True, random_state=7).split(t, t.y):
            for j, fn in enumerate(fns):
                oof[vi, j] = fn(t.iloc[ti], t.iloc[vi])[0]
        meta = LogisticRegression(C=1.0, max_iter=2000).fit(oof, t.y)
        base = [fn(t, *aps) for fn in fns]
        return [meta.predict_proba(np.column_stack([b[i] for b in base]))[:, 1] for i in range(len(aps))]
    return f


# ----------------------------------------------------------------------------------------------- CV machinery
def make_folds(tr, repeats, seed=2026):
    zone = np.digitize(tr.apg, [0.9, 1.1, 1.9, 2.1])
    strat = tr.y.astype(str) + "_" + pd.Series(zone, index=tr.index).astype(str)
    F = np.zeros((len(tr), repeats), int)
    for r in range(repeats):
        for f, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed + r).split(tr, strat)):
            F[v, r] = f
    return F


def run_cv(fn, tr, F, train_mod=None):
    oof = np.zeros((len(tr), F.shape[1])); accs = []
    for r in range(F.shape[1]):
        for f in range(5):
            v = F[:, r] == f
            t = tr[~v]
            if train_mod is not None:
                t = train_mod(t)
            p = fn(t, tr[v])[0]
            oof[v, r] = p
            accs.append(((p > 0.5) == tr.y.values[v]).mean())
    return oof, np.array(accs)


def size_stress(fn, tr, q=0.6, train_mod=None):
    """Extrapolation proxy for the 'big wedding season': train on small weddings, validate on the largest 40%."""
    cut = tr.guests.quantile(q)
    t, v = tr[tr.guests <= cut], tr[tr.guests > cut]
    if train_mod is not None:
        t = train_mod(t)
    return float(((fn(t, v)[0] > 0.5) == v.y.values).mean())


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau-Bengio / Bouckaert-Frank corrected repeated-CV t-test on paired fold accuracies."""
    d = a - b; k = len(d); var = d.var(ddof=1)
    if var == 0:
        return float(d.mean()), 0.0, (1.0 if d.mean() == 0 else 0.0)
    se = np.sqrt((1 / k + n_test / n_train) * var)
    tt = d.mean() / se
    return float(d.mean()), float(se), float(2 * student_t.sf(abs(tt), k - 1))


# ----------------------------------------------------------------------------------------------- hedge
def hedge_pair(p, n_public=N_PUBLIC, sims=20000, seed=0, max_flip=40, tiebreak=None):
    """A = Bayes decisions; B = A with the m most uncertain rows flipped. Choose m maximising
    E[max(private_acc(A), private_acc(B))] under y_i ~ Bernoulli(p_i) and a random public/private split.
    tiebreak: optional secondary sort key for rows with equal |p-0.5| (e.g. distance to the nearest edge)."""
    rng = np.random.default_rng(seed)
    n = len(p); A = (p > 0.5).astype(int)
    order = np.lexsort((tiebreak, np.abs(p - 0.5))) if tiebreak is not None else np.argsort(np.abs(p - 0.5))
    Y = rng.random((sims, n)) < p[None, :]
    priv = np.ones((sims, n), bool)
    for s in range(sims):
        priv[s, rng.choice(n, n_public, replace=False)] = False
    corrA = (Y == A[None, :]) & priv
    accA = corrA.sum(1)
    res = {}
    for m in range(0, max_flip + 1):
        B = A.copy(); B[order[:m]] = 1 - B[order[:m]]
        accB = ((Y == B[None, :]) & priv).sum(1)
        res[m] = float(np.maximum(accA, accB).mean() - accA.mean())
    m_best = max(res, key=res.get)
    B = A.copy(); B[order[:m_best]] = 1 - B[order[:m_best]]
    return A, B, m_best, res


def sharp_band_probabilities(tr, te, bins=(0.0, 0.01, 0.035, 0.1, 0.4, np.inf), repeats=10, seed=0):
    """P(y=1) for test rows under the model the real data selected: a SHARP band whose cut-points sit at the midpoint
    of the training gap (Rafiur midcut), plus an error rate that depends on distance to the nearest edge.

    * the error rate per distance bin is estimated OUT-OF-FOLD (cut-points refit on each training fold), so the
      in-sample 'clean next to the fitted edge' artefact does not leak in;
    * test rows that fall inside the training gap that brackets an edge get P from a uniform prior on the cut location.
    Returns (P_test, info)."""
    r, y = tr.apg.values, tr.y.values
    lo, hi = midcut_rafiur(r, y)
    d_oof, e_oof = [], []
    for k in range(repeats):
        for t, v in StratifiedKFold(5, shuffle=True, random_state=seed + k).split(tr, y):
            l2, h2 = midcut_rafiur(r[t], y[t])
            d_oof.append(np.minimum(abs(r[v] - l2), abs(r[v] - h2)))
            e_oof.append(band(r[v], l2, h2, closed=True) != y[v])
    d_oof, e_oof = np.concatenate(d_oof), np.concatenate(e_oof)
    idx = np.digitize(d_oof, bins[1:-1])
    err = np.array([(e_oof[idx == b].sum() + 0.5) / ((idx == b).sum() + 1.0) for b in range(len(bins) - 1)])
    # training rows that bracket each edge (the cut location is unknown inside this gap)
    below_lo, above_lo = r[r < lo].max(), r[(r >= lo) & (r < 1.5)].min()
    below_hi, above_hi = r[(r <= hi) & (r > 1.5)].max(), r[r > hi].min()
    x = te.apg.values
    d = np.minimum(abs(x - lo), abs(x - hi))
    e = err[np.digitize(d, bins[1:-1])]
    inside = band(x, lo, hi, closed=True)
    P = np.where(inside == 1, 1 - e, e)
    q_lo = np.clip((x - below_lo) / (above_lo - below_lo), 0, 1)      # P(cut < x) inside the lower gap
    q_hi = np.clip((above_hi - x) / (above_hi - below_hi), 0, 1)      # P(cut > x) inside the upper gap
    g_lo, g_hi = (x > below_lo) & (x < above_lo), (x > below_hi) & (x < above_hi)
    P = np.where(g_lo, q_lo * (1 - err[0]) + (1 - q_lo) * err[0], P)
    P = np.where(g_hi, q_hi * (1 - err[0]) + (1 - q_hi) * err[0], P)
    info = dict(cuts=[float(lo), float(hi)], lower_gap=[float(below_lo), float(above_lo)], upper_gap=[float(below_hi), float(above_hi)],
                oof_error_by_distance={f"{bins[b]}-{bins[b + 1]}": round(float(err[b]), 4) for b in range(len(err))},
                test_rows_in_gaps=te.wedding_id[g_lo | g_hi].tolist())
    return P, info


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    found = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True) + glob.glob("/content/**/train.csv", recursive=True))
    ap.add_argument("--data", default=os.path.dirname(found[0]) if found else "data")
    ap.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--bag", type=int, default=100, help="bootstrap replicates for the bagged soft band")
    ap.add_argument("--public-subs", default=None)
    args, _ = ap.parse_known_args()  # ignore the "-f kernel.json" argument Jupyter/Colab/Kaggle kernels pass
    os.makedirs(args.out, exist_ok=True)
    T0 = time.time()

    tr_raw = pd.read_csv(f"{args.data}/train.csv", dtype=str)
    te_raw = pd.read_csv(f"{args.data}/test.csv", dtype=str)
    ss = pd.read_csv(f"{args.data}/sample_submission.csv")
    tr_all, te = clean(tr_raw, True), clean(te_raw, False)
    assert te.apg.notna().all(), "test has missing aloo_count: handle before predicting"
    tr = tr_all.dropna(subset=["apg"]).reset_index(drop=True)

    print("== AUDIT")
    A = audit(tr_raw, te_raw, tr_all, te)
    json.dump(A, open(f"{args.out}/audit.json", "w"), indent=1, default=str)
    for k in ["shapes", "missing", "non_ascii_numeric_cells", "x10_rows", "post_fix_apg_range", "duplicates", "ids",
              "leak_target_vs_id", "leak_band_errors_vs_id", "adversarial_auc"]:
        print(f"  {k}: {A[k]}")
    for c in ["guests", "apg", "mutton_pg", "borhani_pg"]:
        print(f"  shift {c}: {A['shift'][c]}")

    F = make_folds(tr, args.repeats)
    pd.DataFrame(F, columns=[f"rep{r}" for r in range(args.repeats)]).assign(wedding_id=tr.wedding_id).to_csv(f"{args.out}/folds.csv", index=False)
    n_tr, n_te = len(tr) * 4 / 5, len(tr) / 5

    soft_lin, soft_log = m_soft("lin"), m_soft("log")
    E = [  # (id, parent, hypothesis, change, model_fn, features, train_mod, complexity)
        ("E0a", "-", "Reproduce host starter (RF on raw counts)", "RandomForest(300) raw columns", m_rf_raw_starter, "raw", None, 3),
        ("E0", "-", "Reproduce strongest public baseline exactly", "accuracy-optimal band, plateau centre (public 'band_rule')", m_band_accopt, "apg", None, 1),
        ("E0b", "E0", "Public alternative fit: midpoint cuts", "Rafiur midpoint cut-points, closed band", m_band_midcut, "apg", None, 1),
        ("E0c", "E0", "Host prior: round-number band", "fixed 1<apg<2 (no fitting)", m_band_fixed, "apg", None, 0),
        ("E2a", "E0", "x10 fix matters for training", "train WITHOUT x10 fix (raw aloo)", m_band_accopt, "apg_raw", lambda t: t.assign(apg=t.apg_raw, log_apg=np.log(t.apg_raw)), 1),
        ("E2b", "E0", "Dropping x10 rows is as good as fixing", "drop x10 rows from training", m_band_accopt, "apg", lambda t: t[t.aloo_x10 == 0], 1),
        ("E2c", "E3a", "log scale is the natural ratio space", "soft band in log(apg)", soft_log, "log_apg", None, 2),
        ("E3a", "E0", "Smooth likelihood fit beats step-accuracy fit", "soft-band MLE (linear apg), p>0.5", soft_lin, "apg", None, 2),
        ("E3b", "E3a", "Model-averaged soft band (bootstrap, lin+log, AIC-weighted)", "bagged soft band", m_soft_bagged(max(20, args.bag // 4)), "apg", None, 3),
        ("E4a", "E0", "Host hint: depth-2 tree on the ratio", "DecisionTree(max_depth=2) on apg", m_tree_d2, "apg", None, 1),
        ("E5a", "E3a", "Quadratic logit in log space = smooth band", "LogisticRegression on [z, z^2], z=log apg", m_logreg_quad, "log_apg", None, 2),
        ("E5b", "E3a", "Non-parametric neighbour vote (public k=5)", "kNN k=5 on log apg", m_knn(5), "log_apg", None, 2),
        ("E5c", "E5b", "Smoother kNN", "kNN k=25 on log apg", m_knn(25), "log_apg", None, 2),
        ("E5d", "E0a", "Raw-feature logistic (shows ratio necessity)", "LogisticRegression raw", m_logreg_raw, "raw", None, 2),
    ]
    if lgb is not None:
        E += [("E5e", "E3a", "Boosting on the ratio (public lgbm_logapg)", "LightGBM log apg, 3 seeds", m_lgbm_logapg, "log_apg", None, 3),
              ("E5f", "E5e", "Boosting with all per-guest ratios + categoricals", "LightGBM all ratio features", m_lgbm(RATIO), "ratios+cats", None, 4),
              ("E5g", "E0a", "Boosting on raw counts (fails to extrapolate?)", "LightGBM raw features", m_lgbm(RAW), "raw+cats", None, 4)]
    blend_members = [soft_lin, soft_log, m_logreg_quad] + ([m_lgbm_logapg] if lgb is not None else []) + [m_knn(25)]
    E += [("E6", "E3a", "OOF probability blend of decorrelated ratio models", "mean P of soft-lin, soft-log, quad-logit, (lgbm), knn25", m_blend(blend_members, "mean"), "apg", None, 4),
          ("E7", "E6", "Majority vote instead of probability mean", "hard-vote of the same members", m_blend(blend_members, "vote"), "apg", None, 4),
          ("E8", "E6", "Nested stacking (logistic meta-model)", "LR meta on inner-CV OOF", m_stack([soft_lin, soft_log, m_logreg_quad, m_knn(25)]), "apg", None, 5),
          ("E9", "E3a", "Conservative pseudo-labels from confident test rows", "add test rows with P>0.98/<0.02, refit", m_pseudo(soft_lin, te), "apg", None, 4)]

    rows, oofs, accs = [], {}, {}
    print(f"\n== EXPERIMENTS ({args.repeats}x5 repeated CV, {len(tr)} clean rows; stratified on target x apg-zone)")
    for eid, parent, hyp, change, fn, feats, mod, cx in E:
        t0 = time.time()
        reps = args.repeats if eid not in ("E8", "E3b") else min(args.repeats, 3)
        oof, acc = run_cv(fn, tr, F[:, :reps], mod)
        oofs[eid], accs[eid] = oof, acc
        stress = size_stress(fn, tr, 0.6, mod)
        pt = fn(tr if mod is None else mod(tr), te)[0]
        rows.append(dict(experiment_id=eid, parent_id=parent, hypothesis=hyp, change=change, features=feats, folds=f"{reps}x5 RSKF (y x zone)",
                         cv_score=acc.mean(), cv_std=acc.std(), cv_min=acc.min(), cv_max=acc.max(),
                         oof_logloss=float(log_loss(np.repeat(tr.y.values, reps), np.clip(oof[:, :reps].T.ravel(), 1e-6, 1 - 1e-6)))
                         if len(np.unique(oof)) > 2 else np.nan,
                         size_stress_acc=stress, test_pos_rate=float((pt > 0.5).mean()), runtime_s=round(time.time() - t0, 1), complexity=cx,
                         _test_pred=(pt > 0.5).astype(int)))
        print(f"  {eid:5s} cv {acc.mean():.4f} ± {acc.std():.4f} [{acc.min():.3f}, {acc.max():.3f}]  stress {stress if stress == stress else float('nan'):.4f}"
              f"  test+ {(pt > 0.5).mean():.3f}  ({time.time() - t0:.1f}s)  {change}")

    R = pd.DataFrame(rows).set_index("experiment_id")
    # paired corrected t-tests vs parent + decision rule
    dec, why, dmean, dp, ddiff = [], [], [], [], []
    for eid, r in R.iterrows():
        par = r.parent_id
        if par == "-" or par not in accs:
            dec.append("BASELINE"); why.append("reference"); dmean.append(np.nan); dp.append(np.nan); ddiff.append(np.nan); continue
        k = min(len(accs[eid]), len(accs[par]))
        d, se, p = corrected_ttest(accs[eid][:k], accs[par][:k], n_tr, n_te)
        nd = int((R.loc[eid, "_test_pred"] != R.loc[par, "_test_pred"]).sum())
        stress_worse = (r.size_stress_acc == r.size_stress_acc) and (R.loc[par, "size_stress_acc"] == R.loc[par, "size_stress_acc"]) and r.size_stress_acc < R.loc[par, "size_stress_acc"] - 0.01
        if d == 0 and nd == 0:
            dec.append("REJECT"); why.append(f"no effect on CV or on test predictions vs {par}")
        elif d > 0 and p < 0.05 and not stress_worse:
            dec.append("KEEP"); why.append(f"+{d:.4f} (p={p:.3f}) vs {par}")
        elif d < 0 and p < 0.05:
            dec.append("REJECT"); why.append(f"{d:.4f} (p={p:.3f}) vs {par}")
        elif r.complexity > R.loc[par, "complexity"]:
            dec.append("REJECT"); why.append(f"no significant gain ({d:+.4f}, p={p:.2f}); simpler parent kept")
        else:
            dec.append("NEEDS_MORE_EVIDENCE"); why.append(f"{d:+.4f}, p={p:.2f}: within CV noise")
        dmean.append(d); dp.append(p); ddiff.append(nd)
    R["delta_vs_parent"], R["p_corrected"], R["test_rows_changed_vs_parent"], R["decision"], R["reason"] = dmean, dp, ddiff, dec, why

    # ------------------------------------------------------------------ E1: validation scheme comparison
    print("\n== E1 validation schemes (band_accopt)")
    e1 = {}
    for name, strat in [("SKF_y", tr.y), ("SKF_y_x_zone", tr.y.astype(str) + "_" + pd.Series(np.digitize(tr.apg, [0.9, 1.1, 1.9, 2.1])).astype(str))]:
        a = []
        for r in range(args.repeats):
            for t, v in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(tr, strat):
                a.append((m_band_accopt(tr.iloc[t], tr.iloc[v])[0] == tr.y.values[v]).mean())
        e1[name] = dict(mean=float(np.mean(a)), fold_sd=float(np.std(a)), sd_of_repeat_means=float(np.std(np.array(a).reshape(-1, 5).mean(1))))
    for q in (0.5, 0.6, 0.7):
        e1[f"size_stress_q{q}"] = {e: size_stress(fn, tr, q) for e, _, _, _, fn, _, mod, _ in E if mod is None and e in ("E0a", "E0", "E3a", "E5g", "E5f", "E5e")}
    print(json.dumps(e1, indent=1))

    # ------------------------------------------------------------------ E11: residual signal near the edges
    print("\n== E11 residual-signal test near the band edges")
    near = tr[(abs(tr.apg - 1) < 0.15) | (abs(tr.apg - 2) < 0.15)].copy()
    near["m"] = np.minimum(near.log_apg - np.log(1.0), np.log(2.0) - near.log_apg)
    Xb = near[["m"]].values
    Xf = pd.get_dummies(near[["m", "mutton_pg", "borhani_pg", "fairy_pg", "dhol_players", "aunties_asking_when_marriage", "drone", "guests", "event", "city"]],
                        columns=["event", "city"], drop_first=True).astype(float).values
    def cv_ll(X):
        ll, ac = [], []
        for r in range(5):
            for t, v in StratifiedKFold(5, shuffle=True, random_state=r).split(X, near.y):
                sc = StandardScaler().fit(X[t]); m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[t]), near.y.values[t])
                p = m.predict_proba(sc.transform(X[v]))[:, 1]
                ll.append(log_loss(near.y.values[v], p, labels=[0, 1])); ac.append(((p > 0.5) == near.y.values[v]).mean())
        return float(np.mean(ll)), float(np.mean(ac))
    e11 = dict(n_near=len(near), margin_only=cv_ll(Xb), margin_plus_all=cv_ll(Xf))
    print(f"  {e11}")

    # ------------------------------------------------------------------ E12: is aloo/guests the right ratio?
    print("\n== E12 ratio search (in-sample best band acc, optimistic)")
    Al, G = tr.aloo.values, tr.guests.values
    cands = {"aloo/guests": Al / G, "aloo/mutton_kg": Al / tr.mutton_kg.values, "aloo/borhani": Al / tr.borhani_glasses.values}
    for c in ["dhol_players", "aunties_asking_when_marriage", "drone"]:
        for k in (1, 2):
            cands[f"aloo/(guests+{k}*{c})"] = Al / (G + k * tr[c].values)
    for c in (10, 25):
        cands[f"aloo/(guests+{c})"] = Al / (G + c)
    e12 = {k: round(best_band_any(v, tr.y.values)[2], 4) for k, v in cands.items()}
    for k, v in sorted(e12.items(), key=lambda kv: -kv[1]):
        print(f"  {v:.4f}  {k}")

    # ------------------------------------------------------------------ E10 / E13: final model + hedge
    print("\n== E10 final: bagged soft band on all clean train rows")
    pred_fn, info = bagged_soft(tr.apg.values, tr.y.values, B=args.bag)
    p_test = pred_fn(te.apg.values)
    print(f"  {info}")

    # ------------------------------------------------------------------ E15: does edge sharpness depend on wedding size?
    print("\n== E15 size-dependent edge width (log-space soft band, k = exp(a + gamma*log(guests/median)))")
    g_ref = float(tr.guests.median())
    th_h, nll_h = fit_soft_het(tr.log_apg.values, tr.guests.values, tr.y.values, g_ref)
    _, nll_0 = fit_soft(tr.apg.values, tr.y.values, "log")
    lr = 2 * (nll_0 - nll_h); p_lr = float(chi2.sf(max(lr, 0), 1))
    e15 = dict(lo=float(th_h[0]), hi=float(th_h[1]), k_at_median=float(np.exp(th_h[2])), gamma=float(th_h[3]), lr_stat=float(lr), p=p_lr,
               k_at_100_guests=float(np.exp(th_h[2] + th_h[3] * np.log(100 / g_ref))), k_at_1400_guests=float(np.exp(th_h[2] + th_h[3] * np.log(1400 / g_ref))))
    print(f"  {e15}")
    if p_lr < 0.05:
        print("  gamma significant: consider size-aware P for the hedge (not used automatically)")

    # ------------------------------------------------------------------ E16: final decision model chosen by the real data
    # On the real data the smooth soft band (E3a) is significantly worse than the sharp band (E0/E0b), and a sharp
    # midcut band with a distance-dependent error rate has the best near-edge OOF log-loss. So A = sharp midcut band.
    print("\n== E16 final decisions: sharp midcut band + out-of-fold error rate by distance to the edge")
    p_test, e16 = sharp_band_probabilities(tr, te, repeats=args.repeats)
    print(f"  {e16}")
    lo_c, hi_c = e16["cuts"]
    d_edge = np.minimum(abs(te.apg.values - lo_c), abs(te.apg.values - hi_c))
    A_sub, B_sub, m_best, curve = hedge_pair(p_test, tiebreak=d_edge)
    assert (A_sub == band(te.apg, lo_c, hi_c, closed=True)).all(), "A must equal the sharp midcut band"
    print(f"== E13 hedge: flip m={m_best} most-uncertain rows; expected private gain of max(A,B) over A: "
          f"{curve[m_best]:+.3f} rows  (curve: { {k: round(v, 3) for k, v in curve.items() if k <= 20} })")
    unc = te.assign(p=p_test, A=A_sub, B=B_sub).loc[lambda d: (d.p > 0.2) & (d.p < 0.8), ["wedding_id", "guests", "aloo", "apg", "p", "A", "B"]].sort_values("apg")
    print("  uncertain test rows (0.2<p<0.8):"); print(unc.round(4).to_string(index=False))
    unc.to_csv(f"{args.out}/uncertain_test_rows.csv", index=False)

    def write(name, pred):
        s = pd.DataFrame({"wedding_id": te.wedding_id, TARGET: np.asarray(pred).astype(int)})
        assert list(s.wedding_id) == list(ss.wedding_id) and s.shape == ss.shape
        s.to_csv(f"{args.out}/{name}", index=False)
    write("sub_A_primary_sharpband.csv", A_sub)
    write("submission.csv", A_sub)  # what Kaggle's notebook "Submit" button picks up
    write("sub_B_hedge.csv", B_sub)
    write("sub_ref_bagged_softband.csv", (pred_fn(te.apg.values) > 0.5).astype(int))
    write("sub_ref_fixed_1_2.csv", band(te.apg, 1, 2))
    write("sub_ref_accopt_public.csv", R.loc["E0", "_test_pred"])

    if args.public_subs:
        print("\n== diff vs public submission files")
        for f in sorted(glob.glob(f"{args.public_subs}/**/*.csv", recursive=True)):
            p = pd.read_csv(f).set_index("wedding_id")[TARGET].reindex(te.wedding_id).values
            print(f"  {os.path.relpath(f, args.public_subs):70s} vs E0: {(p != R.loc['E0', '_test_pred']).sum():3d}  vs A: {(p != A_sub).sum():3d}  vs B: {(p != B_sub).sum():3d}")

    out = R.drop(columns=["_test_pred"]).reset_index()
    out.to_csv(f"{args.out}/experiments_results.csv", index=False)
    json.dump(dict(E1=e1, E11=e11, E12=e12, E10=info, E13=dict(m=m_best, curve=curve), E15=e15, E16=e16), open(f"{args.out}/extra_results.json", "w"), indent=1, default=float)
    print(f"\nwrote outputs to {args.out}/ in {time.time() - T0:.0f}s")
    print(out[["experiment_id", "parent_id", "cv_score", "cv_std", "size_stress_acc", "delta_vs_parent", "p_corrected", "decision", "reason"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
