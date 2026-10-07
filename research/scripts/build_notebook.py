"""Build solution/kacchi_aloo_pipeline.ipynb from kacchi_pipeline.py's functions, split into explained cells.

The function definitions are taken verbatim from the .py (between section markers) so the two never drift;
the argparse main() is replaced by plain notebook cells."""
import re
import sys

import nbformat as nbf

src = open(sys.argv[1], encoding="utf-8").read()
out = sys.argv[2]


def section(start, end):
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j].rstrip() + "\n"


imports = section("import glob", "TARGET = \"went_back_for_seconds\"")
imports = imports.replace("import argparse\n", "")
cleaning = section("def to_number", "# ----------------------------------------------------------------------------------------------- audit")
audit = section("def psi", "# ----------------------------------------------------------------------------------------------- band helpers")
helpers = section("def band(", "# ----------------------------------------------------------------------------------------------- models")
models = section("def m_band_fixed", "# ----------------------------------------------------------------------------------------------- CV machinery")
cvm = section("def make_folds", "# ----------------------------------------------------------------------------------------------- hedge")
hedge = section("def hedge_pair", "# ----------------------------------------------------------------------------------------------- main")

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []
C.append(md("""# The Great Kacchi Aloo Mystery: robust pipeline

**Goal:** predict `went_back_for_seconds` for 400 test weddings (metric: accuracy) while maximising the chance of a strong **private** leaderboard score.

**The idea in one line:** guests go back for seconds when there are about **1 to 2 potatoes per guest** (`aloo_count / guests`). Every other column is noise, and the test set has much bigger weddings, so only scale-free ratio features generalise.

**What this notebook does**
1. Cleans the traps from Mitu's field notes (Bangla digits, missing aloo, ×10 potato counts).
2. Audits the data: dirt counts, duplicates, ID-order leakage, train/test shift, adversarial validation.
3. Freezes one canonical 10×5 repeated stratified CV (strata = target × distance-to-edge zone) plus a **size-stress** fold that mimics the bigger test weddings.
4. Runs the experiment ladder E0–E15 with a corrected repeated-CV t-test for every KEEP/REJECT decision.
5. Fits the final rule: a bootstrap-averaged soft band, i.e. a posterior-predictive P(seconds | aloo per guest).
6. Builds **two** final submissions: **A** (best decisions) and **B** (A with the most uncertain rows flipped by a fixed rule). Kaggle scores the *best* of your selected finals, so the pair raises the expected private score.

**How to run**
- **Kaggle:** *Add Input → Competitions → The Great Kacchi Aloo Mystery*, then *Run All*. Outputs go to `/kaggle/working`, and `submission.csv` is file **A**.
- **Local / Colab:** put `train.csv`, `test.csv` and `sample_submission.csv` in `./data` (or set the `KACCHI_DATA` environment variable).
- Runtime: about 5–8 min on CPU with the default settings. Set `REPEATS = 3` for a quick run.
"""))
C.append(md("## 0. Setup"))
C.append(code(imports + """
import matplotlib.pyplot as plt
from IPython.display import display

TARGET = "went_back_for_seconds"
NUM = ["guests", "aloo_count", "mutton_kg", "borhani_glasses", "fairy_lights", "dhol_players", "aunties_asking_when_marriage"]
CAT = ["event", "city", "drone_photographer"]
X10_RATIO = 3.0              # clean aloo/guest never exceeds ~2.7, so a ratio above 3 is Tutul's extra zero
N_PUBLIC, N_TEST = 181, 400  # public LB size inferred from leaderboard score granularity

# ---- settings ----
FAST = os.environ.get("KACCHI_FAST") == "1"   # quick smoke-test mode
REPEATS = 2 if FAST else 10                    # CV repeats (x5 folds)
BAG = 20 if FAST else 100                      # bootstrap replicates for the final soft band
PUBLIC_SUBS = None                             # optional folder of public submission CSVs to diff against


def find_data():
    if os.environ.get("KACCHI_DATA"):
        return os.environ["KACCHI_DATA"]
    for pat in ("/kaggle/input/**/train.csv", "/content/**/train.csv", "data/train.csv", "../data/train.csv"):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError("train.csv not found: add the competition data or set KACCHI_DATA")


DATA = find_data()
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
print("data:", DATA, "| outputs:", OUT, "| repeats:", REPEATS, "| bag:", BAG, "| lightgbm:", lgb is not None)
"""))

C.append(md("""## 1. Cleaning Mitu's notebook

- **Bangla digits:** `unicodedata.decimal` converts any Unicode digit (৪৫০ → 450) in **every** numeric column, not just `fairy_lights`.
- **Extra zero:** if aloo per guest is above 3, the count is divided by 10.
- **Empty aloo boxes:** these rows can't be repaired (aloo per guest is independent of every other column), so they are dropped from training. The test set has none.
- **Features:** aloo per guest (`apg`) and its log, plus the other per-guest ratios for comparison."""))
C.append(code(cleaning + """

tr_raw = pd.read_csv(f"{DATA}/train.csv", dtype=str)
te_raw = pd.read_csv(f"{DATA}/test.csv", dtype=str)
ss = pd.read_csv(f"{DATA}/sample_submission.csv")
tr_all, te = clean(tr_raw, True), clean(te_raw, False)
assert te.apg.notna().all(), "test has missing aloo_count: handle before predicting"
tr = tr_all.dropna(subset=["apg"]).reset_index(drop=True)
print(f"train {tr_raw.shape} -> {len(tr)} clean rows | test {te_raw.shape}")
display(tr_raw.head())
"""))

C.append(md("""## 2. Data audit

Checks: dirt counts in train **and** test, the ×10 rows, duplicates, ID-order leakage (does the row order predict the label?), KS / PSI shift for raw counts vs per-guest ratios, and adversarial validation. Raw features should separate train from test (bigger test weddings); ratios should not (AUC ≈ 0.5)."""))
C.append(code(audit + """

A = audit(tr_raw, te_raw, tr_all, te)
json.dump(A, open(f"{OUT}/audit.json", "w"), indent=1, default=str)
for k in ["shapes", "missing", "non_ascii_numeric_cells", "x10_rows", "post_fix_apg_range", "duplicates", "ids",
          "leak_target_vs_id", "leak_band_errors_vs_id", "adversarial_auc"]:
    print(f"{k}: {A[k]}")
shift = pd.DataFrame({c: A["shift"][c] for c in ["guests", "aloo", "mutton_kg", "borhani_glasses", "fairy_lights", "apg", "mutton_pg", "borhani_pg", "fairy_pg"]}).T
display(shift.round(4))
"""))

C.append(md("""## 3. Look at the mystery

Left: the share of weddings that went back for seconds, by aloo per guest. Right: in raw counts the band is a wedge between `aloo = guests` and `aloo = 2 × guests`, and the wedge extends into the bigger test weddings. A tree on raw counts can't follow it past the largest training wedding."""))
C.append(code("""fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
bins = np.arange(0.4, 2.81, 0.1)
g = tr.groupby(pd.cut(tr.apg, bins), observed=True).y.agg(["mean", "count"])
ax[0].bar([i.mid for i in g.index], g["mean"], width=0.09, color="#2a78d6")
for t in (1, 2):
    ax[0].axvline(t, color="k", ls="--", lw=1)
ax[0].set_xlabel("aloo per guest"); ax[0].set_ylabel("share that went back"); ax[0].set_title("The Goldilocks band")
ax[1].scatter(te.guests, te.aloo, s=8, color="#8a8983", alpha=0.5, label="test")
ax[1].scatter(tr.guests, tr.aloo, s=8, c=np.where(tr.y == 1, "#2a78d6", "#eb6834"), label="train (blue = 1)")
gg = np.array([0, 1500]); ax[1].plot(gg, gg, "k--", lw=1); ax[1].plot(gg, 2 * gg, "k--", lw=1)
ax[1].axvline(tr.guests.max(), color="#8a8983", ls=":")
ax[1].set_xlabel("guests"); ax[1].set_ylabel("aloo (fixed)"); ax[1].legend(loc="upper left"); ax[1].set_title("Raw counts: a wedge that grows past the train range")
plt.tight_layout(); plt.show()
display(g.T.round(2))
"""))

C.append(md("""## 4. Band estimators

- `accopt_foysal` / `midcut_rafiur`: exact re-implementations of the two public cut-point fits, so E0 can reproduce the public baseline exactly.
- `fit_soft`: a smooth band P = σ(k(x − lo))·σ(k(hi − x)), fitted by maximum likelihood in linear or log space. It uses every row near the edges, not just the 0/1 errors.
- `bagged_soft`: bootstrap- and AIC-averaged soft band, a cheap posterior-predictive estimate of P(seconds | apg). This is the final model.
- `fit_soft_het`: the edge steepness may depend on wedding size (E15)."""))
C.append(code(helpers))

C.append(md("""## 5. Models

Every model has the same interface: `fn(train_df, *apply_dfs) -> list of P(y=1)` (hard rules return 0/1). That lets one CV loop compare rules, smooth fits, kNN, LightGBM, blends, nested stacking and pseudo-labelling fairly."""))
C.append(code(models))

C.append(md("""## 6. Validation

- **Canonical folds:** `REPEATS × 5` stratified folds, strata = target × apg zone (below / lower edge / inside / upper edge / above). They are frozen to `folds.csv` and never changed between experiments.
- **Size-stress fold:** train on weddings ≤ 60th percentile of guests, validate on the biggest 40%. This mimics the bigger test weddings.
- **Decisions:** Nadeau–Bengio corrected repeated-CV t-test on paired fold accuracies vs the parent experiment. One CV row is 1/776 ≈ 0.0013, so small "gains" are noise."""))
C.append(code(cvm + """

F = make_folds(tr, REPEATS)
pd.DataFrame(F, columns=[f"rep{r}" for r in range(REPEATS)]).assign(wedding_id=tr.wedding_id).to_csv(f"{OUT}/folds.csv", index=False)
n_tr, n_te = len(tr) * 4 / 5, len(tr) / 5
print("folds frozen:", F.shape)
"""))

C.append(md("""## 7. Experiment ladder E0–E9

| ID | Idea |
|---|---|
| E0 / E0a / E0b / E0c | Public band rule (exact), host starter RF, midpoint cuts, fixed 1 < apg < 2 |
| E2a–E2c | ×10-fix ablations; log-space soft band |
| E3a / E3b | Soft-band MLE; bagged soft band |
| E4a | Depth-2 tree (host hint) |
| E5a–E5g | Quadratic logit, kNN, raw logistic, LightGBM on ratio / all ratios / raw counts |
| E6–E9 | Probability blend, hard vote, nested stacking, fold-internal pseudo-labels |

KEEP needs p < 0.05 **and** a size-stress score no more than 1 point worse. On ties the simpler model is kept."""))
C.append(code("""soft_lin, soft_log = m_soft("lin"), m_soft("log")
E = [  # (id, parent, hypothesis, change, model_fn, features, train_mod, complexity)
    ("E0a", "-", "Reproduce host starter (RF on raw counts)", "RandomForest(300) raw columns", m_rf_raw_starter, "raw", None, 3),
    ("E0", "-", "Reproduce strongest public baseline exactly", "accuracy-optimal band, plateau centre (public 'band_rule')", m_band_accopt, "apg", None, 1),
    ("E0b", "E0", "Public alternative fit: midpoint cuts", "Rafiur midpoint cut-points, closed band", m_band_midcut, "apg", None, 1),
    ("E0c", "E0", "Host prior: round-number band", "fixed 1<apg<2 (no fitting)", m_band_fixed, "apg", None, 0),
    ("E2a", "E0", "x10 fix matters for training", "train WITHOUT x10 fix (raw aloo)", m_band_accopt, "apg_raw", lambda t: t.assign(apg=t.apg_raw, log_apg=np.log(t.apg_raw)), 1),
    ("E2b", "E0", "Dropping x10 rows is as good as fixing", "drop x10 rows from training", m_band_accopt, "apg", lambda t: t[t.aloo_x10 == 0], 1),
    ("E2c", "E3a", "log scale is the natural ratio space", "soft band in log(apg)", soft_log, "log_apg", None, 2),
    ("E3a", "E0", "Smooth likelihood fit beats step-accuracy fit", "soft-band MLE (linear apg), p>0.5", soft_lin, "apg", None, 2),
    ("E3b", "E3a", "Model-averaged soft band (bootstrap, lin+log, AIC-weighted)", "bagged soft band", m_soft_bagged(max(20, BAG // 4)), "apg", None, 3),
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

rows, accs = [], {}
for eid, parent, hyp, change, fn, feats, mod, cx in E:
    t0 = time.time()
    reps = REPEATS if eid not in ("E8", "E3b") else min(REPEATS, 3)
    oof, acc = run_cv(fn, tr, F[:, :reps], mod)
    accs[eid] = acc
    stress = size_stress(fn, tr, 0.6, mod)
    pt = fn(tr if mod is None else mod(tr), te)[0]
    rows.append(dict(experiment_id=eid, parent_id=parent, hypothesis=hyp, change=change, features=feats, folds=f"{reps}x5 RSKF (y x zone)",
                     cv_score=acc.mean(), cv_std=acc.std(), cv_min=acc.min(), cv_max=acc.max(),
                     oof_logloss=float(log_loss(np.repeat(tr.y.values, reps), np.clip(oof[:, :reps].T.ravel(), 1e-6, 1 - 1e-6)))
                     if len(np.unique(oof)) > 2 else np.nan,
                     size_stress_acc=stress, test_pos_rate=float((pt > 0.5).mean()), runtime_s=round(time.time() - t0, 1), complexity=cx,
                     _test_pred=(pt > 0.5).astype(int)))
    print(f"{eid:5s} cv {acc.mean():.4f} ± {acc.std():.4f}  stress {stress:.4f}  test+ {(pt > 0.5).mean():.3f}  ({time.time() - t0:.1f}s)  {change}")
"""))
C.append(code("""R = pd.DataFrame(rows).set_index("experiment_id")
dec, why, dmean, dp, ddiff = [], [], [], [], []
for eid, r in R.iterrows():
    par = r.parent_id
    if par == "-" or par not in accs:
        dec.append("BASELINE"); why.append("reference"); dmean.append(np.nan); dp.append(np.nan); ddiff.append(np.nan); continue
    k = min(len(accs[eid]), len(accs[par]))
    d, se, p = corrected_ttest(accs[eid][:k], accs[par][:k], n_tr, n_te)
    nd = int((R.loc[eid, "_test_pred"] != R.loc[par, "_test_pred"]).sum())
    stress_worse = r.size_stress_acc < R.loc[par, "size_stress_acc"] - 0.01
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
R.drop(columns=["_test_pred"]).reset_index().to_csv(f"{OUT}/experiments_results.csv", index=False)
display(R[["parent_id", "cv_score", "cv_std", "size_stress_acc", "delta_vs_parent", "p_corrected", "test_rows_changed_vs_parent", "decision", "reason"]].round(4))
"""))

C.append(md("## 8. E1: validation schemes and the size-stress test"))
C.append(code("""e1 = {}
for name, strat in [("SKF_y", tr.y), ("SKF_y_x_zone", tr.y.astype(str) + "_" + pd.Series(np.digitize(tr.apg, [0.9, 1.1, 1.9, 2.1])).astype(str))]:
    a = []
    for r in range(REPEATS):
        for t, v in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(tr, strat):
            a.append((m_band_accopt(tr.iloc[t], tr.iloc[v])[0] == tr.y.values[v]).mean())
    e1[name] = dict(mean=float(np.mean(a)), fold_sd=float(np.std(a)), sd_of_repeat_means=float(np.std(np.array(a).reshape(-1, 5).mean(1))))
for q in (0.5, 0.6, 0.7):
    e1[f"size_stress_q{q}"] = {e: size_stress(fn, tr, q) for e, _, _, _, fn, _, mod, _ in E if mod is None and e in ("E0a", "E0", "E3a", "E5g", "E5f", "E5e")}
display(pd.DataFrame({k: v for k, v in e1.items() if k.startswith("size")}).round(4))
print({k: v for k, v in e1.items() if k.startswith("SKF")})
"""))

C.append(md("## 9. E11 and E12: is there anything left besides aloo per guest?"))
C.append(code("""# E11: near the edges, does any other column predict the label beyond the distance to the edge?
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
print("E11 (log-loss, accuracy):", e11)

# E12: which ratio gives the best band? (in-sample, optimistic)
Al, G = tr.aloo.values, tr.guests.values
cands = {"aloo/guests": Al / G, "aloo/mutton_kg": Al / tr.mutton_kg.values, "aloo/borhani": Al / tr.borhani_glasses.values}
for c in ["dhol_players", "aunties_asking_when_marriage", "drone"]:
    for k in (1, 2):
        cands[f"aloo/(guests+{k}*{c})"] = Al / (G + k * tr[c].values)
for c in (10, 25):
    cands[f"aloo/(guests+{c})"] = Al / (G + c)
e12 = {k: round(best_band_any(v, tr.y.values)[2], 4) for k, v in cands.items()}
display(pd.Series(e12).sort_values(ascending=False).to_frame("best band accuracy"))
"""))

C.append(md("""## 10. E10 and E15: the smooth model (kept for comparison)

The bagged soft band gives a smooth P(seconds | aloo per guest). E15 checks whether the edges get sharper at bigger weddings. **On the real data both lose:** the soft band is significantly worse in CV than the sharp band (E3a vs E0, p ≈ 0.03), and E15 is not significant (p ≈ 0.42). They are kept here as diagnostics."""))
C.append(code("""pred_fn, info = bagged_soft(tr.apg.values, tr.y.values, B=BAG)
print("E10:", info)

g_ref = float(tr.guests.median())
th_h, nll_h = fit_soft_het(tr.log_apg.values, tr.guests.values, tr.y.values, g_ref)
_, nll_0 = fit_soft(tr.apg.values, tr.y.values, "log")
lr = 2 * (nll_0 - nll_h); p_lr = float(chi2.sf(max(lr, 0), 1))
e15 = dict(lo=float(th_h[0]), hi=float(th_h[1]), k_at_median=float(np.exp(th_h[2])), gamma=float(th_h[3]), lr_stat=float(lr), p=p_lr,
           k_at_100_guests=float(np.exp(th_h[2] + th_h[3] * np.log(100 / g_ref))), k_at_1400_guests=float(np.exp(th_h[2] + th_h[3] * np.log(1400 / g_ref))))
print("E15:", {k: round(v, 4) for k, v in e15.items()})
"""))

C.append(md("""## 11. E16 + E13: final decisions and the hedge

**What the real data chose:** the band edges are **sharp**. On the training set the label flips between aloo/guest 0.9699 and 0.9714 (lower edge) and between 2.00973 and 2.00993 (upper edge), with runs of identical labels right next to each edge. A sharp band, with each cut at the midpoint of that gap, has the best near-edge out-of-fold log-loss and the best CV. All local smoothers (kNN, kernel, LightGBM) do worse.

- **A** = the sharp midcut band.
- **P(seconds)** for each test row = the band's call, softened by an **out-of-fold** error rate that depends on the distance to the nearest edge. Test rows that fall *inside* a training gap get P from a uniform prior on where the cut lies.
- **B** = A with the *m* rows whose P is closest to 0.5 flipped (ties broken by distance to the edge). The rows are chosen by this fixed rule, never by hand (Foundational Rule 4b). *m* maximises E[max(private acc A, private acc B)] over random 181/219 public/private splits. Kaggle scores the **best** of your selected finals."""))
C.append(code(hedge + """

p_test, e16 = sharp_band_probabilities(tr, te, repeats=REPEATS)
print("E16:", e16)
lo_c, hi_c = e16["cuts"]
d_edge = np.minimum(abs(te.apg.values - lo_c), abs(te.apg.values - hi_c))
A_sub, B_sub, m_best, curve = hedge_pair(p_test, tiebreak=d_edge)
assert (A_sub == band(te.apg, lo_c, hi_c, closed=True)).all(), "A must equal the sharp midcut band"
print(f"flip m={m_best}: expected private gain of best-of-(A,B) over A = {curve[m_best]:+.3f} rows")

fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
xs = np.linspace(0.4, 2.75, 600)
ax[0].plot(xs, pred_fn(xs), color="#8a8983", lw=1.5, label="smooth soft band (rejected)")
ax[0].plot(xs, band(xs, lo_c, hi_c, closed=True), color="k", lw=2, label=f"sharp band {lo_c:.4f}-{hi_c:.4f} (A)")
b = tr.groupby(pd.cut(tr.apg, np.arange(0.4, 2.81, 0.05)), observed=True).y.mean()
ax[0].scatter([i.mid for i in b.index], b.values, s=14, color="#2a78d6", label="train share (0.05 bins)")
ax[0].set_xlabel("aloo per guest"); ax[0].legend(fontsize=8); ax[0].set_title("Smooth vs sharp band")
ax[1].plot(list(curve), list(curve.values()), marker="o"); ax[1].set_xlabel("rows flipped in B"); ax[1].set_ylabel("expected gain (private rows)")
ax[1].set_title("Hedge: E[best of A,B] - E[A]")
plt.tight_layout(); plt.show()
unc = te.assign(p=p_test, A=A_sub, B=B_sub).loc[lambda d: (d.p > 0.2) & (d.p < 0.8), ["wedding_id", "guests", "aloo", "apg", "p", "A", "B"]].sort_values("apg")
unc.to_csv(f"{OUT}/uncertain_test_rows.csv", index=False)
display(unc.round(4))
"""))

C.append(md("""## 12. Write submissions

- `submission.csv` = **A** (Kaggle's notebook *Submit* button uses this file).
- `sub_A_primary_sharpband.csv` = A; `sub_B_hedge.csv` = B.
- References: `sub_ref_bagged_softband.csv`, `sub_ref_fixed_1_2.csv`, `sub_ref_accopt_public.csv` (the E0 reproduction of the public band rule).

**Before the deadline:** submit A and B, then on *My Submissions* tick **both** as final. Otherwise Kaggle auto-picks by public score."""))
C.append(code("""def write(name, pred):
    s = pd.DataFrame({"wedding_id": te.wedding_id, TARGET: np.asarray(pred).astype(int)})
    assert list(s.wedding_id) == list(ss.wedding_id) and s.shape == ss.shape
    s.to_csv(f"{OUT}/{name}", index=False)


soft_ref = (pred_fn(te.apg.values) > 0.5).astype(int)
write("submission.csv", A_sub)
write("sub_A_primary_sharpband.csv", A_sub)
write("sub_B_hedge.csv", B_sub)
write("sub_ref_bagged_softband.csv", soft_ref)
write("sub_ref_fixed_1_2.csv", band(te.apg, 1, 2))
write("sub_ref_accopt_public.csv", R.loc["E0", "_test_pred"])
json.dump(dict(E1=e1, E11=e11, E12=e12, E10=info, E13=dict(m=m_best, curve=curve), E15=e15, E16=e16), open(f"{OUT}/extra_results.json", "w"), indent=1, default=float)

refs = {"sub_A_primary_sharpband.csv": A_sub, "sub_B_hedge.csv": B_sub, "sub_ref_bagged_softband.csv": soft_ref,
        "sub_ref_fixed_1_2.csv": band(te.apg, 1, 2).astype(int), "sub_ref_accopt_public.csv": R.loc["E0", "_test_pred"]}
display(pd.DataFrame({"positive_rate": {k: v.mean() for k, v in refs.items()}, "rows_diff_vs_A": {k: int((v != A_sub).sum()) for k, v in refs.items()}}))
print(f"done in {time.time() - T0:.0f}s -> {OUT}")
"""))
C.append(md("## 13. (Optional) Diff against public notebook submissions"))
C.append(code("""if PUBLIC_SUBS:
    for f in sorted(glob.glob(f"{PUBLIC_SUBS}/**/*.csv", recursive=True)):
        p = pd.read_csv(f).set_index("wedding_id")[TARGET].reindex(te.wedding_id).values
        print(f"{os.path.relpath(f, PUBLIC_SUBS):70s} vs E0: {(p != R.loc['E0', '_test_pred']).sum():3d}  vs A: {(p != A_sub).sum():3d}  vs B: {(p != B_sub).sum():3d}")
else:
    print("PUBLIC_SUBS not set: skipped")
"""))

nb = nbf.v4.new_notebook()
nb.cells = C
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}}
nbf.write(nb, out)
print("wrote", out, "cells:", len(C))
