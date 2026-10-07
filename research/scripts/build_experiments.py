"""Write experiments.csv: executed research experiments (S*) + the planned/partly-evidenced E0-E14 ladder."""
import json
import sys

import pandas as pd

SP, WS = sys.argv[1], sys.argv[2]
sim = json.load(open(f"{SP}/analysis/simulation_summary.json"))
abc1 = json.load(open(f"{SP}/analysis/abc_summary.json"))
abc2 = json.load(open(f"{SP}/analysis/abc2_summary.json"))
lin = sim["dgps"]["lin_logit_sym"]; log = sim["dgps"]["log_logit_sym"]
P = lambda d, k: d["procedures"][k]

KEEP = "Corrected repeated-CV t-test (Nadeau-Bengio) p<0.05 with mean gain>0 AND size-stress accuracy not >1pt worse than parent"
REJ = "Mean loss with p<0.05, OR no significant gain while adding complexity, OR size-stress drop >1pt"
PEND = "PENDING: run solution/kacchi_pipeline.py after joining the competition"
rows = []


def add(**kw):
    base = dict(experiment_id="", parent_id="", hypothesis="", change="", dataset="train.csv 776 clean rows (24 missing-aloo dropped)", features="",
                preprocessing="unicode digits->ASCII; x10 fix (apg_raw>3 -> /10); drop missing aloo", model="", hyperparameters="", seed="",
                folds="10x5 RepeatedStratifiedKFold, strata = target x apg-zone (canonical, seed 2026)", validation="OOF accuracy + size-stress fold (train guests<=q60, validate >q60)",
                cv_score="", cv_std="", public_lb="not submitted", private_proxy="", runtime="", memory="<200 MB", leakage_risk="Low", overfit_risk="Low",
                result="", decision="", notes="", expected_cv_impact="", expected_public_impact="", expected_private_impact="", compute_cost="CPU seconds",
                keep_criteria=KEEP, reject_criteria=REJ)
    base.update(kw)
    rows.append(base)


# ------------------------------------------------------------------------------------------- executed research experiments
add(experiment_id="S1", parent_id="-", hypothesis="Public LB size can be recovered from score granularity", change="Search N in 1..400 consistent with all 10 distinct truncated LB scores",
    dataset="public leaderboard snapshot 2026-10-07 08:58 UTC", features="-", preprocessing="-", model="number theory", folds="-", validation="-",
    cv_score="-", result="N in {181, 362}; 181 is the only plausible size -> 1 row = 0.552 pts public; all-zero file = 98/181 -> public has 83 positives (45.9%)",
    decision="KEEP", notes="Kaggle truncates (not rounds) to 5 decimals (88/181=0.486188 shown as 0.48618). Private = 219 rows if no rows are ignored (UNVERIFIED).",
    runtime="<1 s", expected_cv_impact="-", expected_public_impact="-", expected_private_impact="Calibrates how much LB deltas mean", keep_criteria="-", reject_criteria="-")
add(experiment_id="S2", parent_id="-", hypothesis="Public notebook submissions differ only on a few near-edge rows",
    change="Row-wise diff of all 7 public submission files (4 distinct vectors)", dataset="public notebook output CSVs (400 test ids)", features="-", preprocessing="-", model="-",
    folds="-", validation="-", cv_score="-", result="389/400 rows identical across all public files; 11 disputed ids: W0839 W0848 W0865 W0968 W1002 W1022 W1119 W1157 W1163 W1179 W1198",
    decision="KEEP", notes="W0968/W1163 pinned to 0.9700<apg<0.9710 by the V0-vs-Rafiur disagreement; 4 rows in [0.971,0.9911] or [1.9912,2.0087); 5 rows flipped only by kNN5.",
    runtime="<1 s", expected_private_impact="Only ~6 private rows separate public solutions", keep_criteria="-", reject_criteria="-")
for name, d in [("lin_logit_sym", lin), ("log_logit_sym", log)]:
    th = [round(x, 4) for x in d["theta"]]
    add(experiment_id=f"S3-{name}", parent_id="S1", hypothesis=f"Under DGP {name} {th} (fitted to the published binned table), which public fitting procedure generalises best?",
        change=f"Monte Carlo R={sim['R']}: simulate 776 train / 400 test (181 public + 219 private), apply each procedure",
        dataset="simulated from binned public table", features="apg", preprocessing="-", model="fixed(1,2) / accopt band / soft-band MLE lin+log / kNN5 / LGBM", folds="5-fold inside each replicate",
        validation="private-part accuracy across replicates", cv_score="-",
        private_proxy=f"bayes {P(d,'bayes')['private_mean']:.4f}; fixed {P(d,'fixed_1_2')['private_mean']:.4f}; accopt {P(d,'accopt_band')['private_mean']:.4f}; mle_lin {P(d,'mle_lin')['private_mean']:.4f}; mle_log {P(d,'mle_log')['private_mean']:.4f}; knn5 {P(d,'knn5_log')['private_mean']:.4f}"
                      + (f"; lgbm {P(d,'lgbm_log')['private_mean']:.4f}" if "lgbm_log" in d["procedures"] else ""),
        result=(f"Private SD per procedure ~{P(d,'bayes')['private_sd']:.3f} (= {P(d,'bayes')['private_sd']*219:.1f} rows). Mean rows vs accopt: fixed {P(d,'fixed_1_2')['mean_rows_vs_accopt']:+.2f}, "
                f"mle_lin {P(d,'mle_lin')['mean_rows_vs_accopt']:+.2f}, mle_log {P(d,'mle_log')['mean_rows_vs_accopt']:+.2f}, knn5 {P(d,'knn5_log')['mean_rows_vs_accopt']:+.2f}. "
                f"CV gap accopt-minus-fixed: mean {d['cv_gap_mean']:+.4f} sd {d['cv_gap_sd']:.4f}; P(gap >= observed +0.0112) = {d['p_cv_gap_ge_observed']:.3f}. "
                f"Hedge best gain {max(d['hedge'].values())-d['hedge']['0']:+.4f} acc."),
        decision="KEEP", notes="Simulation under a HYPOTHESISED DGP fitted to public outputs; not real-data results.", runtime="several minutes", compute_cost="CPU minutes",
        keep_criteria="-", reject_criteria="-")
add(experiment_id="S4", parent_id="S3", hypothesis="ABC over (lo, hi, k, form) using ALL published evidence (binned rates, error counts 43/33, fitted cuts 0.970/2.0087)",
    change=f"Rejection ABC, {abc1['n_sims']} sims, accept 0.5%; logistic edges, lin/log space", dataset="published summaries", features="apg", preprocessing="-",
    model="ABC posterior", folds="-", validation="posterior predictive", cv_score="-",
    private_proxy="; ".join(f"{k} {v['mean']:.4f} ({v['rows_vs_fixed']:+.2f} rows vs fixed)" for k, v in abc1["private_expectation"].items()),
    result=f"Posterior lo 95% [{abc1['posterior']['lo_q'][0]}, {abc1['posterior']['lo_q'][-1]}], hi 95% [{abc1['posterior']['hi_q'][0]}, {abc1['posterior']['hi_q'][-1]}]; posterior-predictive Bayes cuts {[round(x,4) for x in abc1['bayes_cutpoints']]}",
    decision="KEEP", notes="Accepted sims reproduce err(1,2)~44 (obs 43) but not err(accopt)=33 (sims ~39): sample is luckier than the model or noise is heavier-tailed -> S5.",
    runtime="~25 s (12 cores)", keep_criteria="-", reject_criteria="-")
add(experiment_id="S5", parent_id="S4", hypothesis="Heavy-tailed 'noisy ratio' (Student-t) DGPs explain the data better than logistic edges",
    change=f"ABC model comparison over 4 families, {abc2['n_sims']} sims, accept 0.25%", dataset="published summaries", features="apg", preprocessing="-",
    model="ABC posterior (lin/log x logistic/Student-t)", folds="-", validation="posterior predictive", cv_score="-",
    private_proxy="; ".join(f"{k} {v['mean']:.4f} ({v['rows_vs_fixed']:+.2f} rows vs fixed)" for k, v in abc2["private"].items()),
    result=(f"Family posterior {{{', '.join(f'{k}: {v:.2f}' for k, v in abc2['posterior']['family'].items())}}} (not discriminable). "
            f"Bayes cuts {[round(x,4) for x in abc2['bayes_cutpoints']]}. P(y=1|apg): 0.97->{abc2['pp']['0.970']}, 0.98->{abc2['pp']['0.980']}, 1.00->{abc2['pp']['1.000']}, 2.00->{abc2['pp']['2.000']}, 2.02->{abc2['pp']['2.020']}"),
    decision="KEEP", notes="All candidate cut-point rules are within 0.25 expected private rows of each other: threshold choice is NOT the lever.", runtime="~100 s (12 cores)",
    keep_criteria="-", reject_criteria="-")

# ------------------------------------------------------------------------------------------- planned ladder (real data)
LOGV = "VERIFIED from public notebook execution log (5-fold or 10x5 RSKF, author's folds)"
add(experiment_id="E0", parent_id="-", hypothesis="Reproduce strongest reliable public baseline exactly", change="Accuracy-optimal two-threshold band on aloo/guest, plateau centre (FOYSAL band_rule)",
    features="apg", model="band rule", hyperparameters="lo grid 0.85-1.15, hi grid 1.85-2.2, step 0.0025", seed="-",
    cv_score="0.9558", cv_std="se 0.0018 (10x5)", public_lb="0.95027 claimed in title (UNVERIFIED)", result=LOGV + "; must reproduce 400/400 test predictions of submission_band_rule.csv",
    decision="KEEP (baseline)", overfit_risk="Medium: cut-points fitted to train noise (S3-S5)", expected_cv_impact="0.955 +/- 0.01", expected_public_impact="~0.95 (172/181 if claim holds)",
    expected_private_impact="~0.94-0.96", runtime="<1 s", keep_criteria="Exact match with public file", reject_criteria="Any mismatch -> fix cleaning first")
add(experiment_id="E0a", parent_id="-", hypothesis="Host starter RF on raw counts is a weak, non-extrapolating baseline", change="RandomForest(300) on raw columns",
    features="8 raw numeric", model="RandomForestClassifier", hyperparameters="n_estimators=300", seed="42", cv_score=PEND, public_lb="UNVERIFIED",
    result="Deep EDA log: LightGBM raw features 0.8596 +/- 0.014 (closest verified proxy)", decision="REJECT (reference only)", overfit_risk="High under size shift",
    expected_cv_impact="~0.80-0.86", expected_public_impact="~0.7-0.75 (test is bigger)", expected_private_impact="poor", runtime="~10 s")
add(experiment_id="E0b", parent_id="E0", hypothesis="Rafiur midpoint cut variant", change="Midpoint cuts, median of argmax plateau, closed band", features="apg", model="band rule",
    cv_score="0.9558 +/- 0.0147 (3x5, author folds)", result=LOGV, decision=PEND, expected_private_impact="tie with E0", runtime="<1 s")
add(experiment_id="E0c", parent_id="E0", hypothesis="Round-number generator prior: 1 < apg < 2", change="No fitting; fixed band (1, 2)", features="apg", model="fixed rule",
    cv_score="0.9446", cv_std="se 0.0023 (10x5)", result=LOGV + "; CV gap vs E0 = -0.0112, which S3 shows is very unlikely if (1,2) were the exact truth with logistic edges",
    decision="NEEDS_MORE_EVIDENCE", overfit_risk="None (no fitting)", expected_private_impact="within 0.25 rows of E0 (S5)", runtime="0 s")
add(experiment_id="E1", parent_id="E0", hypothesis="Validation must mimic the hidden test: random rows from same generator + larger weddings",
    change="Compare SKF(y) vs SKF(y x apg-zone) fold SD; add size-stress fold (train small, validate largest 40%)", features="apg", model="band rule + raw RF/LGBM",
    cv_score=PEND, result="Expected: ratio models flat on size-stress, raw-count models collapse", decision=PEND, expected_cv_impact="lower variance of estimates", runtime="~1 min")
add(experiment_id="E2a", parent_id="E0", hypothesis="x10 repair changes nothing for a band fitted only in 0.85-2.2 (typos land at apg>9)", change="Train without x10 fix",
    features="apg_raw", model="band rule", cv_score=PEND, decision=PEND, expected_cv_impact="0", runtime="<1 s")
add(experiment_id="E2b", parent_id="E0", hypothesis="Dropping x10 rows ~ fixing them", change="Drop 6 x10 rows", features="apg", model="band rule", cv_score=PEND, decision=PEND,
    expected_cv_impact="0", runtime="<1 s")
add(experiment_id="E2c", parent_id="E3a", hypothesis="Noise is multiplicative -> log-space soft band fits better (best AIC in S-fits)", change="Soft band in log(apg)",
    features="log_apg", model="soft-band MLE (log)", cv_score="0.9407 (logmle_band, 10x5)", result=LOGV, decision=PEND,
    expected_private_impact="+0.05 rows vs fixed (S5)", runtime="~1 s")
add(experiment_id="E3a", parent_id="E0", hypothesis="Likelihood fit uses all rows near the edge -> lower-variance cut-points than 0-1 accuracy search",
    change="Soft-band MLE (linear apg), predict p>0.5", features="apg", model="soft-band MLE (lin)", cv_score=PEND, decision=PEND,
    expected_private_impact="+0.5 to +0.9 rows vs accopt in S3 (DGP-dependent)", runtime="~1 s")
add(experiment_id="E3b", parent_id="E3a", hypothesis="Bootstrap/AIC-averaged soft band approximates the posterior-predictive Bayes rule",
    change="Bagged soft band (B=100-200, lin+log)", features="apg", model="bagged soft band", cv_score=PEND, decision=PEND,
    expected_private_impact="~+0.25 rows vs fixed (S5 posterior Bayes)", runtime="~1-2 min", compute_cost="CPU minutes")
add(experiment_id="E4a", parent_id="E0", hypothesis="Host hint: depth-2 tree on the ratio", change="DecisionTree(max_depth=2) on apg", features="apg", model="CART depth 2",
    cv_score=PEND, decision=PEND, expected_cv_impact="~= E0", runtime="<1 s")
add(experiment_id="E5a", parent_id="E3a", hypothesis="Quadratic logit in log space is a smooth symmetric band", change="LogisticRegression on [z, z^2], z = log apg",
    features="log_apg, log_apg^2", model="LogisticRegression C=1e4", cv_score="0.9420 (linear apg+apg^2 variant, Deep EDA)", result=LOGV, decision=PEND, runtime="<1 s")
add(experiment_id="E5b", parent_id="E3a", hypothesis="kNN on log apg (public k=5)", change="kNN k=5", features="log_apg", model="kNN", cv_score="0.9513 (10x5)", result=LOGV,
    decision=PEND, overfit_risk="Medium (k small)", runtime="<1 s")
add(experiment_id="E5c", parent_id="E5b", hypothesis="Smoother kNN reduces edge noise", change="kNN k=25", features="log_apg", model="kNN", cv_score=PEND, decision=PEND, runtime="<1 s")
add(experiment_id="E5d", parent_id="E0a", hypothesis="Raw-feature logistic cannot represent a band", change="LogisticRegression raw", features="raw", model="LR",
    cv_score="0.5554 +/- 0.0141 (Deep EDA)", result=LOGV, decision="REJECT", runtime="<1 s")
add(experiment_id="E5e", parent_id="E3a", hypothesis="Boosting on log ratio", change="LightGBM log apg, 3 seeds", features="log_apg", model="LightGBM",
    hyperparameters="n=300 lr=0.02 leaves=4 mcs=15 subsample=0.8", seed="0,1,2", cv_score="0.9541 (10x5)", result=LOGV, decision=PEND, runtime="~5 s")
add(experiment_id="E5f", parent_id="E5e", hypothesis="All per-guest ratios + categoricals add nothing", change="LightGBM on ratio features", features="ratios + event/city",
    model="LightGBM", cv_score="0.9407 +/- 0.0094 (engineered, Deep EDA)", result=LOGV, decision=PEND, overfit_risk="Medium (noise features)", runtime="~3 s")
add(experiment_id="E5g", parent_id="E0a", hypothesis="Boosting on raw counts fails under size shift", change="LightGBM raw", features="raw + cats", model="LightGBM",
    cv_score="0.8596 +/- 0.0140 (Deep EDA)", result=LOGV + "; size-stress expected far lower", decision="REJECT", overfit_risk="High", runtime="~3 s")
add(experiment_id="E6", parent_id="E3a", hypothesis="Probability blend of decorrelated ratio models", change="Mean P of soft-lin, soft-log, quad-logit, LGBM, kNN25",
    features="apg", model="blend", cv_score=PEND, decision=PEND, expected_private_impact="~0 (members agree off-edge; disagree only on coin-flip rows)", runtime="~10 s")
add(experiment_id="E7", parent_id="E6", hypothesis="Hard vote / rank averaging", change="Majority vote of members", features="apg", model="vote", cv_score=PEND, decision=PEND,
    notes="Rank averaging is meaningless for a non-monotone band under accuracy: needs a calibrated 0.5 threshold.", runtime="~10 s")
add(experiment_id="E8", parent_id="E6", hypothesis="Nested stacking (LR meta on inner-CV OOF)", change="Nested 5x5 stacking", features="member OOF P", model="LR meta",
    cv_score=PEND, decision=PEND, overfit_risk="Medium", expected_private_impact="~0", runtime="~1 min", folds="3x5 outer, 5 inner")
add(experiment_id="E9", parent_id="E3a", hypothesis="Pseudo-labels from confident test rows help", change="Add test rows with P>0.98 or <0.02 (fold-internal), refit",
    features="apg", model="soft band + PL", cv_score=PEND, decision=PEND, leakage_risk="Low if fold-internal",
    notes="Confident rows are far from edges and carry ~no information about cut-points -> expected no effect.", expected_private_impact="0", runtime="~10 s")
add(experiment_id="E10", parent_id="E3b", hypothesis="Final robust single decision rule = posterior-predictive Bayes band",
    change="Bagged soft band on all 776 rows (or ABC cuts ~0.982 / ~2.0075 from S5)", features="apg", model="bagged soft band", cv_score=PEND, decision=PEND,
    expected_private_impact="Best expected accuracy, but only ~0.25 rows above any public rule", runtime="~1 min")
add(experiment_id="E11", parent_id="E3a", hypothesis="Errors near the edges are pure label noise (no other column predicts them)",
    change="Near-edge rows: LR on signed log-margin vs margin + all other features (5x5 CV log-loss)", features="margin + others", model="LR",
    cv_score=PEND, decision=PEND, notes="Public evidence: mutton p=0.22-0.34 near edges; no event/city/drone effect (Deep EDA, Rafiur logs).", runtime="~5 s")
add(experiment_id="E12", parent_id="E0", hypothesis="aloo/guests is the right ratio (vs extra eaters, mutton, borhani denominators)",
    change="In-sample best band accuracy for 11 candidate ratios", features="ratios", model="best band (prefix sums)",
    cv_score="Rafiur log: aloo/guests 0.9588 best; +dhol 0.9562; +aunties 0.9536; +50 0.8814", result="VERIFIED from Rafiur execution log", decision="KEEP aloo/guests", runtime="<5 s")
add(experiment_id="E13", parent_id="E10", hypothesis="Two final selections that differ on the most uncertain test rows raise E[best-of-two private score]",
    change="A = Bayes decisions; B = A flipped on the m rows with P closest to 0.5; m chosen by Monte Carlo on real test P",
    features="apg", model="hedge construction", cv_score="-", private_proxy="Simulation: +0.5 to +0.9 expected private rows (S3; replica run m=18 -> +0.88 rows)",
    decision=PEND, overfit_risk="None (no labels used)", leakage_risk="None", notes="Requires selecting BOTH files manually as final submissions; Kaggle scores the best of the selected.",
    expected_private_impact="Largest available lever (> any threshold choice)", runtime="~30 s")
add(experiment_id="E14", parent_id="-", hypothesis="Integrity checks: no ID-order leak, no train/test duplicates, ratio features are shift-free",
    change="Spearman(id, y), Spearman(id, band errors), exact/near duplicate hashes, adversarial AUC raw vs ratio, KS/PSI", features="all", model="audit",
    cv_score=PEND, decision=PEND, notes="Expected: adversarial AUC high on raw (guests), ~0.5 on ratios", runtime="~10 s")

add(experiment_id="E15", parent_id="E2c", hypothesis="Edge sharpness grows with wedding size (counting noise in integer aloo_count / guests matters less at big weddings)",
    change="Log-space soft band with k = exp(a + gamma*log(guests/median)); likelihood-ratio test vs gamma=0", features="log_apg, guests (width only)",
    model="heteroscedastic soft band", cv_score="-", decision=PEND, private_proxy="If significant: re-ranks uncertain test rows for the E13 hedge",
    notes="Synthetic replica with integer rounding of aloo_count: gamma=0.38, p=0.011 (code test only, not competition data).",
    expected_private_impact="small; better hedge targeting", runtime="~10 s", keep_criteria="LR test p<0.05 and gamma>0", reject_criteria="p>=0.05")

df = pd.DataFrame(rows)
df.to_csv(f"{WS}/experiments.csv", index=False, encoding="utf-8")
print("experiments.csv rows:", len(df))
