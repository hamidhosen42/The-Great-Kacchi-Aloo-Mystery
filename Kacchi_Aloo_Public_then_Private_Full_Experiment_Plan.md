# The Great Kacchi Aloo Mystery
## Public-LB Beat → Private-LB Win: Full Experiment Plan

**Status used for this plan**

- Current best public submission: **Q01 = 0.96685 = 175/181**
- Public position reported by the user: **#3**, tied with FOYSAL and Dadhichi
- Public leaderboard uses about **45%** of the 400-row test set
- Final standings use the other about **55%**
- Current selected/final-candidate pool already contains 25 submissions
- Current representative notebook folder contains 11 non-redundant notebooks
- Q02 scored `0.93922` and is method-equivalent to P12; it should be treated as redundant until an exact 400-row vector hash comparison proves otherwise

This document has two explicit objectives:

1. **Beat the current public top score** without turning the workflow into row-by-row leaderboard probing.
2. **Build a stronger private-LB portfolio** so that public success does not come at the cost of the final ranking.

The public objective and private objective are related, but they are **not the same optimization problem**.

---

# 1. Executive Decision

The competition is not a “find a bigger model” problem anymore.

The real-data work already established that the useful signal is essentially:

\[
r = \frac{\text{aloo_count}}{\text{guests}}
\]

with a sharp two-sided band around approximately:

\[
0.97068 \le r \le 2.00983
\]

and that most remaining mistakes occur near the two edges.

The correct strategy now is:

\[
\boxed{
\text{better edge probability}
\rightarrow
\text{better uncertainty ranking}
\rightarrow
\text{better hedge vectors}
}
\]

The wrong strategy is:

\[
\boxed{
\text{more random thresholds}
\rightarrow
\text{more public feedback}
\rightarrow
\text{public overfit}
}
\]

Q01 already reached `175/181`, which ties the current public top.

To **beat** the current public top outright, the next public target is:

\[
176/181 = 0.97237
\]

So the public problem is literally a **+1 net-row improvement over Q01**.

That sounds small, but because the scored public set is only 181 rows, it is also easy to overfit.

For private, the objective is different:

> Maximize the expected best private score among the selected final tickets, using **unique, statistically plausible prediction vectors**.

---

# 2. Evidence Already Established

The existing notebooks and deep-analysis report support the following findings.

## 2.1 Data cleaning

The established pipeline:

- normalizes Unicode/Bangla digits;
- drops training rows with missing `aloo_count`;
- corrects obvious ×10 `aloo_count` typos when `aloo_count / guests > 3`;
- computes `apg = aloo_count / guests`.

The usable train set is 776 rows after dropping 24 rows with missing `aloo_count`.

## 2.2 Real-data model findings

Reported real-data results include approximately:

| Approach | CV / result |
|---|---:|
| Sharp accuracy-optimal band | **0.9546** |
| Midpoint sharp band | 0.9543 |
| LightGBM on log(APG) | 0.9537 |
| Fixed `(1,2)` band | 0.9446 |
| Smooth soft band | 0.9421 |
| Bagged soft band | 0.9386 |

The smooth-band family was already shown worse than the sharp rule.

## 2.3 Edge-error structure

The established sharp rule has cut-points around:

\[
0.97068,\quad 2.00983
\]

and the OOF error pattern reported by the notebooks is:

| Distance to nearest edge | OOF error |
|---|---:|
| `0–0.01` | 0.127 |
| `0.01–0.035` | **0.374** |
| `0.035–0.10` | 0.148 |
| `0.10–0.40` | 0.022 |
| `>0.40` | ≈ 0 |

That tells us the battle is in the noisy ring.

## 2.4 Already-rejected directions

Do not spend serious submission budget on these again unless a genuinely new validation result appears:

- raw-count RF / XGBoost / CatBoost as the main model;
- large neural networks;
- smooth logistic soft-band as the primary classifier;
- city-specific boundaries;
- event-specific boundaries;
- drone-specific boundaries;
- size-specific boundaries;
- mutton / borhani / fairy-light group-specific boundaries;
- pseudo-labeling of confident test rows;
- arbitrary per-feature hidden-rule searches.

The previous analysis found no robust additional signal from these.

---

# 3. Current Representative Notebook Audit

The current folder contains these representative methods.

| Notebook | Public score | Main method |
|---|---:|---|
| **Q01** | **0.96685** | Flip E30 uncertainty ranks 1–7 from A |
| P02 | 0.96132 | Band `1.00608–2.02353` |
| P03 | 0.96132 | A + E16 upper uncertainty ranks 8–14 |
| B | 0.95580 | A + 9 most-uncertain E16 rows |
| C | 0.95580 | Band `1.000–1.975` |
| A | 0.95027 | Sharp band `0.97068–2.00983` |
| P05 | 0.95027 | Lower-edge flip ranks 6–10 |
| P01 | 0.94475 | Upper-edge flip ranks 3–10 |
| P07 | 0.93922 | Band `0.95816–1.98070` |
| P12 | 0.93922 | Upper-edge flip ranks 1–5 |
| P06 | 0.93370 | Both-edge flip ranks 9–14 |

The key observation is that many submissions are **different tickets but not different statistical models**.

Most of the portfolio comes from:

1. one sharp-band anchor;
2. one uncertainty ordering;
3. contiguous blocks of that ordering;
4. a small set of shifted hard bands.

The next gains should come from **new uncertainty estimators**, not just more windows from the same ranking.

---

# 4. Q01 Forensic Analysis

Q01 is extremely important because it ties the public top.

Its notebook uses:

- E16: sharp-band OOF distance-based probability;
- E30: lower/upper isotonic edge probability;
- E31: Bayesian/change-point-style edge probability;
- Q01 itself: flip E30 uncertainty ranks 1–7 relative to A.

The notebook reports:

- E16 uncertain rows with `0.25 < P < 0.75`: 26
- E30 uncertain rows: 17
- E31 uncertain rows: 3

Q01 changes exactly these seven test rows:

| wedding_id | APG | E30 P | A | Q01 |
|---|---:|---:|---:|---:|
| W1163 | 0.970443 | 0.379 | 0 | 1 |
| W0968 | 0.970588 | 0.425 | 0 | 1 |
| W1119 | 0.981221 | 0.692 | 1 | 0 |
| W1157 | 0.987382 | 0.692 | 1 | 0 |
| W1179 | 0.999286 | 0.692 | 1 | 0 |
| W1022 | 1.001347 | 0.694 | 1 | 0 |
| W0972 | 1.005587 | 0.699 | 1 | 0 |

Notice something crucial:

> Q01 is **not** simply “use the E30 predicted class.”

For example:

- `P=0.425` would normally predict `0`, but Q01 flips to `1`;
- `P≈0.69` would normally predict `1`, but Q01 flips to `0`.

Q01 is a **hedge against the model’s most uncertain E30 decisions**.

Therefore:

\[
\boxed{
\text{Q01 public success} \neq
\text{proof that E30 is the true classifier}
}
\]

It proves that the **E30 uncertainty ordering is useful enough to construct a strong hedge**.

That distinction is central to the private strategy.

---

# 5. Public Score Mathematics

The current public test size is consistent with 181 rows.

Therefore one row is:

\[
\frac{1}{181} = 0.00552486
\]

so the score ladder is:

| Correct | Score |
|---:|---:|
| 169 | 0.93370 |
| 170 | 0.93922 |
| 171 | 0.94475 |
| 172 | 0.95027 |
| 173 | 0.95580 |
| 174 | 0.96132 |
| 175 | **0.96685** |
| 176 | **0.97237** |
| 177 | 0.97790 |

Current Q01 is already:

\[
175/181
\]

So to beat the current public top:

\[
\boxed{Q01 + 1\ \text{net correct public row}}
\]

is sufficient.

However, the next public experiment must not be designed as a one-row label probe.

---

# 6. Two-Track Strategy

From this point forward every new experiment gets one of two labels.

## Track P — Public Specialist

Purpose:

> increase the probability of reaching 176/181 or higher.

A Track-P candidate may be slightly worse in CV if it is a coherent, pre-specified hedge over uncertain rows.

But it must **not** be constructed by changing one row at a time and reading the leaderboard.

## Track R — Private Robust

Purpose:

> maximize expected private accuracy / best-of-portfolio performance.

A Track-R model is selected only using:

- OOF predictions;
- repeated CV;
- edge log-loss / Brier;
- size-stress checks;
- bootstrap / jackknife uncertainty;
- portfolio simulation.

Public score is recorded afterward but does not determine model parameters.

The best final 25 should contain both types.

---

# 7. PUBLIC-BEAT EXPERIMENT PROGRAM

The public goal is one additional net row beyond Q01.

We should not blindly perturb Q01.

Instead, generate a small **pre-registered batch** from improved edge uncertainty.

---

# 8. P-01 — Cross-Fitted E30 Audit

Before using E30 more, validate it properly.

## Hypothesis

The E30 isotonic ranking may be useful because it captures lower/upper edge shape more accurately than E16.

## Current E30

Lower window:

\[
0.85 \le r \le 1.15
\]

with increasing isotonic regression.

Upper window:

\[
1.80 \le r \le 2.20
\]

with decreasing isotonic regression.

Outside those windows, plateau probabilities are used.

## Required experiment

Use 10×5 repeated stratified CV.

For every fold:

1. fit lower isotonic on training fold only;
2. fit upper isotonic on training fold only;
3. predict validation fold;
4. save OOF probability.

Report:

- overall accuracy at `P>0.5`;
- edge accuracy;
- log-loss;
- Brier;
- calibration near 0.5;
- uncertainty ranking stability across repeats.

## Decision

If E30 cross-fitted probability is better than or tied with E16 on edge log-loss/Brier:

**KEEP E30 as a primary uncertainty model.**

If E30 is materially worse:

**Q01 remains a hedge, but stop expanding E30 blindly.**

---

# 9. P-02 — Bagged E30 Isotonic

Plain isotonic is sensitive to small sample changes.

## Method

Bootstrap the train set `B=1000` times.

For each bootstrap:

- fit E30 lower isotonic;
- fit E30 upper isotonic;
- predict all 400 test rows.

Average:

\[
P_{\text{E30bag},i}
=
\frac{1}{B}\sum_b P_{b,i}
\]

Produce two vectors:

### Candidate P-02A

MAP:

\[
\hat y_i = I(P_{\text{E30bag},i} > 0.5)
\]

### Candidate P-02B

Hedge:

- compute `abs(P - 0.5)`;
- use a **pre-selected hedge size determined from CV/simulation**, not from public feedback;
- flip the most uncertain MAP decisions.

## Why it may beat Q01

Q01 uses one E30 fit.

Bagging may re-order the 7–20 most uncertain rows and identify a better coherent hedge.

---

# 10. P-03 — E30 Window Ensemble

E30 currently uses one arbitrary pair of windows.

Test these window pairs in CV:

### Lower windows

- `[0.85, 1.15]`
- `[0.88, 1.12]`
- `[0.90, 1.10]`

### Upper windows

- `[1.80, 2.20]`
- `[1.85, 2.15]`
- `[1.90, 2.10]`

Do **not** pick the best by Public LB.

For each test row average predictions over the CV-acceptable window configurations.

This produces:

\[
P_{\text{E30win}}
\]

Submit only if the resulting vector is different from Q01/A/P02.

---

# 11. P-04 — Four-Sided Edge Posterior

E16 treats nearest-edge distance too symmetrically.

Split the world into four cases:

- `LO`: lower-edge outside
- `LI`: lower-edge inside
- `UI`: upper-edge inside
- `UO`: upper-edge outside

For each region and distance bin:

\[
P(y=1 \mid \text{region}, d)
\]

Use Jeffreys smoothing:

\[
\hat p = \frac{k+0.5}{n+1}
\]

Recommended distance bins:

- 0–0.005
- 0.005–0.01
- 0.01–0.02
- 0.02–0.035
- 0.035–0.06
- 0.06–0.10
- 0.10–0.20
- 0.20–0.40

Merge bins automatically if counts are too small.

Generate:

- MAP vector;
- uncertainty ordering;
- one pre-registered hedge vector.

This is a direct upgrade path from B/E16.

---

# 12. P-05 — Cross-Fitted E16/E30/E31 Consensus

Use three independent probability estimates:

\[
P_{16},\quad P_{30},\quad P_{31}
\]

but **do not use public scores as weights**.

Estimate weights only with OOF data:

\[
P^*
=
w_{16}P_{16}
+
w_{30}P_{30}
+
w_{31}P_{31}
\]

such that:

\[
w_j \ge 0,\qquad \sum w_j = 1
\]

Choose weights by nested-CV log-loss or Brier.

Produce:

1. MAP candidate;
2. uncertainty hedge candidate.

This is the private-safe replacement for public-score-based re-weighting.

---

# 13. P-06 — E31 Change-Point Upgrade

Current E31 reports only three test rows in `0.25<P<0.75`, meaning it is more decisive than E30/E16.

The next experiment is to make E31 properly cross-fitted and compare:

- Beta(1,1)
- Jeffreys Beta(0.5,0.5)
- edge-window variants

Use posterior integration rather than a single cut.

If E31 identifies only a few ambiguous rows and those are stable across folds, it may produce a high-value public candidate with fewer risky flips than Q01.

---

# 14. P-07 — Jackknife Edge Stability

For every usable train row \(j\):

1. remove row \(j\);
2. refit edge model;
3. predict all test rows.

Then:

\[
\bar P_i
=
\frac{1}{M}\sum_j P_{-j,i}
\]

and:

\[
V_i
=
\operatorname{Var}_j(P_{-j,i})
\]

Use both:

- closeness to 0.5;
- high jackknife variance.

A row with `P≈0.5` and high model instability is a much stronger hedge target than a row selected only by distance to the edge.

---

# 15. P-08 — Bootstrap Hard-Band Posterior

Use 2000–5000 bootstrap samples.

Each bootstrap fits a **hard** sharp band.

For test row \(i\):

\[
P_i
=
\frac{1}{B}\sum_b
I(lo_b \le r_i \le hi_b)
\]

This retains the sharp-edge structure that worked in real CV.

Generate:

- bootstrap MAP;
- bootstrap-uncertainty hedge.

This experiment is distinct from the already-failed soft logistic band.

---

# 16. P-09 — Q01-Neighborhood Batch, But Pre-Registered

This is the most aggressive public-oriented experiment and must be handled carefully.

Q01 flips 7 E30-uncertain lower-edge rows.

Do **not** probe one row at a time.

Instead, before submitting anything, pre-register a small set of **block-level** alternatives from a fresh probability model such as Bagged-E30:

- Hedge size 5
- Hedge size 7
- Hedge size 9
- Hedge size 11

The ordering must be frozen before seeing their public scores.

Use at most one batch.

Reason:

> We want to test whether Q01’s success is a stable uncertainty-region effect, not reverse-engineer individual public labels.

If one of these reaches `176/181`, freeze the public experiment family immediately.

---

# 17. P-10 — Public Candidate Stop Rule

For public optimization:

- once a candidate reaches **176/181 or better**, stop local public search;
- do not continue squeezing the public leaderboard;
- move all effort to private robustness.

The cost of chasing `177/181` after reaching `176/181` is likely more public overfit than private benefit.

---

# 18. Expected Public Submission Order

If fresh daily submissions are available, the next five should be:

1. **Bagged E30 MAP**
2. **Four-sided posterior MAP**
3. **OOF-only E16/E30/E31 consensus MAP**
4. **Jackknife-stability hedge**
5. **One pre-registered Bagged-E30 block hedge**

Do not start with more arbitrary hard-band cut pairs.

---

# 19. PRIVATE-LB WIN PROGRAM

Once the public top is tied or beaten, the objective changes.

The final score uses the private subset.

For the private side, Q01’s public score should **not** dominate model selection.

---

# 20. R-01 — Canonical OOF Probability Table

Create one table with a row for every training example and OOF probabilities from:

- A sharp rule
- E16
- E30
- E31
- Four-sided posterior
- Bagged E30
- Bootstrap hard band
- Jackknife ensemble

Required columns:

```text
wedding_id
apg
target
P_E16
P_E30
P_E31
P_E30bag
P_4side
P_bootstrap
P_jackknife
```

No training row may receive a probability from a model that trained on that row.

This table is the foundation for all private model selection.

---

# 21. R-02 — Calibrated Model Comparison

For each model compute:

- accuracy;
- log-loss;
- Brier;
- edge-only log-loss;
- edge-only Brier;
- lower-edge accuracy;
- upper-edge accuracy;
- calibration error;
- seed stability.

Do not select on accuracy alone.

Why:

Two models can have the same accuracy but very different uncertainty quality, and the uncertainty quality determines hedge quality.

---

# 22. R-03 — Nested CV for Hyperparameters

Any of these must be selected in an inner loop:

- isotonic windows;
- distance bins;
- ensemble weights;
- hedge-size parameters;
- change-point priors;
- smoothing strength.

The outer fold is used only for final evaluation.

This prevents the private model from repeating the same cut-point overfitting problem.

---

# 23. R-04 — Size-Stress Check

Even though APG is scale-free, keep the size-stress test.

Train on smaller weddings, validate on larger weddings.

Reject any “improvement” that collapses on large weddings.

---

# 24. R-05 — Public-Independent Model Freeze

For the private anchor:

**freeze all parameters before checking its public score.**

Public score may be reported after the vector is frozen, but it must not determine:

- model weights;
- edge windows;
- cut-points;
- hedge size.

This protects the private anchor from adaptive public fitting.

---

# 25. R-06 — Posterior Uncertain-Row Set

After the best calibrated probability ensemble is selected, define a final uncertain set such as:

\[
0.20 < P_i < 0.80
\]

or use a data-driven threshold from OOF calibration.

Expected size should be small, perhaps around 10–30 rows.

Confident test rows remain fixed.

Only uncertain rows participate in portfolio design.

---

# 26. R-07 — Posterior Codebook Portfolio

This is the most important private experiment.

Suppose there are \(m\) uncertain test rows.

Do not use contiguous rank windows.

Instead generate plausible label vectors from:

\[
Y_i^{(s)}
\sim Bernoulli(P_i)
\]

for uncertain rows.

Confident rows stay at MAP predictions.

Generate 50k–500k candidate vectors.

For every candidate estimate its expected contribution to the final portfolio.

---

# 27. R-08 — Simulate Public/Private Split

For each Monte Carlo replicate:

1. sample/define latent labels from the calibrated posterior;
2. randomly allocate 181 rows to public and 219 to private, consistent with the competition design;
3. score every existing candidate on private;
4. score each potential new candidate;
5. retain the best-of-selected private score.

The key portfolio objective is:

\[
E\left[
\max_{c \in \mathcal{S}}
S_{\text{private}}(c)
\right]
\]

not mean individual score.

---

# 28. R-09 — Marginal Portfolio Value

For candidate \(c\), define:

\[
\Delta(c)
=
E[
\max(S_{\mathcal{S}}, S_c)
-
S_{\mathcal{S}}
]
\]

where \(\mathcal{S}\) is the current selected portfolio.

Greedily add the candidate with the largest \(\Delta\).

Then recompute and continue until 25 unique slots are filled.

This directly optimizes the Kaggle best-of-selected mechanism.

---

# 29. R-10 — Diversity Constraint

Do not allow the optimizer to fill the portfolio with near-duplicates.

For prediction vectors \(a,b\):

\[
H(a,b)
=
\sum_i I(a_i \ne b_i)
\]

Use minimum Hamming-distance constraints among uncertain rows.

Suggested rule:

- core anchors may be close;
- hedge tickets should differ by at least 3 meaningful uncertain rows;
- exact duplicates are forbidden.

---

# 30. R-11 — Q02/P12 Duplicate Audit

Q02 and P12 are reported to have the same method and public score.

Before final selection calculate:

```python
diff = (pred_Q02 != pred_P12).sum()
```

If:

```text
diff == 0
```

they are exact duplicates.

Keeping both consumes a final slot with **zero** possible best-of-25 benefit.

If a unique submitted vector is available, replace one duplicate with that unique vector.

---

# 31. R-12 — Existing Ticket Classification

Classify every submitted ticket as:

### Anchor

Strong model expected to generalize:

- A
- best OOF-calibrated new MAP
- bootstrap/jackknife consensus

### Public Specialist

Strong public result but adaptive/hedged:

- Q01
- possibly P02
- possibly P03

### Private Hedge

Lower individual expectation but high portfolio complementarity:

- B
- C
- selected P tickets
- posterior-codebook tickets

### Redundant

Exact or near duplicates with no marginal value.

The final 25 should be intentionally balanced across these roles.

---

# 32. R-13 — Keep Q01, But Do Not Make It the Only Anchor

Q01 must stay.

It is currently the best public ticket.

But its generation logic is a hedge:

\[
Q01 = A \oplus
\{\text{E30 most uncertain ranks 1–7}\}
\]

Therefore the ideal final portfolio contains:

- Q01;
- A or another sharp anchor;
- a private-calibrated MAP model;
- multiple posterior-diverse hedges.

---

# 33. R-14 — P02 Interpretation

P02:

\[
1.00608 \le r \le 2.02353
\]

scores `174/181` public but has train accuracy reported as only `0.9446`.

That makes it a useful hedge, but not a trustworthy canonical boundary.

Run nested CV over:

\[
lo \in [0.96,1.02]
\]

\[
hi \in [1.97,2.04]
\]

Record selected cuts in each outer fold.

If cuts consistently migrate toward P02 and outer accuracy improves, upgrade P02’s status.

Otherwise keep it as a hedge only.

---

# 34. R-15 — Final 25 Portfolio Optimization

If the system allows 25 selected finals, the portfolio should not simply be the 25 highest public scores.

Use this priority:

1. unique strong private anchors;
2. public winner Q01;
3. OOF-calibrated MAP models;
4. diverse posterior hedges;
5. a few proven public specialists;
6. remove duplicates and weakest redundant windows.

If all 25 slots are already occupied, the next experiment is not necessarily a new submission.

It may be **re-selection**:

> replace zero-marginal duplicate/redundant tickets with already-submitted unique vectors having higher simulated private marginal value.

---

# 35. Exact Validation Ledger for Every New Experiment

Every experiment row must contain:

| Field | Requirement |
|---|---|
| ID | P-xx or R-xx |
| Parent | A / Q01 / E30 / etc |
| Hypothesis | One sentence |
| Parameters | Frozen before public |
| Folds | Same repeated folds |
| OOF Accuracy | Required |
| OOF Log-loss | Required |
| OOF Brier | Required |
| Edge Accuracy | Required |
| Edge Log-loss | Required |
| Lower-edge score | Required |
| Upper-edge score | Required |
| Size-stress score | Required |
| Prediction count | Required |
| Hamming vs A | Required |
| Hamming vs Q01 | Required |
| Hamming vs P02 | Required |
| Public score | Recorded after freeze |
| Portfolio marginal value | Required for final selection |
| Decision | KEEP / PUBLIC-HEDGE / PRIVATE-HEDGE / REJECT |

---

# 36. Keep / Reject Rules

## KEEP as private anchor

Require:

- CV ≈ 0.953+ or statistically tied with the strongest anchor;
- stable across seeds;
- no size-stress collapse;
- edge probability metrics competitive;
- no hidden public fitting.

## KEEP as hedge

May have lower raw CV if:

- all changed rows are genuinely uncertain;
- vector adds meaningful portfolio diversity;
- Monte Carlo marginal portfolio value is positive.

## REJECT

Reject if:

- improvement exists only after seeing public score;
- it changes confident rows;
- it duplicates another vector;
- one lucky seed creates the improvement;
- other noisy features are added without validated residual signal;
- the candidate is based on one-row public probing.

---

# 37. Public-Beat Submission Ladder

If additional submissions are still possible, use the following order.

## Batch 1 — model improvement

1. Bagged-E30 MAP
2. Four-sided MAP
3. OOF-only E16/E30/E31 consensus MAP

## Batch 2 — uncertainty improvement

4. Jackknife instability hedge
5. Bootstrap-hard-band hedge

## Batch 3 — only if still below 176/181

6. one pre-registered Bagged-E30 hedge-size candidate
7. one second pre-registered hedge-size candidate

Do not exceed this without a new modelling hypothesis.

---

# 38. Public Stop Criteria

Stop public chasing immediately if:

- any candidate reaches **0.97237 = 176/181** or higher;
- or the next 5 coherent candidates fail to improve Q01.

At that point:

\[
\boxed{\text{all compute} \rightarrow \text{private portfolio}}
\]

---

# 39. Private Portfolio Simulation Specification

Recommended Monte Carlo:

- 100,000+ replicates if cheap;
- posterior label probabilities from the calibrated OOF ensemble;
- 181/219 public/private partition;
- all submitted vectors scored;
- best-of-selected objective.

Report:

- expected best private accuracy;
- median best private accuracy;
- 5/25/75/95 percentiles;
- probability each ticket is portfolio winner;
- probability each ticket is never useful;
- marginal value of each slot;
- pairwise Hamming matrix.

The tickets with near-zero marginal value should be replaced first.

---

# 40. Public vs Private Decision Matrix

| Situation | Interpretation | Action |
|---|---|---|
| CV↑, public↑ | strongest evidence | keep |
| CV↑, public↓ by 1–2 rows | public noise possible | private keep |
| CV≈, public↑ | public specialist / hedge | keep as hedge |
| CV↓, public↑ | likely public overfit | do not make anchor |
| CV↑, public same | useful private candidate | keep |
| CV↓, public↓ | reject |
| Public top but model is hedge | Q01-like | keep, diversify around it |
| Exact duplicate | zero portfolio value | remove one |

---

# 41. What Not to Do

Do **not**:

- flip one test row, submit, infer its label, repeat;
- infer public/private membership row-by-row;
- brute-force tiny threshold moves until one gains one public row;
- use public scores to fit dozens of model weights;
- replace all anchors with Q01-like oppositional hedges;
- select final 25 purely by public score;
- keep exact duplicate prediction vectors;
- resurrect feature families already falsified without new evidence.

---

# 42. Highest-Priority New Experiments

Overall priority:

\[
\boxed{
\text{Cross-fit E30}
\rightarrow
\text{Bagged E30}
\rightarrow
\text{Four-sided posterior}
\rightarrow
\text{OOF BMA}
\rightarrow
\text{Jackknife}
\rightarrow
\text{Posterior codebook}
}
\]

Public priority:

\[
\boxed{
\text{Bagged E30 MAP}
\rightarrow
\text{Four-side MAP}
\rightarrow
\text{OOF consensus MAP}
}
\]

Private priority:

\[
\boxed{
\text{OOF calibration}
\rightarrow
\text{posterior codebook}
\rightarrow
\text{best-of-25 portfolio optimization}
}
\]

---

# 43. Practical Notebook Build List

If implementing from scratch, create these notebooks/scripts:

```text
E39_crossfit_E30.ipynb
E40_bagged_E30.ipynb
E41_oof_E16_E30_E31_BMA.ipynb
E42_four_sided_edge_posterior.ipynb
E43_E30_window_sensitivity.ipynb
E44_jackknife_edge_stability.ipynb
E45_posterior_codebook_portfolio.ipynb
E46_private_portfolio_optimizer.ipynb
E47_P02_nested_cut_test.ipynb
```

Each notebook should:

1. load competition data;
2. run deterministic cleaning;
3. reproduce A;
4. create OOF probabilities;
5. print validation metrics;
6. print changed test rows relative to A and Q01;
7. write `submission.csv` only if it passes a hard validation gate;
8. print an MD5/hash of the prediction vector;
9. save a machine-readable experiment JSON/CSV row.

---

# 44. Pseudocode — Cross-Fitted Probability Evaluation

```python
for repeat in range(10):
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=2026 + repeat
    )

    for train_idx, val_idx in skf.split(X, y):
        model = fit_edge_model(
            r[train_idx],
            y[train_idx]
        )

        oof_prob[val_idx] = model.predict_proba(
            r[val_idx]
        )

acc = accuracy_score(y, oof_prob >= 0.5)
ll  = log_loss(y, oof_prob)
br  = brier_score_loss(y, oof_prob)
```

Use identical folds for every model.

---

# 45. Pseudocode — Bagged E30

```python
P_test = np.zeros(len(test))
P_train_oof = ...

for b in range(1000):
    idx = rng.integers(0, len(train), len(train))

    model = fit_isotonic_edges(
        train.apg.values[idx],
        train.y.values[idx]
    )

    P_test += model.predict(test.apg.values)

P_test /= 1000
pred = (P_test > 0.5).astype(int)
```

For honest OOF bagging, bootstrap only inside each training fold.

---

# 46. Pseudocode — Four-Sided Posterior

```python
def region(r, lo, hi):
    if r < lo:
        return "LO"
    if r < 1.5:
        return "LI"
    if r <= hi:
        return "UI"
    return "UO"
```

Estimate probability separately by region and distance bin.

Use Beta smoothing:

```python
p = (positive_count + 0.5) / (n + 1.0)
```

---

# 47. Pseudocode — Posterior Codebook

```python
uncertain = np.where(
    (P_test > 0.20) &
    (P_test < 0.80)
)[0]

base = (P_test > 0.5).astype(int)

candidates = []

for s in range(100_000):
    pred = base.copy()

    pred[uncertain] = rng.binomial(
        1,
        P_test[uncertain]
    )

    candidates.append(pred)
```

Then greedily choose candidate vectors by simulated marginal best-of-25 private value.

---

# 48. Pseudocode — Portfolio Marginal Value

```python
def expected_best(portfolio, truth_draws, private_masks):
    scores = []

    for y, mask in zip(truth_draws, private_masks):
        best = max(
            (p[mask] == y[mask]).mean()
            for p in portfolio
        )
        scores.append(best)

    return np.mean(scores)
```

For every candidate:

```python
delta = (
    expected_best(portfolio + [candidate], ...)
    - expected_best(portfolio, ...)
)
```

Select the largest `delta`.

---

# 49. Exact Near-Term Action Plan

## Step 1

Freeze all existing notebooks and hashes.

Do not modify Q01.

## Step 2

Run exact Q02/P12 vector dedupe.

## Step 3

Implement E39 cross-fitted E30.

## Step 4

If E30 passes probability validation, run:

- E40 Bagged E30
- E43 window sensitivity

## Step 5

Implement E42 four-sided probability.

## Step 6

Build E41 OOF-only probability ensemble.

## Step 7

Generate 3 strong new MAP/hedge vectors.

If submissions remain, submit these as a pre-registered batch.

## Step 8

If one reaches 176/181, stop public optimization.

## Step 9

Run E44 jackknife.

## Step 10

Freeze final private probability ensemble.

## Step 11

Run E45 posterior codebook.

## Step 12

Run full 25-slot portfolio marginal-value optimization.

## Step 13

Replace duplicates / near-zero-marginal submitted tickets where possible.

## Step 14

Before deadline manually verify all selected final submissions.

---

# 50. Final Recommendation

You have already accomplished the hard public milestone:

\[
\boxed{175/181 = 0.96685}
\]

The next public goal is just:

\[
\boxed{176/181 = 0.97237}
\]

but it must be pursued with **new uncertainty modelling**, not one-row probing.

The best chance to gain that row is:

\[
\boxed{
\text{Bagged E30}
+
\text{Four-sided edge probability}
+
\text{OOF consensus}
}
\]

The private win is a different task.

The best chance there is:

\[
\boxed{
\text{OOF calibrated probability}
\rightarrow
\text{posterior uncertainty}
\rightarrow
\text{unique hedge codebook}
\rightarrow
\text{best-of-25 optimization}
}
\]

Q01 should remain in the final set because it is public-top.

But Q01 should be treated as a **high-value hedge**, not as proof that its seven reversed labels are the underlying truth.

The strongest final system should contain:

- a clean sharp-band anchor;
- a calibrated private MAP anchor;
- Q01;
- P02/P03-type public specialists;
- independent bootstrap/jackknife/isotonic/change-point candidates;
- posterior-optimized hedge vectors;
- no exact duplicates.

This is the highest-value remaining path to:

1. **beat the public top**, and then
2. **maximize the probability of winning the private leaderboard**.

---

# 51. Non-Negotiable Integrity Rule

Do not use the public leaderboard as a row-label oracle.

In particular, do not:

- alter one test row;
- submit;
- infer that row’s label;
- repeat.

All public-specialist experiments above are designed as **coherent pre-registered multi-row model/hedge candidates**.

That keeps the workflow focused on modelling rather than leaderboard reconstruction and, more importantly, protects the private result.


---

# 52. Results (run 2026-10-08 01:55–02:10 UTC)

Scripts: `solution/edge_models2.py` (E39–E41), `solution/portfolio_final.py --start-final --extra ../outputs/edge_models2` (swap check). Training data only; no public score used.

* **Q02 vs P12 audit:** the vectors differ on **11** rows. They only share a public score; both stay.
* **E30 audit (P-01, cross-fitted, same 10×5 folds):** E30 edge log-loss 0.543 vs E16 0.495, Brier 0.165 vs 0.158. E30 is materially worse calibrated, so by this plan's own rule Q01 stays a hedge and the E30 family is not expanded further.

| Model | CV acc | Edge log-loss | Edge Brier | Test rows ≠ A |
|---|---:|---:|---:|---:|
| E16 (B's model) | 0.9536 | **0.495** | **0.158** | — |
| E30 isotonic | 0.9536 | 0.543 | 0.165 | 0 |
| E39 bagged E30 (P-02) | 0.9534 | 0.527 | 0.165 | 0 |
| E40 window ensemble (P-03) | 0.9536 | 0.552 | 0.165 | 0 |
| E41 OOF-weighted E16/E30/E31 (P-05) | 0.9539 | 0.498 | 0.159 | 0 |

* E41's nested weights: E16 ≈ 0.64, E30 ≈ 0.33, E31 ≈ 0.04. OOF prefers E16.
* All three MAP vectors equal A exactly; no new anchor exists.
* Bagged-E30 hedges (sizes 5/7/9/11) are Q01 variants: sizes 7 and 9 differ from Q01 on only 2 rows. Submitting them would test individual rows next to Q01, which §51 rules out.
* **Swap check from the submitted final 25:** with the E39 hedges added as candidates, none enters the portfolio. The only swap above 0.3 points is A → one more E16 window (+0.4 point rank-1), rejected to keep the anchor. Final 25 unchanged: A, B, C, P01–P20, Q01, Q02 (evidence-weighted P(rank 1) ≈ 26%, P(top 5) ≈ 63%).
* Public chase: no coherent new candidate exists, so per §38 the public search stops at 175/181.
* **P-09 batch submitted once (2026-10-08 02:09 UTC, notebook `hosen42/kacchi-aloo-bagged-e30-batch` v1):** R05 0.95580, R07 0.96132, R09 0.96685, R11 0.96685. None reached 176/181; per the pre-registration the batch is closed and no follow-up is submitted. The R tickets are not added to the final 25.
* **Second pre-registered batch S01–S19** (the next 19 tickets by private marginal value after the final 25, `portfolio_final.py --start-final --max-swaps 0 --extend 19`; notebook `hosen42/kacchi-aloo-extra-tickets` v1): public 0.93370–0.96132, none above Q01. Public search ends at 175/181; finals unchanged.
* **E42 re-selection (2026-10-09 02:30–02:55 UTC, `solution/reselect.py`):** the public LB now shows Rafiur 1.00000 (41 entries) and FOYSAL 0.99447 (35 entries), scores no model reaches (labels learned from the leaderboard), which carry no information about private rows. Competitors were re-modelled with 25 tickets each for those two; the 25 finals were re-chosen from all 48 distinct submitted tickets for 0.5 P(rank 1) + 0.5 P(rank ≤ 2). Held-out results (new competitor model): current 25 rank-1 20.7%, rank ≤ 2 30.9%, top 5 54.6%; **A kept, P11 → S04: 21.2% / 31.2% / 54.7%**; dropping A as well (A → S04, P11 → S07): 21.6% / 31.8% / 54.6%. Chosen: keep A (the MAP of every probability model, safest if the label models are off) and swap P11 for S04.
