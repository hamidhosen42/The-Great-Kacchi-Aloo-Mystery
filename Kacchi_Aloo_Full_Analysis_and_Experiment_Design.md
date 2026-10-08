# The Great Kacchi Aloo Mystery
## Full Competition Analysis, Submission Audit, and Next Experiment Design

**Prepared from:** all uploaded A/B/C notebooks, P01–P22 notebooks, the P01–P22 portfolio notebook, the theory notebook, README, deep-analysis report, experiments ledger, resource catalogue, and the latest leaderboard screenshot supplied in this chat.

**Latest leaderboard snapshot supplied by the user:** 2026-10-08 around 00:34 Bangladesh time  
**Your current best public score:** **0.96132**  
**Current public rank:** **#6**  
**Current top public score:** **0.96685**  
**Your entries shown:** **25**  
**Leaderboard split:** approximately **45% public / 55% private**

---

# 1. Executive Summary

The competition is already structurally close to solved.

The strongest verified predictive structure is

\[
r = \frac{\text{aloo_count}}{\text{guests}}
\]

and the target is approximately a **two-sided band** in this one variable. The strongest scientifically defensible single rule from the real training data is the sharp band

\[
0.97068 \le r \le 2.00983.
\]

The main remaining difficulty is **not model capacity**. It is uncertainty around the two decision edges. The real-data experiments already show that:

- smooth soft-band modelling was significantly worse than the sharp band;
- adding the other columns did not produce robust residual signal;
- event/city/drone/group-specific edges did not improve CV;
- size-dependent edge-width was not supported;
- the remaining mistakes are concentrated in a narrow “noisy ring” around the lower and upper cut-points.

Therefore, the next round of work should **not** be “try another large model.” The highest-value direction is:

> **better estimation of edge uncertainty, better calibration of the four directional edge regions, and diversified final tickets that are generated from new uncertainty models rather than from more arbitrary threshold nudges.**

Your current best is `0.96132 = 174/181`. The public leader is `0.96685 = 175/181`.

So:

- **tie the leader:** gain **1 net public row**;
- **beat the leader outright:** reach at least `176/181 = 0.97237`, i.e. gain **2 net public rows** over your current best.

That gap is tiny. It also means the public leaderboard is very noisy: one row changes the score by about

\[
1/181 \approx 0.005525.
\]

The goal should therefore be **Private-LB-safe improvement**, not blindly fitting the 181 public rows.

---

# 2. Competition Structure

## 2.1 Task

Synthetic tabular binary classification.

Target:

`went_back_for_seconds ∈ {0,1}`

Main useful engineered feature:

`apg = aloo_count / guests`

Train/test size from the supplied analysis:

- train: 800 rows originally;
- usable rows with non-missing `aloo_count`: 776;
- test: 400 rows.

## 2.2 Metric

Accuracy:

\[
\text{Accuracy} = \frac{1}{n} \sum_i I(\hat y_i = y_i).
\]

For this metric:

- ranking quality is irrelevant by itself;
- calibration matters mainly near 0.5;
- the final binary decision is what matters;
- moving a threshold enough to change one row changes public score by about 0.5525 percentage points.

## 2.3 Public/private split

The latest Kaggle UI explicitly states approximately:

- **45% public**
- **55% private**

The observed score granularity is consistent with roughly:

- public: 181 rows
- private: 219 rows

Thus the top of the public leaderboard can move substantially because of only one or two weddings.

---

# 3. Data Forensics and Cleaning

The uploaded deep analysis and notebooks consistently use the following cleaning logic.

## 3.1 Missing `aloo_count`

24 training rows have no usable aloo count.

The competition notebooks drop these rows for ratio-based fitting because `apg` cannot be recovered reliably.

## 3.2 ×10 aloo-count typo

A small number of training rows have an extra zero in `aloo_count`.

The notebooks use:

```python
if aloo_count / guests > 3:
    aloo_count /= 10
```

This produces a stable `apg` feature.

## 3.3 Bangla digits

Some numeric-looking cells contain Bengali Unicode digits. The supplied notebooks normalize all Unicode decimal digits to ASCII-compatible numeric values before conversion.

## 3.4 Distribution shift

The key train/test shift is wedding size.

The supplied analysis found many test weddings larger than the maximum training wedding. This explains why raw-count RF/GBM models can look reasonable in ordinary CV yet extrapolate poorly.

The ratio `aloo_count / guests` is approximately scale-free and is therefore the correct representation.

---

# 4. What the Real Data Already Tells Us

The supplied real-data update reports approximately:

- sharp accuracy-optimal band CV: **0.9546**
- midpoint band CV: **0.9543**
- LightGBM on log-APG: **0.9537**
- fixed `(1,2)` band: **0.9446**
- smooth soft band: **0.9421**
- bagged soft band: **0.9386**

The smooth approach was reported significantly worse than the sharp band.

The edge pattern is also sharply localized. Reported OOF error by distance to nearest edge:

| Distance to nearest edge | Approx. OOF error rate |
|---|---:|
| 0–0.01 | 0.127 |
| 0.01–0.035 | 0.374 |
| 0.035–0.10 | 0.148 |
| 0.10–0.40 | 0.022 |
| >0.40 | ~0 |

This is the central fact for the next experiments:

> **Almost all useful remaining information is concentrated close to the two band boundaries.**

---

# 5. Current Submission Portfolio Audit

`r` below means `aloo_count / guests`.

| ID | Family | Rule | Public | Correct | Rows vs A | Predicted 1s |
|---|---|---|---:|---:|---:|---:|
| A | Sharp band | `0.97068 ≤ r ≤ 2.00983` | 0.95027 | 172/181 | 0 | 184 |
| B | Uncertainty hedge | `A + 9 most-uncertain rows flipped` | 0.95580 | 173/181 | 9 | — |
| C | Shifted band | `1.000 < r < 1.975` | 0.95580 | 173/181 | — | — |
| P01 | Upper-edge flip | `A + uncertainty ranks 3–10, upper` | 0.94475 | 171/181 | 8 | 186 |
| P02 | Band | `1.00608 ≤ r ≤ 2.02353` | 0.96132 | 174/181 | 10 | 182 |
| P03 | Upper-edge flip | `A + uncertainty ranks 8–14, upper` | 0.96132 | 174/181 | 7 | 181 |
| P04 | Band | `1.00689 ≤ r ≤ 1.98867` | 0.95580 | 173/181 | 8 | 174 |
| P05 | Lower-edge flip | `A + uncertainty ranks 6–10, lower` | 0.95027 | 172/181 | 5 | 183 |
| P06 | Both-edge flip | `A + uncertainty ranks 9–14, both` | 0.93370 | 169/181 | 6 | 184 |
| P07 | Band | `0.95816 ≤ r ≤ 1.98070` | 0.93922 | 170/181 | 9 | 181 |
| P08 | Both-edge flip | `A + uncertainty ranks 20–26, both` | 0.96132 | 174/181 | 7 | 179 |
| P09 | Band | `0.97052 ≤ r ≤ 2.02641` | 0.95580 | 173/181 | 7 | 189 |
| P10 | Lower-edge flip | `A + uncertainty ranks 3–6, lower` | 0.95027 | 172/181 | 4 | 182 |
| P11 | Both-edge flip | `A + uncertainty ranks 17–20, both` | 0.95580 | 173/181 | 4 | 180 |
| P12 | Upper-edge flip | `A + uncertainty ranks 1–5, upper` | 0.93922 | 170/181 | 5 | 183 |
| P13 | Upper-edge flip | `A + uncertainty ranks 14–18, upper` | 0.94475 | 171/181 | 5 | 179 |
| P14 | Both-edge flip | `A + uncertainty ranks 13–16, both` | 0.95580 | 173/181 | 4 | 186 |
| P15 | Both-edge flip | `A + uncertainty ranks 2–6, both` | 0.95580 | 173/181 | 5 | 183 |
| P16 | Band | `0.98430 ≤ r ≤ 1.97931` | 0.95027 | 172/181 | 7 | 175 |
| P17 | Band | `0.99333 ≤ r ≤ 2.01654` | 0.95027 | 172/181 | 5 | 183 |
| P18 | Both-edge flip | `A + uncertainty ranks 24–28, both` | 0.96132 | 174/181 | 5 | 179 |
| P19 | Both-edge flip | `A + uncertainty ranks 16–22, both` | 0.96132 | 174/181 | 7 | 179 |
| P20 | Band | `0.96938 ≤ r ≤ 1.95509` | 0.95027 | 172/181 | 11 | 175 |
| P21 | Band | `1.00347 ≤ r ≤ 1.98148` | Pending | — | 8 | 174 |
| P22 | Band | `0.96462 ≤ r ≤ 2.01953` | Pending | — | 7 | 189 |

## 5.1 Score tiers

### Tier 1 — Current best: 0.96132 = 174/181

- P02
- P03
- P08
- P18
- P19

### Tier 2 — 0.95580 = 173/181

- B
- C
- P04
- P09
- P11
- P14
- P15

### Tier 3 — 0.95027 = 172/181

- A
- P05
- P10
- P16
- P17
- P20

### Lower tiers

- P01, P13: 171/181
- P07, P12: 170/181
- P06: 169/181

P21 and P22 were still pending in the uploaded README.

---

# 6. What We Learn from A, B and C

## 6.1 A — Sharp-band anchor

A uses:

\[
0.97068 \le r \le 2.00983.
\]

This is the most defensible single-model anchor because it comes directly from the training data’s sharp edge structure.

Public score:

`0.95027 = 172/181`

A should **not** be discarded simply because some hedges score higher publicly. It is the most useful reference vector for measuring every later ticket.

## 6.2 B — Model-aware uncertainty hedge

B starts from A and flips the 9 most uncertain test rows.

Uncertainty is estimated from repeated OOF error rates as a function of distance to the nearest edge.

This is much stronger than random flipping because the row ordering is produced algorithmically.

Public score:

`0.95580 = 173/181`

The important weakness is that B’s uncertainty estimator primarily compresses the problem into **distance from the nearest edge**. It does not fully distinguish four directional situations:

1. lower-edge outside,
2. lower-edge inside,
3. upper-edge inside,
4. upper-edge outside.

That observation motivates one of the strongest new experiments below.

## 6.3 C — Aggressive shifted-band hedge

C uses:

\[
1.000 < r < 1.975.
\]

This deliberately narrows the positive region.

Public:

`0.95580 = 173/181`

C is useful as diversity, but its lower training fit means it should be interpreted as a **hedge**, not proof that these are the true generator boundaries.

---

# 7. What the P01–P22 Portfolio Tells Us

The portfolio contains only two fundamental families:

1. **band variants**: shift `(lo, hi)`;
2. **window flips**: start from A and flip a contiguous block of rows from the same uncertainty ranking.

This is important.

The portfolio is broad in *tickets*, but relatively narrow in *model assumptions*.

## 7.1 Best band ticket: P02

P02:

\[
1.00608 \le r \le 2.02353
\]

Public:

`0.96132 = 174/181`

It differs from A on 10 test rows.

This is currently the best public band, but the public result does **not** establish that these are better true cut-points. A 181-row public set is too small to distinguish a one-row advantage reliably.

## 7.2 Best flip tickets

Also at `0.96132`:

- P03: upper uncertainty ranks 8–14
- P08: both-edge ranks 20–26
- P18: both-edge ranks 24–28
- P19: both-edge ranks 16–22

This is revealing.

Several very different uncertainty windows all produce the exact same public number. That strongly suggests:

- many tested changes hit a small collection of near-edge rows;
- public score plateaus are common;
- there is no evidence that a single particular uncertainty block is uniquely correct.

## 7.3 Weak tickets are informative

P06 = `0.93370`

P07/P12 = `0.93922`

P01/P13 = `0.94475`

These failures tell us which regions of the existing uncertainty ranking are dangerous.

However, they do **not** justify reverse-engineering public labels row-by-row. The correct lesson is that the current uncertainty ordering is imperfect and needs a better model.

---

# 8. Why More Random Threshold Search Is Low Value

Suppose we now try:

- `(1.004, 2.022)`
- `(1.005, 2.024)`
- `(1.007, 2.021)`
- and dozens more.

If we select them based on Public LB, we are adaptively fitting 181 hidden labels.

That creates two problems:

1. **Public overfitting**
2. **No reason the selected threshold transfers to the 55% private split**

The uploaded resource review already identifies optimal cut-point search and repeated holdout adaptation as major sources of selection bias.

Therefore:

> P02 should be treated as a hypothesis to validate, not a center around which to brute-force public thresholds.

---

# 9. Already-Tested Ideas That Should Not Be Repeated

Based on the supplied real-data analysis:

## 9.1 Smooth soft band

Rejected.

It scored materially worse than the sharp rule.

## 9.2 Raw-feature boosting

Low priority.

Raw-count models suffer from the test wedding-size shift.

## 9.3 Adding all other columns near the edges

No robust residual signal was found.

## 9.4 Event/city/drone/group-specific thresholds

Tested and worse than the global band.

## 9.5 Size-dependent edge width

No significant evidence.

## 9.6 Alternative ratios/denominators

`aloo_count / guests` remained the strongest ratio.

## 9.7 Pseudo-labeling

Low expected value because confident test rows are far from the boundary and therefore provide almost no new information about the cut-points.

---

# 10. Correct Objective for the Next Stage

The next research question is not:

> “Which model has more capacity?”

It is:

> **“Can we estimate the probability of each near-edge test row more accurately than the current distance-bucket method?”**

That requires independent uncertainty models.

The current portfolio has substantial ticket diversity but limited **epistemic diversity**.

The next experiments should generate new probability estimates using different statistical principles.

---

# 11. Frozen Validation Protocol

Do not change the validation split from experiment to experiment.

## 11.1 Primary CV

Use the existing repeated CV framework:

- 5 folds
- 10 repeats
- fixed seeds
- stratification by target and, where implemented, APG zone

Report:

- mean OOF accuracy
- SD
- per-repeat mean
- exact OOF prediction vector

## 11.2 Size-stress holdout

Because test weddings are larger, retain a holdout where training uses smaller weddings and validation uses the largest training weddings.

A ratio model should remain stable.

## 11.3 Edge metrics

Overall accuracy alone is insufficient.

For every candidate report:

- accuracy for `distance_to_edge < 0.01`
- accuracy for `< 0.035`
- accuracy for `< 0.10`
- edge log-loss
- edge Brier score
- calibration around 0.5

## 11.4 Paired comparison

Compare each candidate against A using the **same OOF rows**.

Useful diagnostics:

- number of discordant OOF predictions
- McNemar exact test
- corrected repeated-CV comparison
- mean gain in number of correct rows rather than only decimal accuracy

## 11.5 Test-vector diversity

Before submitting, compute:

- Hamming distance vs A
- Hamming distance vs P02
- Hamming distance vs B
- Hamming distance vs each other new candidate

A new submission that is identical to an existing one adds no portfolio value.

---

# 12. NEW EXPERIMENT E29 — Bootstrap Hard-Band Posterior

## Hypothesis

A single full-data sharp cut can be overly controlled by a few noisy edge examples. Bootstrap aggregation can quantify cut uncertainty while preserving the empirically successful hard-edge assumption.

## Procedure

For `B = 5000` bootstrap samples:

1. sample the 776 usable train rows with replacement;
2. fit the same sharp `midcut_cuts`;
3. store `(lo_b, hi_b)`.

For test row `j`:

\[
p_j =
\frac{1}{B}
\sum_b I(lo_b \le r_j \le hi_b).
\]

## Candidate E29-A

\[
\hat y_j = I(p_j > 0.5).
\]

## Candidate E29-B

Use the same posterior, but create a complementary hedge by flipping only rows with posterior closest to 0.5.

The hedge size must be selected **inside simulation/CV**, not from Public LB.

## Diagnostics

Report:

- bootstrap distribution of lower cut;
- bootstrap distribution of upper cut;
- 2.5/25/50/75/97.5 percentiles;
- test rows with `0.25 < p < 0.75`;
- Hamming distance vs A/P02.

## Why this is genuinely new

This is **not** the previously rejected smooth logistic band.

Each bootstrap member is still a hard classifier.

---

# 13. NEW EXPERIMENT E30 — Cross-Fitted Edge-Specific Isotonic Model

## Hypothesis

The probability curve is sharp but does not need to be symmetric, logistic, or identical on both boundaries.

Fit the two edges separately.

### Lower edge

Use a local range, e.g.

\[
0.85 < r < 1.15.
\]

Fit an increasing isotonic model:

\[
P(y=1|r).
\]

### Upper edge

Use, for example,

\[
1.80 < r < 2.20.
\]

Fit a decreasing isotonic model.

### Away from edges

- central plateau: probability near 1
- far outside: probability near 0

## Critical requirement

Probabilities used to evaluate training rows must be **cross-fitted OOF**.

Never train and score an isotonic model on the same edge rows.

## Why it can help

It allows:

- different lower/upper edge shapes;
- asymmetric inside/outside behavior;
- no pre-imposed logistic functional form.

---

# 14. NEW EXPERIMENT E31 — Bayesian Change-Point Posterior

## Hypothesis

There may not be one precisely known cut. The correct Bayesian action is to integrate over plausible cut locations.

For each candidate lower cut `c_L`, estimate likelihood.

Likewise for upper cut `c_U`.

Posterior:

\[
P(c_L,c_U|D)
\propto
P(D|c_L,c_U)P(c_L,c_U).
\]

Then for a test row:

\[
P(y_j=1|D)
=
\sum_{c_L,c_U}
P(y_j=1|c_L,c_U)
P(c_L,c_U|D).
\]

## Practical version

To keep computation small:

- treat lower and upper edge windows separately;
- use Beta-Binomial probabilities on the two sides;
- combine the resulting edge posteriors.

## Output

Produce:

- posterior median cuts;
- credible intervals;
- posterior `P(y=1)` for each test row;
- majority candidate;
- uncertainty hedge candidate.

---

# 15. NEW EXPERIMENT E32 — Jackknife Hard Band

## Hypothesis

If one or two training examples determine the cut, deleting them should move the boundary strongly.

For each usable train row `i`:

1. remove row `i`;
2. fit sharp cuts;
3. predict the 400 test rows.

This gives 776 hard models.

For test row `j`:

\[
p_j =
\frac{1}{776}
\sum_i \hat y_j^{(-i)}.
\]

## Benefits

- detects influential training points;
- gives a model-independent instability score;
- preserves sharp edges;
- can produce a new ranking of uncertain test rows.

## Important output

List which training observations move `lo` or `hi` the most.

This can explain whether A’s boundaries are data-stable or hinge on individual edge observations.

---

# 16. NEW EXPERIMENT E33 — Nested-CV One-Standard-Error Band

## Hypothesis

The cut with the maximum training/CV accuracy can overfit boundary noise.

Instead:

1. inner CV searches `(lo, hi)`;
2. calculate best mean score and standard error;
3. among all cuts within one SE of the best, choose the simplest/most regular cut — for example the one closest to `(1,2)`;
4. evaluate only in the outer fold.

This produces an **honest model-selection score**.

## Key question

Does the nested selected cut repeatedly move toward P02 (`~1.006, 2.024`)?

- **Yes:** P02 may represent a real structural shift.
- **No:** P02’s public advantage is probably sampling luck.

---

# 17. NEW EXPERIMENT E34 — Four-Sided Edge Posterior

This is one of the highest-priority experiments.

## Limitation of B

B mainly uses:

\[
d = \min(|r-lo|, |r-hi|)
\]

and estimates error from distance buckets.

But these four cases need not behave identically:

- `LO`: lower edge, outside
- `LI`: lower edge, inside
- `UI`: upper edge, inside
- `UO`: upper edge, outside

## New estimator

For each region and distance bucket estimate a smoothed error rate.

For `k` observed errors among `n` OOF examples use Jeffreys/Beta smoothing:

\[
\hat e = \frac{k+0.5}{n+1}.
\]

Then obtain separate functions:

\[
e_{LO}(d),\quad
e_{LI}(d),\quad
e_{UI}(d),\quad
e_{UO}(d).
\]

Convert them into row-level probabilities.

## Success criterion

E34 should improve at least one of:

- OOF edge log-loss;
- OOF edge Brier score;
- edge accuracy;

without degrading overall CV materially.

If it does, its uncertainty ranking should replace B’s ranking for future hedges.

---

# 18. NEW EXPERIMENT E35 — Consensus of Independent Edge Models

Do not average arbitrary models.

Average only the new, independently justified edge-probability estimates:

- E29 bootstrap
- E30 isotonic
- E31 change-point
- E32 jackknife
- E34 four-sided posterior

For row `j`:

\[
p_j^{cons}
=
\frac{1}{M}
\sum_m p_{jm}.
\]

Candidate:

\[
\hat y_j = I(p_j^{cons} > 0.5).
\]

Also measure disagreement across models.

High disagreement is a stronger measure of epistemic uncertainty than distance to A’s cut alone.

---

# 19. NEW EXPERIMENT E36 — P02 Structural Test

Do **not** manually tune around P02 based on its public score.

Instead test the hypothesis honestly.

## Search space

For example:

\[
lo \in [0.98, 1.02]
\]

\[
hi \in [1.98, 2.04].
\]

## Method

Nested CV.

Record the selected `(lo, hi)` in every outer fold.

Then answer:

- How frequently is `lo > 1.00`?
- How frequently is `hi > 2.01`?
- What are median selected cuts?
- Does the outer-CV score improve over A?

If the selected cuts repeatedly cluster near P02, P02 deserves more trust.

If not, keep P02 as a public-successful hedge, not as the new canonical model.

---

# 20. New Candidate Acceptance Rules

## Core-model candidate

Keep only if:

- overall repeated-CV accuracy is roughly `≥ 0.953`;
- no meaningful size-stress degradation;
- edge log-loss/Brier is at least competitive;
- test vector is not a duplicate;
- no evidence that the improvement exists only in one lucky CV seed.

## Hedge candidate

A hedge may have slightly worse single-model CV if:

- every changed row is genuinely uncertain;
- it is algorithmically generated;
- its test vector adds useful diversity;
- its expected best-of-portfolio private score improves in simulation.

## Reject

Reject if:

- gain exists only on Public LB;
- CV materially worsens without a clear hedge rationale;
- changed rows are far from the boundary;
- the candidate duplicates an existing vector;
- it uses extra columns whose residual signal has already failed;
- it is selected after row-by-row leaderboard inference.

---

# 21. Next Five Submissions I Would Run

If you have a fresh five-submission budget, use genuinely new model families:

| Order | Submission | Purpose |
|---:|---|---|
| 1 | **E29 bootstrap hard-band majority** | new robust cut estimator |
| 2 | **E30 cross-fitted isotonic edge model** | non-parametric edge probability |
| 3 | **E31 Bayesian change-point majority** | integrated cut uncertainty |
| 4 | **E34 four-sided edge posterior** | direct upgrade of B |
| 5 | **E35 consensus** | aggregate independent uncertainty models |

Do **not** spend the first five slots on more arbitrary cut pairs.

---

# 22. Which Existing Tickets Should Stay

The strongest portfolio anchors currently are:

- A — scientifically defensible sharp anchor
- B — original uncertainty hedge
- C — materially shifted hedge
- P02 — best public band
- P03 — best public upper-edge window
- P08 — best public both-edge window
- P18 — best public both-edge window
- P19 — best public both-edge window

These should generally stay in a 25-slot private portfolio unless a new candidate is an exact duplicate.

---

# 23. Which Existing Tickets Are First to Replace

If validated E29–E35 candidates need slots, replace the weakest/redundant tickets first.

Suggested order:

1. P06 — 0.93370
2. P07 — 0.93922
3. P12 — 0.93922
4. P01 — 0.94475
5. P13 — 0.94475

Then consider replacing redundant 0.95027 variants if the new models add genuinely different prediction vectors.

Do **not** remove A simply because its public score is lower. It remains the clean reference/anchor.

---

# 24. Public Score Interpretation

Current top cluster:

| Public | Correct |
|---:|---:|
| 0.96685 | 175/181 |
| 0.96132 | 174/181 |
| 0.95580 | 173/181 |
| 0.95027 | 172/181 |

This means your displayed rank #6 is misleading if interpreted as a large modelling gap.

You are only **one public wedding behind the leader**.

At this sample size, a one-row gap is not strong evidence that the top model has a higher true private accuracy.

The final result can reorder substantially.

---

# 25. What “Beat the Top” Actually Means

## Public goal

Current best:

\[
174/181 = 0.96132
\]

Current leader:

\[
175/181 = 0.96685
\]

To tie:

\[
175/181.
\]

To beat outright:

\[
176/181 = 0.97237.
\]

So the practical research objective is:

> Find a candidate that changes only a small number of genuinely uncertain decisions, with enough independent statistical support that **two additional public decisions could plausibly improve without sacrificing private robustness**.

This cannot be guaranteed.

Any claim that a new threshold “must score 0.97237+” would be unsupported.

---

# 26. Private-LB Strategy

Public optimisation and final optimisation are different.

For the private leaderboard:

1. keep a strong canonical anchor;
2. include multiple statistically justified uncertainty models;
3. include a small number of deliberately different hedges;
4. avoid filling 25 slots with variations from one identical ranking;
5. never use the public score to infer individual hidden labels;
6. preserve early valid submissions when ties matter;
7. manually check final selections before the deadline.

A 25-ticket portfolio is valuable only when it spans **different plausible error models**, not when it is 25 near-copies.

---

# 27. Experiment Log Template

Use this table for every new run.

| Field | Value |
|---|---|
| Experiment ID | E29 / E30 / ... |
| Parent | A / B / other |
| Hypothesis | |
| Exact method | |
| Random seeds | |
| Folds | |
| Overall CV | |
| CV SD | |
| Δ vs A | |
| McNemar discordant counts | |
| Edge accuracy <0.01 | |
| Edge accuracy <0.035 | |
| Edge accuracy <0.10 | |
| Edge log-loss | |
| Edge Brier | |
| Size-stress accuracy | |
| Test Hamming vs A | |
| Test Hamming vs P02 | |
| Test Hamming vs B | |
| Number predicted positive | |
| Public score | only after candidate is frozen |
| Decision | KEEP / HEDGE / REJECT |

---

# 28. Stop Conditions

Stop searching a family when:

- 3–5 independent variations fail the same validation test;
- the new vector is almost identical to an existing ticket;
- improvement is below one OOF-row equivalent and unstable across seeds;
- it requires additional noise features already shown uninformative;
- only Public LB supports it.

The competition is small enough that endless experimentation can make model selection worse.

---

# 29. Theory Notebook Recommendation

The theory notebook is already strong because it explains:

- why the ratio matters;
- why raw models fail under size shift;
- why the signal is a Goldilocks band;
- why the edges are noisy;
- why more features do not help.

One wording should remain conservative:

Instead of saying “95% is the absolute ceiling,” prefer:

> “Under the observed out-of-fold edge-error pattern, achievable accuracy appears to be roughly in the mid-90% range; a few rows above that can occur because the public/private samples are small.”

This separates observed evidence from an unprovable exact Bayes ceiling.

---

# 30. Final Recommendation

## What to do now

### Do

1. Preserve **A, B, C, P02, P03, P08, P18, P19**.
2. Run **E29 Bootstrap Hard Band**.
3. Run **E34 Four-Sided Edge Posterior**.
4. Run **E30 Cross-Fitted Isotonic**.
5. Run **E31 Bayesian Change-Point**.
6. Run **E35 Consensus**.
7. Use E36 nested CV to test whether P02 is structurally justified.
8. Replace the weakest old tickets with validated new model-family tickets.

### Do not

- brute-force dozens of cuts around P02 using public feedback;
- return to raw-count RF/XGBoost/CatBoost as the main path;
- use city/event/drone-specific edges again;
- infer hidden row labels from leaderboard changes;
- assume `0.96132` means you are materially behind `0.96685`.

---

# 31. Bottom Line

Your current best is already one public row from first place.

The existing P01–P22 portfolio has explored many outputs, but most are generated from the **same sharp band and the same uncertainty ordering**.

Therefore the next real opportunity is not a larger portfolio of the same idea.

It is:

\[
\boxed{
\text{new edge-probability estimators}
\rightarrow
\text{new uncertainty rankings}
\rightarrow
\text{validated diverse tickets}
}
\]

The highest-priority experiments are:

\[
\boxed{
E29 \rightarrow E34 \rightarrow E30 \rightarrow E31 \rightarrow E35
}
\]

If there are legitimate extra 1–2 rows of predictive headroom left, these experiments are substantially more likely to find them than another round of arbitrary threshold search.

At the same time, because the public leaderboard is only 181 rows, **no experiment can honestly guarantee beating 0.96685**. The correct target is to maximize the probability of improving both public and, more importantly, private performance while controlling adaptive overfitting.

---

# Source Basis

This analysis was grounded in the uploaded project materials, especially:

- `01_A_sharp_band_0.95027.ipynb`
- `02_B_hedge_0.95580.ipynb`
- `03_A_theory_notebook_0.95027.ipynb`
- `04_C_band_1.000-1.975_0.95580.ipynb`
- `05_P01...` through `26_P22...`
- `P01-P22_portfolio_kaggle_notebook.ipynb`
- `README(1).md`
- `The_Great_Kacchi_Aloo_Mystery_Deep_Analysis(1).md`
- `experiments(1).csv`
- `TOP_100_RESOURCES(1).md`
- the latest leaderboard screenshot supplied in the conversation.

Where the uploaded materials contained an unsupported simulation claim without its referenced script (for example external `solution/...` scripts not present in the uploaded set), this document does not treat the exact claimed probability as independently verified.

---

# 32. Results (run 2026-10-07 19:00–19:45 UTC)

Scripts: `solution/edge_models.py` (E29–E36), `solution/portfolio_robust.py` (E37), `solution/portfolio_final.py` (E38). Outputs: `outputs/edge_models/`, `outputs/portfolio_robust/`, `outputs/portfolio_final/`. Training data only; no public score was used to build any vector.

## 32.1 Edge models under the frozen protocol (10×5 CV, seed 2026)

| Model | CV acc | Δ vs A (corrected p) | Edge log-loss (d < 0.1) | Test rows ≠ A |
|---|---:|---:|---:|---:|
| A midcut band | 0.9543 | — | hard 0/1 | 0 |
| E16 B's distance model | 0.9536 | −0.0006 (0.79) | **0.495** | 0 |
| E29 bootstrap hard band | 0.9539 | −0.0004 (0.71) | 0.825 | 0 |
| E30 isotonic edges | 0.9536 | −0.0006 (0.60) | 0.543 | 0 |
| E31 Bayes change-point | 0.9532 | −0.0010 (0.47) | 0.528 | 0 |
| E32 jackknife hard band | 0.9543 | 0.0000 (1.00) | hard | 0 |
| E34 four-sided posterior | 0.9492 | −0.0050 (0.15) | 0.528 | 1 |
| E35 consensus | 0.9543 | 0.0000 (1.00) | 0.531 | 0 |
| E33 nested 1-SE band | 0.9482 | −0.0061 (0.18) | hard | 0 |
| E36 P02-range band | 0.9459 | **−0.0084 (0.04)** | hard | 4 |

* Every independent model makes **the same 400 test decisions as A** (E34 differs on 1 row, E36 on 4). Submitting E29/E30/E31/E35 would only duplicate A.
* E34 does not beat E16 on edge log-loss and loses CV accuracy, so B's ranking stays the best-calibrated one.
* Cut uncertainty is small: bootstrap lower cut 95% range 0.959–1.004 (median 0.9707), upper 2.006–2.053 (median 2.0098); E31 posterior lower 0.968–0.975, upper 2.006–2.013.
* **P02 structural test (E36):** with the grid forced to lo ∈ [0.98, 1.02], hi ∈ [1.98, 2.04], nested CV picks hi > 2.01 in only 20% of outer folds (lo > 1.00 in 48%), median cuts (0.9996, 2.0098), and the band is significantly worse than A. P02's upper cut (2.0235) is not supported by the training data; its 174/181 is best read as sampling luck. E33 (1-SE) median cuts (0.9767, 2.0082).

## 32.2 Robustness of the 25 finals to the label model (E37)

Held-out scenarios conditioned on A = 172, B = 173, C = 173 and 83 public ones. P(private rank 1) / P(top 5):

| Portfolio | E16 OOF | parametric band | E30 isotonic | E31 change-point |
|---|---:|---:|---:|---:|
| A + B + C | 4.7% / 15.9% | 8.5% / 22.9% | 1.7% / 10.7% | 1.0% / 21.9% |
| current 25 (A, B, C, P01–P22) | 25.6% / 61.3% | 33.9% / 71.1% | 14.0% / 48.7% | 6.3% / 40.5% |
| re-chosen 25 (equal weights) | 21.8% / 59.6% | 30.4% / 71.2% | 18.0% / 59.9% | 13.7% / 61.0% |

How often each generator reproduces our known public scores (evidence): E16 1.6e-4, parametric 1.1e-4, isotonic 0.7e-4, change-point 0.08e-4. The change-point model explains the public facts ~20× worse than E16, so it gets ~2% weight. The re-chosen set kept P06 and P12, the two lowest public scorers: public score is not a good guide to a hedge's private value.

## 32.3 Decision (E38, evidence-weighted: E16 0.46, parametric 0.32, isotonic 0.22)

| Final 25 | P(rank 1) | P(top 3) | P(top 5) |
|---|---:|---:|---:|
| A, B, C, P01–P22 | 25.8% | 47.2% | 62.2% |
| **A, B, C, P01–P20, Q01, Q02** | **26.1%** | **48.0%** | **63.4%** |
| same, A swapped for one more window | 26.5% | 48.7% | 63.5% |

* P21/P22 (not yet submitted) are replaced by two windows on the E30 isotonic ranking: **Q01** = A + flip ranks 1–7 (W0968, W0972, W1022, W1119, W1157, W1163, W1179) and **Q02** = A + flip ranks 8–13 (W0832, W0846, W0886, W1153, W1191, W1199). Notebook: `hosen42/kacchi-aloo-final-tickets` v2, outputs verified identical.
* A is kept: dropping it gains < 0.5 points, less than the model uncertainty, and A is the anchor with the earliest timestamp.
* No other swap improves the objective by more than 0.3 points.

**Final selection (25):** `sub_A_primary_sharpband.csv` (10:28), `sub_B_hedge.csv` (10:29), `sub_C_band_1p00_1p975.csv` (16:35), `P01.csv`–`P20.csv`, `Q01.csv`, `Q02.csv`. Not the duplicate `submission.csv` entries (10:53 = A, 17:45 = C). All probabilities above are model-based; none is a guarantee.
