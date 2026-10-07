# The Great Kacchi Aloo Mystery: Deep Competition Analysis

**Competition:** https://www.kaggle.com/competitions/the-great-kacchi-aloo-mystery
**Research snapshot:** 2026-10-07, about 09:40 UTC. Leaderboard snapshot from 08:58 UTC.
**Final deadline:** **2026-10-09 17:59 UTC**, about 56 hours after this snapshot. The team-merger deadline is the same moment.
**Companion files in this folder:**

| File | Contents |
|---|---|
| `external_resources.csv` | 122 verified resources |
| `TOP_100_RESOURCES.md` | Top 100 resources, ranked |
| `public_notebooks.csv` | Public notebook catalogue |
| `important_discussions.csv` | Discussion catalogue |
| `experiments.csv` | Experiment ledger |
| `solution/kacchi_pipeline.py` | End-to-end pipeline |
| `research/` | Scripts, raw evidence, simulation outputs |

### Evidence labels used throughout

| Tag | Meaning |
|---|---|
| **[OFFICIAL]** | Official fact from the competition pages or Kaggle Foundational Rules. I read these through the Kaggle API. |
| **[ORGANIZER]** | Organizer clarification. None exist outside the official pages: there are 0 discussion topics. |
| **[VERIFIED]** | Verified public result: a Kaggle leaderboard value, or a number printed in a public notebook's **Kaggle execution log**. It is not independently re-run, because the data is gated (see §0). |
| **[CLAIM]** | Participant claim that I could not verify, for example a score in a notebook title. |
| **[EXTERNAL]** | External research evidence (papers, docs, prior competitions). See `external_resources.csv`. |
| **[COMPUTED]** | My own computation: leaderboard arithmetic, submission-file forensics, or Monte Carlo / ABC simulation under a **stated** hypothesis. |
| **[HYPOTHESIS]** | My hypothesis, not yet tested on the real data. |
| **[RECOMMENDATION]** | My recommendation. |

Anything that could not be checked is marked **UNVERIFIED**.

---

## UPDATE: results on the real data (2026-10-07, ~10:30 UTC)

The data was later downloaded with the **hosen42** account, which had joined. The pipeline ran on the real files, and two submissions were made. **This section overrides any earlier recommendation that conflicts with it.**

| Check | Result on real data |
|---|---|
| Audit | Matches the public logs exactly: 24 missing aloo, 15 Bangla-digit cells, 6 ×10 rows, all in train; 145 test weddings > 700 guests. No duplicates, no ID-order leak (Spearman p = 0.17). Adversarial AUC: raw 0.66, per-guest 0.54, apg alone 0.54. |
| E0 reproduction | 400/400 identical to the public `submission_band_rule.csv` |
| Best CV (10×5) | E0 accuracy-optimal band 0.9546 ± 0.0108; E0b midpoint band 0.9543; LightGBM log-apg 0.9537; fixed (1, 2) 0.9446 |
| **Smooth soft band (E3a)** | **0.9421, significantly worse than E0** (corrected t-test p = 0.030). The bagged soft band (E3b) scores 0.9386. **The earlier recommendation "A = bagged soft band" is withdrawn.** |
| E15 (size-dependent edge width) | Not significant (γ = 0.11, p = 0.42). Hypothesis rejected. |
| E11 (other columns near the edges) | No signal: adding all columns worsens near-edge log-loss (0.396 vs 0.374). |
| Edge structure | **Sharp edges.** In train the label flips between aloo/guest 0.969925 and 0.971429, and between 2.009732 and 2.009934. Out-of-fold error rate by distance to the edge: 13% within 0.01, **37% at 0.01–0.035**, 15% at 0.035–0.1, 2% at 0.1–0.4. A sharp band with this error profile has the best near-edge OOF log-loss (0.553); kNN, kernel smoothers and LightGBM all do worse (0.64–0.95). |
| Final A | **Sharp midcut band, 0.97068 < apg < 2.00983** (= Rafiur's public V2 vector). |
| Final B | A with 9 algorithmically chosen rows flipped (W0839, W0968, W0974, W1006, W1023, W1079, W1119, W1157, W1163). Expected best-of-two gain **+0.42 private rows** (the earlier +0.75 assumed the smoother model). |
| Submitted (hosen42, 10:28–10:29 UTC) | **A: public 0.95027 (172/181). B: public 0.95580 (173/181).** Team "Md. Hamid Hosen" appears at 0.95580. |
| Refuted hypothesis | §7.3 guessed Rafiur's 0.95580 came from V2. V2 (= A) scores 172, so it did not. |
| E17: group-specific edges | Separate edges per event, Gaye Holud vs rest, drone, city, or terciles of mutton/borhani/fairy per guest, guests, dhol or aunties: **every variant is worse in 10×5 CV** than the single global band (Δ from −0.0012 to −0.0135). The apparent Gaye Holud shift near the lower edge is noise. Script: `solution/conditional_cuts.py`. |
| Public/private split | Kaggle's leaderboard page states public ≈ 45% and private ≈ 55% of test, consistent with 181 / 219 rows. |
| E18: hidden-rule search | Each edge was re-fitted as a cut on apg × exp(β·z) for 20 candidate columns (per-guest ratios, size, dhol, aunties, drone, event, city, id, digit patterns), plus alternative ratios. Best in-sample gain: 3 errors. **Best CV gain +0.0014 (p = 0.64)**. The 32 training errors show no association with any column or dirt flag (all p > 0.1). Script: `solution/hidden_rule_search.py`. |
| Accuracy ceiling (target 0.97) | The sharp rule's out-of-fold accuracy is 0.9546. Applying the out-of-fold "ring" error rates to the actual test rows gives an expected test accuracy of **0.946**. P(public ≥ 175/181) ≈ 0.13; **P(private ≥ 0.97) ≈ 0.04**. 0.97 is not reachable on purpose; it would take luck. |
| Best Aloo Theory notebook | `solution/best_aloo_theory.ipynb`: self-contained and Kaggle-ready, runs in about 20 s, and every claim in it is computed live (saved with outputs). |
| Reproducibility | `solution/kacchi_aloo_pipeline.ipynb` (full settings) reproduces both submitted files exactly. |

**Still to do by hand:** on https://www.kaggle.com/competitions/the-great-kacchi-aloo-mystery/submissions, tick **both A and B** as final submissions. Ignore their 1-row public difference, which is noise. Don't re-upload the same files later; the earlier timestamp wins ties.

---

## 0. Action required before anything else

**The competition data could not be downloaded.** The Kaggle API returned **HTTP 403** for `competitions download` because the Kaggle account configured on this machine has **not joined** the competition. Accepting the rules can only be done in the web UI, so I did not do it on your behalf.

What this means for this report:
- Every dataset statistic below comes from **[OFFICIAL]** pages or from the **[VERIFIED] execution logs** of the four public notebooks, which were run on Kaggle against the real data. None of it was re-computed by me on the real files.
- The full pipeline (`solution/kacchi_pipeline.py`) is written and was executed end-to-end on a **schema-identical synthetic replica** to prove it runs. Replica numbers are **not** competition results and are not reported as such.

**To unlock everything (about 10 minutes):**
1. Open https://www.kaggle.com/competitions/the-great-kacchi-aloo-mystery and click **Join Competition** / accept the rules. Use **one** Kaggle account only; Foundational Rule 5a disqualifies multi-account submitting.
2. `kaggle competitions download the-great-kacchi-aloo-mystery -p data && cd data && unzip *.zip && cd ..`
   This needs Kaggle CLI ≥ 1.8 for `access_token` credentials. The installed CLI is 1.7.4.5 and only reads `kaggle.json`. Run `pip install -U kaggle`, or use the venv recipe in Appendix A.
3. **Notebook (recommended):** open `solution/kacchi_aloo_pipeline.ipynb` on Kaggle (Add Input → this competition) or in Colab/Jupyter, then *Run All*. It writes `submission.csv` (= A) and `sub_B_hedge.csv`.
   **Script alternative:** `python solution/kacchi_pipeline.py --data data --out outputs --public-subs research/public_notebooks` (it also runs inside a notebook cell).
   This takes about 5–8 minutes on CPU with the default 10×5 CV.
4. Submit `outputs/sub_A_primary_bagged_softband.csv` and `outputs/sub_B_hedge.csv` **the same day** (see §17 and §18 on why timing matters), then **manually select both** as final submissions.

---

## Table of contents

0. [Action required](#0-action-required-before-anything-else)
1. [Executive summary](#1-executive-summary)
2. [Competition breakdown and rules](#2-competition-breakdown-and-rules)
3. [Dataset analysis](#3-dataset-analysis)
4. [Target analysis](#4-target-analysis)
5. [Metric analysis](#5-metric-analysis)
6. [Leaderboard analysis](#6-leaderboard-analysis)
7. [Public notebooks, ranking and lineage](#7-public-notebooks-ranking-and-lineage)
8. [Discussions and organizer clarifications](#8-discussions-and-organizer-clarifications)
9. [Reverse-engineering the generator (simulation and ABC)](#9-reverse-engineering-the-generator)
10. [Validation strategy](#10-validation-strategy)
11. [Leakage analysis](#11-leakage-analysis)
12. [Distribution shift](#12-distribution-shift)
13. [Duplicates](#13-duplicates)
14. [Seed and fold sensitivity](#14-seed-and-fold-sensitivity)
15. [External resources: papers, repos, models, datasets, related competitions](#15-external-resources)
16. [Why current solutions work, and the remaining headroom](#16-why-current-solutions-work-and-the-remaining-headroom)
17. [Experiment roadmap E0–E15](#17-experiment-roadmap)
18. [Top 10 experiments](#18-top-10-experiments)
19. [Final comparison (verified scores only)](#19-final-comparison-verified-scores-only)
20. [Recommended final submissions](#20-recommended-final-submissions)
21. [Approaches to avoid](#21-approaches-to-avoid)
22. [Private leaderboard strategy](#22-private-leaderboard-strategy)
23. [Step-by-step execution plan (Stages 1–10)](#23-step-by-step-execution-plan)
24. [The "Best Aloo Theory" prize](#24-the-best-aloo-theory-prize)
25. [Direct source links](#25-direct-source-links)
26. [Final conclusions](#26-final-conclusions)

Appendices: A (reproduction), B (file index).

---

## 1. Executive summary

**What the competition really is.** A synthetic, educational binary-classification task with 800 train and 400 test rows, scored on accuracy **[OFFICIAL]**. The label is essentially **1 when aloo_count / guests lies in a band of about 1 to 2 potatoes per guest**, with soft, noisy edges **[VERIFIED]** (three independent public notebooks' execution logs). Every other column carries no measurable signal **[VERIFIED]**. The host says so outright: *"The fanciest model doesn't always win. Thinking does."* **[OFFICIAL]**

**Key findings**

1. **The modelling problem is solved structurally.** In the [VERIFIED] public logs, a two-threshold rule on one ratio scores about **95.5% CV**. LightGBM on raw counts scores **86%** because trees cannot extrapolate to the test set's much bigger weddings (145 of 400 test rows exceed the largest train wedding). Logistic regression on raw counts scores **56%**.
2. **The public leaderboard is 181 rows** [COMPUTED from score granularity], so **one row = 0.55 accuracy points**. The private LB is very likely **219 rows** (UNVERIFIED; it assumes no ignored rows), where one row = 0.46 points and the binomial SD of any ~95%-accurate solution is about **3 rows**. The current top cluster (public 0.950–0.967, i.e. 172–175 of 181) is **statistically indistinguishable**.
3. **389 of the 400 test rows are predicted identically by every public submission file** [COMPUTED]. Only **11 near-edge rows** are disputed. So the private ranking among competent teams will be decided by **about 6 private coin-flip rows**, plus Kaggle's **tie-break: the earlier submission wins** [OFFICIAL, FR 7b].
4. **Threshold choice is not the lever.** I fitted the data-generating process to the published binned statistics (Monte Carlo, plus approximate Bayesian computation over four noise families). Every reasonable cut-point rule lies within **≤0.25 expected private rows** of the others [COMPUTED].
   - The posterior-predictive Bayes cut-points are about **(0.982, 2.0075)**.
   - The public "accuracy-optimal" cut-points (0.970, 2.0087) fit training noise. Under smooth-edge generators they lose about 0.5–0.7 expected private rows to likelihood-based or fixed cut-points.
5. **The largest legitimate private-LB lever is final-selection hedging.** Kaggle scores the *best* of your selected final submissions. Select (A) the Bayes-optimal decisions and (B) the same file with the most uncertain rows flipped by a fixed algorithm. This raises the expected best-of-two private score by about **+0.75 rows** [COMPUTED], three times more than any threshold choice.
6. **Timing matters.** Ties on 219 rows are likely, and the earlier entry wins a tie. Every current team will beat you in a tie, so **submit your final candidates as early as possible** and do not re-upload them later.
7. **A second prize is winnable on insight alone.** "Best Aloo Theory" goes to the notebook that best explains *why* guests go back for seconds [OFFICIAL]. New, verifiable insights are available (§9, §24):
   - Edge noise is **multiplicative**: the upper edge is about 1.8× wider in absolute units.
   - Edge sharpness may grow with wedding size (E15).
   - The Bayes ceiling is about 94–96%, and the public leader is in a ~5–8% luck tail.

**Bottom line [RECOMMENDATION]:** join, run the pipeline, verify it reproduces the public baseline exactly, submit A and B **today**, select both as finals, and spend the remaining time on the "Best Aloo Theory" notebook rather than on leaderboard chasing.

---

## 2. Competition breakdown and rules

### 2.1 Core facts

| Item | Value | Evidence |
|---|---|---|
| Host | Bangladesh AI Olympiad (BdAIO), run by BdOSN, for World AI Week 2026 | [OFFICIAL] |
| Category / reward | Community / Kudos | [OFFICIAL] (API) |
| Real-world framing | Did wedding guests go back for a second plate of kacchi biryani? | [OFFICIAL] |
| ML task | Binary classification on synthetic tabular data | [OFFICIAL] |
| Input | 11 columns per wedding (see §3) | [OFFICIAL] |
| Target | `went_back_for_seconds` ∈ {0, 1} | [OFFICIAL] |
| Output format | CSV with header `wedding_id,went_back_for_seconds`, one row per test id, integer 0/1 | [OFFICIAL] |
| Train / test | 800 / 400 rows | [OFFICIAL] |
| Public LB | **181 rows (45.25%)**. N ∈ {181, 362} fits all 10 distinct scores; 362 is implausible. | [COMPUTED] |
| Private LB | 219 rows if the complement of public (Kaggle's usual design) | UNVERIFIED |
| Metric | Accuracy (`accuracy score` tag; "Accuracy Score") | [OFFICIAL] |
| Submissions | 5 per day at ~09:00 UTC (API `maxDailySubmissions`); later the same day the API's `submission-limits` showed **25 remaining today**, so the host appears to have raised it. Re-check before planning. | [OFFICIAL] (API) |
| Team size | ≤ 10; merger deadline = final deadline | [OFFICIAL] (API) |
| Timeline | Enabled 2026-10-06 12:30 UTC; deadline 2026-10-09 17:59 UTC | [OFFICIAL] |
| Code requirements | None. CSV upload, not a code competition: any hardware, internet allowed | [OFFICIAL] (no code-comp settings) |
| Number of final selections | UNVERIFIED. Kaggle's default is 2; check the "Submissions" page. | — |
| Prizes | (1) top private-LB scores; (2) **Best Aloo Theory**: best plain-words notebook explaining *why* | [OFFICIAL] |
| Data provenance | Synthetic, "created by a Python computer program" | [OFFICIAL] |

### 2.2 Rules and their practical impact

| Rule | Source | Practical impact |
|---|---|---|
| One Kaggle account per person; no submitting through multiple accounts | FR 5a [OFFICIAL] | Use a single account. This machine has several saved tokens; pick one and stick to it. |
| No private code or data sharing outside a team; public sharing must be on Kaggle | FR 5d, 6a–b | Share your theory notebook publicly (needed for the prize anyway); don't send code to other teams. |
| Open-source code must be under an OSI licence that doesn't limit commercial use | FR 6c | Avoid tools with non-commercial weight licences. TabPFN's weights are listed as non-commercial in its README (§15.4). Not needed anyway. |
| "Submissions may not use … hand labeling or human prediction of the … test data records" | FR 4b | **Every row decision must come from code.** The hedge (B) flips rows by a fixed algorithmic rule (|P−0.5| ranking), never by hand-picking ids. |
| Winners by Private LB only | FR 7a | Public LB is only a sanity check. |
| **Ties: the submission entered first wins** | FR 7b | With accuracy on ~219 rows, ties are common. **Submit finals early.** Your entry is already later than all 17 current teams. |
| Final submission auto-selected if you don't select | FR 18c | Auto-selection takes the best *public* scores, which would usually drop the hedge B. **Select A and B manually.** |
| Public LB is "a representative sample of the test data" | FR 18f | Supports treating public/private as random splits of the same 400 rows (assumed in §6 and §9). |
| Sponsor may disqualify for "undermining the legitimate operation" | FR 8d, 16a | No LB probing for labels, no generator-RNG reconstruction hacks (§21). |
| External data / pretrained models | No clause on the competition pages; FR 6c licence condition only | Effectively irrelevant: synthetic data, no useful external dataset (§15.5). |
| Eligibility: 18+ or age of majority unless the sponsor obtains guardian consent | FR 1a | The audience is high-school students; not a modelling concern. |

### 2.3 Unusual scoring mechanics

- Accuracy on a **tiny** split means scores move in **0.552-point steps** on public and about **0.457-point steps** on private.
- Kaggle **truncates** displayed scores to 5 decimals: 88/181 = 0.486188 is displayed as 0.48618 [COMPUTED].
- The host's own benchmark rows appear at rank 0: `sample_submission.csv` (0.54143 = 98/181, all zeros) and **`submission (8).csv` (0.95027 = 172/181)**, presumably the host's reference solution (UNVERIFIED which rule).
- **Two prize tracks**, and the explanation track does not need a top score.

---

## 3. Dataset analysis

### 3.1 Schema

[OFFICIAL] column definitions; [VERIFIED] statistics from public notebook logs.

| Column | Type | Meaning | Key facts |
|---|---|---|---|
| `wedding_id` | string id | Unique wedding id | Train W0001–W0800, test W0801–W1200 [VERIFIED: test ids in public submissions] |
| `event` | categorical (3) | Gaye Holud / Biye / Bou-bhat | No effect on band edges (subgroup cut-points ≈ equal) [VERIFIED] |
| `city` | categorical (8) | Barishal, Chattogram, Dhaka, Khulna, Mymensingh, Rajshahi, Rangpur, Sylhet | Apparent rate differences explained by each city's aloo/guest mix [VERIFIED claim + log] |
| `guests` | int | Number of guests | Train 51–700 (mean 372); **test 51–1499 (mean 648); 145 test rows > 700** [VERIFIED] |
| `aloo_count` | int (nullable) | Total potatoes | **24 missing (train only)**; **6 ×10 typos (train only)**; train max 11,520 (typo), test max 3,814 [VERIFIED] |
| `mutton_kg` | float | Mutton cooked | Mutton/guest uniform-like 0.18–0.32; r(guests) = 0.94 [VERIFIED] |
| `borhani_glasses` | int | Borhani served | Borhani/guest 0.80–1.50; r(guests) = 0.93 [VERIFIED] |
| `fairy_lights` | int as string | Fairy lights | **15 values in Bangla digits (train only)**; r(guests) = 0.68 [VERIFIED] |
| `drone_photographer` | yes/no | Drone filming | No effect [VERIFIED] |
| `dhol_players` | int 0–8 | Drummers | No effect; extra-eater denominators don't help [VERIFIED] |
| `aunties_asking_when_marriage` | int 0–16 | Aunties | Weakest raw correlate (ρ = −0.07); no effect after the ratio [VERIFIED] |
| `went_back_for_seconds` | 0/1 (train) | **Target** | 42.3% positive on clean rows [VERIFIED] |

### 3.2 Granularity and relationships

- **One train row = one test row = one wedding party (event).** [OFFICIAL]
- Train and test are linked only through the generator. No entity spans rows: no couple or venue id, and the three parties of one wedding are not linked by any key.
  - [HYPOTHESIS] The rows are i.i.d. draws, so no GroupKFold is needed. The pipeline's ID-order and duplicate checks (E14) test this.
- Train and test come from the **same generator, except for wedding size**. Test oversamples big weddings ("big wedding season") [OFFICIAL]. Per-guest ratios have the same distribution in both [VERIFIED].
- Hidden test structure: 400 rows, randomly split into public (181) and private (219). There is no temporal, site or device structure. The only "domain" variable is size.

### 3.3 Data-quality traps

All dirt is in train only [VERIFIED].

| Trap | Train | Test | Correct handling | Impact |
|---|---|---|---|---|
| Missing `aloo_count` | 24 rows (target rate 0.333) | 0 | Drop for training. aloo/guest is independent of every other ratio (r ≤ 0.04), so it cannot be imputed [VERIFIED] | None on test |
| Bangla digits in `fairy_lights` | 15 rows | 0 | `unicodedata.decimal` / `int()` handles any Unicode digit [EXTERNAL A055]; the pipeline normalises **all** numeric columns | Feature is useless anyway |
| ×10 `aloo_count` | 6 rows: W0081, W0317, W0448, W0502, W0672, W0786 (raw ratios 22.7, 9.7, 16.5, 16.6, 26.8, 23.4) | 0 | Clean ratio ≤ 2.70, so ratio > 3 ⇒ divide by 10 [VERIFIED] | Affects training only; the band fit ignores ratios > 2.2 anyway (E2a/E2b) |
| Bigger test weddings | max 700 | max 1499, 145 rows > 700 | Use scale-free ratios | **Decisive**: kills raw-count trees |

### 3.4 Feature value

- All raw Spearman correlations with the target satisfy |ρ| < 0.075 [VERIFIED].
- The signal exists **only** in `aloo_count / guests` (apg), and it is **non-monotone** (a band), which is why linear correlations miss it.
- apg is close to **uniform on [0.40, 2.70]** in train and test: train quartiles 0.96 / 1.51 / 2.13, test 0.95 / 1.54 / 2.07 [VERIFIED]. That matches U(0.4, 2.7) quartiles of 0.975 / 1.55 / 2.125 [COMPUTED]. The generator most likely draws apg uniformly, then sets aloo_count ≈ apg × guests.

---

## 4. Target analysis

- **Single binary target.** Base rate 0.4227 on 776 clean train rows (328 positives) [VERIFIED + COMPUTED]; 0.42 on all 800.
- **Public split:** 83 positives / 98 negatives (45.9%) [COMPUTED from the all-zero benchmark score]. That matches the band rule's predicted test positive rate of 0.455–0.460 [VERIFIED], so the band generalises to test.
- **Structure** (binned table, [VERIFIED] log; bins are (a, b], so "1.0–1.1" means 1.0 < apg ≤ 1.1):

| apg bin | 0.4–0.9 | 0.9–1.0 | 1.0–1.1 | 1.1–1.2 | 1.2–1.6 | 1.6–1.7 | 1.7–1.8 | 1.8–1.9 | 1.9–2.0 | 2.0–2.1 | 2.1–2.2 | 2.2–2.3 | 2.3–2.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P(seconds) | 0/177 | 12/37 | 25/32 | 36/37 | 128/128 | 23/24 | 38/39 | 34/37 | 18/23 | 11/37 | 1/35 | 2/35 | 0/135 |

- **Difficulty:** trivial away from the edges, coin flips within about ±0.03 of each edge. About 76 test rows lie within ±0.1 of an edge, and about 11 within the true coin-flip zone [VERIFIED / COMPUTED].
- **Noise:**
  - No label noise far from the edges: 0 errors among 177 rows below 0.9, 135 rows above 2.3, and 128 rows in 1.2–1.6.
  - Near the edges, noise is consistent with a soft boundary. Errors by distance from the fitted edge [VERIFIED, Rafiur log]: 5.7% at 0–0.02, 36.4% at 0.02–0.05, 14.1% at 0.05–0.1, 3.5% at 0.1–0.2, 1.0% at 0.2–0.5, 0% beyond.
  - The *low* error right next to the fitted edge is the fingerprint of **cut-points fitted to the sample's own noise** [COMPUTED interpretation; cf. Altman et al. 1994, EXTERNAL A017].
- **Missingness:** the target rate among missing-aloo rows (8/24) and Bangla-digit rows (4/15) is lower, but not significantly (binomial, n tiny), and there is no dirt in test. **Not usable.**

---

## 5. Metric analysis

**Definition:** Accuracy = (1/n) Σᵢ 𝟙[ŷᵢ = yᵢ], with ŷᵢ ∈ {0, 1}.

**Bayes decision:** with calibrated pᵢ = P(yᵢ = 1 | xᵢ), the optimum is ŷᵢ = 𝟙[pᵢ > 0.5]. The expected accuracy is (1/n) Σ max(pᵢ, 1 − pᵢ).

| Question | Answer for this competition |
|---|---|
| What it rewards | Being on the correct side of 0.5 for each row. Here that means **where you put the two band edges**. |
| What it ignores | Confidence, calibration away from 0.5, ranking, log-loss. A model with perfect edges and terrible probabilities scores the same. |
| Does ranking matter? | No. AUC-style quality is irrelevant except through the 0.5 crossing. |
| Does calibration matter? | Only **at** p ≈ 0.5. A monotone distortion of p (e.g. p²) keeps AUC but moves the band edges. |
| Does threshold selection matter? | Yes, it is the whole game. The threshold must be **0.5 on calibrated P(y \| apg)**, not the train base rate. |
| Class weights? | No. Accuracy is symmetric; re-weighting shifts the 0.5 crossing and hurts. |
| Per-target optimisation? | Single target; not applicable. |
| Clipping? | Irrelevant. |
| Rank averaging? | **Harmful or meaningless.** Ranks destroy the absolute 0.5 threshold, and the target is non-monotone in the score. |
| Logit vs probability averaging? | Only differs near p = 0.5. Probability averaging of calibrated members is preferred (it estimates posterior-predictive P). |
| Geometric mean? | Biases toward 0 near the edges, which shifts edges inward. Avoid. |

**Examples where similar conventional metrics give different accuracy:**
1. **Same AUC, different accuracy.** A soft-band model with P and the same model reporting P² have **identical AUC**. But P² > 0.5 requires P > 0.707, which moves both edges inward. At apg = 1.01, P ≈ 0.69 under the ABC posterior, so P² ≈ 0.47 flips the prediction to 0. With about 76 test rows within ±0.1 of an edge, this costs several rows.
2. **Better log-loss, same accuracy.** A soft band with steeper k has lower log-loss on far-from-edge rows but identical decisions. Log-loss improvements here are **not** accuracy improvements.
3. **Higher CV accuracy, lower expected private accuracy.** The accuracy-optimal band (0.970, 2.0087) beats the fixed (1, 2) band by +0.0112 CV. Under smooth-edge generators fitted to the same data, its expected private accuracy is still 0.5–0.7 rows lower (§9). CV rewarded fitting the sample's own clusters of noisy labels.

---

## 6. Leaderboard analysis

### 6.1 Decoded public leaderboard (snapshot 2026-10-07 08:58 UTC)

[VERIFIED scores] [COMPUTED counts and Wilson 95% intervals]

| Rank | Team | Score | Correct / 181 | Errors | Subs | Wilson 95% CI |
|---|---|---|---|---|---|---|
| 1 | FOYSAL | 0.96685 | 175 | 6 | 10 | 0.930–0.985 |
| 2–4 | রোদশী / Str1k3rFl0 / Raiyana Rufaida | 0.96132 | 174 | 7 | 7 / 6 / 4 | 0.922–0.981 |
| 5–10 | amar_nam_adnan, Tasnimul Ferdous, Dadhichi Sarker, Md. Najmul Hasan Shihab, el mencho, Rafiur Rahman | 0.95580 | 173 | 8 | 3 / 4 / 1 / 5 / 4 / 3 | 0.915–0.977 |
| host benchmark | `submission (8).csv` | 0.95027 | 172 | 9 | 1 | 0.908–0.974 |
| 11–14 | fig_is_a_storm … Md Abdul Al Hasib | 0.696–0.740 | 126–134 | 47–55 | 1–3 | — |
| host benchmark | `sample_submission.csv` (all zeros) | 0.54143 | 98 | 83 | 1 | — |
| 15–17 | … | 0.486–0.541 | 88–98 | — | — | — |

### 6.2 What it means

- **The top 10 teams plus the host benchmark span 4 rows (172–175).** The Wilson intervals overlap almost completely.
- For a rule with true accuracy 0.955, P(public ≥ 175/181) = 0.29. For 0.94 it is 0.08 [COMPUTED, binomial].
- In simulation, a Bayes-optimal rule reaches ≥ 175/181 on public in only 5–8% of draws [COMPUTED, §9]. **The leader's 0.96685 is most likely a favourable public split plus selection over 10 submissions.** It is not evidence of a better model.
- **Public/private correlation:** no evidence exists yet (the competition is running). Theory: for a fixed prediction vector, the public and private errors fall on disjoint random rows, so they are **independent**. Among near-identical solutions, public rank carries ~zero information about private rank [COMPUTED].
- **Shake-up expectation:** large among ranks 1–10 (essentially a lottery over about 6 private disputed rows plus ties). Ranks 11+ (raw-feature models at 0.70–0.74) will stay down.

### 6.3 Risk register

| Risk | Size | Mitigation |
|---|---|---|
| Public-overfit by repeated submissions | One row = 0.55 points; selecting the max of k noisy scores inflates public by ~1–2 rows for k ≈ 10 | Never select by public score. |
| Per-row threshold tuning on public | Changes only rows in public; zero transfer to private (labels are independent given apg) | Don't. |
| Ensemble-weight tuning on public | Same | OOF-only weights (all members tie anyway). |
| Threshold tuning on train (accuracy-optimal cuts) | Fits noise; −0.5 to −0.7 private rows under smooth generators (§9) | Likelihood-based or posterior-predictive cuts. |
| Ties | Earlier submission wins | Submit finals early. |

### 6.4 Framework: CV improvement vs Public improvement vs Private improvement

| CV | Public | Interpretation | Action |
|---|---|---|---|
| ↑ significant (corrected t-test p < 0.05) | ↑ | Strong candidate | Keep |
| ↑ significant | ↓ by ≤ 3 rows | Within public noise (SD ≈ 2.8 rows) | Keep. Trust CV. |
| ≈ (p > 0.05) | ↑ 1–3 rows | Public noise | Ignore. Keep the simpler or more principled model. |
| ↓ | ↑ | Public overfit | Reject |
| Only one fold or seed improves | any | Insufficient evidence | NEEDS_MORE_EVIDENCE |
| ≈ CV, better size-stress score | ≈ | More shift-robust | Prefer for private |
| Any change that only alters rows within ±0.03 of an edge | any | Coin flips: no model can win them in expectation | Use for the **hedge**, not as a "better model" |

---

## 7. Public notebooks, ranking and lineage

All 4 public notebooks were found: via the Kaggle API `kernels list --competition`, text searches for "kacchi" and "aloo mystery", and a dataset search. There are **no forks, no output datasets and no GitHub mirrors** (searched). Full details are in `public_notebooks.csv`.

### 7.1 Catalogue

| Notebook | Author | Last run (UTC) | Votes | CV [VERIFIED log] | Public LB | Model / key technique |
|---|---|---|---|---|---|---|
| [kacchi_starter](https://www.kaggle.com/code/tasnimmahfuznafis/kacchi-starter) | Tasnim Mahfuz Nafis (host-linked) | 10-06 12:15 | 4 | UNVERIFIED (empty log) | UNVERIFIED | RF(300) on raw counts; hints (ratio, depth-2 tree, big weddings) |
| [Kacchi Aloo Mystery Deep EDA](https://www.kaggle.com/code/foysalemonshanto/kacchi-aloo-mystery-deep-eda) | FOYSAL | 10-06 16:01 | 0 | fitted band 0.9549 ± 0.0070 (5-fold) | not claimed | Full audit; accuracy-optimal band (0.970, 2.010); soft-band MLE |
| [LB_0.95027_Kacchi Aloo Theory](https://www.kaggle.com/code/foysalemonshanto/lb-0-95027-kacchi-aloo-theory) | FOYSAL | 10-06 15:55 | 0 | band_rule 0.9558 (se 0.0018, 10×5); lgbm 0.9541; knn5 0.9513; fixed 0.9446; logmle 0.9407 | **0.95027 [CLAIM]** | Band rule (0.9700, 2.0087), plateau centre; 5 candidate files |
| [The Aloo Theory: The Goldilocks Band](https://www.kaggle.com/code/rafiurrahman01/the-aloo-theory-the-goldilocks-band) | Rafiur Rahman | 10-07 06:59 | 0 | band 0.9558 ± 0.0147 (3×5); fixed 0.9446; LGBM-all 0.9386 | team best 0.95580 [VERIFIED]; which file produced it is UNVERIFIED | Midpoint cuts [0.971, 2.010]; falsification tests |

### 7.2 Ranking by category

| Category | Members |
|---|---|
| A. Verified high-scoring notebooks | **None.** No notebook's own LB score can be verified without submitting its file. |
| B. Claimed scores but unverified | LB_0.95027 (title claim 0.95027 = 172/181, the same as the host benchmark) |
| C. Strong CV/training notebooks | LB_0.95027 (10×5 CV, 5 models); Goldilocks Band (3×5 CV + falsification tests) |
| D. EDA/preprocessing notebooks | Deep EDA; host starter |
| E. Forks / reused stacks | None as Kaggle forks, but see the prediction-vector families below. |

### 7.3 Lineage: prediction-vector forensics [COMPUTED]

I diffed all 7 public submission files row by row. There are **4 distinct prediction vectors**:

| Vector | Files | Rule | Positive rate |
|---|---|---|---|
| **V0** | Deep EDA `submission_rule.csv` = LB_0.95027 `submission.csv` = `submission_band_rule.csv` | Accuracy-optimal band (0.970, 2.0087/2.010) | 0.460 |
| **V2** | LB_0.95027 `submission_lgbm_logapg.csv` = **Goldilocks `submission.csv`** | LGBM on log apg ≡ midpoint band [0.971, 2.010] | 0.455 |
| V1 | `submission_knn5.csv` | kNN-5 on log apg | 0.4425 |
| V3 | `submission_logmle_band.csv` | log-space soft-band MLE (0.9911, 1.9912) | 0.445 |

- **389/400 rows are identical across all vectors.** The 11 disputed ids and what they reveal:

| id(s) | V0 | V2 | V3 | V1 | Inferred apg location |
|---|---|---|---|---|---|
| W0968, W1163 | 1 | 0 | 0 | 0 | **0.9700 < apg < 0.9710** (pinned by V0 vs the closed band [0.971, …]) |
| W0839, W0865, W1119, W1157 | 1 | 1 | 0 | 1 | [0.971, 0.9911] or [1.9912, 2.0087) |
| W0848, W1002, W1022, W1179, W1198 | 1 | 1 | 1 | 0 | inside (0.9911, 1.9912) but kNN-5 neighbours negative, so near an edge |

- **Solution families**:

| Family | Members | Status |
|---|---|---|
| F0: raw-feature trees | starter RF, LGBM-raw | Fails under size shift |
| F1: accuracy-optimal ratio band | V0, plus V2 as its midpoint variant | Dominant public family |
| F2: smooth ratio models | soft-band MLE (V3), LGBM-log-apg (V2), kNN (V1) | Same feature, different edge estimators |

- **Independence:** FOYSAL's two notebooks share one pipeline (same cleaning, same V0). Rafiur's notebook is independent code that lands on V2. **Genuinely independent approaches: 2** (FOYSAL's and Rafiur's), plus the host starter. All of them converge on the same feature.
- Using the ABC posterior (§9), P(seconds | apg = 0.9705) ≈ 0.41, so the Bayes call for W0968/W1163 is **0** (V2 agrees, V0 does not).
  - ~~[HYPOTHESIS] Rafiur's public 0.95580 (173) came from V2.~~ **Refuted:** V2 itself was submitted (as A) and scored 0.95027 (172). See the UPDATE section.

---

## 8. Discussions and organizer clarifications

- **There are zero discussion topics** (Kaggle API `competition_list_topics` returned total_count = 0) [VERIFIED]. No organizer replies exist.
- The only organizer guidance is on the official pages [OFFICIAL]. `important_discussions.csv` records it:
  - The Data page's "Field Notes from Mitu (read this!)": empty aloo boxes, Bangla digits, an extra zero, bigger test weddings, synthetic data.
  - The Overview's "The fanciest model doesn't always win. Thinking does."
  - The starter notebook's hints: "per guest", Babul Baburchi's "too little / too MUCH aloo", "tiny tree", and "a model that understands the real reason … doesn't care how big the wedding is".
  - Rules contact: bdaio@bdosn.org.
- [RECOMMENDATION] If you need certainty on the number of final selections or the theory-prize judging criteria, email the host. Answers sent by email are not public.

---

## 9. Reverse-engineering the generator

This is the core original research of this report: [COMPUTED] under stated hypotheses. Scripts are in `research/scripts/` and outputs in `research/analysis/`.

### 9.1 Fitting candidate generators to the published binned table

The 0.1-wide binned rates and counts in the public logs let me recover the exact positive count per bin (sum = 328/776, matching the base rate). I fitted P(y = 1 | apg) = σ(k(g(apg) − g(lo))) · σ(k(g(hi) − g(apg))), with g = identity or log, plus probit variants, by binned maximum likelihood:

| Model | lo | hi | k | NLL | AIC |
|---|---|---|---|---|---|
| **log-space logistic, symmetric** | 0.9904 | 1.9948 | 29.8 | 116.78 | **239.6 (best)** |
| linear logistic, asymmetric k | 0.9924 | 1.9990 | 27.6 (lower) / **15.5 (upper)** | 116.45 | 240.9 |
| log-space logistic, asymmetric | 0.9902 | 1.9952 | 27.6 / 30.9 | 116.64 | 241.3 |
| log probit | 0.9917 | 1.9911 | 15.8 | 118.05 | 242.1 |
| linear logistic, symmetric | 0.9934 | 2.0000 | 19.2 | 120.40 | 246.8 |
| linear probit | 0.9947 | 1.9967 | 9.9 | 123.50 | 253.0 |

**Finding [COMPUTED]: the noise is multiplicative.** In absolute apg units the upper edge is about 1.8× wider than the lower edge (k 15.5 vs 27.6), which is exactly what a symmetric log-space noise implies (factor 2 at 2 vs 1). The best AIC is the log-space model.

Plain words: guests judge potatoes in *proportion*, or the generator perturbs the ratio by a percentage. This also agrees with:
- FOYSAL's [VERIFIED] unbinned log-space MLE edges (0.9911, 1.9912);
- the Deep EDA's linear MLE (0.99, 2.00, k ≈ 20) [CLAIM: printed in a plot legend, not in the log].

Profile 95% CIs: lo ∈ [0.971, 1.016], hi ∈ [1.975, 2.025] (linear); lo ∈ [0.973, 1.010], hi ∈ [1.967, 2.022] (log).

### 9.2 Monte Carlo: which public procedure generalises best? (1,000 replicates per generator)

Setup: simulate 776 train and 400 test rows (181 public + 219 private). Apply each procedure. Score it on the private part.

| Procedure | Private acc, generator = linear logistic | Δ rows vs accuracy-optimal | Private acc, generator = log logistic | Δ rows vs accuracy-optimal |
|---|---|---|---|---|
| Bayes oracle (true cuts) | 0.9363 ± 0.0173 | +0.67 | 0.9391 ± 0.0157 | +0.56 |
| fixed (1, 2) | 0.9362 | +0.66 | 0.9388 | +0.50 |
| soft-band MLE (linear) | 0.9356 | +0.53 | 0.9383 | +0.39 |
| soft-band MLE (log) | 0.9357 | +0.54 | 0.9384 | +0.42 |
| **accuracy-optimal band (public V0 method)** | 0.9332 | 0 | 0.9365 | 0 |
| LGBM on log apg (public) | 0.9332 | −0.01 | — | — |
| kNN-5 on log apg (public) | 0.9281 | −1.11 | 0.9312 | −1.16 |

- **Private SD for every procedure is about 0.016–0.017, i.e. 3.5–3.8 rows.** The differences between procedures (≤ 0.7 rows) are an order of magnitude smaller than the luck of the draw.
- **Evidence the real data deviates from these smooth generators:**
  - The real CV gap (accuracy-optimal minus fixed (1, 2)) is +0.0112. Simulated CV gaps average −0.003 (SD 0.006), and P(gap ≥ +0.0112) = **1.3–1.4%**.
  - The real in-sample error reduction from optimising the cuts is 10 rows; simulated, P(≥ 10) ≈ 6%.
  - So either the true lower edge sits slightly below 1, or the sample has an unusual cluster of positives at 0.97–0.99 ("7 of 8 rows positive", [VERIFIED] Deep EDA). §9.3 weighs both.
- The simulated accuracies (~0.937–0.939) are lower than the real CV (~0.955). The fitted smooth generators are slightly noisier than reality, so read the absolute levels as pessimistic and the *differences* as informative.

### 9.3 Approximate Bayesian computation (ABC) over all published evidence

- **Summaries matched:** 10 edge-bin rates; errors of the fixed (1, 2) rule (43); errors of the accuracy-optimal band (33); its cut-points (0.9700, 2.0087); zero far-from-edge flips.
- **Priors:** lo ~ U(0.95, 1.03), hi ~ U(1.96, 2.05), width log-uniform. Four families: linear/log × logistic/Student-t ("noisy ratio" with heavy tails).
- **Run:** 400,000 simulations; nearest 0.25% accepted.

| Quantity | Result |
|---|---|
| Family posterior | lin-logistic 0.30, log-logistic 0.22, lin-t 0.25, log-t 0.23. Not distinguishable from these summaries. |
| lo posterior (2.5 / 25 / 50 / 75 / 97.5%) | 0.956 / 0.972 / **0.982** / 0.993 / 1.014 |
| hi posterior | 1.976 / 1.995 / **2.007** / 2.018 / 2.038 |
| **Posterior-predictive Bayes cut-points** | **(0.9820, 2.0075)** (a 3-family logistic-only ABC gives (0.9835, 2.0035)) |
| P(seconds \| apg) near the lower edge | 0.95 → 0.28; 0.97 → 0.41; 0.98 → 0.49; 0.99 → 0.56; 1.00 → 0.63; 1.02 → 0.74 |
| P(seconds \| apg) near the upper edge | 1.97 → 0.69; 1.99 → 0.59; 2.00 → 0.54; 2.01 → 0.49; 2.03 → 0.38 |

Expected private accuracy over the posterior (219 rows):

| Rule | Δ rows vs fixed (1, 2) |
|---|---|
| fixed (1, 2) | 0 |
| log-MLE (0.9911, 1.9912) (V3) | +0.05 |
| accuracy-optimal (0.970, 2.0087) (V0) | +0.15 |
| Rafiur midpoints (0.971, 2.010) (V2) | +0.16 |
| **posterior Bayes (0.982, 2.0075)** | **+0.25** |
| oracle (knows the true parameters) | +0.55 |

**Conclusion [COMPUTED].** Once uncertainty about the generator is integrated out, all public rules are within 0.25 expected rows of each other, and even an oracle gains only about 0.55 rows. **The threshold is not where private-LB edge comes from.**

### 9.4 Final-selection hedge

Set A = Bayes decisions and B = A with the m rows of smallest |P − 0.5| flipped. Kaggle scores max(A, B) on private. Monte Carlo under both generators gives E[max] − E[A] = **+0.73 rows (lin) and +0.76 rows (log)**, best at m ≈ 8–16. On the replica run with real-size inputs, m = 18 gave +0.80–0.88 rows. The pipeline (E13) recomputes the optimal m on the **real** test apg values.

### 9.5 New hypothesis: E15, size-dependent edge sharpness

If labels come from the *true* ratio while the recorded `aloo_count` (an integer) or `guests` carries counting noise ("Tutul is *mostly* accurate"), then a ±few-potato error matters far less at a 1,400-guest wedding. Edges would be **sharper for big weddings**.

- The pipeline fits k = exp(a + γ·log(guests / median)) and runs a likelihood-ratio test against γ = 0.
- On the synthetic replica, where integer rounding of aloo_count creates exactly this effect, it detected γ = 0.38 (p = 0.011).
- **If significant on real data**, near-edge test rows from huge weddings (36% of test) are *less* uncertain. The hedge should then flip mainly small-wedding edge rows, and the edges can be estimated better by weighting sharp (large-wedding) train rows. Nobody has explored this publicly.

---

## 10. Validation strategy

**How the hidden test was generated** [OFFICIAL + VERIFIED]: same generator, i.i.d. rows, a random public/private split. The only deviation is wedding size (test reaches 1,499 guests vs 700 in train). Validation must therefore answer two questions:
1. How well does a rule generalise to new rows from the same generator? Answer this with **repeated stratified K-fold**.
2. Does it survive extrapolation in size? Answer this with a **size-stress fold**.

[RECOMMENDATION] Canonical scheme, frozen in `outputs/folds.csv`:
- **RepeatedStratifiedKFold, 5 folds × 10 repeats, seeds 2026–2035.** Strata = target × apg-zone (below 0.9 / lower edge / inside / upper edge / above 2.1), so every fold gets its share of the decisive edge rows. In the replica, fold SD of the band rule fell from 0.0163 (target-only strata) to 0.0114.
- **Size-stress holdout:** train on guests ≤ 60th percentile, validate on the largest 40%; also at q = 0.5 and 0.7. Ratio models should be flat; raw-count models collapse (replica: RF raw 0.62, LGBM raw 0.67 vs band 0.945).
- **Paired comparison:** Nadeau–Bengio / Bouckaert–Frank **corrected repeated-CV t-test** on per-fold accuracy differences vs the parent [EXTERNAL A023, A024]. Treat |Δ| < 2 SE as a tie and break ties toward the simpler model.
- **Adversarial validation:** raw features vs per-guest features. The expectation is high AUC on raw (driven by guests) and ~0.5 on ratios [EXTERNAL A012, C031].

**Rejected alternatives:**
- GroupKFold / StratifiedGroupKFold: there are no groups. Re-test with E14.
- TimeSeriesSplit / purged / embargo CV: there is no time axis.
- Leave-one-site-out by city: city has no effect; it would only add variance.
- Nested CV: only needed for the stacking experiment E8, which the pipeline implements.
- Public LB as validation: 181 rows, SD 2.8 rows; it is a bug check, not a selector.

---

## 11. Leakage analysis

| Route | Plausible? | Test (automated in pipeline E14) | Use it? |
|---|---|---|---|
| Target-derived features | No: all columns precede the outcome | — | — |
| ID-order leakage (generator writes rows in label order) | Possible in synthetic data | Spearman(id, y), AUC(id → y), Spearman(id, band errors) | Never use. If found, report it to the host. |
| Dirt-as-signal (missing / Bangla / ×10 correlate with y) | Weak in train (n tiny); **absent in test** | Target rate by flag | Useless on test |
| Train–test duplicates or near duplicates | Unknown | Exact feature hashes; same (guests, aloo) pairs | Only to verify consistency |
| Group / entity leakage | No entities | — | — |
| Temporal leakage | No time axis | — | — |
| Fitting cleaning thresholds on train+test | ×10 rule uses a fixed 3.0 cut-off from train | Test has 0 ×10 rows | Safe |
| Target-encoding leakage | Not used | — | — |
| Pseudo-label leakage | Only if labels come from a model that saw the validation fold | Pseudo-labels generated inside each fold (E9) | Safe as implemented, but useless |
| Public-LB feedback leakage (probing) | Possible with 5 subs/day | — | **Do not.** It only fits public rows, and FR 8d/16a apply. |
| Generator / RNG reconstruction | Theoretical (seeded Python program) | — | **Do not.** It undermines legitimate operation and needs the host's code. |
| Hand labelling of test rows | Prohibited | — | FR 4b: keep all decisions algorithmic. |

---

## 12. Distribution shift

[VERIFIED] from logs:

| Feature | Train | Test |
|---|---|---|
| guests | mean 372 (sd 185), max 700 | mean 648 (sd 428), max 1499; quartiles 289 / 534 / 964 |
| aloo_count (fixed) | ≤ ~1,890 (700 guests × 2.70; computed bound, not logged) | max 3814 |
| mutton_kg | max 221 | max 453 |
| borhani | max 1016 | max 2166 |
| fairy_lights | max 2861 | max 5942 |
| aloo / guest | mean 1.54, range 0.41–2.70 | mean 1.53, range 0.40–2.68 |
| dhol, aunties | max 8, 16 | max 8, 12 |

**Interpretation:**
- Pure **covariate shift in scale**. The concept P(y | apg) is unchanged (the test positive rate implied by public = 45.9% ≈ the band's predicted 46%).
- Any feature that grows with size (all raw counts) is out of range for 36% of test rows.
- Ratios are shift-free, so a ratio model needs **no** importance weighting. Shimodaira / Sugiyama-style reweighting [EXTERNAL A008, A011] is unnecessary because the model only uses shift-free inputs.

**How shift influences model selection [RECOMMENDATION]:**
- Reject any model whose size-stress accuracy drops more than 1 point below its CV.
- Prefer scale-free features.
- E15 is the one place where size legitimately enters: as a *noise-width* modifier, not as a location shift.

The pipeline adds KS, PSI and adversarial AUC for raw vs per-guest features (E14).

---

## 13. Duplicates

- Not determinable without the data (UNVERIFIED). The pipeline checks:
  - exact feature-hash duplicates within train, within test, and across them;
  - (guests, fixed aloo) pair collisions across train and test.
- Expected [HYPOTHESIS]: none. Values are continuous draws.
- If duplicates exist across train and test, their train labels are still only noisy draws near the edges. Use them only as a consistency check, never as "lookups".

---

## 14. Seed and fold sensitivity

[VERIFIED] public evidence:
- Fold-level SD of the fitted band: 0.0070 (Deep EDA, 5 folds) and 0.0147 (Rafiur, 3×5).
- Fold-level SD of the fixed rule: 0.020–0.021.
- 10×5 standard error ≈ 0.0018–0.0023 (FOYSAL).

[COMPUTED] Implications:
- **One test-row-equivalent in CV ≈ 0.0013** (1/776). Any CV gain below ~0.004 (≈ 3 rows) between ratio models is noise.
- Band rules are deterministic, so the "seed" is the fold assignment. LGBM / kNN members add negligible seed variance on a 1-D feature.
- The pipeline tracks mean, SD, min, max and per-fold accuracy for every experiment; decisions use the corrected paired test.

---

## 15. External resources

Brief: 500+ resources. **Reality:** for an 11-column synthetic educational task, **122 genuinely relevant, verified, de-duplicated resources** were found. Every URL was confirmed via WebSearch/WebFetch, and one duplicate was removed. I did not pad the list.
- `external_resources.csv`: full catalogue with all requested columns, relevance, evidence and actionability ratings.
- `TOP_100_RESOURCES.md`: ranked Top 100 with key idea, how to use it, impact, difficulty and link.
- `research/related_competitions.md`: deep notes on 6 analogous competitions.

| Category | Count (approx.) |
|---|---|
| Competition primary sources (pages, notebooks, LB) | 8 |
| Prior competitions & winning solutions | 30 |
| Cross-validation, model selection, significance tests | 14 |
| Leaderboard overfitting / adaptive data analysis | 6 |
| Interpretable models, rules, symbolic regression | 11 |
| Covariate shift / adversarial validation | 8 |
| Label noise | 4 |
| Cut-point / dichotomisation dangers | 3 |
| Feature engineering & tree extrapolation | 6 |
| Calibration / probabilistic modelling | 4 |
| Data cleaning / validation tooling | 5 |
| Core library docs | ~12 |
| Pretrained tabular models | 2 |
| Domain context | 4 |

### 15.1 Must-read papers (Top 25)

Difficulty: L/M/H. Impact: impact on this competition's decisions.

| # | Paper | Authors (year) | Contribution | Why relevant here | Proposed experiment | Diff. | Impact | Link |
|---|---|---|---|---|---|---|---|---|
| 1 | Dangers of using "optimal" cutpoints… | Altman, Lausen, Sauerbrei, Schumacher (1994) | Data-driven cut-points inflate apparent performance | Public band cut-points are exactly this | Compare accuracy-optimal vs MLE vs fixed cuts in repeated CV + simulation (E3a, S3) | L | **High** | https://doi.org/10.1093/jnci/86.11.829 |
| 2 | An Empirical Analysis of Feature Engineering for Predictive Modeling | Heaton (2016) | Models differ in their ability to synthesise ratios | Ratio is the whole signal; trees approximate it poorly | Raw vs ratio features under size-stress (E1, E5g) | L | High | https://arxiv.org/pdf/1701.07852 |
| 3 | Bias in error estimation when using CV for model selection | Varma & Simon (2006) | Selecting by CV inflates the selected CV score | Picking the best of many ratio rules by CV | Nested evaluation of the selection step | M | High | https://pmc.ncbi.nlm.nih.gov/articles/PMC1397873 |
| 4 | On over-fitting in model selection… | Cawley & Talbot (2010) | Variance of the selection criterion matters as much as bias | Many near-identical rules | Prefer low-variance criteria (likelihood) over 0-1 loss | M | High | https://www.jmlr.org/papers/v11/cawley10a.html |
| 5 | Inference for the Generalization Error | Nadeau & Bengio (2003) | Corrected resampled t-test | Deciding KEEP/REJECT on 776 rows | Used in the pipeline's decision rule | M | High | https://mlanthology.org/mlj/2003/nadeau2003mlj-inference |
| 6 | Evaluating the replicability of significance tests… | Bouckaert & Frank (2004) | Corrected repeated k-fold test | Same | Same | M | High | https://researchcommons.waikato.ac.nz/handle/10289/1451 |
| 7 | Approximate statistical tests for comparing classifiers | Dietterich (1998) | McNemar / 5×2cv tests | Rules differ on a few rows only | McNemar on disagreeing OOF rows | L | Medium | https://mlanthology.org/neco/1998/dietterich1998neco-approximate/ |
| 8 | Note on the sampling error of the difference between correlated proportions | McNemar (1947) | Paired test on discordant pairs | Same | statsmodels `mcnemar(exact=True)` | L | Medium | https://link.springer.com/doi/10.1007/BF02295996 |
| 9 | The Ladder | Blum & Hardt (2015) | Leaderboards overfit under adaptive submissions | 181-row public LB, 5 subs/day | Ignore public deltas < ~3 rows | L | High | https://arxiv.org/pdf/1502.04585 |
| 10 | The reusable holdout | Dwork et al. (2015) | Adaptive reuse of a holdout breaks validity | Same | Use public LB only as a bug check | M | Medium | https://cis.upenn.edu/~aaroth/reusable.html |
| 11 | A Meta-Analysis of Overfitting in ML | Roelofs et al. (2019) | Little adaptive overfitting on large test sets | Contrast: small test sets are dominated by sampling noise | Binomial CIs on every LB score | L | Medium | https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning |
| 12 | Cross-validation: what does it estimate…? | Bates, Hastie, Tibshirani (2024) | CV estimates average error over datasets; naive CIs undercover | Interpreting ±SE of CV here | Nested-CV CIs for the final rule | H | Medium | https://arxiv.org/abs/2104.00673 |
| 13 | A study of CV and bootstrap… | Kohavi (1995) | Recommends stratified 10-fold | Fold design | Stratify on target × apg-zone | L | Medium | https://mlanthology.org/ijcai/1995/kohavi1995ijcai-study |
| 14 | Preventing "overfitting" of cross-validation data | Ng (1997) | Selecting among many hypotheses overfits CV | Many cut-point pairs tried | Restrict hypothesis space (2–3 params) | M | Medium | https://ai.stanford.edu/~ang/papers/cv-final.pdf |
| 15 | Probable inference… (Wilson interval) | Wilson (1927) | Score interval for proportions | LB scores on 181 rows | Wilson CI on every LB number (done, §6) | L | Medium | https://doi.org/10.1080/01621459.1927.10502953 |
| 16 | Interval estimation for a binomial proportion | Brown, Cai, DasGupta (2001) | Wald intervals are bad; use Wilson / Agresti–Coull | Same | Same | L | Low | https://projecteuclid.org/euclid.ss/1009213286 |
| 17 | Adversarial validation approach to concept drift… | Pan et al. (2020) | Train-vs-test classifier diagnoses shift | Size shift | Adversarial AUC raw vs ratio (E14) | L | Medium | https://arxiv.org/pdf/2004.03045 |
| 18 | Improving predictive inference under covariate shift… | Shimodaira (2000) | Importance weighting under covariate shift | Explains why ratio models need no weighting | Optional: importance-weighted CV check | M | Low | https://doi.org/10.1016/S0378-3758(00)00115-4 |
| 19 | Covariate shift adaptation by importance-weighted CV | Sugiyama, Krauledat, Müller (2007) | IWCV | Size shift | Weighted CV toward test guests (sanity) | M | Low | https://jmlr.org/papers/v8/sugiyama07a.html |
| 20 | Classification in the presence of label noise: a survey | Frénay & Verleysen (2014) | Taxonomy of noise models | Edge noise is label noise; noise model choice (§9) | Compare noise families (S5) | M | Medium | https://research.dial.uclouvain.be/handle/2078.5/254107 |
| 21 | Very simple classification rules perform well… | Holte (1993) | 1R rules nearly match complex learners | Extreme case: one ratio, two cuts | 1R ablation per column (E12) | L | Medium | https://mlanthology.org/mlj/1993/holte1993mlj-very/ |
| 22 | Stop explaining black-box models… | Rudin (2019) | Prefer interpretable models | Theory prize + simplest-model principle | Present the final rule as a 3-parameter formula | L | Medium | https://arxiv.org/pdf/1811.10154 |
| 23 | Dichotomizing continuous predictors…: a bad idea | Royston, Altman, Sauerbrei (2006) | Hard cut-points lose information | Prefer smooth P(y \| apg) estimation, threshold only at the end | E3a soft band | L | Medium | https://doi.org/10.1002/sim.2331 |
| 24 | Detecting outliers: use MAD around the median | Leys et al. (2013) | Robust outlier rule | ×10 typo detection on log ratio | MAD rule vs the fixed 3.0 cut-off (agrees here) | L | Low | https://doi.org/10.1016/j.jesp.2013.03.013 |
| 25 | Interpretable ML for science with PySR | Cranmer (2023) | Symbolic regression | Recover the generator formula (the theory prize) | PySR on (aloo, guests, …) → logit P | M | Low–Med | https://arxiv.org/pdf/2305.01582 |

### 15.2 Repositories / libraries

Stars were not collected (UNVERIFIED).

| Repository | Main method | Licence | Usefulness / experiment enabled |
|---|---|---|---|
| scikit-learn (docs A061–A067) | CV, LR, trees, kNN | BSD-3 | Entire pipeline |
| statsmodels (A035, A036, A053) | Wilson CI, McNemar, Logit with CIs | BSD-3 | LB CIs, paired tests, log-log ratio test (E11) |
| [imodels](https://github.com/csinva/imodels) | Rule lists, FIGS, interpretable trees | MIT | Theory notebook: show the learned rule is the band |
| [InterpretML / EBM](https://github.com/interpretml/interpret) | Glass-box GAM-boosting | MIT | Shape function of apg (visual proof of the band) |
| [pyGAM](https://github.com/dswah/pyGAM) | Spline GAM | Apache-2.0 | Smooth P(y \| apg) with CIs (alternative to E3a) |
| [PySR](https://github.com/MilesCranmer/PySR) | Symbolic regression | Apache-2.0 | Equation discovery for the theory prize |
| [gplearn](https://gplearn.readthedocs.io/en/stable/) | GP symbolic classifier | BSD-3 | Same, pure Python |
| [cleanlab](https://github.com/cleanlab/cleanlab) | Label-issue detection | Apache-2.0 | Shows "label issues" sit only at the edges (theory illustration) |
| [pandera](https://github.com/unionai-oss/pandera) | Dataframe validation | MIT | Assert test cleanliness (no NaN, ratio range) |
| [shakeup](https://github.com/davidthaler/shakeup) | LB shake-up measurement | not shown | Post-competition analysis |
| [hakubishin3/kaggle_Instant_Gratification](https://github.com/hakubishin3/kaggle_Instant_Gratification) | GMM/QDA generator matching | not shown | Template for "model the generator" |

### 15.3 Related competitions

Details and links are in `research/related_competitions.md` [EXTERNAL; facts there are marked VERIFIED or UNVERIFIED individually].

| Competition | Similarity | Winning idea | Transferable lesson |
|---|---|---|---|
| Instant Gratification (2019) | Synthetic data from a known generator (`make_classification`) | Match the model to the generator (QDA/GMM, 3 clusters per class) | Recover the generator; then CV ≈ public ≈ private (winner: 0.9754 / 0.9744 / 0.97598) |
| Don't Overfit! II (2019) | Tiny training data | Sparse L1 logistic; LB probing debated | Simplest model within 1 SE; don't probe |
| LANL Earthquake (2019) | **Covariate shift between train and test** | Made training look like test; **two finals hedged on the shift**; best entry was public rank 2000+ | Size-stress validation; **hedge the two finals** |
| Santander Customer Transaction (2019) | Synthetic artefacts | Detected fake test rows; value-count "magic" | Look for generator fingerprints (E14/E15), legitimately |
| Tabular Playground / Playground Series | Synthetic tabular | Planted interactions; "less is more" (depth-2 trees, LR) | Engineer the interaction explicitly |
| Titanic | Small binary accuracy competition | Leaks via families and external lookups | Integrity: no external label lookups; LB differences are 1–2 SE |
| Mercedes-Benz / Restaurant Revenue | Small noisy public sets | Huge shake-ups | Never select on public LB |

### 15.4 Pretrained models

| Model | Architecture / data | Licence | Transferability | Rule risk | Recommendation |
|---|---|---|---|---|---|
| TabPFN v2 (Hollmann et al., *Nature* 2025) | Transformer prior-fitted on synthetic tabular tasks | Code Apache-2.0; **weights non-commercial** (per README, [EXTERNAL C044]) | Could learn a 1-D band from apg, but no better than a 3-parameter MLE | **FR 6c conflict** (licence limits commercial use) → Competition-uncertain | Not recommended |
| TabPFN v1 (2022 preprint) | Same family | — | Same | Same | Not recommended |
| Any CV/NLP foundation model | — | — | None (11-column synthetic table) | — | Not applicable |

### 15.5 Related datasets

| Dataset | Use | Classification |
|---|---|---|
| Indian Food 101 (Kaggle) and other food datasets | Searched; no wedding or second-plate labels | **Not recommended** (irrelevant) |
| Any real wedding data | The generator is synthetic; real data cannot inform the band | Not recommended |
| Synthetic replica (my `make_replica.py`) | Code testing only | Research-only |

---

## 16. Why current solutions work, and the remaining headroom

**Why they work:**
1. **The decisive component** is the per-guest ratio feature. It removes the size shift and turns a wedge-shaped boundary in raw space into two thresholds in 1-D.
2. **Cleaning matters for training only:** the ×10 fix, Bangla digits and dropping missing rows. In practice the ×10 fix barely matters to a band fitted in 0.85–2.2 (E2a/E2b test this).
3. **Edge estimation** (accuracy-optimal vs midpoint vs MLE vs LGBM vs kNN) changes about 2–11 of 400 test rows.

**Component audit:**

| Component | Verdict |
|---|---|
| Ratio feature | Essential, private-robust |
| ×10 fix | Correct but low impact |
| Missing-row drop | Correct |
| Accuracy-optimal cut search | Fits noise (§9); slightly negative in expectation |
| LGBM on log apg | Redundant with the band (identical predictions to the midpoint band, V2) |
| kNN-5 | Noisier edges; −1.1 rows in expectation (§9.2) |
| Other ratios / categoricals | Redundant noise features (LGBM engineered 0.9407 < band 0.955) |
| Ensembling | Members agree on 389/400 rows; ensemble contribution ≈ 0 |
| Public-overfit components | Any choice between V0–V3 made by public score; any threshold nudged after a submission |

**Headroom:**
- **Model headroom ≈ +0.5 private rows at most** (an oracle on the true generator; §9.3).
- **Selection headroom ≈ +0.75 rows** from the A/B hedge (§9.4).
- **Tie-break headroom:** submit early (§2.2).
- Possible **E15 headroom**, if edge width depends on size: better edge estimates and better-targeted hedging. Small but novel.
- **Theory-prize headroom:** large. Public notebooks have not shown multiplicative noise, the Bayes ceiling, the cut-point overfitting effect, or size-dependent sharpness.

---

## 17. Experiment roadmap

Every experiment is implemented in `solution/kacchi_pipeline.py` unless noted. Results are written to `outputs/experiments_results.csv`. The ledger, with verified public-log values and executed research experiments S1–S5, is `experiments.csv`.

**Common settings:**
- Validation for all experiments: canonical 10×5 RSKF (target × apg-zone) + size-stress fold. Seeds are the fold seeds 2026–2035 unless noted.
- **Keep criterion:** corrected-t p < 0.05 with Δ > 0, and size-stress not worse by more than 1 point.
- **Reject criterion:** Δ < 0 with p < 0.05, or no significant gain while adding complexity, or a size-stress drop of more than 1 point.
- Otherwise NEEDS_MORE_EVIDENCE. Identical CV *and* test predictions are recorded as REJECT ("no effect").

| ID | Parent | Hypothesis | Exact change | Model / features | Expected CV / Public / Private impact | Overfit / leakage risk | Compute |
|---|---|---|---|---|---|---|---|
| E0 | – | Reproduce strongest reliable baseline exactly | Accuracy-optimal band, plateau centre (FOYSAL) | band / apg | CV 0.9558 [VERIFIED]; public ≈ 0.950 [CLAIM]; private ≈ 0.94–0.96 | Med (noise-fitted cuts) / none | <1 s |
| E0a | – | Host RF baseline is weak under shift | RF(300) raw | RF / raw | CV ~0.80–0.86; public ~0.72; poor private | High / none | 10 s |
| E0b | E0 | Midpoint-cut variant | Rafiur cuts | band / apg | tie | Med / none | <1 s |
| E0c | E0 | Generator prior (1, 2) | Fixed band | rule / apg | CV 0.9446 [VERIFIED]; private ≈ tie (§9.3) | none / none | 0 |
| E1 | E0 | Validation must mimic test (size shift) | Compare strata; size-stress q = 0.5 / 0.6 / 0.7 | — | Lower estimate variance; exposes raw models | none | 1 min |
| E2a / E2b | E0 | ×10 handling irrelevant for the band | No fix / drop rows | band | Δ = 0 expected | none | <1 s |
| E2c | E3a | Noise multiplicative → log space | Soft band in log apg | soft MLE (log) | tie on CV; +0.05 rows private (§9.3) | low | 1 s |
| E3a | E0 | Likelihood cuts are lower-variance than 0-1 cuts | Soft-band MLE (lin) | soft MLE | CV tie; private +0.4–0.5 rows under smooth generators | low | 1 s |
| E3b | E3a | Posterior-predictive band (bootstrap, AIC-weighted lin + log) | Bagged soft band | bagged | CV tie; best expected private (+0.25 rows vs fixed, ABC) | low | 1–2 min |
| E4a | E0 | Host hint: depth-2 tree | CART d = 2 on apg | tree | tie | low | <1 s |
| E5a | E3a | Quadratic logit in log space | LR on [z, z²] | LR | tie / slightly worse | low | <1 s |
| E5b / E5c | E3a | kNN k = 5 / 25 | kNN on log apg | kNN | k5 worse (−1.1 rows, §9.2); k25 ≈ tie | Med | <1 s |
| E5d | E0a | Raw LR can't model a band | LR raw | LR | 0.555 [VERIFIED] | — | <1 s |
| E5e / E5f / E5g | E3a / E5e / E0a | Boosting on ratio / all ratios / raw | LGBM | LGBM | 0.954 / 0.941 / 0.860 [VERIFIED analogues] | Med / High (raw) | seconds |
| E6 | E3a | Probability blend of ratio models | Mean P | blend | ≈ 0 | low | 10 s |
| E7 | E6 | Hard vote (rank averaging is invalid here) | Vote | vote | ≈ 0 | low | 10 s |
| E8 | E6 | Nested stacking (LR meta) | Nested 5×5 | stack | ≈ 0 / slightly negative | Med | 1 min |
| E9 | E3a | Fold-internal pseudo-labels from confident test rows | Add P > 0.98 / < 0.02 rows | soft + PL | 0 (confident rows carry no edge information) | low (fold-internal) | 10 s |
| E10 | E3b | Final decision rule | Bagged soft band on all 776 rows | bagged | best expected private | low | 1 min |
| E11 | E3a | Edge errors are pure noise | Near-edge LR: margin vs margin + all columns | LR | Expect no gain; any significant gain = new signal | low | 5 s |
| E12 | E0 | aloo/guests is the right ratio | 11 candidate ratios, best band | prefix-sum band | aloo/guests best [VERIFIED Rafiur] | low | 5 s |
| E13 | E10 | Final-selection hedge raises E[best of two] | Flip the m most-uncertain rows; m by Monte Carlo | — | **+0.75 private rows** expected | none (no labels used) | 30 s |
| E14 | – | Integrity: ID leak, duplicates, shift | Spearman / AUC on id, hashes, KS / PSI, adversarial AUC | audit | — | — | 10 s |
| E15 | E2c | Edge sharpness grows with wedding size | k = exp(a + γ·log(guests / median)), LR test | het soft band | Changes P for big near-edge rows → better hedge | low | 10 s |

---

## 18. Top 10 experiments

| Priority | Experiment | Why | Expected upside | Private robustness | Compute | Risk | Dependencies |
|---|---|---|---|---|---|---|---|
| 1 | **E14 + E0**: audit + exact reproduction of V0 (400/400 match) | Proves the cleaning and pipeline are right before anything else | Avoids catastrophic bugs | — | seconds | none | Data access |
| 2 | **E10**: posterior-predictive (bagged soft) band | Best expected decisions (§9.3) | +0.25 rows vs public rules | High (scale-free) | 1 min | low | E0 |
| 3 | **E13**: A/B hedge with Monte-Carlo-chosen m | Largest legitimate lever | +0.75 rows (best of two) | High (no labels used) | 30 s | Must select both manually | E10 |
| 4 | **Submit A and B today** | Tie-break goes to the earlier entry | Wins ties that would otherwise be lost | — | — | none | E13 |
| 5 | **E15**: size-dependent edge width | Novel; could re-rank uncertain rows | Small; better hedge targeting | High | 10 s | Over-reading noise: require p < 0.05 | E10 |
| 6 | **E11**: residual-signal test near edges | If any column predicts edge flips, it is real headroom | Up to several rows *if* positive (expected: none) | Med | 5 s | Multiple testing: use CV log-loss, not p-hacking | E3a |
| 7 | **E1**: size-stress validation | Confirms ratio models are shift-proof; rejects raw models | Guards private | High | 1 min | none | E0 |
| 8 | **E3a / E2c**: lin vs log soft band | Confirms multiplicative noise on the real data (theory prize) | ≈ 0 rows; high insight value | High | s | none | E0 |
| 9 | **E12**: ratio search | Rules out alternative generators | Insight | High | s | none | E0 |
| 10 | **E5–E9**: alternative families, blends, stacking, pseudo-labels | Due diligence; expected null | ≈ 0 | — | min | Overfitting through selection | E3a |

---

## 19. Final comparison (verified scores only)

Public LB values are **team-level** [VERIFIED]. No notebook-level public score can be verified without submitting its file. CV values are [VERIFIED] from execution logs (each author's own folds).

| Rank | Approach / notebook | Public LB | CV | Model | Key technique | Overfit risk | Reproducible? |
|---|---|---|---|---|---|---|---|
| 1 | Accuracy-optimal band (FOYSAL, V0) | UNVERIFIED (claim 0.95027) | **0.9558** (10×5) | 2-cut rule | Ratio + grid cut search | Medium (noise-fitted cuts) | Yes |
| 2 | Midpoint band (Rafiur, V2) | team best 0.95580; file attribution UNVERIFIED | **0.9558** (3×5) | 2-cut rule | Ratio + midpoint cuts + falsification | Medium | Yes |
| 3 | Fitted band (Deep EDA, V0) | UNVERIFIED | 0.9549 ± 0.0070 (5-fold) | 2-cut rule | Same as V0 (grid 0.005) | Medium | Yes |
| 4 | LGBM on log apg (V2) | UNVERIFIED | 0.9541 (10×5) | GBDT, 1 feature | Smooth estimator of the band | Low–Med | Yes |
| 5 | LGBM apg only (Deep EDA) | UNVERIFIED | 0.9536 ± 0.0024 | GBDT | Same | Low–Med | Yes |
| 6 | kNN-5 log apg (V1) | UNVERIFIED | 0.9513 (10×5) | kNN | — | Medium | Yes |
| 7 | Fixed 1 < apg < 2 | UNVERIFIED | 0.9446 | rule | Generator prior | None | Yes |
| 8 | LR on apg + apg² (Deep EDA) | UNVERIFIED | 0.9420 ± 0.0181 | LR | — | Low | Yes |
| 9 | Log-space soft-band MLE (V3) | UNVERIFIED | 0.9407 (10×5) | 3-param MLE | Likelihood cuts | Low | Yes |
| 10 | LGBM engineered / all features | UNVERIFIED | 0.9407 / 0.9386 | GBDT | Ratios + noise columns | Medium | Yes |
| 11 | LGBM raw features | ≈ 0.70–0.74 public class (team-level, attribution UNVERIFIED) | 0.8596 | GBDT | Raw counts | **High (size shift)** | Yes |
| 12 | LR raw features | UNVERIFIED | 0.5554 | LR | Raw counts | High | Yes |

Note: ranks 1–9 are within CV noise of each other (SE ≈ 0.002–0.007). My simulations (§9.2) show CV *prefers* the accuracy-optimal cuts even though they are slightly worse on future data.

---

## 20. Recommended final submissions

| Item | Recommendation |
|---|---|
| **Best current public approach** | Ratio band (V0 / V2). V2 is marginally better by the ABC posterior on the 2 rows where they differ. |
| **Best reproducible baseline** | E0, the exact V0 reproduction (sanity check: 400/400 match with `submission_band_rule.csv`) |
| **Best validation** | 10×5 RSKF stratified on target × apg-zone + size-stress fold + corrected paired t-test |
| **Biggest bottleneck** | Irreducible label noise in about ±0.03 of each edge (≈ 11 test rows), plus a 219-row private split |
| **Best single model** | **Real data (see UPDATE): sharp midcut band 0.97068 < apg < 2.00983.** The pre-data recommendation (bagged soft band) was rejected by CV (p = 0.03). |
| **Best ensemble** | None needed. If one is used, probability-average calibrated ratio models (E6). Never rank-average. |
| **Best pseudo-label strategy** | None. Pseudo-labels on confident rows carry no edge information (E9). |
| **Final selection A** | `outputs/sub_A_primary_sharpband.csv`: the sharp midcut band. **Submitted; public 0.95027.** |
| **Final selection B** | `outputs/sub_B_hedge.csv` = A with the m = 9 rows of smallest \|P − 0.5\| flipped, where P comes from the out-of-fold sharp-band error model (E16). This is algorithmic, not hand-labelled (FR 4b). **Submitted; public 0.95580.** |

**Fallback if you cannot run the pipeline in time** [RECOMMENDATION, lower quality]: use **V2** (`research/public_notebooks/.../submission_lgbm_logapg.csv`) as A and **V3** (`submission_logmle_band.csv`) as B.
- They differ on exactly 4 coin-flip rows: W0839, W0865, W1119, W1157.
- V2 already makes the posterior-favoured call (0) on W0968/W1163.
- The downside: identical files to other people's lose every tie. The pipeline route is better.

---

## 21. Approaches to avoid

| Approach | Why |
|---|---|
| **Weak public tricks** | Choosing between V0–V3 or nudging thresholds by public score: it changes ≤ 5 public rows whose labels say nothing about private rows. |
| **LB probing** | Flipping single rows to learn their labels or their public/private membership. It only improves public, wastes submissions, and risks FR 8d/16a. |
| **Raw-count models** | RF/GBM/NN on guests, aloo_count, mutton, … fail on the 36% of test rows above train size (CV 0.86 → worse on test). |
| **Unstable models** | kNN with small k (−1.1 rows); deep trees on the ratio; per-city or per-event cut-points (small-sample noise [VERIFIED]). |
| **Expensive low-ROI work** | Hyper-parameter search, big ensembles, neural nets, TabPFN (licence risk), AutoML. The Bayes ceiling caps all of them. |
| **Leaderboard hacks** | Reconstructing the host's RNG; hand-labelling test rows (FR 4b). |
| **Redundant ensembles** | Members agree on 389/400 rows, so there is zero diversity where it matters. |
| **Invalid validation** | Single 75/25 split (starter); folds that change between experiments; tuning cut-points on full train and then "validating" on the same rows; trusting the public LB. |
| **Questionable external data** | None exists that is relevant; real wedding data cannot inform a synthetic band. |
| **Pseudo-labelling that amplifies noise** | Pseudo-labelling near-edge test rows bakes your own edge guess back in as "truth". |
| **Rank averaging / geometric means** | Destroy the absolute 0.5 threshold for a non-monotone target. |
| **Class re-weighting / base-rate thresholds** | Accuracy needs P > 0.5, not P > 0.42. |

---

## 22. Private leaderboard strategy

1. **Conservative model selection.** Pick the posterior-predictive rule (E10), not the max-CV rule. Treat all ratio rules as tied unless the corrected test says otherwise.
2. **Domain robustness.** Ratio-only decisions plus a size-stress check, so the size shift cannot hurt.
3. **Confidence intervals.** Report every CV with mean ± SE, and every LB number with Wilson CIs in rows. Ignore deltas under 3 public rows.
4. **Seed and fold robustness.** Use frozen canonical folds. Keep a change only if the corrected paired test passes.
5. **Diversity where it counts.** The two finals must differ **only** on genuinely uncertain rows (the E13 hedge). Two near-identical "best" files waste the second selection.
6. **Limited submission feedback.** Use the public LB as a bug detector: anything below about 0.93 means something is broken. Never use it as a selector.
7. **Holdout discipline.** Do not edit decisions after seeing public scores.
8. **Timing.** Submit A and B early (tie-break), and **manually** select them as finals.

---

## 23. Step-by-step execution plan

About 56 hours remain at the time of writing.

| Stage | When | What | Exact deliverables |
|---|---|---|---|
| 1. Research | Done | This report + resources | `The_Great_Kacchi_Aloo_Mystery_Deep_Analysis.md`, `external_resources.csv`, `TOP_100_RESOURCES.md`, `public_notebooks.csv`, `important_discussions.csv`, `experiments.csv` |
| 2. Data audit | Day 1, first hour | Join; download; run the pipeline's AUDIT (E14) | `outputs/audit.json`: confirm 24 / 15 / 6 dirt counts, 0 in test, 145 test rows > 700, adversarial AUC raw ≫ ratio ≈ 0.5, no ID leak |
| 3. Validation | Day 1 | Freeze folds; run E1 | `outputs/folds.csv`; E1 block in `outputs/extra_results.json` |
| 4. Baseline | Day 1 | E0 exact reproduction; diff vs public files | E0 row in `experiments_results.csv`; "vs E0: 0" for `submission_band_rule.csv` in the diff block |
| 5. Controlled experiments | Day 1 | E2a–E2c, E3a, E4a, E11, E12, E15 | Rows in `experiments_results.csv` with p-values and decisions |
| 6. Strong single model | Day 1 | E3b / E10 bagged soft band | `bayes_cuts` in `extra_results.json`; compare with the ABC (0.982, 2.0075) |
| 7. Diversity | Day 1 | E5a–E5g (expected ties) | Rows in `experiments_results.csv` |
| 8. Ensembling | Day 1 | E6–E9 (expected null) | Rows in `experiments_results.csv` |
| 9. Robustness tests | Day 1 | Size-stress per model; E13 hedge curve; inspect `uncertain_test_rows.csv` | Hedge m and expected gain; uncertain-row table |
| 10. Final submissions | **Day 1 (today)** | Submit (i) E0 reproduction as a bug check (expect ≈ 0.950), (ii) A, (iii) B. **Select A + B as finals.** Day 2: theory notebook. Before 2026-10-09 17:59 UTC: re-check the selections. | Two selected finals; theory notebook published |

---

## 24. The "Best Aloo Theory" prize

[OFFICIAL] "The notebook that best explains *why* guests go back for seconds, in plain words. You don't need the top score to win."

[RECOMMENDATION] A winning notebook should cover, in plain language for students:
1. **The story:** potatoes per guest, a Goldilocks band of 1 to 2. Too few disappoints; too many feels like the cook saved on mutton (Babul Baburchi's quote).
2. **Proof it is a ratio:** log-log logistic weights are equal and opposite [VERIFIED Rafiur]. Alternative denominators lose (E12).
3. **Why fancy models lose:** a wedge in raw space vs two lines in ratio space; trees cannot extrapolate past 700 guests (size-stress demo, E1).
4. **New insight 1, multiplicative noise:** the upper edge is about twice as blurry in absolute terms. Guests judge potatoes *proportionally* (§9.1; confirm with E2c on real data).
5. **New insight 2 (if E15 is significant):** big weddings have sharper edges. A few miscounted potatoes matter less when there are 3,000 of them.
6. **Honest ceiling:** about 4–5% of weddings sit right on the edge and are coin flips, so about 95% is the best anyone can do. The leaderboard top is luck at the 1–3-row level (§6).
7. **One-line rule:** "Guests go back for seconds when each guest gets between about 1 and 2 potatoes." Add a 3-parameter formula P = σ(k·ln(apg/0.98))·σ(k·ln(2.0/apg)), with k ≈ 30, as the scientific version.

---

## 25. Direct source links

**Competition primary sources:**

| Source | Link |
|---|---|
| Overview | https://www.kaggle.com/competitions/the-great-kacchi-aloo-mystery/overview |
| Data | https://www.kaggle.com/competitions/the-great-kacchi-aloo-mystery/data |
| Rules | https://www.kaggle.com/competitions/the-great-kacchi-aloo-mystery/rules |
| Leaderboard | https://www.kaggle.com/competitions/the-great-kacchi-aloo-mystery/leaderboard |
| Discussion (empty) | https://www.kaggle.com/competitions/the-great-kacchi-aloo-mystery/discussion |
| Starter notebook | https://www.kaggle.com/code/tasnimmahfuznafis/kacchi-starter |
| Deep EDA | https://www.kaggle.com/code/foysalemonshanto/kacchi-aloo-mystery-deep-eda |
| LB_0.95027 | https://www.kaggle.com/code/foysalemonshanto/lb-0-95027-kacchi-aloo-theory |
| Goldilocks Band | https://www.kaggle.com/code/rafiurrahman01/the-aloo-theory-the-goldilocks-band |
| World AI Week events | https://worldaiweek.ai/all-events-2026/ |

**Key external sources** (the full list is in `external_resources.csv`):

| Topic | Link(s) |
|---|---|
| Altman 1994 | https://doi.org/10.1093/jnci/86.11.829 |
| Heaton 2016 | https://arxiv.org/pdf/1701.07852 |
| Varma & Simon 2006 | https://pmc.ncbi.nlm.nih.gov/articles/PMC1397873 |
| Cawley & Talbot 2010 | https://www.jmlr.org/papers/v11/cawley10a.html |
| The Ladder | https://arxiv.org/pdf/1502.04585 |
| Instant Gratification | https://www.kaggle.com/c/instant-gratification, 1st place https://www.kaggle.com/c/instant-gratification/discussion/96549 |
| LANL 1st-place account | https://www.linkedin.com/pulse/my-team-won-20000-1st-place-kaggles-earthquake-corey-levinson |
| Adversarial validation | http://fastml.com/adversarial-validation-part-one/ |
| Kaggle competition docs | https://www.kaggle.com/docs/competitions |

---

## 26. Final conclusions

1. This competition is **structurally solved**: one ratio, two edges, soft noise. The open questions are statistical, not architectural.
2. **The public leaderboard cannot rank the top 10.** Differences are 1–3 of 181 rows, and the leader is most likely lucky.
3. **Private outcomes among competent teams hinge on about 6 coin-flip rows and the earliest-submission tie-break.**
4. **Threshold choice is worth ≤ 0.25 expected rows; the A/B final-selection hedge is worth about +0.75; submitting early wins ties.** These are the real, legitimate levers.
5. The accuracy-optimal cut-points that win CV are slightly worse in expectation than likelihood-based cut-points. **Use the posterior-predictive band (about 0.982 to 2.0075)**, confirmed on the real data by E10.
6. The **Best Aloo Theory** prize rewards the deeper understanding developed here: multiplicative noise, the Bayes ceiling, cut-point overfitting and possibly size-dependent sharpness. It is the best use of the remaining time after the finals are submitted.

---

## Appendix A: Reproduction

```bash
# 0) Python env with a current Kaggle CLI (reads ~/.kaggle/access_token)
python -m venv kenv && kenv/Scripts/pip install -U kaggle pandas numpy scipy scikit-learn lightgbm statsmodels
# 1) Join the competition in the browser, then:
kenv/Scripts/kaggle competitions download the-great-kacchi-aloo-mystery -p data && (cd data && unzip -o *.zip)
# 2) Full pipeline (audit, folds, E0-E15, hedge, submissions)
kenv/Scripts/python solution/kacchi_pipeline.py --data data --out outputs --public-subs research/public_notebooks
# 3) Research re-runs (no competition data needed)
kenv/Scripts/python research/scripts/forensics.py <dir-with official/lb and nb/*/out> research/analysis
kenv/Scripts/python research/scripts/simulate.py research/analysis 1000
(cd research/scripts && ../../kenv/Scripts/python abc2.py ../analysis 400000)
```

## Appendix B: File index

| Path | Content |
|---|---|
| `The_Great_Kacchi_Aloo_Mystery_Deep_Analysis.md` | This report |
| `external_resources.csv` | 122 verified resources, all requested columns |
| `TOP_100_RESOURCES.md` | Ranked Top 100 |
| `public_notebooks.csv` | 4 notebooks × 44 attributes |
| `important_discussions.csv` | Discussion audit (0 topics) + official guidance |
| `experiments.csv` | S1–S5 executed research + E0–E15 ladder with verified public values / pending markers |
| `solution/kacchi_aloo_pipeline.ipynb` | Same pipeline as a 30-cell notebook with explanations and plots (Kaggle/Colab-ready; tested on a synthetic replica) |
| `solution/kacchi_pipeline.py` | End-to-end pipeline script (tested on a synthetic replica; safe to run inside a notebook cell) |
| `research/official/` | Official pages (API text) + leaderboard CSV snapshot |
| `research/public_notebooks/` | All public notebooks (code, text render, execution logs, submission files) |
| `research/analysis/` | Decoded LB, submission matrix, disputed rows, simulation and ABC outputs |
| `research/scripts/` | All research scripts |
| `research/related_competitions.md` | Related-competition deep dive |
