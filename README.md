# The Great Kacchi Aloo Mystery

My work on the Kaggle competition [The Great Kacchi Aloo Mystery](https://www.kaggle.com/competitions/the-great-kacchi-aloo-mystery):
predict whether guests at a Bangladeshi wedding `went_back_for_seconds` (binary, metric = accuracy).

> **Keep this repository private until the competition ends (2026-10-09 17:59 UTC).**
> The competition rules (2d, 3a) forbid sharing code or data outside Kaggle during the competition.
> `data/` also holds the competition files.

## Status (2026-10-08)

| | |
|---|---|
| Best public score | **0.96132** (174 / 181 public rows) |
| Public rank | #6 (leader 0.96685 = 175 / 181) |
| Public / private split | ~45% / 55% (181 / ~219 test rows; 1 row = 0.55 points public) |
| Submissions | A, B, C + portfolio tickets P01–P22 (all from Kaggle notebooks) |
| Final selections | up to 25: A, B, C, P01–P22 |
| Deadline | 2026-10-09 17:59 UTC |

## Key finding

The target is a **two-sided band on one ratio**, `apg = aloo_count / guests`. The best single rule is the
sharp band:

```
0.97068 <= aloo_count / guests <= 2.00983   ->   went_back_for_seconds = 1
```

It scores 0.9546 in 10x5 repeated CV. Nothing else beat it. The remaining errors sit in a narrow noisy
ring around the two cuts. These ideas were all tested and rejected:

- soft (smooth) bands
- boosting on the raw columns
- adding the other columns near the edges
- per-event / city / drone / group edges
- size-dependent edge width
- other ratios
- pseudo-labelling

The ratio also matters because test weddings are larger than any training wedding. Models on raw counts
extrapolate badly; the ratio does not depend on wedding size.

Data cleaning before computing `apg`:

1. Normalise Bangla (Unicode) digits to ASCII.
2. Drop the 24 training rows without a usable `aloo_count`.
3. Fix the x10 typo: if `aloo_count / guests > 3`, divide `aloo_count` by 10.

Because the private ranking is decided by roughly 11 near-edge rows, the final stage optimises the
**set of final selections** rather than a single model. Kaggle scores a team by its best selected
submission on the private leaderboard, so each ticket is a different rule-based guess about the edge
rows. No test row is labelled by hand (rule 4b).

## Submissions

| Ticket | Rule | Public |
|---|---|---|
| **A** | sharp band 0.97068 – 2.00983 | 0.95027 |
| **B** | A with its 9 most uncertain rows flipped (hedge) | 0.95580 |
| **C** | band 1.000 – 1.975 | 0.95580 |
| P02 | band 1.00608 – 2.02353 | 0.96132 |
| P03, P08, P18, P19 | A with uncertainty ranks flipped (8–14 upper, 20–26, 24–28, 16–22 both edges) | 0.96132 |
| P01, P04–P07, P09–P17, P20 | other band / flip tickets | 0.93370 – 0.95580 |
| P21, P22 | bands 1.00347 – 1.98148 and 0.96462 – 2.01953 | pending |

Each submission has a self-contained notebook in [submission_notebooks/](submission_notebooks/), named
`<rank>_<public score>_<ticket>_<rule>.ipynb`. Every notebook reproduces its submitted file exactly.

## Repository layout

```
data/                      competition files (train 800 rows, test 400 rows)
solution/                  pipeline, experiments and Kaggle kernels
  kacchi_pipeline.py       audit, 10x5 CV, experiments E0–E14, submissions A/B/C
  conditional_cuts.py      E17  group-specific edges
  hidden_rule_search.py    E18  hidden-rule search on modified ratios
  portfolio.py, portfolio2.py         E19–E20  best-of-K final-selection portfolio
  win_prob.py, win_prob2.py           E21–E22  P(private rank 1 / top 5) for the final pair
  portfolio25.py           E23  25-ticket portfolio (P01–P22)
  edge_models.py           E29–E36  edge-probability models (bootstrap, isotonic, Bayesian change-point, ...)
  portfolio_robust.py      E37  robustness of the 25 finals to the label model
  portfolio_final.py       E38  evidence-weighted swap of final tickets
  kaggle_*_kernel/         notebooks pushed to Kaggle (band, portfolio, theory)
submission_notebooks/      one reproducible notebook per submission, ranked by public score
outputs/                   CV results, submission CSVs, Kaggle submission log, portfolio files
research/                  official pages, public notebooks, leaderboard forensics, helper scripts
experiments.csv            experiment ledger (S1–S5, E0–E15)
*.md                       analysis write-ups (see below)
```

## Documents

- [Kacchi_Aloo_Full_Analysis_and_Experiment_Design.md](Kacchi_Aloo_Full_Analysis_and_Experiment_Design.md):
  latest full analysis, submission audit and the E29+ experiment design.
- [The_Great_Kacchi_Aloo_Mystery_Deep_Analysis.md](The_Great_Kacchi_Aloo_Mystery_Deep_Analysis.md):
  first deep analysis (its "UPDATE" section overrides the rest).
- [TOP_100_RESOURCES.md](TOP_100_RESOURCES.md): curated external resources. Also
  `external_resources.csv`, `public_notebooks.csv` and `important_discussions.csv`.

## Reproduce

Python 3.11+ with `numpy`, `pandas`, `scipy`, `scikit-learn` and `nbformat`. Run the scripts from inside `solution/`:

```bash
cd solution
python kacchi_pipeline.py --data ../data --out ../outputs   # A/B/C + experiment table (--repeats 3 for a quick run)
python portfolio25.py --data ../data --out ../outputs --write
python edge_models.py --data ../data --out ../outputs
python portfolio_robust.py --data ../data --out ../outputs
```

Submit through Kaggle notebooks, not by uploading CSVs directly. For example:

```bash
python build_portfolio_kernel.py
kaggle kernels push -p kaggle_portfolio_kernel
```

To rebuild `submission_notebooks/` after new scores arrive (run from the repository root):

```bash
kaggle competitions submissions the-great-kacchi-aloo-mystery --csv --page-size 100 > outputs/kaggle_submissions.csv
python research/scripts/build_submission_notebooks.py
```