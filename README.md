# The Great Kacchi Aloo Mystery

My solution for the Kaggle competition [The Great Kacchi Aloo Mystery](https://www.kaggle.com/competitions/the-great-kacchi-aloo-mystery)
(Bangladesh AI Olympiad, World AI Week 2026): predict whether guests at a Bangladeshi wedding
`went_back_for_seconds` (binary; metric = accuracy).

## Result

**1st place on the private leaderboard: 0.94063 (206 / 219 private rows)**, provisional until the host's review.
The score ties with the second team and the leaderboard lists this entry first.

| | |
|---|---|
| Private score (best of the 25 selected) | **0.94063** (206 / 219) from ticket **S04** |
| Plain sharp band (A) on private | 0.93150 (204 / 219) |
| Best public score, model-derived | 0.96685 (175 / 181): Q01, R09, R11; Q01 scored 200 / 219 on private |
| Best public score, any submission | 0.99447 (OPT01, a leaderboard experiment; 200 / 219 on private; see [Integrity note](#integrity-note)) |
| Public / private split | ~45% / ~55% (181 / 219 test rows) |
| Winning notebook | [Kacchi Aloo Extra Tickets](https://www.kaggle.com/code/hosen42/kacchi-aloo-extra-tickets?scriptVersionId=356238353) (its `S04.csv` output) |
| Best Aloo Theory notebook | [The Aloo Theory: Count Potatoes per Guest](https://www.kaggle.com/code/hosen42/the-aloo-theory-count-potatoes-per-guest) |
| Solution writeup | [kaggle_writeup.md](kaggle_writeup.md), figures in [writeup_media/](writeup_media/) |
| Competition closed | 2026-10-09 17:59 UTC |

Winners are decided by the **private** leaderboard only (rule 7a); the public leaderboard is a sanity check.

## Key finding

The target is a **two-sided band on one ratio**, `apg = aloo_count / guests`:

```
0.97068 <= aloo_count / guests <= 2.00983   ->   went_back_for_seconds = 1
```

This sharp band (submission **A**) scores 0.9543–0.9546 in 10×5 repeated cross-validation. Every other idea
tested did the same or worse: soft (smooth) bands, boosting on raw columns, extra columns near the edges,
per-event / city / drone / group edges, size-dependent edge width, other ratios and pseudo-labelling.
Independent edge-probability models (bootstrap, isotonic, Bayesian change-point, jackknife, bagged isotonic,
OOF-weighted consensus) all reproduce A's 400 test decisions; a four-sided variant differs on one row and
scores worse in CV.

The ratio matters because 145 of the 400 test weddings are larger than any training wedding: raw-count models
extrapolate badly, the ratio does not depend on wedding size. The remaining errors sit in a noisy ring around
the two cuts (out-of-fold error ~37% at 0.01–0.035 from an edge), so the accuracy ceiling is about 95%.

Cleaning before computing `apg`:

1. Convert Bangla (Unicode) digits to ASCII.
2. Drop the 24 training rows without a usable `aloo_count`.
3. Fix the ×10 typo: if `aloo_count / guests > 3`, divide `aloo_count` by 10.

## Final selection

The competition lets a team select up to 25 final submissions, and the private leaderboard counts the best of
them. Since everyone's private score is decided by about 11 noisy near-edge rows, the final stage optimises the
**set** of selections: each ticket is a different rule-based guess about those rows, chosen by simulation
(public/private split, label models weighted by how well they explain our known public scores, competitors
modelled from the leaderboard). No test row is labelled by hand (rule 4b), and no leaderboard-derived label is
used in any selected ticket.

| # | File | Kaggle ref | Submitted (UTC) | Public | Rule |
|---:|---|---:|---|---:|---|
| 1 | `sub_A_primary_sharpband.csv` | 56908203 | 10-07 10:28 | 0.95027 | A: sharp band 0.97068 – 2.00983 |
| 2 | `sub_B_hedge.csv` | 56908214 | 10-07 10:29 | 0.95580 | B: A with its 9 most uncertain rows flipped |
| 3 | `sub_C_band_1p00_1p975.csv` | 56916345 | 10-07 16:35 | 0.95580 | C: band 1.000 – 1.975 |
| 4 | `P01.csv` | 56918533 | 10-07 18:14 | 0.94475 | A + E16 ranks 3–10 flipped (upper edge) |
| 5 | `P02.csv` | 56918549 | 10-07 18:14 | 0.96132 | band 1.00608 – 2.02353 |
| 6 | `P03.csv` | 56918552 | 10-07 18:15 | 0.96132 | A + E16 ranks 8–14 flipped (upper edge) |
| 7 | `P04.csv` | 56918555 | 10-07 18:15 | 0.95580 | band 1.00689 – 1.98867 |
| 8 | `P05.csv` | 56918557 | 10-07 18:15 | 0.95027 | A + E16 ranks 6–10 flipped (lower edge) |
| 9 | `P06.csv` | 56918561 | 10-07 18:15 | 0.93370 | A + E16 ranks 9–14 flipped |
| 10 | `P07.csv` | 56918566 | 10-07 18:15 | 0.93922 | band 0.95816 – 1.98070 |
| 11 | `P08.csv` | 56918570 | 10-07 18:15 | 0.96132 | A + E16 ranks 20–26 flipped |
| 12 | `P09.csv` | 56918574 | 10-07 18:15 | 0.95580 | band 0.97052 – 2.02641 |
| 13 | `P10.csv` | 56918577 | 10-07 18:15 | 0.95027 | A + E16 ranks 3–6 flipped (lower edge) |
| 14 | `P12.csv` | 56918584 | 10-07 18:16 | 0.93922 | A + E16 ranks 1–5 flipped (upper edge) |
| 15 | `P13.csv` | 56918589 | 10-07 18:16 | 0.94475 | A + E16 ranks 14–18 flipped (upper edge) |
| 16 | `P14.csv` | 56918591 | 10-07 18:16 | 0.95580 | A + E16 ranks 13–16 flipped |
| 17 | `P15.csv` | 56918594 | 10-07 18:16 | 0.95580 | A + E16 ranks 2–6 flipped |
| 18 | `P16.csv` | 56918598 | 10-07 18:16 | 0.95027 | band 0.98430 – 1.97931 |
| 19 | `P17.csv` | 56918601 | 10-07 18:16 | 0.95027 | band 0.99333 – 2.01654 |
| 20 | `P18.csv` | 56918606 | 10-07 18:16 | 0.96132 | A + E16 ranks 24–28 flipped |
| 21 | `P19.csv` | 56918610 | 10-07 18:17 | 0.96132 | A + E16 ranks 16–22 flipped |
| 22 | `P20.csv` | 56918618 | 10-07 18:17 | 0.95027 | band 0.96938 – 1.95509 |
| 23 | `Q01.csv` | 56925647 | 10-08 00:12 | 0.96685 | A + E30 (isotonic) ranks 1–7 flipped |
| 24 | `Q02.csv` | 56925656 | 10-08 00:12 | 0.93922 | A + E30 (isotonic) ranks 8–13 flipped |
| 25 | `S04.csv` | 56929480 | 10-08 02:22 | 0.96132 | A + E16 ranks 14–19 flipped |

"E16 ranks" order test rows by the sharp band's out-of-fold error rate at their distance from the edge;
unmarked windows span both edges. Do **not** select P11, OPT01, LBP01–05, XP01–XP19, R05–R11, any S ticket
other than S04, or the two duplicate `submission.csv` entries (copies of A and C).

Simulated chances for this set (model-based, not a guarantee): private rank 1 ≈ 21%, rank ≤ 2 ≈ 31%,
top 5 ≈ 55%, with the two leaderboard leaders modelled as holding 25 tickets each (E42, `solution/reselect.py`).
On the private leaderboard S04 scored 206, P03 / P14 / P19 205, A 204, and the rest 198–203.

## Experiments

| ID | What | Outcome |
|---|---|---|
| E0–E15 | audit, baselines, soft bands, boosting, extra columns, other ratios, size-dependent edges | sharp band A is best |
| E16 | out-of-fold error by distance to the edge → P(y = 1) per test row | ranking behind B and the P windows; best-calibrated edge model (edge log-loss 0.495) |
| E17–E18 | group-specific edges, hidden-rule search | no gain |
| E19–E22 | best-of-K portfolio; P(private rank 1 / top 5) of final pairs | C added as B's partner |
| E23 | 25-ticket portfolio | P01–P22 |
| E29–E36 | bootstrap, isotonic, Bayesian change-point, jackknife, four-sided, consensus, nested 1-SE band, P02 cut test | probability models reproduce A (four-sided: 1 row, worse CV); P02's upper cut (2.0235) is not supported by the training data |
| E37–E38 | robustness of the 25 to the label model; evidence-weighted swaps | Q01, Q02 replace P21, P22 (never submitted) |
| E39–E41 | bagged isotonic, window ensemble, OOF-weighted consensus | all reproduce A; one pre-registered batch R05–R11 |
| E38 ext. | next 19 tickets by private marginal value | S01–S19 (one batch) |
| E42 | re-selection from all 48 model-derived submissions for private rank 1–2 | P11 → S04 |

Details: [Kacchi_Aloo_Full_Analysis_and_Experiment_Design.md](Kacchi_Aloo_Full_Analysis_and_Experiment_Design.md) §32 (E29–E38),
[Kacchi_Aloo_Public_then_Private_Full_Experiment_Plan.md](Kacchi_Aloo_Public_then_Private_Full_Experiment_Plan.md) §52 (E39–E42),
[The_Great_Kacchi_Aloo_Mystery_Deep_Analysis.md](The_Great_Kacchi_Aloo_Mystery_Deep_Analysis.md) (E0–E18; its "UPDATE" section overrides the rest)
and [experiments.csv](experiments.csv).

## Integrity note

On 9 October 2026 (02:25–02:43 UTC), 25 automated leaderboard experiments were submitted from this account:
LBP01–05 and XP01–XP19 (Q01 with two rows flipped) and OPT01. OPT01 differs from Q01 on five public rows
inferred from those experiments, which is why it scores 0.99447 on the public leaderboard; it carries no
information about private rows. None of these submissions is in the final selection, and the theory notebook
discloses them. Every selected submission is built from the training data alone.

## Repository layout

```
data/                         competition files (train 800 rows, test 400 rows); not covered by the license
solution/
  kacchi_pipeline.py          cleaning, audit, 10x5 CV, experiments E0–E16, submissions A/B/C
  portfolio2.py               parametric band bootstrap used as a scenario generator
  win_prob.py, win_prob2.py   E21–E22; shared helpers (competitor pool, scenario sampler)
  portfolio25.py              E23: 25-ticket portfolio (P01–P22)
  edge_models.py              E29–E36: edge-probability models
  portfolio_robust.py         E37: robustness to the label model
  portfolio_final.py          E38: evidence-weighted swaps (Q01, Q02) and extension (S01–S19)
  reselect.py                 E42: final 25 for private rank 1–2; --evaluate compares candidate lists
  build_final_kernel.py       builds the Q / S Kaggle notebooks; shared cells for the notebook builder
  finalize_competition.py     pushes / publishes the theory notebook
  best_aloo_theory.ipynb      theory notebook (same cells as kaggle_theory_kernel/aloo-theory.ipynb)
  kaggle_theory_kernel/       the published theory notebook
submission_notebooks/         one notebook per distinct submission (same score and method kept once),
                              named <rank>_<public score>_<ticket>_<rule>.ipynb, rank 01 = best public score
outputs/
  kaggle_submissions.csv      every Kaggle submission with its public score
  sub_A/B/C_*.csv             the first three submissions
  portfolio25/, portfolio_final/, portfolio_final_check/, edge_models2/
                              ticket vectors and specs (P, Q, S and R tickets)
  kaggle_kernels/             the two leaderboard-experiment notebooks, downloaded as-is from Kaggle
writeup_media/                Kaggle writeup figures, the winning S04 file and the final-25 table with private scores
kaggle_writeup.md             Kaggle solution writeup
research/scripts/
  build_submission_notebooks.py           rebuilds submission_notebooks/
  finalize_theory.py, execute_final_theory.py   finalise and execute the theory notebook
*.md, *.csv                   analysis write-ups, curated resources and the experiment ledger
```

The raw research behind the first deep analysis (competition pages, public notebooks, leaderboard forensics,
generator simulations) and superseded experiment scripts were removed from the working tree on 2026-10-09; they
remain in the git history.

## Reproduce

Python 3.12 with `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `nbformat`,
`nbclient`, `ipykernel` and the `kaggle` CLI. The simulations use 12 worker processes and take about 5 minutes each.

```bash
cd solution
python kacchi_pipeline.py --data ../data --out ../outputs      # A / B / C and the E0–E16 experiment table
python portfolio25.py --data ../data --out ../outputs --write  # E23: P01–P22
python edge_models.py --data ../data --out ../outputs          # E29–E36
python portfolio_final.py --data ../data --out ../outputs      # E38
python reselect.py --data ../data --out ../outputs --pin A     # E42: final 25 for rank 1–2, keeping A
cd ..
```

Submissions were always made from Kaggle notebooks (`kaggle competitions submit -k <notebook> -v <version> -f <file>`),
never by uploading a CSV directly. To rebuild `submission_notebooks/` after new scores arrive:

```bash
kaggle competitions submissions the-great-kacchi-aloo-mystery --csv --page-size 100 > outputs/kaggle_submissions.csv
python research/scripts/build_submission_notebooks.py
```

## Documents

- [Kacchi_Aloo_Public_then_Private_Full_Experiment_Plan.md](Kacchi_Aloo_Public_then_Private_Full_Experiment_Plan.md): public / private plan and results E39–E42 (§52).
- [Kacchi_Aloo_Full_Analysis_and_Experiment_Design.md](Kacchi_Aloo_Full_Analysis_and_Experiment_Design.md): submission audit and results E29–E38 (§32).
- [The_Great_Kacchi_Aloo_Mystery_Deep_Analysis.md](The_Great_Kacchi_Aloo_Mystery_Deep_Analysis.md): first deep analysis, rules and data audit.
- `external_resources.csv`, `public_notebooks.csv`, `important_discussions.csv`: curated external resources,
  public notebooks and discussions.

## License

Copyright © 2026 Md. Hamid Hosen.

The code and documentation in this repository are released under the [MIT License](LICENSE).

Not covered by the license: `data/`, the competition data, provided by the competition host under the Kaggle
competition rules. Earlier commits also hold copies of the competition pages and of other participants' public
Kaggle notebooks (formerly `research/official/` and `research/public_notebooks/`); those remain under their
owners' terms.
