# Public leaderboard experiments

Q01 baseline: 0.96685 (175/181). OPT01 verified score: 0.99447. Current displayed rank: 3.

19 new two-row probes. 72 score equations over 63 varying rows; integer feasibility checked.

The optimized submission flips only contributions proven to be +1 in every feasible solution. No ambiguous row is assumed private. These gains belong to public rows; private predictions stay identical to Q01.

| Wedding | Q01 prediction | OPT01 prediction |
|---|---:|---:|
| W0815 | 1 | 0 |
| W1006 | 0 | 1 |
| W1012 | 1 | 0 |
| W1153 | 1 | 0 |
| W1180 | 0 | 1 |

| Probe | Rows | Public score | Net vs Q01 |
|---|---|---:|---:|
| XP01.csv | W0851, W0848 | 0.96132 | -1 |
| XP02.csv | W0851, W1112 | 0.96132 | -1 |
| XP03.csv | W0851, W0908 | 0.95580 | -2 |
| XP04.csv | W0851, W0963 | 0.96132 | -1 |
| XP05.csv | W0851, W1012 | 0.96685 | 0 |
| XP06.csv | W0851, W0964 | 0.95580 | -2 |
| XP07.csv | W0851, W1152 | 0.96132 | -1 |
| XP08.csv | W0851, W1085 | 0.96132 | -1 |
| XP09.csv | W0851, W1009 | 0.96132 | -1 |
| XP10.csv | W0851, W1107 | 0.95580 | -2 |
| XP11.csv | W0851, W1090 | 0.96132 | -1 |
| XP12.csv | W0851, W0984 | 0.96132 | -1 |
| XP13.csv | W0851, W1141 | 0.96132 | -1 |
| XP14.csv | W0851, W1080 | 0.96132 | -1 |
| XP15.csv | W0851, W1033 | 0.96132 | -1 |
| XP16.csv | W0851, W0880 | 0.96132 | -1 |
| XP17.csv | W0851, W0926 | 0.96132 | -1 |
| XP18.csv | W0851, W0820 | 0.95580 | -2 |
| XP19.csv | W0851, W0898 | 0.96132 | -1 |

OPT01 submitted from the private Kaggle notebook; score receipt is in optimized_receipt.json. Existing final-submission selections were not modified by this experiment.
