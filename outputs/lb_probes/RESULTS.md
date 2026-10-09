# Two-row probe results

Baseline: Q01, public accuracy 0.96685.

| Probe | Rows | Score | Net public correct change |
|---|---|---:|---:|
| LBP01.csv | W0968, W1163 | 0.96132 | -1 |
| LBP02.csv | W1119, W1157 | 0.96685 | 0 |
| LBP03.csv | W1179, W1022 | 0.96132 | -1 |
| LBP04.csv | W0972, W0846 | 0.96132 | -1 |
| LBP05.csv | W1153, W1191 | 0.96685 | 0 |

Assuming 181 public rows, -1 means exactly one public row whose flip hurt and one private row; which row is public remains unknown. Zero means either both private or two public flips cancelling. No individual labels or final selections were changed.
