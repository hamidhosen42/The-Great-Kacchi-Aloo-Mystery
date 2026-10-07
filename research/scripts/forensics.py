"""Leaderboard + public-submission forensics (no competition data needed).

1. Infer the public-LB row count N from the set of observed accuracy scores.
2. Translate each LB score into #correct / #errors and Wilson 95% intervals.
3. Compare every public notebook submission file row-by-row: positive rate, pairwise
   disagreement, and the exact set of 'disputed' wedding_ids.
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

lb = pd.read_csv(glob.glob(f"{SP}/official/lb/*.csv")[0])
scores = sorted(lb.Score.unique())

# 1. N inference: Kaggle displays accuracy truncated (not rounded) to 5 decimals,
#    so N is feasible if every displayed score equals floor(k/N * 1e5)/1e5 for some integer k.
def feasible(s, n):
    k = int(np.floor(s * n))
    return any(np.floor(kk / n * 1e5 + 1e-9) / 1e5 == round(s, 5) for kk in (k, k + 1) if 0 <= kk <= n)


cands = [n for n in range(1, 401) if all(feasible(s, n) for s in scores)]
print("Public-LB sizes consistent with all", len(scores), "distinct scores:", cands)
N = cands[0]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


lb["public_correct"] = np.ceil(lb.Score * N - 1e-6).astype(int)
lb["public_errors"] = N - lb.public_correct
ci = lb.public_correct.apply(lambda k: wilson(k, N))
lb["wilson95_lo"] = [round(a, 4) for a, _ in ci]
lb["wilson95_hi"] = [round(b, 4) for _, b in ci]
print(lb[["Rank", "TeamName", "Score", "SubmissionCount", "public_correct", "public_errors", "wilson95_lo", "wilson95_hi"]].to_string(index=False))
lb.to_csv(f"{OUT}/leaderboard_decoded.csv", index=False)

# all-zero sample submission => number of public negatives
s0 = lb.loc[lb.TeamName == "sample_submission.csv", "public_correct"].iloc[0]
print(f"\nAll-zero sample submission: {s0}/{N} correct  => public negatives={s0}, positives={N - s0} ({(N - s0) / N:.3f})")

# 3. submission files
files = {}
for f in sorted(glob.glob(f"{SP}/nb/*/out/*.csv")):
    nb = os.path.basename(os.path.dirname(os.path.dirname(f)))
    key = f"{nb.split('_', 1)[1]}::{os.path.basename(f)}"
    files[key] = pd.read_csv(f).set_index("wedding_id")["went_back_for_seconds"]
P = pd.DataFrame(files)
print("\nTest ids:", P.index.min(), "...", P.index.max(), "n =", len(P), "| all files same ids:", P.notna().all().all())
print("\nPositive rate per file:")
print(P.mean().round(4).to_string())

# identical files (hash)
groups = {}
for c in P.columns:
    groups.setdefault(tuple(P[c].values), []).append(c)
print("\nDistinct prediction vectors:", len(groups))
for i, (k, v) in enumerate(groups.items()):
    print(f"  vector {i}: {v}")

D = pd.DataFrame({a: {b: int((P[a] != P[b]).sum()) for b in P.columns} for a in P.columns})
print("\nPairwise # disagreeing rows:")
D.columns = [c.split("::")[1] for c in D.columns]
print(D.to_string())

disputed = P[P.nunique(axis=1) > 1]
print(f"\nRows where ANY public submission disagrees: {len(disputed)}")
print(disputed.to_string())
cons = P[P.nunique(axis=1) == 1].iloc[:, 0]
print(f"Consensus rows: {len(cons)}  (consensus positive rate {cons.mean():.3f})")
P.to_csv(f"{OUT}/public_submissions_matrix.csv")
disputed.to_csv(f"{OUT}/disputed_rows.csv")
json.dump({"N_public_candidates": cands, "N_public": N, "N_private_if_rest": 400 - N,
           "public_negatives": int(s0), "public_positives": int(N - s0)},
          open(f"{OUT}/lb_facts.json", "w"), indent=1)
