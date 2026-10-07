"""Write the Kaggle notebook that produces a band submission from the competition data (kernel submissions only).

    python build_band_kernel.py --lo 1.0 --hi 1.975 --tag C   # then: kaggle kernels push -p kaggle_band_kernel
"""
import argparse
import json
import os

KERNEL_ID = "hosen42/kacchi-aloo-band-submission"

CLEAN = r'''import glob, unicodedata
import numpy as np, pandas as pd

root = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True) + glob.glob("../data/train.csv"))[0].rsplit("/", 1)[0]
tr_raw, te_raw = pd.read_csv(f"{root}/train.csv", dtype=str), pd.read_csv(f"{root}/test.csv", dtype=str)

def to_number(v):
    """Float from a cell; any Unicode decimal digit (e.g. Bangla) becomes ASCII."""
    if pd.isna(v):
        return np.nan
    s = "".join(str(unicodedata.decimal(c)) if unicodedata.decimal(c, None) is not None else c
                for c in str(v).strip().replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return np.nan

def aloo_per_guest(d):
    guests, aloo = d.guests.map(to_number), d.aloo_count.map(to_number)
    apg = aloo / guests
    return np.where(apg > 3.0, apg / 10, apg)   # a few aloo counts were typed x10

tr = tr_raw.assign(apg=aloo_per_guest(tr_raw), y=tr_raw.went_back_for_seconds.astype(int)).dropna(subset=["apg"])
te = te_raw.assign(apg=aloo_per_guest(te_raw))
print(f"data from {root}: train {len(tr)} rows with aloo, test {len(te)} rows")'''

PREDICT = r'''def band(x, lo, hi):
    return ((x > lo) & (x < hi)).astype(int)

train_acc = (band(tr.apg.values, LO, HI) == tr.y.values).mean()
pred = band(te.apg.values, LO, HI)
sub = pd.DataFrame({"wedding_id": te.wedding_id, "went_back_for_seconds": pred})
sub.to_csv("submission.csv", index=False)
print(f"band {LO} < aloo/guest < {HI}: train accuracy {train_acc:.4f}; test ones {pred.sum()} / {len(pred)}")
sub.head()'''


_ids = iter(range(100))


def md(s):
    return {"cell_type": "markdown", "id": f"cell{next(_ids)}", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "id": f"cell{next(_ids)}", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", default="kaggle_band_kernel")
    a = ap.parse_args()
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": [md(f"# Kacchi Aloo band submission ({a.tag})\n\n"
                       "A wedding's guests go back for seconds when the potatoes per guest fall inside a band. "
                       "This notebook cleans `aloo_count / guests` (Bangla digits, x10 typos, missing aloo) and "
                       "predicts 1 inside the band. The cut-points were chosen by the final-selection simulation "
                       "(E21/E22) from the training data."),
                    code(CLEAN),
                    code(f"LO, HI = {a.lo!r}, {a.hi!r}   # band cut-points, variant {a.tag}"),
                    code(PREDICT)]}
    os.makedirs(a.dir, exist_ok=True)
    with open(f"{a.dir}/kacchi-aloo-band-submission.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    meta = {"id": KERNEL_ID, "title": "Kacchi Aloo Band Submission", "code_file": "kacchi-aloo-band-submission.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [],
            "competition_sources": ["the-great-kacchi-aloo-mystery"], "kernel_sources": [], "model_sources": [],
            "docker_image": "gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a"}
    with open(f"{a.dir}/kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {a.dir} for band ({a.lo}, {a.hi}) tag {a.tag}")


if __name__ == "__main__":
    main()
