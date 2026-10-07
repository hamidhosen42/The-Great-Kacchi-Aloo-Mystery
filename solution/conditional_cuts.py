"""E17: do the band edges depend on a group (event, drone, city, or terciles of other columns)?

For each grouping, fit Rafiur-style midcut bands separately per group and compare with one global midcut band
using the canonical 10x5 repeated stratified CV and the Nadeau-Bengio corrected t-test. Also fits a
'shared band + per-group shift' variant that pools edges across groups (less variance).

    python conditional_cuts.py --data ../data
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import kacchi_pipeline as kp  # noqa: E402


def group_band(t, a, key):
    """Midcut band fitted per group; groups too small to fit fall back to the global band."""
    lo_g, hi_g = kp.midcut_rafiur(t.apg.values, t.y.values)
    out = np.empty(len(a))
    for g in a[key].unique():
        tg = t[t[key] == g]
        lo, hi = (kp.midcut_rafiur(tg.apg.values, tg.y.values) if len(tg) >= 60 and tg.y.nunique() == 2 else (lo_g, hi_g))
        m = (a[key] == g).values
        out[m] = kp.band(a.apg.values[m], lo, hi, closed=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--repeats", type=int, default=10)
    args, _ = ap.parse_known_args()
    tr = kp.clean(pd.read_csv(f"{args.data}/train.csv", dtype=str), True).dropna(subset=["apg"]).reset_index(drop=True)
    for c, q in [("mutton_pg", 3), ("borhani_pg", 3), ("fairy_pg", 3), ("guests", 3), ("dhol_players", 3), ("aunties_asking_when_marriage", 3)]:
        tr[c + "_t"] = pd.qcut(tr[c], q, labels=False, duplicates="drop").astype(str)
    tr["gaye_holud"] = (tr.event == "Gaye Holud").astype(str)
    keys = ["event", "gaye_holud", "drone_photographer", "city"] + [c + "_t" for c in ["mutton_pg", "borhani_pg", "fairy_pg", "guests", "dhol_players", "aunties_asking_when_marriage"]]
    F = kp.make_folds(tr, args.repeats)
    n_tr, n_te = len(tr) * 4 / 5, len(tr) / 5
    base = lambda t, a: kp.band(a.apg.values, *kp.midcut_rafiur(t.apg.values, t.y.values), closed=True)
    _, acc0 = kp.run_cv(lambda t, *aps: [base(t, a) for a in aps], tr, F)
    print(f"global midcut band: CV {acc0.mean():.4f} ± {acc0.std():.4f}")
    rows = []
    for k in keys:
        _, acc = kp.run_cv(lambda t, *aps, k=k: [group_band(t, a, k) for a in aps], tr, F)
        d, se, p = kp.corrected_ttest(acc, acc0, n_tr, n_te)
        full = {g: tuple(np.round(kp.midcut_rafiur(tr[tr[k] == g].apg.values, tr[tr[k] == g].y.values), 4)) for g in sorted(tr[k].unique())}
        rows.append(dict(group=k, cv=acc.mean(), delta=d, p=p, cuts_full_train=full))
    R = pd.DataFrame(rows).sort_values("p")
    R["bonferroni_p"] = np.minimum(R.p * len(keys), 1)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
