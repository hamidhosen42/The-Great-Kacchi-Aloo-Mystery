"""Build the 'Best Aloo Theory' notebook (self-contained, Kaggle-ready)."""
import sys

import nbformat as nbf

out = sys.argv[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# 🥔 The Aloo Theory: two sharp edges, a noisy ring, and a 95% ceiling

**Mitu's question:** what makes guests go back for a second plate of kacchi?

**The answer in plain words**
1. **Count the potatoes per guest** (`aloo_count ÷ guests`). Guests go back for seconds when every guest gets **between about 0.97 and 2.01 potatoes**. Too few, and someone misses out on the golden aloo. Too many, and the plate feels like potato instead of mutton.
2. **Nothing else Mitu wrote down decides it:** not fairy lights, drones, dhol players, aunties, mutton, borhani, the city, or which party it was. Every one of those theories is tested below, and every one fails.
3. **The two edges are sharp,** much sharper than the "soft" curves people usually fit. A sharp rule beats a smooth one in cross-validation.
4. **Around the edges there is a noisy ring:** about 1 in 3 weddings that sit 0.01–0.035 potatoes per guest away from an edge break the rule, and about 1 in 7 at 0.035–0.1 away. No column in the data predicts which ones. That randomness sets a **ceiling of about 95%** for everyone.
5. **Why fancy models lose:** test weddings are much bigger than training weddings. A model that learned "more than 800 potatoes" breaks on a 1,500-guest wedding. Dividing by guests makes every wedding the same size.

Everything below is computed live from the competition data. Run all cells to reproduce it."""))

C.append(md("## 0. Setup and Mitu's three traps"))
C.append(code('''import glob, os, unicodedata, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr, t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from IPython.display import display
warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "figure.dpi": 110})
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8a8983", "#0b0b0b"

def find_data():
    if os.environ.get("KACCHI_DATA"):
        return os.environ["KACCHI_DATA"]
    for pat in ("/kaggle/input/**/train.csv", "data/train.csv", "../data/train.csv"):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError("add the competition data")

DATA = find_data()
NUM = ["guests", "aloo_count", "mutton_kg", "borhani_glasses", "fairy_lights", "dhol_players", "aunties_asking_when_marriage"]

def to_number(v):
    """Any Unicode digit (Bangla ৪৫০, ...) -> ASCII number."""
    if pd.isna(v):
        return np.nan
    s = "".join(str(unicodedata.decimal(c)) if unicodedata.decimal(c, None) is not None else c for c in str(v).strip())
    return float(s)

def clean(raw):
    d = raw.copy()
    d["bangla_digits"] = d.fairy_lights.astype(str).str.contains("[০-৯]", regex=True)
    for c in NUM:
        d[c] = d[c].map(to_number)
    d["x10"] = d.aloo_count / d.guests > 3                      # Tutul's extra zero
    d["aloo"] = np.where(d.x10, d.aloo_count / 10, d.aloo_count)
    d["apg"] = d.aloo / d.guests                                # aloo per guest
    for c, n in [("mutton_kg", "mutton_pg"), ("borhani_glasses", "borhani_pg"), ("fairy_lights", "fairy_pg")]:
        d[n] = d[c] / d.guests
    d["drone"] = (d.drone_photographer == "yes").astype(int)
    return d

raw_tr, raw_te = pd.read_csv(f"{DATA}/train.csv", dtype=str), pd.read_csv(f"{DATA}/test.csv", dtype=str)
tr_all, te = clean(raw_tr), clean(raw_te)
tr_all["y"] = tr_all.went_back_for_seconds.astype(int)
tr = tr_all.dropna(subset=["apg"]).reset_index(drop=True)      # empty aloo boxes can't be repaired
display(pd.DataFrame({"empty aloo boxes": [tr_all.aloo_count.isna().sum(), te.aloo_count.isna().sum()],
                      "Bangla digits": [tr_all.bangla_digits.sum(), te.bangla_digits.sum()],
                      "extra-zero potatoes": [tr_all.x10.sum(), te.x10.sum()],
                      "weddings > 700 guests": [(tr_all.guests > 700).sum(), (te.guests > 700).sum()]}, index=["train", "test"]))
print(f"clean training weddings: {len(tr)} | test weddings: {len(te)}")
'''))
C.append(md("""**What this shows:** all the messy data (empty boxes, Bangla digits, extra zeros) is in the training notebook only, so the test set is clean. The fourth column is Mitu's "big wedding season": many test weddings are bigger than anything in training."""))

C.append(md("## 1. Nothing Mitu counted works on its own… until you divide by guests"))
C.append(code('''cols = ["guests", "aloo", "mutton_kg", "borhani_glasses", "fairy_lights", "dhol_players", "aunties_asking_when_marriage", "drone"]
rho = {c: spearmanr(tr[c], tr.y)[0] for c in cols}
inband = ((tr.apg > 1) & (tr.apg < 2)).astype(int)
rho["aloo per guest inside 1-2? (yes/no)"] = spearmanr(inband, tr.y)[0]
display(pd.Series(rho).round(3).to_frame("correlation with going back"))

fig, ax = plt.subplots(figsize=(12, 4))
bins = np.arange(0.4, 2.81, 0.1)
g = tr.groupby(pd.cut(tr.apg, bins), observed=True).y.agg(["mean", "count"])
ax.bar([i.mid for i in g.index], g["mean"], width=0.085, color=BLUE)
for x, (m, n) in zip([i.mid for i in g.index], zip(g["mean"], g["count"])):
    ax.text(x, m + 0.02, f"{n}", ha="center", fontsize=7, color=GRAY)
ax.set_xlabel("potatoes per guest"); ax.set_ylabel("share that went back for seconds")
ax.set_title("The Goldilocks band: seconds happen only between about 1 and 2 potatoes per guest (numbers = weddings per bar)")
plt.tight_layout(); plt.show()
'''))
C.append(md("""**What this shows:** every raw count has almost zero correlation with going back. The same potato count can be generous at a small wedding and stingy at a big one. Once you divide by guests, a simple yes/no question ("is it between 1 and 2 per guest?") is very strongly linked to seconds. The pattern is not "more is better". It is a **band**: too few and too many both fail."""))

C.append(md("## 2. Why the fancy model fails: the big-wedding season"))
C.append(code('''def band(x, lo, hi):
    x = np.asarray(x, float)
    return ((x >= lo) & (x <= hi)).astype(int)

def midcut(r, y):
    """Sharp two-cut band: each cut sits halfway between the two training weddings where the answer flips."""
    def best(x, yy, side):
        xs = np.sort(np.unique(x)); c = (xs[:-1] + xs[1:]) / 2
        acc = np.array([((x >= t) == yy).mean() if side == "lo" else ((x <= t) == yy).mean() for t in c])
        b = np.flatnonzero(acc == acc.max())
        return c[b[len(b) // 2]]
    m = r < 1.5
    return best(r[m], y[m], "lo"), best(r[~m], y[~m], "hi")

fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(te.guests, te.aloo, s=8, color=GRAY, alpha=0.5, label="test (answers hidden)")
ax.scatter(tr.guests, tr.aloo, s=8, c=np.where(tr.y == 1, BLUE, ORANGE), label="train (blue = went back)")
xx = np.array([0, 1500]); ax.plot(xx, xx, "k--", lw=1); ax.plot(xx, 2 * xx, "k--", lw=1)
ax.axvline(tr.guests.max(), color=GRAY, ls=":"); ax.text(tr.guests.max() + 10, 100, "largest training wedding", color=GRAY)
ax.set_xlabel("guests"); ax.set_ylabel("potatoes"); ax.legend(loc="upper left")
ax.set_title("In raw counts the 'yes' zone is a wedge between 1x and 2x guests, and it keeps going into the big test weddings")
plt.tight_layout(); plt.show()

# Stress test: learn from small weddings only, then predict the biggest 40%
cut = tr.guests.quantile(0.6); small, big = tr[tr.guests <= cut], tr[tr.guests > cut]
raw_cols = ["guests", "aloo", "mutton_kg", "borhani_glasses", "fairy_lights", "dhol_players", "aunties_asking_when_marriage", "drone"]
rf = RandomForestClassifier(300, random_state=42, n_jobs=-1).fit(small[raw_cols], small.y)
lo_s, hi_s = midcut(small.apg.values, small.y.values)
display(pd.Series({"Rafi's MEGA AI (300 trees on raw counts)": (rf.predict(big[raw_cols]) == big.y).mean(),
                   "two cuts on potatoes per guest": (band(big.apg, lo_s, hi_s) == big.y).mean()}).round(3)
        .to_frame(f"accuracy on the biggest weddings (> {cut:.0f} guests), trained on the smaller ones"))
'''))
C.append(md("""**What this shows:** in raw counts the "yes" zone is a wedge, and a tree can only cover it with little boxes that stop where its training data stops. When it learns from small weddings and is tested on big ones, the 300-tree forest drops close to a coin toss. The two-cut potato-per-guest rule doesn't care about size at all."""))

C.append(md("## 3. Is it really *potatoes per guest*? Nodi asks \"but why?\""))
C.append(code('''import statsmodels.api as sm
rows = {}
for name, zone in [("lower edge", tr.apg.between(0.8, 1.2)), ("upper edge", tr.apg.between(1.8, 2.25))]:
    z = tr[zone]
    X = sm.add_constant(np.column_stack([np.log(z.aloo), np.log(z.guests), np.log(z.mutton_kg)]))
    m = sm.Logit(z.y.values, X).fit(disp=0)
    rows[name] = {"weight on log(potatoes)": m.params[1], "weight on log(guests)": m.params[2],
                  "weight on log(mutton)": m.params[3], "p-value of mutton": m.pvalues[3], "weddings": len(z)}
display(pd.DataFrame(rows).T.round(3))

def best_band_acc(r, y):
    o = np.argsort(r); ys = y[o]; n = len(r)
    cp, cn = np.r_[0, np.cumsum(ys)], np.r_[0, np.cumsum(1 - ys)]
    i, j = np.arange(n + 1)[:, None], np.arange(n + 1)[None, :]
    err = np.where(j >= i, cp[i] + (cn[j] - cn[i]) + (cp[n] - cp[j]), n)
    return 1 - err.min() / n
A, G, Y = tr.aloo.values, tr.guests.values, tr.y.values
alts = {"potatoes / guests": A / G, "potatoes / (guests + dhol players)": A / (G + tr.dhol_players.values),
        "potatoes / (guests + aunties)": A / (G + tr.aunties_asking_when_marriage.values), "potatoes / (guests + 25)": A / (G + 25),
        "potatoes / mutton kg": A / tr.mutton_kg.values, "potatoes / borhani glasses": A / tr.borhani_glasses.values}
display(pd.Series({k: best_band_acc(v, Y) for k, v in alts.items()}).sort_values(ascending=False).round(4).to_frame("best possible two-cut accuracy"))
'''))
C.append(md("""**What this shows:**
- Near each edge, the weights on log(potatoes) and log(guests) are about the same size with opposite signs. That is exactly what a *ratio* looks like, because log(potatoes ÷ guests) = log(potatoes) − log(guests).
- Mutton gets a small weight that could easily be chance (large p-value). **Babul Baburchi's mutton theory doesn't hold up.**
- Counting dhol players or aunties as extra eaters, or comparing potatoes with mutton or borhani instead of guests, never beats plain potatoes per guest."""))

C.append(md("## 4. The edges are sharp"))
C.append(code('''lo, hi = midcut(tr.apg.values, tr.y.values)
fig, axes = plt.subplots(1, 2, figsize=(14, 2.8))
for ax, centre, name in [(axes[0], lo, "lower edge"), (axes[1], hi, "upper edge")]:
    z = tr[(tr.apg > centre - 0.06) & (tr.apg < centre + 0.06)]
    ax.scatter(z.apg, np.zeros(len(z)), c=np.where(z.y == 1, BLUE, ORANGE), s=60, marker="|", linewidths=3)
    ax.axvline(centre, color=INK, ls="--", lw=1)
    ax.set_yticks([]); ax.set_xlabel("potatoes per guest"); ax.set_title(f"{name}: cut at {centre:.4f} (blue = went back, orange = didn't)")
plt.tight_layout(); plt.show()

def bracket(side):
    if side == "lo":
        a = tr[tr.apg < lo].nlargest(1, "apg"); b = tr[(tr.apg > lo) & (tr.apg < 1.5)].nsmallest(1, "apg")
    else:
        a = tr[(tr.apg < hi) & (tr.apg > 1.5)].nlargest(1, "apg"); b = tr[tr.apg > hi].nsmallest(1, "apg")
    return pd.concat([a, b])[["wedding_id", "guests", "aloo", "apg", "y"]]
display(pd.concat([bracket("lo"), bracket("hi")]).assign(apg=lambda d: d.apg.round(6)))
print(f"Sharp rule: go back for seconds if {lo:.5f} <= potatoes per guest <= {hi:.5f}")
'''))
C.append(md("""**What this shows:** right at each edge the answer flips cleanly. At the lower edge, a wedding with 0.96992 potatoes per guest said *no* and one with 0.97143 said *yes*. At the upper edge, 2.00973 said *yes* and 2.00993 said *no*. Just inside the upper edge there is a run of 10 *yes* weddings in a row, and just outside it a run of 8 *no*. A smooth, blurry edge would almost never produce runs like these."""))

C.append(md("## 5. Sharp beats smooth (honest cross-validation)"))
C.append(code('''def soft_band_fit(x, y):
    """Smooth band in log space: P = s(k(ln x - ln lo)) * s(k(ln hi - ln x)), fitted by maximum likelihood."""
    z = np.log(x)
    def nll(t):
        if t[2] <= 0 or t[0] >= t[1]:
            return 1e9
        p = np.clip(expit(t[2] * (z - np.log(t[0]))) * expit(t[2] * (np.log(t[1]) - z)), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    return minimize(nll, [1.0, 2.0, 30], method="Nelder-Mead", options=dict(maxiter=4000)).x

rules = {"sharp two-cut band (midpoint cuts)": lambda t, v: band(v.apg, *midcut(t.apg.values, t.y.values)),
         "smooth (soft) band, max-likelihood": lambda t, v: (lambda th: ((v.apg > th[0]) & (v.apg < th[1])).astype(int))(soft_band_fit(t.apg.values, t.y.values)),
         "fixed 1 < potatoes per guest < 2": lambda t, v: ((v.apg > 1) & (v.apg < 2)).astype(int)}
zone = np.digitize(tr.apg, [0.9, 1.1, 1.9, 2.1]); strat = tr.y.astype(str) + "_" + pd.Series(zone).astype(str)
acc = {k: [] for k in rules}
for rep in range(10):
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=2026 + rep).split(tr, strat):
        t, v = tr.iloc[ti], tr.iloc[vi]
        for k, f in rules.items():
            acc[k].append((f(t, v) == v.y.values).mean())
acc = {k: np.array(v) for k, v in acc.items()}
ref = acc["sharp two-cut band (midpoint cuts)"]
def corrected_p(a, b):        # Nadeau-Bengio corrected repeated-CV t-test
    d = a - b; k = len(d); se = np.sqrt((1 / k + 0.25) * d.var(ddof=1))
    return 2 * student_t.sf(abs(d.mean() / se), k - 1) if se > 0 else 1.0
display(pd.DataFrame({"CV accuracy (10x5)": {k: v.mean() for k, v in acc.items()},
                      "difference vs sharp": {k: v.mean() - ref.mean() for k, v in acc.items()},
                      "p-value": {k: (corrected_p(v, ref) if k != "sharp two-cut band (midpoint cuts)" else np.nan) for k, v in acc.items()}}).round(4))
'''))
C.append(md("""**What this shows:** in 50 honest train/test splits, the sharp rule beats both the smooth curve and the "round numbers" rule (1 to 2). The smooth curve looks more scientific, but it blurs edges that are actually sharp."""))

C.append(md("## 6. The noisy ring around each edge"))
C.append(code('''bins_d = [0, 0.01, 0.035, 0.1, 0.4, 3]
d_all, e_all = [], []
for rep in range(10):
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=rep).split(tr, tr.y):
        t, v = tr.iloc[ti], tr.iloc[vi]
        l2, h2 = midcut(t.apg.values, t.y.values)
        d_all.append(np.minimum(abs(v.apg.values - l2), abs(v.apg.values - h2)))
        e_all.append(band(v.apg, l2, h2) != v.y.values)
d_all, e_all = np.concatenate(d_all), np.concatenate(e_all)
ring = pd.Series(e_all).groupby(pd.cut(d_all, bins_d)).mean()
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar([str(i) for i in ring.index], ring.values, color=ORANGE)
ax.set_xlabel("distance from the nearest edge (potatoes per guest)"); ax.set_ylabel("share that breaks the rule")
ax.set_title("Out-of-fold: rule-breakers live in a ring 0.01-0.1 away from the edges")
plt.tight_layout(); plt.show()
display(ring.round(3).to_frame("rule-breaking rate (out-of-fold)"))
'''))
C.append(md("""**What this shows:** these numbers are measured on weddings the rule never saw (out-of-fold). Far from the edges almost nobody breaks the rule. In a ring around each sharp edge, about 1 wedding in 3 breaks it (0.01–0.035 away) and about 1 in 7 further out (0.035–0.1 away). That is never most of them, so the rule still wins there. Next we check whether anything in Mitu's notebook can tell us *which* weddings break it."""))

C.append(md("## 7. Testing everyone's theories: can any column move the edges?"))
C.append(code('''def best_cut_errors(r, y, side):
    o = np.argsort(r); ys = y[o]
    if side == "lo":
        e = np.r_[0, np.cumsum(ys)] + np.r_[np.cumsum((1 - ys)[::-1])[::-1], 0]
    else:
        e = np.r_[0, np.cumsum(1 - ys)] + np.r_[np.cumsum(ys[::-1])[::-1], 0]
    k = int(np.argmin(e)); rs = r[o]
    return int(e[k]), ((rs[k - 1] + rs[k]) / 2 if 0 < k < len(rs) else rs[min(k, len(rs) - 1)])

def theory_columns(d):
    Z = {"Rafi: fairy lights per guest": np.log(d.fairy_pg.values), "Babul: mutton per guest": np.log(d.mutton_pg.values),
         "borhani per guest": np.log(d.borhani_pg.values), "wedding size (guests)": np.log(d.guests.values),
         "dhol players": d.dhol_players.values.astype(float), "aunties asking": d.aunties_asking_when_marriage.values.astype(float),
         "drone": d.drone.values.astype(float), "Gaye Holud party": (d.event == "Gaye Holud").values.astype(float),
         "Dhaka": (d.city == "Dhaka").values.astype(float)}
    return Z

BETAS = np.arange(-0.2, 0.2001, 0.01)
def fit_shift(t, name):
    """Each edge becomes a cut on potatoes-per-guest x exp(beta * column): the column may push the edge up or down."""
    zt = theory_columns(t)[name]; mu, sd = zt.mean(), zt.std() or 1.0
    out = {}
    for side, m in (("lo", t.apg.values < 1.5), ("hi", t.apg.values >= 1.5)):
        best = None
        for b in BETAS:
            e, c = best_cut_errors(t.apg.values[m] * np.exp(b * (zt[m] - mu) / sd), t.y.values[m], side)
            if best is None or e < best[0] or (e == best[0] and abs(b) < abs(best[1])):
                best = (e, b, c)
        out[side] = best
    return out, mu, sd

def predict_shift(fit, v, name):
    out, mu, sd = fit
    z = (theory_columns(v)[name] - mu) / sd
    lo_ok = v.apg.values * np.exp(out["lo"][1] * z) > out["lo"][2]
    hi_ok = v.apg.values * np.exp(out["hi"][1] * z) < out["hi"][2]
    return (lo_ok & hi_ok).astype(int)

res = {}
for name in theory_columns(tr):
    a = []
    for rep in range(10):
        for ti, vi in StratifiedKFold(5, shuffle=True, random_state=2026 + rep).split(tr, strat):
            t, v = tr.iloc[ti], tr.iloc[vi]
            a.append((predict_shift(fit_shift(t, name), v, name) == v.y.values).mean())
    a = np.array(a)
    res[name] = {"CV accuracy": a.mean(), "change vs sharp rule": a.mean() - ref.mean(), "p-value": corrected_p(a, ref)}
display(pd.DataFrame(res).T.sort_values("change vs sharp rule", ascending=False).round(4))
'''))
C.append(md("""**What this shows:** each theory gets its best chance here. The column is allowed to push the edges up or down by as much as it likes, and the result is checked on weddings the rule didn't see. **None of them helps by more than noise.** Rafi's fairy lights, Babul's mutton, the aunties, the dhol, the drone, the Gaye Holud party and the city all leave the edges where they were. So the rule-breakers near the edges are random as far as Mitu's notebook can tell."""))

C.append(md("## 8. How good can anyone get?"))
C.append(code('''oof_acc = 1 - e_all.mean()
lo_t, hi_t = midcut(tr.apg.values, tr.y.values)
d_test = np.minimum(abs(te.apg.values - lo_t), abs(te.apg.values - hi_t))
# each test wedding gets the out-of-fold rule-breaking rate of its distance ring (section 6)
p_break = ring.values[np.clip(np.digitize(d_test, bins_d[1:-1]), 0, len(ring) - 1)]
rng = np.random.default_rng(0); S = 40000
broken = rng.random((S, len(te))) < p_break[None, :]
public = np.zeros((S, len(te)), bool)
for s in range(S):
    public[s, rng.choice(len(te), 181, replace=False)] = True      # the leaderboard uses ~45% / ~55% of the 400 test weddings
pub_ok, priv_ok = (~broken & public).sum(1), (~broken & ~public).sum(1)
display(pd.Series({
    "out-of-fold accuracy of the sharp rule (training weddings)": round(oof_acc, 4),
    "expected accuracy on these 400 test weddings": round(1 - p_break.mean(), 4),
    "test weddings within 0.1 of an edge (where luck lives)": int((d_test < 0.1).sum()),
    "one wedding on the public leaderboard is worth": f"{100 / 181:.2f} points",
    "chance the sharp rule scores >= 175/181 on public": round((pub_ok >= 175).mean(), 3),
    "chance it scores >= 0.97 (213/219) on private": round((priv_ok >= 213).mean(), 3)}).to_frame("value"))
'''))
C.append(md("""**What this shows:** the best honest rule gets about 95–96% of weddings right. One wedding moves the public leaderboard by more than half a point, so scores of 95.0, 95.6, 96.1 and 96.7 are only 1–3 weddings apart, and that is mostly luck. Even the perfect rule would only reach 97% now and then."""))

C.append(md("""## 🥔 The Aloo Theory (final answer)

> **Guests go back for seconds when every guest gets between about 0.97 and 2.01 potatoes.**

- **It's about sharing, not size.** A 1,500-guest wedding and a 60-guest wedding follow the same rule once you count potatoes *per guest*.
- **Too little aloo:** below about 0.97 per guest, someone at the table misses out, and nobody goes back.
- **Too much aloo:** above about 2.01 per guest, the plate is mostly potato and not enough mutton and rice. That feels stingy, and nobody goes back.
- **The edges are sharp,** but in a thin ring around them between 1 wedding in 3 and 1 in 7 surprises us. Nothing Mitu wrote down (fairy lights, drones, dhol, aunties, mutton, borhani, city, party) explains those surprises. They are the luck of the kacchi pot.
- **So no model can be perfect.** About 95% is the ceiling, and fancy models only do worse when the weddings get bigger than anything they've seen.

*Nodi's "but why?", answered: it's not how much food there is, it's how the aloo is shared.*"""))

C.append(code('''sub = pd.DataFrame({"wedding_id": te.wedding_id, "went_back_for_seconds": band(te.apg, lo_t, hi_t)})
sub.to_csv("submission.csv", index=False)
print(f"submission.csv written: {lo_t:.5f} <= potatoes per guest <= {hi_t:.5f} | share predicted to go back: {sub.went_back_for_seconds.mean():.3f}")
'''))

nb = nbf.v4.new_notebook()
nb.cells = C
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, out)
print("wrote", out, "cells:", len(C))
