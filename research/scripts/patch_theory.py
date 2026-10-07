import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
reps = [
    ("4. **Around the edges there is a noisy ring:** about 1 in 4 weddings that sit just outside the sharp edge (0.01–0.035 potatoes per guest away) break the rule, and no column in the data predicts which ones. That randomness sets a **ceiling of about 95%** for everyone.",
     "4. **Around the edges there is a noisy ring:** about 1 in 3 weddings that sit 0.01–0.035 potatoes per guest away from an edge break the rule, and about 1 in 7 at 0.035–0.1 away. No column in the data predicts which ones. That randomness sets a **ceiling of about 95%** for everyone."),
    ("In a ring just outside each sharp edge, roughly 1 wedding in 4–6 does, but never most of them, so the rule still wins there.",
     "In a ring around each sharp edge, about 1 wedding in 3 breaks it (0.01–0.035 away) and about 1 in 7 further out (0.035–0.1 away). That is never most of them, so the rule still wins there."),
    ("- **The edges are sharp,** but in a thin ring around them about 1 wedding in 4 surprises us.",
     "- **The edges are sharp,** but in a thin ring around them between 1 wedding in 3 and 1 in 7 surprises us."),
]
for a, b in reps:
    assert a in s, a[:60]
    s = s.replace(a, b)
start = s.index("C.append(code('''oof_acc = 1 - e_all.mean()")
end = s.index('C.append(md("""**What this shows:** the best honest rule')
new = '''C.append(code(\'\'\'oof_acc = 1 - e_all.mean()
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
\'\'\'))
'''
s = s[:start] + new + s[end:]
open(p, "w", encoding="utf-8").write(s)
print("patched")
