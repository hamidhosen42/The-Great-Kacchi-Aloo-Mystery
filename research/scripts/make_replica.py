"""Schema-identical SYNTHETIC replica (for testing code only; NOT competition data)."""
import sys
import numpy as np, pandas as pd
from scipy.special import expit
out = sys.argv[1]; rng = np.random.default_rng(7)
CITIES = ["Barishal", "Chattogram", "Dhaka", "Khulna", "Mymensingh", "Rajshahi", "Rangpur", "Sylhet"]
EVENTS = ["Gaye Holud", "Biye", "Bou-bhat"]
def gen(n, start, big):
    g = rng.integers(51, 701, n) if not big else np.where(rng.random(n) < 0.36, rng.integers(701, 1500, n), rng.integers(51, 701, n))
    apg = rng.uniform(0.4028, 2.70, n)
    z = np.log(apg); p = expit(30 * (z - np.log(0.99))) * expit(30 * (np.log(1.995) - z))
    d = pd.DataFrame({"wedding_id": [f"W{start + i:04d}" for i in range(n)], "event": rng.choice(EVENTS, n), "city": rng.choice(CITIES, n),
                      "guests": g, "aloo_count": np.round(apg * g).astype(int), "mutton_kg": np.round(g * rng.uniform(0.18, 0.32, n), 1),
                      "borhani_glasses": np.round(g * rng.uniform(0.8, 1.5, n)).astype(int), "fairy_lights": np.round(g * rng.uniform(1, 4, n)).astype(int),
                      "drone_photographer": rng.choice(["yes", "no"], n), "dhol_players": rng.integers(0, 9, n),
                      "aunties_asking_when_marriage": rng.integers(0, 17, n)})
    return d, rng.binomial(1, p)
tr, y = gen(800, 1, False); te, _ = gen(400, 801, True)
tr["went_back_for_seconds"] = y
tr = tr.astype({"aloo_count": object, "fairy_lights": object})
bn = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
for i in rng.choice(800, 15, replace=False): tr.loc[i, "fairy_lights"] = str(tr.loc[i, "fairy_lights"]).translate(bn)
x10 = rng.choice(800, 6, replace=False)
for i in x10: tr.loc[i, "aloo_count"] = int(tr.loc[i, "aloo_count"]) * 10
for i in rng.choice(np.setdiff1d(np.arange(800), x10), 24, replace=False): tr.loc[i, "aloo_count"] = np.nan
tr.to_csv(f"{out}/train.csv", index=False); te.to_csv(f"{out}/test.csv", index=False)
te[["wedding_id"]].assign(went_back_for_seconds=0).to_csv(f"{out}/sample_submission.csv", index=False)
print("replica written", tr.shape, te.shape)
