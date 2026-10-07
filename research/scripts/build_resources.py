"""Merge verified resource catalogs, add competition primary sources, and write
external_resources.csv + TOP_100_RESOURCES.md into the workspace."""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP, WS = sys.argv[1], sys.argv[2]
a = pd.read_csv(f"{SP}/resources/resources_academic.csv")
c = pd.read_csv(f"{SP}/resources/resources_competitions.csv")
c = c[c.resource_id != "C034"]  # duplicate of A029 (same NeurIPS paper/URL)

# A026: andrewng.org page now 308-redirects to Google Scholar -> use Stanford-hosted PDF
m = a.resource_id == "A026"
a.loc[m, "url"] = "https://ai.stanford.edu/~ang/papers/cv-final.pdf"
a.loc[m, "paper_url"] = "https://ai.stanford.edu/~ang/papers/cv-final.pdf"
a.loc[m, "notes"] = (str(a.loc[m, "notes"].iloc[0]) + " | 2026-10-07 re-check: original andrewng.org URL 308-redirects to Google Scholar; "
                     "Stanford PDF URL resolves (167 KB PDF) but its text layer is not machine-readable, so title was not re-confirmed from the PDF itself.")

COLS = list(a.columns)
K = "https://www.kaggle.com"
k_rows = [
    ("K001", "Competition primary source", "The Great Kacchi Aloo Mystery: Data page (data description + Mitu's field notes)", "BdAIO (host)", 2026, "Competition page", "Kaggle",
     f"{K}/competitions/the-great-kacchi-aloo-mystery/data", "Binary classification", "Synthetic tabular", "-", "High", "Technical documentation", "Official",
     "Immediate experiment", "Field notes disclose every trap: empty aloo_count, Bangla-digit fairy_lights, x10 potato counts, and test weddings much bigger than train.",
     "Audit each trap in train AND test; build the size-stress validation fold.", "Content read via Kaggle API (competition_list_pages) 2026-10-07."),
    ("K002", "Competition primary source", "The Great Kacchi Aloo Mystery: Rules page + Kaggle Foundational Rules", "BdAIO / Kaggle", 2026, "Competition page", "Kaggle",
     f"{K}/competitions/the-great-kacchi-aloo-mystery/rules", "-", "-", "-", "High", "Technical documentation", "Official", "Conceptual",
     "Ties are broken by earliest submission (FR 7b); no hand-labelling of test rows (FR 4b); one account per person (FR 5a); max 5 subs/day, team size 10.",
     "Submit final candidates early; select a hedge pair manually.", "Content read via Kaggle API 2026-10-07."),
    ("K003", "Competition primary source", "The Great Kacchi Aloo Mystery: Public leaderboard", "Kaggle", 2026, "Leaderboard", "Kaggle",
     f"{K}/competitions/the-great-kacchi-aloo-mystery/leaderboard", "-", "-", "-", "High", "Technical documentation", "Official", "Conceptual",
     "Score granularity implies public LB = 181 rows (362 is the only other consistent size and is implausible); host benchmark 'submission (8).csv' = 0.95027 (172/181); all-zero = 98/181.",
     "Translate every LB delta into rows; ignore deltas < 3 rows.", "Downloaded via Kaggle API 2026-10-07 08:58 UTC snapshot."),
    ("K004", "Public notebook", "kacchi_starter (host starter notebook)", "Tasnim Mahfuz Nafis", 2026, "Notebook", "Kaggle",
     f"{K}/code/tasnimmahfuznafis/kacchi-starter", "Binary classification", "Synthetic tabular", "RandomForest raw + hints", "High", "Technical documentation", "Full code",
     "Immediate experiment", "Host hints: per-guest quantity, 'too little / too much aloo', depth-2 tree, and big test weddings defeat Random Forests.",
     "E0a baseline + E4a depth-2 tree on the ratio.", "Pulled via Kaggle API; execution log empty (no outputs)."),
    ("K005", "Public notebook", "Kacchi Aloo Mystery Deep EDA", "FOYSAL (foysalemonshanto)", 2026, "Notebook", "Kaggle",
     f"{K}/code/foysalemonshanto/kacchi-aloo-mystery-deep-eda", "Binary classification", "Synthetic tabular", "Band rule on aloo/guest; soft-band MLE", "High", "Community/blog",
     "Full code + log", "Immediate experiment", "Full audit: 24 missing aloo, 15 Bangla digits, 6 x10 rows (train only); 145/400 test weddings > train max; band 0.97<apg<2.01; LGBM raw 0.86 CV.",
     "Reproduce audit; E0 baseline.", "Pulled via Kaggle API; outputs taken from Kaggle execution log."),
    ("K006", "Public notebook", "LB_0.95027_Kacchi Aloo Theory", "FOYSAL (foysalemonshanto)", 2026, "Notebook", "Kaggle",
     f"{K}/code/foysalemonshanto/lb-0-95027-kacchi-aloo-theory", "Binary classification", "Synthetic tabular", "Band rule / LGBM / kNN / log-MLE, 10x5 RSKF", "High", "Community/blog",
     "Full code + log", "Immediate experiment", "10x5 repeated CV: band_rule 0.9558, lgbm_logapg 0.9541, knn5 0.9513, fixed(1,2) 0.9446, logmle 0.9407; writes 5 candidate files.",
     "E0 exact reproduction (must match 400/400 test predictions).", "Pulled via Kaggle API; title score is a participant claim."),
    ("K007", "Public notebook", "The Aloo Theory: The Goldilocks Band", "Rafiur Rahman", 2026, "Notebook", "Kaggle",
     f"{K}/code/rafiurrahman01/the-aloo-theory-the-goldilocks-band", "Binary classification", "Synthetic tabular", "Midpoint band rule + falsification tests", "High", "Community/blog",
     "Full code + log", "Immediate experiment", "Log-log logit near edges gives equal/opposite weights (it is a ratio); mutton not significant; alternative denominators never beat aloo/guests.",
     "E12 ratio search; E11 residual-signal test.", "Pulled via Kaggle API; outputs from execution log."),
    ("K008", "Domain context", "World AI Week 2026: all events (BdAIO Kaggle contest listing)", "World AI Week", 2026, "Event page", "-",
     "https://worldaiweek.ai/all-events-2026/", "-", "-", "-", "Low", "Community/blog", "-", "Background",
     "Confirms the contest window (6-9 Oct 2026) and its educational purpose.", "-", "Linked from the competition rules page; URL verified via WebFetch 2026-10-07."),
]
kdf = pd.DataFrame([dict(zip(["resource_id", "category", "title", "authors", "year", "resource_type", "venue", "url", "task", "domain", "method",
                              "relevance", "evidence_level", "reproducibility", "competition_applicability", "key_insight", "experiment_inspired", "notes"], r))
                    for r in k_rows])
for col in COLS:
    if col not in kdf:
        kdf[col] = ""
kdf["license"] = ""
R = pd.concat([kdf[COLS], a[COLS], c[COLS]], ignore_index=True)
R["year"] = R.year.apply(lambda v: "" if pd.isna(v) or v == "" else str(int(float(v))))
assert not R.resource_id.duplicated().any() and not R.url.duplicated().any()
R.to_csv(f"{WS}/external_resources.csv", index=False, encoding="utf-8")
print("external_resources.csv rows:", len(R))

# ---------------------------------------------------------------------------- Top 100
manual = ["K001", "K006", "K005", "K007", "K004", "K003", "K002", "A017", "A001", "A005", "C031", "A012", "C026", "C025", "A027", "C033",
          "A036", "A034", "A035", "A031", "A020", "A021", "A022", "A024", "A023", "A025", "C005", "C002", "C001", "C039", "C037", "A045",
          "A053", "A066", "A062", "A055", "A059", "A013", "A008", "A011", "C024", "C022", "C023", "C009", "C008", "C028", "C030", "C029",
          "C012", "A029", "A026", "A028", "A003", "A018", "A064", "A041", "A040", "A047", "A048", "C003", "C004", "C016", "C015", "C014",
          "C017", "C019", "C020", "C032", "A004", "A006", "A015", "A016", "A014", "A010", "A009", "A030", "A032", "A033", "A019", "A037",
          "A038", "A044", "A043", "A042", "A046", "A054", "A052", "A056", "A057", "A058", "A061", "A063", "A065", "A067", "C036", "C038",
          "C035", "C042", "C044", "C010"]
assert len(manual) == len(set(manual)) == 100, len(set(manual))
missing = set(manual) - set(R.resource_id)
assert not missing, missing
idx = R.set_index("resource_id")


def impact(r):
    app, rel = r.competition_applicability, r.relevance
    if r.name.startswith("K") or (app == "Immediate experiment" and rel == "High"):
        return "High"
    if app in ("Immediate experiment", "Potential experiment") or rel == "High":
        return "Medium"
    return "Low"


def difficulty(r):
    t = str(r.resource_type).lower()
    if any(s in t for s in ("documentation", "notebook", "leaderboard", "competition page", "event", "blog", "news", "encyclopedia", "tutorial")):
        return "Low"
    if "book" in t:
        return "High"
    return "Medium"


lines = ["# TOP 100 Resources: The Great Kacchi Aloo Mystery", "",
         "Ranked for usefulness to **this** competition (synthetic 11-column tabular data, binary accuracy, ratio-band target, size shift, "
         "181-row public LB). Ranking is a manual judgement: competition primary sources first, then resources that change a decision "
         "(cut-point overfitting, extrapolation, shift, LB noise, final-selection hedging), then supporting theory and tooling.", "",
         f"Source catalog: `external_resources.csv` ({len(R)} verified, de-duplicated resources). The original brief asked for 500+; "
         "only about 120 genuinely relevant, verifiable resources exist for a toy synthetic task like this, and the list was not padded.", "",
         "Columns: Impact = expected effect on final private-LB decisions; Difficulty = effort to apply.", "",
         "| Rank | Resource | Type | Year | Key idea | Relevance | How to use it here | Expected impact | Difficulty | Link |",
         "|---:|---|---|---|---|---|---|---|---|---|"]
esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ") if pd.notna(s) else ""
for i, rid in enumerate(manual, 1):
    r = idx.loc[rid]
    how = r.experiment_inspired if pd.notna(r.experiment_inspired) and str(r.experiment_inspired).strip() not in ("-", "") else r.competition_applicability
    lines.append(f"| {i} | {esc(r.title)} ({rid}) | {esc(r.resource_type)} | {esc(r.year)} | {esc(r.key_insight)} | {esc(r.relevance)} | {esc(how)} | "
                 f"{impact(r)} | {difficulty(r)} | {esc(r.url)} |")
lines += ["", "## Not in the Top 100", "", "Lower-ranked catalog entries (still verified) are in `external_resources.csv`: "
          + ", ".join(f"{rid} {idx.loc[rid].title}" for rid in R.resource_id if rid not in manual) + "."]
open(f"{WS}/TOP_100_RESOURCES.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("TOP_100_RESOURCES.md written;", len(R) - 100, "resources not in top 100")
