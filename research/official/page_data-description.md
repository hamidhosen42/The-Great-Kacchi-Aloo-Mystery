# Data Description

Each row is **one Bangladeshi wedding party** from Mitu's research notebook.

## Files
- **train.csv**: 800 weddings, *with* the answer (`went_back_for_seconds`)
- **test.csv**: 400 weddings, *without* the answer. You predict these!
- **sample_submission.csv**: an example of the file you upload

## Columns

| Column | What it means |
|---|---|
| `wedding_id` | A unique name for each wedding |
| `event` | Which party: Gaye Holud, Biye or Bou-bhat |
| `city` | Where it happened |
| `guests` | How many guests came |
| `aloo_count` | Total potatoes in all the kacchi 🥔 |
| `mutton_kg` | Total mutton cooked, in kilograms |
| `borhani_glasses` | Glasses of borhani served |
| `fairy_lights` | Number of fairy lights decorating the venue ✨ |
| `drone_photographer` | Was there a drone filming? (yes / no) |
| `dhol_players` | Number of dhol drummers 🥁 |
| `aunties_asking_when_marriage` | How many aunties asked an unmarried guest "when is YOUR wedding?" |
| `went_back_for_seconds` | **The target!** 1 = guests went back for seconds, 0 = they didn't (train only) |

## Field Notes from Mitu (read this!)
- Some `aloo_count` boxes are **empty**. We got caught by the bride's uncle and had to help serve food. 
- My cousin Tutul wrote some `fairy_lights` numbers in **Bangla digits** (৪৫০ instead of 450).
- Tutul also says he *might* have added an extra zero to a few potato counts. Look carefully!
- Some weddings in the test set are **much bigger** than anything in the training set. Big wedding season!
- The data is **synthetic**: made up by a Python computer program for this competition.
