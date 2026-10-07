## TL;DR
- **What:** Predict if wedding guests went back for a **second plate of kacchi biryani** (1 = yes, 0 = no).
- **How:** Train on `train.csv` → predict for `test.csv` → upload a file shaped like `sample_submission.csv`.
- **Score:** Accuracy, the % of weddings you got right. Higher is better!
- **Start here:** https://www.kaggle.com/code/tasnimmahfuznafis/kacchi-starter
- **Secret tip:** The fanciest model doesn't always win. Thinking does.

---

## The Story
Mitu is in class 7, she loves Machine Learning, and she has a research method: go to as many Bangladeshi weddings as possible. With her friends Tutul (potato counter), Rafi (believer in MEGA AI) and Nodi (asker of "but why?"), she recorded what happened at hundreds of weddings, from the number of guests to the number of aunties asking "when are YOU getting married?" 

Her big question: **what makes guests go back for seconds?**

She has a hunch it has something to do with the **aloo**, the potato in Bangladeshi kacchi biryani, which people take *very* seriously. Can your model crack the mystery?

## A Tiny Guide to Bangladeshi Weddings
- **Kacchi biryani:** the star of the show. Fragrant rice and mutton cooked slowly together, with a golden potato hiding inside.
- **Aloo:** potato. In Bangladesh, finding a potato in your biryani is a big deal. Arguments have started over less!
- **Borhani:** a spicy yogurt drink served with biryani. Very refreshing.
- **Three parties, not one:** *Gaye Holud* (turmeric ceremony, lots of yellow), *Biye* (the wedding day) and *Bou-bhat* (reception at the groom's home).
- **Dhol:** a big drum. More dhol = more dancing.

##  Evaluation
Submissions are scored on **accuracy**: the share of weddings where your prediction (0 or 1) matches what really happened.

## Submission Format
For every `wedding_id` in `test.csv`, predict `went_back_for_seconds`. Your file needs a header and should look like this:

```
wedding_id,went_back_for_seconds
W0801,0
W0802,1
W0803,0
```

## Ranks
Climb the leaderboard and earn your title:

| Accuracy | Title |
|---|---|
| below 60% |  Borhani Beginner |
| 60% to 80% | Jorda Junior |
| 80% to 90% |  Kacchi Commander |
| 90%+ | **Aloo Legend** |

##  Prizes
- **Leaderboard winners:** top scores on the private leaderboard
- **Best Aloo Theory:** the notebook that best explains *why* guests go back for seconds, in plain words. You don't need the top score to win this one!


## About
Made by the **Bangladesh AI Olympiad (BdAIO)** for **World AI Week 2026**. The data is **synthetic**: it was created by a Python computer program to tell a fun story, so please don't use it to plan a real wedding. (Though more aloo never hurt anyone. Or did it? )
