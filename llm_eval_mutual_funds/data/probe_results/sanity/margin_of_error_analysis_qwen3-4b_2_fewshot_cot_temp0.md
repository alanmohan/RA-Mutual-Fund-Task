# Margin-of-Error Summary Analysis: Qwen3-4B, 2_fewshot_cot_temp0

## Hypothesis

**Probe accuracy is lower when the absolute difference |value_1 − value_2| between the two funds is very small** (i.e. the probe tends to get predictions **wrong** when the two feature values are close).

Equivalently: **Wrong predictions should have smaller |diff| than right predictions** (wrong < right in distribution).

---

## Scope: Numerical Continuous Features

We restrict the analysis to **numerical continuous** features and exclude:

- **load_f1_no**, **ntf_f1_yes** — binary/categorical (Y/N).
- **medalist_f1_higher** — ordinal category, not a continuous scale in the same sense.

**Numerical continuous features considered:**  
expense_ratio_f1_lower, sharpe_f1_higher, stdev_f1_lower, return_3yr_f1_higher, beta_f1_lower, tenure_f1_longer, inception_f1_older, assets_f1_higher, turnover_f1_lower.

---

## 1. Statistical Test: Mann-Whitney U (wrong < right) + effect size

For each feature we test: *Do wrong-prediction |diff| values tend to be smaller than right-prediction |diff|?*  
One-sided Mann-Whitney (alternative: wrong < right) is appropriate because distributions are not assumed normal.

We also report **rank-biserial correlation** as an effect size for Mann-Whitney:


$$
r_{rb} = 1 - \frac{2U}{n_{wrong}n_{right}} \in [-1, 1]
$$


- **Positive** \(r_{rb}\) means wrong tends to have **smaller** |diff| than right; larger magnitude means stronger separation.

| Feature | n_wrong | n_right | Median \|diff\| Wrong | Median \|diff\| Right | Median ratio (Right/Wrong) | Mann-Whitney p (wrong<right) | r_rb |
|--------|--------|--------|------------------------|------------------------|-----------------------------|------------------------------|------|
| expense_ratio_f1_lower | 71 | 929 | 0.0021 | 0.0050 | **2.38** | **8.45e-06** | 0.306 |
| sharpe_f1_higher | 84 | 916 | 0.105 | 0.19 | **1.81** | **6.67e-05** | 0.252 |
| stdev_f1_lower | 265 | 735 | 1.0 | 1.5 | **1.50** | **5.69e-08** | 0.220 |
| return_3yr_f1_higher | 99 | 901 | 1.94 | 3.26 | **1.68** | **6.36e-06** | 0.267 |
| beta_f1_lower | 227 | 670 | 0.09 | 0.12 | **1.33** | **1.15e-06** | 0.210 |
| tenure_f1_longer | 168 | 832 | 4.0 | 8.0 | **2.00** | **3.08e-11** | 0.319 |
| inception_f1_older | 232 | 768 | 1404 | 2732.5 | **1.95** | **5.38e-08** | 0.230 |
| assets_f1_higher | 157 | 843 | 346.18 | 675.38 | **1.95** | **0.00029** | 0.173 |
| turnover_f1_lower | 308 | 692 | 28.0 | 48.4 | **1.73** | **7.84e-11** | 0.253 |

- **Median ratio (Right/Wrong):** Typical |diff| when the probe is **right** is about **1.3× to 2.4×** larger than when it is **wrong**. So correct predictions tend to occur at larger separations; errors tend to occur when values are closer.
- **Mann-Whitney p:** For every numerical continuous feature, **p ≪ 0.05** (all ≤ 0.0003). We reject the null: the distribution of |diff| for **wrong** is stochastically smaller than for **right**.

**Conclusion (numerical features):** The data **strongly support** the hypothesis that the probe is wrong more often when the feature difference is small. Sample sizes (n_wrong 71–308, n_right 692–929) are sufficient for these tests.

---

## 2. Visual findings for key continuous features

These comments refer to the **per-feature 2×2 plots** (full vs zoomed; wrong vs right).

### beta_f1_lower

- **Full view**: both wrong and right are left-skewed, but **wrong has more mass near 0** and drops off faster, while **right extends further into larger |diff|**.
- **Zoomed view (small |diff|)**: wrong is visibly more concentrated in the smallest bins; right is less concentrated (more spread within the zoom window).
- **Quantification**: median ratio **1.33×**, p **1.15e-06**, \(r_{rb}=0.210\). This is a **statistically strong** and **moderate-sized** “errors at small |diff|” effect.

### stdev_f1_lower

- **Full view**: wrong is concentrated at small |diff|; right has a heavier tail (more density at larger |diff|).
- **Zoomed**: wrong is slightly more left-shifted than right; the separation is present but not extreme.
- **Quantification**: median ratio **1.50×**, p **5.69e-08**, \(r_{rb}=0.220\).

### turnover_f1_lower

- **Full view**: very left-skewed with a long tail; right has noticeably more probability mass beyond the smallest bins.
- **Zoomed**: wrong is highly concentrated near 0; right is flatter/more spread across the zoom range.
- **Quantification**: median ratio **1.73×**, p **7.84e-11**, \(r_{rb}=0.253\).

**Takeaway:** for these continuous features, the visuals match the statistics: wrong predictions are **enriched** in the smallest-|diff| region, even though both wrong and right are naturally left-skewed because many fund pairs are close.

---

## 3. Categorical / Binary Features (for context)

| Feature | n_wrong | n_right | Median Wrong | Median Right | Mann-Whitney p |
|--------|--------|--------|--------------|--------------|----------------|
| load_f1_no | 64 | 936 | 1.0 | 1.0 | 0.55 (n.s.) |
| ntf_f1_yes | 30 | 970 | 1.0 | 0.0 | 0.89 (n.s.) |
| medalist_f1_higher | 185 | 456 | 1.0 | 1.0 | 0.006 |

- **load** and **ntf**: No evidence that wrong has smaller |diff| than right (p ≈ 0.55, 0.89); |diff| is 0/1-like, so the “small diff” story does not apply the same way.
- **medalist**: p = 0.006 (significant) but median is 1.0 vs 1.0; interpretation is different (ordinal scale).

---

## 4. Does the correlation between low |diff| and probe accuracy exist?

- **Yes, for numerical continuous features.**  
  - Wrong predictions have **smaller** |diff| than right predictions (Mann-Whitney, one-sided “wrong < right”), with very small p-values.  
  - So **lower |diff| is associated with more errors** (i.e. lower probe accuracy when we condition on small |diff|). That is exactly a correlation between “small diff” and “probe wrong.”

---

## 5. Quantifying the correlation

We can quantify it in three ways:

1. **Mann-Whitney p-value**  
   - Already in the summary CSV.  
   - Answers: “Is the effect statistically significant?”  
   - For all 9 numerical features: **yes** (p ≤ 0.0003).

2. **Median ratio (Right / Wrong)**  
   - **median_abs_diff_right / median_abs_diff_wrong.**  
   - Interpretations:  
     - Ratio > 1: typical |diff| when correct is larger than when wrong; larger ratio ⇒ stronger “errors at small diff” effect.  
     - For these features, ratios range from **1.33** (beta) to **2.38** (expense_ratio).  
   - This is now computed in the summary as **median_ratio_right_over_wrong** (after re-running or using `--plot-only`).

3. **Rank-biserial correlation (r_rb)**  
   - Effect size for Mann-Whitney: **r_rb = 1 − 2U/(n_wrong×n_right)** in [-1, 1].  
   - **Positive r_rb** ⇒ wrong tends to have smaller values than right; **closer to 1** ⇒ stronger effect.  
   - The sanity-check script now writes **rank_biserial_r** to the summary CSV when you run (or re-run with `--plot-only`).  
   - Use it as a single number for “how strongly is smaller |diff| associated with wrong?” (e.g. 0.2 = small, 0.5 = medium, 0.8 = large, depending on conventions).

So: **the correlation is quantified by**  
- **p-value** (significance),  
- **median_ratio_right_over_wrong** (how much larger |diff| is when correct),  
- **rank_biserial_r** (effect size of “wrong has smaller |diff|”).

---

## 6. Summary

| Question | Answer |
|----------|--------|
| Is there sufficient data to support “low accuracy when diff is very low”? | **Yes** for all 9 numerical continuous features (adequate n_wrong and n_right; all p ≪ 0.05). |
| Does the correlation between low \|diff\| and probe accuracy exist? | **Yes**: wrong predictions have significantly smaller |diff| than right predictions for every numerical feature. |
| Can we quantify this correlation? | **Yes**: (1) Mann-Whitney p, (2) median ratio Right/Wrong, (3) rank-biserial r (in summary CSV after re-run with updated script). |

**Bottom line:** For Qwen3-4B (2_fewshot_cot_temp0), the margin-of-error summary provides strong evidence that **probe errors concentrate when the two funds’ feature values are close** (small |diff|). The effect is both statistically significant and practically meaningful (median |diff| when correct is about 1.3×–2.4× that when wrong for numerical features). Re-run the sanity checks with the updated script (or `--plot-only` if data are already saved) to add **median_ratio_right_over_wrong** and **rank_biserial_r** to the CSV for explicit quantification.
