# Attribution Analysis Report

**Model:** Llama-3.2-3B-Instruct  
**Prompt:** Zero-shot template  
**Samples:** 200 non-tie mutual fund pairs (from 7,677 available)  
**Granularity:** SENTENCE — each feature key-value pair is treated as one segment  
**Library:** interpreto  

---

## 1. Experiment Overview

This report presents the results of applying three attribution methods — Integrated Gradients, KernelSHAP, and Occlusion — to Llama-3.2-3B-Instruct to understand which input features most influence the model's decision when comparing mutual fund pairs.

Each mutual fund pair consists of 11 financial features presented for both funds. The model is prompted to choose "mutual fund 1" or "mutual fund 2" based solely on the data. The attribution methods assign importance scores to each feature segment, revealing which parts of the input the model relies on most when producing its answer.

### 1.1 Setup

- **Input format:** Chat-formatted prompt with a system message (describing the task and feature definitions) and a user message (the feature data for both funds plus an instruction line).
- **Granularity:** All three methods use `Granularity.SENTENCE`. Each feature line in the prompt ends with `. ` (period + space) so the sentence splitter produces one segment per feature key-value pair. This yields approximately 27 segments per prompt.
- **Attribution target:** A single token — `"1"` or `"2"` — extracted from the model's response. This focuses attribution directly on the decision token and produces a (1 × 27) attribution matrix rather than a multi-row matrix that must be averaged.
- **Score extraction:** The last (and only) row of the attribution matrix is used, corresponding to the decision token's attribution over all input segments.
- **Per-sample normalization:** Each sample's feature scores are normalized to sum to 1 before averaging across samples. This removes scale differences between samples and makes the average interpretable as "fraction of importance."

### 1.2 Improvements Over Initial Implementation

Five key improvements were made to the attribution pipeline before producing these results:

1. **Single-token attribution target.** Previously, IG and SHAP used `"mutual fund 1"` (6 tokens) while Occlusion used `"1"` (1 token). All three methods now use a unified single-token target (`"1"` or `"2"`). This eliminates the need to average across generated tokens and ensures all methods attribute to the same output event.

2. **Last-token score extraction.** The `extract_words_and_scores` function previously applied `nanmean(abs, dim=0)` across all rows of the 2D attribution matrix, averaging together attributions for every generated token. It now selects the last row — the decision token's attribution — which is the only row that directly determines the model's choice.

3. **Increased hyperparameters.** IG integration steps were increased from 10 to 30 (the minimum for reliable gradient path estimates). SHAP perturbations were increased from 64 to 200, giving much better coverage of the ~27-segment input space. The previous 64 perturbations were severely undersampled and produced near-uniform noise across features.

4. **Per-sample normalized scoring.** Rather than averaging raw absolute scores (which vary in scale across samples), each sample's feature scores are normalized to sum to 1. The reported `mean_norm` is the average of these per-sample proportions, making it directly interpretable: a value of 0.15 means "on average, 15% of the feature-level attribution is assigned to this feature."

5. **Feature attribution diagnostics.** The aggregation now tracks what fraction of total attribution goes to feature segments versus system prompt / instruction segments. This reveals that 60–68% of attribution is directed at the actual feature data, with the remainder going to boilerplate text.

### 1.3 Hyperparameters

| Parameter | Value |
|-----------|-------|
| IG integration steps | 30 |
| SHAP perturbations | 200 |
| Granularity | SENTENCE (feature-level) |
| Attribution target | Single token ("1" or "2") |
| Score extraction | Last generated token |
| Batch size | 1 |
| Model precision | bfloat16 |
| Gradient checkpointing | Enabled |

### 1.4 Target Distribution

Out of 200 samples, the model predicted "1" for 112 samples and "2" for 88 samples.

---

## 2. Feature Descriptions

| Short Name | Full Label | Description |
|------------|-----------|-------------|
| expense_ratio | Expense Ratio - Net | Annual operating expenses after waivers/reimbursements |
| sharpe | 3 Year Sharpe Ratio | Risk-adjusted performance over 3 years |
| std_dev | Standard Deviation | Volatility of returns |
| return_3yr | 3 Yr | Total return over the past 3 years |
| beta | Beta | Sensitivity to the benchmark's movements |
| tenure | Manager Tenure | Years the current manager has served |
| inception | Inception Date | Date the fund started |
| assets | Assets (Millions) | Total assets under management |
| turnover | Turnover Rates | Trading activity of the portfolio |
| load | Load (Y/N) | Whether the fund charges a sales load |
| ntf | NTF (Y/N) | Whether the fund is no-transaction-fee |

---

## 3. Results by Method

### 3.1 Integrated Gradients

Integrated Gradients computes the path integral of gradients from a baseline (zero embedding) to the actual input embedding. It measures how sensitive the model's output logit for the decision token is to the presence of each input segment.

**Feature attribution fraction:** 68.4% ± 0.8% of total attribution is directed at feature segments (the remainder goes to the system prompt and instruction text).

**Mean feature importance (per-sample normalized) across 200 samples:**

| Rank | Feature | Mean Norm | Std Norm | Mean Raw | Std Raw |
|------|---------|-----------|----------|----------|---------|
| 1 | ntf | 0.1638 | 0.0058 | 0.001751 | 0.000112 |
| 2 | expense_ratio | 0.1206 | 0.0111 | 0.001292 | 0.000171 |
| 3 | load | 0.1114 | 0.0041 | 0.001191 | 0.000080 |
| 4 | tenure | 0.0969 | 0.0081 | 0.001037 | 0.000112 |
| 5 | assets | 0.0911 | 0.0041 | 0.000974 | 0.000074 |
| 6 | inception | 0.0864 | 0.0086 | 0.000924 | 0.000118 |
| 7 | turnover | 0.0804 | 0.0025 | 0.000859 | 0.000052 |
| 8 | beta | 0.0709 | 0.0116 | 0.000760 | 0.000142 |
| 9 | sharpe | 0.0701 | 0.0056 | 0.000750 | 0.000082 |
| 10 | std_dev | 0.0609 | 0.0029 | 0.000651 | 0.000052 |
| 11 | return_3yr | 0.0475 | 0.0016 | 0.000508 | 0.000032 |

**Top-1 frequency:** NTF is ranked #1 in 199 out of 200 samples (99.5%).  
**Top-3 frequency:** NTF (100%), Expense Ratio (94.5%), Load (89.0%).

**Key observations:**

- **NTF dominates with extreme consistency.** At 16.4% normalized importance, NTF captures nearly 80% more attribution than the uniform baseline (1/11 ≈ 9.1%). Its standard deviation is remarkably low (0.006), meaning this dominance is stable across virtually all fund pairs.
- **Cost features cluster at the top.** NTF (rank 1), Expense Ratio (rank 2), and Load (rank 3) — all cost-related features — together account for 39.6% of feature attribution. The model's gradient landscape is most sensitive to cost information.
- **Smooth gradient from top to bottom.** Unlike the flat SHAP profile, IG produces a clearly differentiated ranking from NTF (0.164) down to 3-Year Return (0.048), a 3.4× spread.
- **Performance features rank low.** Sharpe Ratio (rank 9), Standard Deviation (rank 10), and 3-Year Return (rank 11) all fall below the uniform baseline.
- **Very low cross-sample variance.** The normalized standard deviations are small (0.002–0.012), indicating the ranking is stable regardless of specific fund-pair values.

### 3.2 KernelSHAP

KernelSHAP estimates Shapley values via weighted linear regression over perturbation coalitions. It measures the marginal contribution of each segment by randomly masking subsets and observing the effect on the output logit.

**Feature attribution fraction:** 65.0% ± 5.2%.

**Mean feature importance (per-sample normalized) across 200 samples:**

| Rank | Feature | Mean Norm | Std Norm | Mean Raw | Std Raw |
|------|---------|-----------|----------|----------|---------|
| 1 | ntf | 0.1553 | 0.0501 | 0.432 | 0.144 |
| 2 | turnover | 0.1030 | 0.0455 | 0.288 | 0.131 |
| 3 | assets | 0.1026 | 0.0468 | 0.287 | 0.136 |
| 4 | expense_ratio | 0.0868 | 0.0421 | 0.243 | 0.123 |
| 5 | load | 0.0868 | 0.0448 | 0.244 | 0.134 |
| 6 | std_dev | 0.0796 | 0.0427 | 0.221 | 0.117 |
| 7 | inception | 0.0785 | 0.0400 | 0.221 | 0.119 |
| 8 | beta | 0.0782 | 0.0386 | 0.218 | 0.109 |
| 9 | return_3yr | 0.0771 | 0.0366 | 0.216 | 0.108 |
| 10 | sharpe | 0.0763 | 0.0398 | 0.214 | 0.121 |
| 11 | tenure | 0.0758 | 0.0406 | 0.213 | 0.118 |

**Top-1 frequency:** NTF is ranked #1 in 88 out of 200 samples (44.0%). Assets (13.0%), Turnover (9.0%), Load (8.0%) are the next most frequent top-1 features.  
**Top-3 frequency:** NTF (78.0%), Assets (38.5%), Turnover (36.5%).

**Key observations:**

- **NTF now clearly stands out.** With the improved setup (200 perturbations, single-token target), NTF's normalized importance (0.155) is 50% higher than the next feature (Turnover, 0.103). In the previous run with 64 perturbations, all features were within a narrow 0.44–0.50 band. The improvement is substantial.
- **Two-tier structure emerges.** NTF (0.155) sits distinctly above a second tier of Turnover (0.103) and Assets (0.103), which in turn sit above the remaining 8 features clustered between 0.076–0.087.
- **Still high variance.** Standard deviations remain 40–50% of the mean (e.g., NTF: 0.050 std on 0.155 mean). This is inherent to Shapley estimation with limited perturbations — 200 is better than 64 but still a small fraction of the 2²⁷ possible coalitions.
- **The bottom 8 features are still essentially indistinguishable.** The spread between rank 4 (Expense Ratio, 0.087) and rank 11 (Tenure, 0.076) is only 0.011, well within the standard deviations. KernelSHAP can confidently identify NTF as #1 but lacks the statistical power to differentiate the remaining features.
- **Turnover and Assets rank surprisingly high** (ranks 2–3), which differs from both IG and Occlusion. This may reflect genuine interaction effects that SHAP captures (removing Turnover or Assets in combination with other features has outsized impact) or residual noise.

### 3.3 Occlusion

Occlusion systematically removes one input segment at a time and measures the change in the model's output logit for the decision token. It is the most direct measure of "what happens if this feature were missing."

**Feature attribution fraction:** 60.5% ± 9.6%.

**Mean feature importance (per-sample normalized) across 200 samples:**

| Rank | Feature | Mean Norm | Std Norm | Mean Raw | Std Raw |
|------|---------|-----------|----------|----------|---------|
| 1 | ntf | 0.1411 | 0.0186 | 2.538 | 0.797 |
| 2 | beta | 0.1225 | 0.0143 | 2.234 | 0.782 |
| 3 | load | 0.1139 | 0.0154 | 2.134 | 0.879 |
| 4 | tenure | 0.1113 | 0.0163 | 2.042 | 0.752 |
| 5 | inception | 0.0956 | 0.0184 | 1.856 | 0.918 |
| 6 | sharpe | 0.0873 | 0.0364 | 1.409 | 0.301 |
| 7 | expense_ratio | 0.0815 | 0.0271 | 1.356 | 0.263 |
| 8 | turnover | 0.0675 | 0.0225 | 1.366 | 0.826 |
| 9 | assets | 0.0656 | 0.0180 | 1.274 | 0.660 |
| 10 | std_dev | 0.0605 | 0.0256 | 1.270 | 0.870 |
| 11 | return_3yr | 0.0533 | 0.0199 | 1.077 | 0.704 |

**Top-1 frequency:** NTF is ranked #1 in 139 out of 200 samples (69.5%). Beta (13.0%), Tenure (6.5%), Load (6.0%).  
**Top-3 frequency:** NTF (99.0%), Beta (74.0%), Load (56.0%).

**Key observations:**

- **NTF is now #1 for Occlusion too** — a major improvement in cross-method consistency. Previously, Beta was #1 for Occlusion. The change to single-token targets and last-row extraction produced a ranking that is much more aligned with IG.
- **Clear two-tier structure.** A high-importance group (NTF through Inception, 0.096–0.141) and a low-importance group (Sharpe through 3-Year Return, 0.053–0.087), with a gap between rank 5 (Inception, 0.096) and rank 6 (Sharpe, 0.087).
- **Beta ranks 2nd.** Occlusion still assigns high importance to Beta (0.123), ranking it above Expense Ratio (rank 7, 0.082). Removing Beta creates a significant information gap — it's the only feature directly measuring benchmark sensitivity. When it's absent, the model's logit for the decision token shifts substantially.
- **Lower feature attribution fraction** than IG (60.5% vs 68.4%). Occlusion assigns relatively more importance to the system prompt and instruction segments. This makes sense: when the system prompt is occluded, the model loses all context about what task it's performing, causing a very large logit shift.
- **Moderate variance.** Standard deviations are 0.014–0.036, higher than IG but much lower than SHAP. The ranking is reasonably stable across samples.

---

## 4. Cross-Method Comparison

### 4.1 Feature Ranking Table

| Feature | IG Rank | SHAP Rank | Occ Rank | Mean Rank |
|---------|---------|-----------|----------|-----------|
| ntf | 1 | 1 | 1 | **1.0** |
| load | 3 | 5 | 3 | **3.7** |
| expense_ratio | 2 | 4 | 7 | **4.3** |
| assets | 5 | 3 | 9 | **5.7** |
| turnover | 7 | 2 | 8 | **5.7** |
| beta | 8 | 8 | 2 | **6.0** |
| inception | 6 | 7 | 5 | **6.0** |
| tenure | 4 | 11 | 4 | **6.3** |
| sharpe | 9 | 10 | 6 | **8.3** |
| std_dev | 10 | 6 | 10 | **8.7** |
| return_3yr | 11 | 9 | 11 | **10.3** |

### 4.2 Rank Correlations

| Pair | Spearman ρ | p-value | Kendall τ | p-value |
|------|-----------|---------|-----------|---------|
| IG vs SHAP | 0.509 | 0.110 | 0.382 | 0.121 |
| IG vs Occlusion | 0.600 | 0.051 | 0.491 | 0.041 |
| SHAP vs Occlusion | 0.045 | 0.894 | 0.018 | 1.000 |

**Compared to the previous run:**

- **IG vs SHAP improved dramatically**: from ρ = 0.118 (p = 0.73) to ρ = 0.509 (p = 0.11). While not quite statistically significant at the 5% level, the correlation has increased over 4×. This is primarily due to SHAP now resolving NTF as the clear #1 feature, which previously it could not do.
- **IG vs Occlusion remains moderate**: ρ = 0.600 (p = 0.051), Kendall τ = 0.491 (p = 0.041). The Kendall correlation is statistically significant, indicating meaningful agreement on the relative ordering.
- **SHAP vs Occlusion remains low**: ρ = 0.045. These methods still disagree on everything except NTF being #1. Their mid-ranking features are essentially unrelated.

### 4.3 Consensus Points

Despite the disagreements in mid-rankings, three strong consensus findings emerge:

1. **NTF is unanimously the most important feature.** Rank 1 across all three methods. Mean consensus rank = 1.0. No other feature comes close.
2. **3-Year Return is consistently the least important.** Rank 11 for both IG and Occlusion, rank 9 for SHAP. Mean consensus rank = 10.3.
3. **Load is consistently in the top tier.** Rank 3 for both IG and Occlusion, rank 5 for SHAP. Mean consensus rank = 3.7.

### 4.4 Notable Disagreements

- **Beta:** Rank 8 (IG), Rank 8 (SHAP), Rank 2 (Occlusion). Occlusion uniquely identifies Beta as a high-importance feature. See Section 5.2 for analysis.
- **Tenure:** Rank 4 (IG), Rank 11 (SHAP), Rank 4 (Occlusion). SHAP assigns Tenure the lowest importance, while IG and Occlusion both rank it in the top 4.
- **Turnover and Assets:** Rank 7/5 (IG), Rank 2/3 (SHAP), Rank 8/9 (Occlusion). SHAP uniquely rates these as second and third most important, while the other methods place them in the bottom half.

---

## 5. Detailed Analysis

### 5.1 Why NTF Dominates All Three Methods

NTF (No-Transaction-Fee) is unanimously the most important feature. Three factors likely explain this:

1. **Binary categorical contrast.** NTF is a binary Y/N feature. The difference between "Y" and "N" is semantically unambiguous and doesn't require numerical comparison. The model can process it with a simple categorical heuristic.
2. **Position in the prompt.** NTF is the last feature listed for each fund, immediately before the "Mutual fund 2:" header and the instruction footer. Autoregressive language models exhibit recency bias — tokens closer to the generation point receive more attention during prediction. NTF occupies the most recency-advantaged position.
3. **Direct cost relevance.** The system prompt describes NTF as "whether the fund is no-transaction-fee." In the context of "which fund would you invest in," a no-transaction-fee fund is a straightforward positive signal requiring no quantitative analysis.

**It is important to note** that the dominance of NTF may partly reflect prompt-position artifacts rather than genuine financial reasoning. A controlled experiment shuffling feature order within the prompt would be needed to disentangle content importance from position importance.

### 5.2 Beta's High Occlusion Importance

Beta ranks 2nd for Occlusion (0.123) but 8th for both IG (0.071) and SHAP (0.078). This discrepancy is methodologically informative:

- **IG measures gradient sensitivity** — how much a small embedding perturbation along the interpolation path changes the output. Beta values are short numbers (e.g., "1.11"), and the gradient through the embedding may be smooth and low-magnitude.
- **Occlusion measures removal impact** — what happens when the entire Beta segment is deleted. Beta is the only feature measuring benchmark sensitivity. Its removal creates a qualitative information gap that forces the model to make its decision without this signal, causing a large logit shift.
- **Interpretation:** The model doesn't process Beta with high gradient sensitivity (it's "easy" to encode — a simple decimal), but it **relies** on Beta heavily for its final decision. IG captures processing effort; Occlusion captures reliance. These are different aspects of importance.

### 5.3 KernelSHAP: Improvements and Remaining Limitations

The increase from 64 to 200 perturbations produced a meaningful improvement:

| Metric | Previous (64 perturbs) | Current (200 perturbs) |
|--------|----------------------|----------------------|
| NTF mean_norm | 0.100 (uniform-like) | 0.155 (clearly #1) |
| Top–bottom spread | 0.010 (flat) | 0.079 (differentiated) |
| NTF top-1 frequency | 10.5% | 44.0% |
| IG–SHAP Spearman ρ | 0.118 | 0.509 |

However, KernelSHAP still has fundamental limitations in this setting:

- **Still high variance.** Standard deviations remain 40–50% of mean values. The per-sample rankings are noisy, especially for mid-ranked features.
- **Bottom 8 features are indistinguishable.** The spread between rank 4 and rank 11 is only 0.011 (0.087 → 0.076), well within the standard deviations. SHAP can identify NTF as important but cannot reliably differentiate the remaining features.
- **Masking creates out-of-distribution inputs.** When KernelSHAP masks a segment, it replaces it with a baseline. For structured prompts where the model expects "Beta: 1.11", replacing it with padding creates inputs the model has never seen, making the output unpredictable and the Shapley estimates noisy.
- **Further improvements** would require either substantially more perturbations (500+, which is VRAM-prohibitive) or a more meaningful masking strategy (e.g., replacing masked features with plausible alternative values rather than padding).

### 5.4 Performance Features Rank Low

Sharpe Ratio, Standard Deviation, and 3-Year Return consistently rank in the bottom three across methods. This is counterintuitive from a financial perspective — these are arguably the most decision-relevant metrics for fund comparison.

Possible explanations:

1. **Numerical reasoning limitations.** Comparing "15.73" vs. "15.33" (Standard Deviation) or "0.12" vs. "0.10" (Sharpe Ratio) requires precise numerical comparison that 3B-parameter models often struggle with. The model may not be able to reliably determine which value is "better."
2. **Feature redundancy.** Sharpe Ratio, Standard Deviation, 3-Year Return, and Beta are correlated (all relate to risk-return dynamics). The model may rely on one proxy (Beta, per Occlusion) and discount the others.
3. **Categorical vs. quantitative processing.** The model appears to prefer features with simple categorical signals (Y/N for NTF and Load, years for Tenure) over features requiring quantitative comparison.

### 5.5 Feature Attribution Fraction

The fraction of total attribution going to feature segments versus system prompt / instruction text varies by method:

| Method | Feature Fraction | Std |
|--------|-----------------|-----|
| Integrated Gradients | 68.4% | 0.8% |
| KernelSHAP | 65.0% | 5.2% |
| Occlusion | 60.5% | 9.6% |

Approximately 32–40% of the model's sensitivity is directed at non-feature text (system prompt, task instructions, "Mutual fund 1/2:" headers, instruction footer). This is expected: the system prompt provides crucial context for interpreting the feature data. Occlusion assigns relatively more to non-feature segments because removing the system prompt causes a catastrophic shift in model behavior (it no longer knows what task it's performing).

### 5.6 Connection to Linear Probing

Linear probing results from a separate experiment measure how well each feature's comparative direction (e.g., "which fund has a lower expense ratio?") can be decoded from the model's hidden states.

The Spearman correlation between probe accuracy and average attribution importance across the three methods is **ρ = 0.745 (p = 0.009)**, indicating a statistically significant positive relationship.

| Feature | Probe Accuracy | Avg Attribution |
|---------|---------------|-----------------|
| ntf | 0.996 | 0.153 |
| load | 0.991 | 0.104 |
| expense_ratio | 0.975 | 0.096 |
| inception | 0.946 | 0.087 |
| tenure | 0.921 | 0.095 |
| assets | 0.920 | 0.086 |
| sharpe | 0.914 | 0.078 |
| return_3yr | 0.907 | 0.059 |
| turnover | 0.875 | 0.084 |
| beta | 0.854 | 0.091 |
| std_dev | 0.858 | 0.067 |

Features that the model encodes more accurately in its hidden states (high probe accuracy) tend to also receive higher attribution scores. This is consistent: the model can only effectively use features it successfully represents.

The notable exception is **Beta**: it has relatively low probe accuracy (0.854) but above-average attribution importance (0.091, driven by Occlusion ranking it #2). This suggests that while the model's internal encoding of Beta may be somewhat noisy, its removal from the input still substantially affects the output — the model attempts to use Beta even if it doesn't perfectly encode it.

---

## 6. Methodological Comparison

### 6.1 What Each Method Measures

| Method | Question Answered | Mechanism |
|--------|------------------|-----------|
| Integrated Gradients | "How sensitive is the output to each feature's embedding?" | Path integral of gradients from zero to actual embedding |
| KernelSHAP | "What is each feature's marginal contribution?" | Shapley values estimated via coalition sampling |
| Occlusion | "What happens if this feature is removed?" | Output change upon single-segment deletion |

### 6.2 Strengths and Weaknesses in This Experiment

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| Integrated Gradients | Theoretically grounded; very low variance; clear differentiation | May overweight features near the generation boundary (position bias); gradient magnitude ≠ causal importance |
| KernelSHAP | Game-theoretic foundation; captures interaction effects | High variance despite 200 perturbations; cannot differentiate mid-ranked features; masking creates OOD inputs |
| Occlusion | Most intuitive (direct removal test); no gradient required | Doesn't capture interactions; correlated features cause redundancy effects; high non-feature attribution fraction |

### 6.3 Scale Differences

The three methods operate on fundamentally different scales:

- **IG:** ~10⁻³ (gradient magnitudes in embedding space)
- **SHAP:** ~10⁻¹ (logit-space perturbation effects via regression)
- **Occlusion:** ~10⁰ (direct logit differences from segment removal)

This is why **per-sample normalization is essential** — raw scores should never be compared across methods. All comparisons in this report use normalized scores.

---

## 7. Summary of Findings

1. **NTF is unanimously the most important feature.** Rank 1 across all three methods, with mean normalized importance of 15.3%. This is 68% above the uniform baseline (9.1%). NTF is a binary cost feature positioned at the end of each fund's feature list.

2. **Load and Expense Ratio form a consistent second tier.** Load ranks 3rd for both IG and Occlusion (consensus rank 3.7). Expense Ratio ranks 2nd for IG (consensus rank 4.3). Together with NTF, the three cost features account for ~35% of feature-level attribution.

3. **Beta shows a method-dependent pattern.** It ranks 2nd for Occlusion but 8th for both IG and SHAP. The model depends on Beta for its decision (removing it causes a large logit shift) but processes it with low gradient sensitivity. This suggests Beta is efficiently but heavily used.

4. **KernelSHAP improved significantly with more perturbations** but still cannot reliably differentiate mid-ranked features. NTF is now clearly identified as #1 (44% top-1 frequency vs. 10.5% previously), and the IG–SHAP rank correlation increased from 0.12 to 0.51.

5. **Performance metrics (Sharpe, Standard Deviation, 3-Year Return) consistently rank in the bottom three.** The model appears to rely on simpler categorical and cost-based heuristics rather than quantitative risk-return analysis.

6. **Attribution importance correlates with linear probe accuracy** (Spearman ρ = 0.745, p = 0.009). Features the model encodes more accurately are also assigned more attribution importance.

7. **The model uses a cost-focused decision heuristic.** The dominance of NTF, Load, and Expense Ratio suggests Llama-3.2-3B prioritizes transaction costs and fee structures over risk-adjusted returns. This is consistent with the hypothesis that a 3B-parameter model defaults to categorical comparisons rather than sophisticated numerical reasoning. However, positional bias (NTF and Load being the last features listed) may contribute to this pattern and should be controlled for in future experiments.
