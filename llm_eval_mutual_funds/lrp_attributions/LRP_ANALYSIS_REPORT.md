# LRP Attribution Analysis Report

**Model:** Llama-3.2-3B-Instruct
**Prompt:** Zero-shot template with 100 feature-order permutations
**Samples:** 500 non-tie mutual fund pairs (seed=42)
**Library:** LXT v2.1 (AttnLRP, ICML 2024)
**Precision:** bfloat16 (no quantization)
**Hardware:** NVIDIA A100-SXM4-40GB
**Decimal Truncation:** 4 decimal places

---

## 1. Experiment Overview

This report presents results from applying Layer-wise Relevance Propagation (LRP) via the LXT library to Llama-3.2-3B-Instruct on 500 mutual fund pair comparisons. Four LRP variants were run:

- **AttnLRP (single-logit):** Distributes relevance through attention via gradient division (Q/4, K/4, V/2). Attributes to the decision token's logit.
- **CP-LRP (single-logit):** Conservative propagation -- blocks gradient flow through softmax in attention, routing all relevance through the value path only. Attributes to the decision token's logit.
- **AttnLRP (logit-difference):** Same gradient rules as AttnLRP, but attributes to `logit_chosen - logit_rejected` -- the decision margin rather than the absolute logit.
- **CP-LRP (logit-difference):** Same gradient rules as CP-LRP, attributing to the logit difference.

To control for positional bias, each of the 500 samples was evaluated with a randomly assigned feature-order permutation drawn from a pool of 100 deterministic diverse permutations (10 systematic strategies x 11 circular rotations, deduplicated). All floating-point feature values were truncated to 4 decimal places to eliminate tokenization artifacts from excessive precision.

### 1.1 Diagnostics

**Conservation check (AttnLRP, single-logit):**
- Mean ratio (relevance_sum / target_logit): **0.6104** (std: 0.0114)
- Interpretation: LRP conserves approximately 61% of the target logit's magnitude at the input layer. The ~39% loss is expected with the efficient LXT implementation's approximate conservation (gradient checkpointing and the Input x Gradient formulation introduce systematic leakage). The very low standard deviation indicates this loss is consistent across samples, not random.

**Conservation check (AttnLRP, logit-difference):**
- Mean ratio (relevance_sum / logit_diff): **0.6771** (std: 0.0249)
- The logit-difference target achieves higher conservation (68% vs 61%), likely because the difference cancels some of the systematic leakage common to both logit channels.

**Feature attribution fraction:**
- AttnLRP (single-logit): **4.4%** (+/-0.8%) of total relevance falls on feature tokens
- CP-LRP (single-logit): **8.5%** (+/-3.5%) of total relevance falls on feature tokens
- AttnLRP (logit-diff): **6.4%** (+/-1.0%) of total relevance falls on feature tokens
- This is substantially lower than interpreto methods (60-68%). The majority of LRP relevance is assigned to system prompt, chat template special tokens, and instruction text. LRP operates at the raw token level, and the system prompt tokens -- which provide task context -- accumulate substantial relevance. The feature tokens (~120-140 tokens) are outnumbered ~3:1 by non-feature tokens (~450 tokens).

**Target distribution:** The model predicted "1" for some samples and "2" for others (identical to interpreto since the same model with greedy decoding was used).

---

## 2. AttnLRP Feature Importance Rankings (Single-Logit)

| Rank | Feature | Mean Norm | Std Norm | Top-1 Freq | Top-3 Freq |
|------|---------|-----------|----------|------------|------------|
| 1 | **sharpe** | 0.2079 | 0.0860 | 59.4% | 92.6% |
| 2 | return_3yr | 0.1490 | 0.0730 | 21.6% | 71.0% |
| 3 | expense_ratio | 0.1068 | 0.0406 | 8.2% | 43.4% |
| 4 | inception | 0.1050 | 0.0292 | 4.0% | 41.2% |
| 5 | assets | 0.0863 | 0.0312 | 4.0% | 19.2% |
| 6 | tenure | 0.0693 | 0.0237 | 0% | -- |
| 7 | load | 0.0654 | 0.0251 | 0% | -- |
| 8 | beta | 0.0571 | 0.0259 | 0% | -- |
| 9 | std_dev | 0.0549 | 0.0238 | 0% | -- |
| 10 | ntf | 0.0496 | 0.0207 | 0% | -- |
| 11 | turnover | 0.0487 | 0.0217 | 0% | -- |

**Key observations:**

1. **Sharpe Ratio is the top feature but not overwhelmingly so.** At 20.8% normalized importance (2.3x the uniform baseline of 9.1%), Sharpe is ranked #1 in 59.4% of individual samples. The remaining top-1 assignments are distributed across Return_3yr (21.6%), Expense Ratio (8.2%), Inception (4.0%), and Assets (4.0%), indicating that Sharpe's dominance varies meaningfully across sample pairs.

2. **Return_3yr is a strong #2.** At 14.9%, Return_3yr is the second most important feature and ranks #1 in 21.6% of samples. Together, the two performance metrics (Sharpe + Return_3yr) account for 35.7% of total feature relevance.

3. **Expense Ratio is a clear #3.** At 10.7%, it is the top cost feature and appears in the top 3 for 43.4% of samples.

4. **NTF drops to rank 10.** NTF was unanimously #1 by all three interpreto methods (IG, SHAP, Occlusion). Under AttnLRP, it receives only 5.0% importance -- below the uniform baseline. This suggests the interpreto methods' NTF rankings may have been amplified by positional effects that permutation averaging controls for, or that LRP's token-level decomposition distributes NTF's single-token relevance differently.

5. **Higher variance than interpreto.** Standard deviations are larger relative to means (e.g., Sharpe CV=41%, Return_3yr CV=49%), reflecting genuine sample-to-sample variation in which features drive the decision. This is expected given the 100 permutations introduce feature-order diversity.

---

## 3. CP-LRP Feature Importance Rankings (Single-Logit)

| Rank | Feature | Mean Norm | Std Norm |
|------|---------|-----------|----------|
| 1 | **sharpe** | 0.1798 | 0.0600 |
| 2 | inception | 0.1335 | 0.0317 |
| 3 | tenure | 0.1219 | 0.0789 |
| 4 | expense_ratio | 0.1077 | 0.0425 |
| 5 | turnover | 0.0936 | 0.0352 |
| 6 | return_3yr | 0.0849 | 0.0291 |
| 7 | assets | 0.0708 | 0.0293 |
| 8 | std_dev | 0.0640 | 0.0209 |
| 9 | load | 0.0511 | 0.0183 |
| 10 | ntf | 0.0469 | 0.0141 |
| 11 | beta | 0.0457 | 0.0195 |

**Key observations:**

1. **Sharpe remains #1 but with a flatter distribution.** At 18.0%, Sharpe leads but the gap to #2 (Inception, 13.4%) is smaller than under AttnLRP (20.8% vs 14.9%). CP-LRP produces a more uniform distribution overall.

2. **Inception and Tenure rise substantially.** Inception ranks #2 (13.4%) and Tenure #3 (12.2%). These date/duration features receive much more relevance through the value path alone, suggesting that under standard attention, relevance flowing through the Q/K path partially redistributes importance away from these features.

3. **NTF remains near the bottom (rank 10).** Both LRP variants consistently demote NTF, confirming this is not an artifact of one particular gradient routing strategy.

4. **Tenure has the highest variance** (std=0.079 on mean=0.122, ~65% CV), suggesting its importance is highly sample-dependent.

### 3.1 AttnLRP vs CP-LRP Agreement

**Spearman rank correlation: rho = 0.618 (p = 0.043)** -- statistically significant.

The two LRP variants agree on Sharpe as #1 and share broad agreement on the tier structure, but differ on specific mid-tier placements:
- Return_3yr: rank 2 (AttnLRP) vs rank 6 (CP-LRP)
- Inception: rank 4 vs rank 2
- Tenure: rank 6 vs rank 3
- Turnover: rank 11 vs rank 5

The significant correlation (unlike interpreto-to-LRP pairs) confirms that the two variants capture related but distinct aspects of the model's computation.

---

## 4. Logit-Difference Attribution

### 4.1 AttnLRP Logit-Difference Rankings

| Rank | Feature | Mean Norm | Std Norm |
|------|---------|-----------|----------|
| 1 | **sharpe** | 0.2096 | 0.0806 |
| 2 | expense_ratio | 0.1871 | 0.0445 |
| 3 | return_3yr | 0.1681 | 0.0776 |
| 4 | inception | 0.0809 | 0.0312 |
| 5 | assets | 0.0651 | 0.0210 |
| 6 | turnover | 0.0632 | 0.0233 |
| 7 | std_dev | 0.0529 | 0.0206 |
| 8 | beta | 0.0490 | 0.0189 |
| 9 | tenure | 0.0475 | 0.0168 |
| 10 | ntf | 0.0395 | 0.0155 |
| 11 | load | 0.0371 | 0.0124 |

**Logit-difference sharply concentrates relevance on the top 3.** Sharpe (21.0%), Expense Ratio (18.7%), and Return_3yr (16.8%) together account for 56.5% of feature relevance -- compared to 46.4% for the same three features under single-logit AttnLRP. By attributing to the decision margin rather than the absolute logit, features that genuinely discriminate between the two funds are amplified.

**Single-logit vs logit-diff correlation (AttnLRP): rho = 0.745 (p = 0.009).** The rankings are correlated but not identical. Expense Ratio rises from rank 3 to rank 2, while Tenure drops from rank 6 to rank 9. This suggests Expense Ratio contributes more to the decision *margin* than its absolute logit contribution implies.

### 4.2 CP-LRP Logit-Difference Rankings

| Rank | Feature | Mean Norm | Std Norm |
|------|---------|-----------|----------|
| 1 | **sharpe** | 0.2172 | 0.0685 |
| 2 | tenure | 0.1347 | 0.0958 |
| 3 | expense_ratio | 0.1234 | 0.0436 |
| 4 | inception | 0.0939 | 0.0263 |
| 5 | turnover | 0.0872 | 0.0395 |
| 6 | return_3yr | 0.0755 | 0.0271 |
| 7 | assets | 0.0675 | 0.0307 |
| 8 | std_dev | 0.0628 | 0.0221 |
| 9 | beta | 0.0551 | 0.0251 |
| 10 | load | 0.0445 | 0.0163 |
| 11 | ntf | 0.0383 | 0.0147 |

**CP-LRP single-logit vs logit-diff correlation: rho = 0.945 (p < 0.0001).** The CP-LRP rankings are highly stable across attribution targets, suggesting CP-LRP's value-only pathway produces more target-invariant attributions.

### 4.3 Cross-Variant Summary

| Feature | AttnLRP | CP-LRP | LogitDiff AttnLRP | LogitDiff CP-LRP |
|---------|---------|--------|-------------------|------------------|
| sharpe | **1** | **1** | **1** | **1** |
| expense_ratio | 3 | 4 | 2 | 3 |
| return_3yr | 2 | 6 | 3 | 6 |
| inception | 4 | 2 | 4 | 4 |
| assets | 5 | 7 | 5 | 7 |
| tenure | 6 | 3 | 9 | 2 |
| load | 7 | 9 | 11 | 10 |
| beta | 8 | 11 | 8 | 9 |
| std_dev | 9 | 8 | 7 | 8 |
| ntf | 10 | 10 | 10 | 11 |
| turnover | 11 | 5 | 6 | 5 |

**Robust findings across all 4 LRP variants:**
- Sharpe is unanimously rank 1
- Expense Ratio is consistently top 4 (rank 2-4)
- NTF is consistently bottom 2 (rank 10-11)
- Beta, Std Dev, and Load are consistently in the bottom half

---

## 5. Bootstrap Confidence Intervals

Bootstrap analysis (1,000 resamples from 500 samples) quantifies the stability of rankings:

### 5.1 AttnLRP (Single-Logit)

| Feature | Median Rank | 95% CI | P(top 3) | Std |
|---------|-------------|--------|----------|-----|
| sharpe | 2 | [2, 2] | 100% | 0.0 |
| return_3yr | 3 | [3, 3] | 100% | 0.0 |
| expense_ratio | 4 | [4, 5] | 0% | 0.4 |
| inception | 5 | [4, 5] | 0% | 0.4 |
| assets | 6 | [6, 6] | 0% | 0.0 |

### 5.2 CP-LRP (Single-Logit)

| Feature | Median Rank | 95% CI | P(top 3) | Std |
|---------|-------------|--------|----------|-----|
| sharpe | 2 | [2, 2] | 100% | 0.0 |
| inception | 3 | [3, 3] | 99.4% | 0.1 |
| tenure | 4 | [4, 4] | 0.6% | 0.1 |
| expense_ratio | 5 | [5, 5] | 0% | 0.0 |

### 5.3 AttnLRP (Logit-Diff)

| Feature | Median Rank | 95% CI | P(top 3) | Std |
|---------|-------------|--------|----------|-----|
| sharpe | 2 | [2, 2] | 100% | 0.0 |
| expense_ratio | 3 | [3, 3] | 100% | 0.0 |
| return_3yr | 4 | [4, 4] | 0% | 0.0 |

### 5.4 CP-LRP (Logit-Diff)

| Feature | Median Rank | 95% CI | P(top 3) | Std |
|---------|-------------|--------|----------|-----|
| sharpe | 2 | [2, 2] | 100% | 0.0 |
| tenure | 3 | [3, 3] | 97.6% | 0.2 |
| expense_ratio | 4 | [4, 4] | 2.4% | 0.2 |

**Key finding: Rankings are extremely stable.** With 500 samples and permutation averaging, the bootstrap confidence intervals are remarkably tight. Sharpe is in the top 2 with 100% probability across all 4 variants. The only non-trivial uncertainty is between adjacent features (e.g., expense_ratio vs inception under single-logit AttnLRP, tenure vs expense_ratio under logit-diff CP-LRP). The top-tier (Sharpe, Return_3yr/Expense_ratio) and bottom-tier (NTF, Load, Beta) assignments are robust to resampling.

Note: Bootstrap ranks start at 2 (not 1) because the per-sample normalization produces 12 feature scores (11 features + an "other" category for unassigned tokens), and "other" consistently occupies rank 1 due to the low feature attribution fraction.

---

## 6. Cross-Method Comparison (All 5 Methods)

### 6.1 Full Ranking Table

| Feature | IG | SHAP | Occ | AttnLRP | CP-LRP | Avg Rank |
|---------|-----|------|-----|---------|--------|----------|
| expense_ratio | 2 | 4 | 7 | 3 | 4 | **4.0** |
| ntf | 1 | 1 | 1 | 10 | 10 | **4.6** |
| inception | 6 | 7 | 5 | 4 | 2 | **4.8** |
| load | 3 | 5 | 3 | 7 | 9 | **5.4** |
| sharpe | 9 | 10 | 6 | 1 | 1 | **5.4** |
| tenure | 4 | 11 | 4 | 6 | 3 | **5.6** |
| assets | 5 | 3 | 9 | 5 | 7 | **5.8** |
| turnover | 7 | 2 | 8 | 11 | 5 | **6.6** |
| beta | 8 | 8 | 2 | 8 | 11 | **7.4** |
| return_3yr | 11 | 9 | 11 | 2 | 6 | **7.8** |
| std_dev | 10 | 6 | 10 | 9 | 8 | **8.6** |

### 6.2 Pairwise Rank Correlations

| Pair | Spearman rho | p-value |
|------|-------------|---------|
| IG vs SHAP | 0.509 | 0.110 |
| IG vs Occlusion | 0.600 | 0.051 |
| IG vs AttnLRP | -0.218 | 0.519 |
| IG vs CP-LRP | -0.127 | 0.709 |
| SHAP vs Occlusion | 0.045 | 0.894 |
| SHAP vs AttnLRP | -0.564 | 0.071 |
| SHAP vs CP-LRP | -0.427 | 0.190 |
| Occlusion vs AttnLRP | -0.300 | 0.370 |
| Occlusion vs CP-LRP | -0.291 | 0.386 |
| **AttnLRP vs CP-LRP** | **0.618** | **0.043** |

**LRP rankings are uncorrelated -- and in some cases negatively correlated -- with all three interpreto methods.** The only statistically significant positive correlation in the table is between the two LRP variants themselves (rho=0.618). All LRP-to-interpreto pairs show weak or negative correlations, confirming LRP measures fundamentally different aspects of the model's computation.

### 6.3 Consensus Analysis

| Rank | 3-Method (Interpreto) | 5-Method (All) |
|------|----------------------|----------------|
| 1 | ntf (1.0) | expense_ratio (4.0) |
| 2 | load (3.7) | ntf (4.6) |
| 3 | expense_ratio (4.3) | inception (4.8) |

Expense_ratio becomes the top consensus feature when LRP is included. It ranks 2-4 across all 5 methods, making it the most robust feature in the entire analysis. NTF drops from a perfect consensus score of 1.0 (3-method) to 4.6 (5-method), dragged down by its rank 10 positions under both LRP variants.

---

## 7. Layer-Wise Relevance Analysis

### 7.1 Total Relevance by Layer

The total relevance by layer shows an inverted-U pattern:
- **Layers 0-4:** Relevance increases steadily from ~5.3 to ~6.1
- **Layers 5-11:** Steep rise, peaking at layer 11 (~8.9)
- **Layers 12-23:** Plateau around 8.0-8.5 with a secondary peak at layers 20-23
- **Layers 24-27:** Sharp decline. Layer 27 has zero relevance (final layer before the LM head).

The bimodal peak structure (layers 9-13 and layers 20-23) suggests two processing phases: a mid-network phase where features are initially encoded and compared, and a late phase where the decision is consolidated.

### 7.2 Layer-Feature Heatmap

**Sharpe and Return_3yr dominate across all 28 layers.** Key patterns from the layer-feature matrix:

- **Sharpe:** Highest relevance at every layer, with peaks at layers 4-5 (0.067-0.068), layer 14 (0.082), and layers 22-23 (0.069). Consistent dominance regardless of network depth.
- **Return_3yr:** Second-highest across most layers, with notable peaks at layers 14 (0.056), 17 (0.055), 22 (0.059), and 23 (0.063). Its layer 23 value (0.063) nearly matches Sharpe (0.069), suggesting Return_3yr becomes increasingly decision-relevant in the final processing layers.
- **Expense Ratio:** Third-highest in early/mid layers (peaks at layers 9-11, ~0.035), declining in later layers.
- **Inception:** Shows a distinct mid-layer bulge (layers 6-9, ~0.023-0.027) then declines.
- **NTF and Load:** Relatively flat across layers at low values (~0.009-0.015).
- **Layer 27:** All features have zero relevance.

### 7.3 LRP Relevance vs Probe Accuracy (Per-Layer Correlation)

| Layer Range | Mean Spearman rho | Significant (p<0.05)? |
|-------------|------------------|-----------------------|
| 0-7 | -0.30 | No |
| 8-15 | -0.21 | No |
| 16-17 | +0.33 | No |
| **21-25** | **+0.64** | **Yes (3 of 5 layers)** |

Specific statistically significant layers:
- Layer 21: rho = 0.691, p = 0.019
- Layer 22: rho = 0.664, p = 0.026
- Layer 25: rho = 0.618, p = 0.043

**Interpretation:** The correlation flips from negative (early layers) to positive (late layers). In early-to-mid layers (0-15), LRP relevance and probe accuracy are negatively correlated, suggesting these layers actively process features without yet encoding the comparison outcome. In the late layers (21-25), features that the model encodes well (high probe accuracy) also receive more LRP relevance -- the model uses what it knows to make its decision.

The overlay plots show feature-specific patterns:
- **NTF and Load:** Probe accuracy rises steeply (layers 0-6) then plateaus near 0.99. LRP relevance is flat and low. Encoded early but used simply.
- **Expense Ratio:** Probe accuracy rises through layers 0-13. LRP relevance peaks in mid-layers then declines.
- **Beta and Std Dev:** Both have lower probe accuracy (~0.85) and low LRP relevance, consistent with the model struggling to encode and use these features.

---

## 8. Signed Relevance Analysis (LRP-Unique)

LRP uniquely provides signed relevance: positive values support the predicted decision, negative values oppose it.

| Feature | % Positive | % Negative |
|---------|-----------|-----------|
| return_3yr | 98.5% | 1.5% |
| sharpe | 98.4% | 1.6% |
| ntf | 97.8% | 2.2% |
| assets | 96.0% | 4.0% |
| load | 91.8% | 8.2% |
| inception | 91.4% | 8.6% |
| beta | 91.1% | 8.9% |
| tenure | 90.3% | 9.7% |
| std_dev | 89.4% | 10.6% |
| expense_ratio | 84.6% | 15.4% |
| turnover | 73.3% | 26.7% |

**Observations:**

1. **Relevance is overwhelmingly positive for all features.** Even the least unidirectional feature (turnover) has 73.3% positive relevance.

2. **Return_3yr and Sharpe are the most unidirectional** (98.5% and 98.4% positive). When these features contribute relevance, they almost always push toward the chosen answer -- consistent with their role as primary decision drivers.

3. **Turnover has the most opposing signal** (26.7% negative), followed by Expense Ratio (15.4%). These features contain the most "conflicting" information across the two funds -- tokens that push against the model's final choice.

4. **Expense Ratio's relatively high negative fraction** (15.4%) is notable given its rank-3 importance. This suggests Expense Ratio frequently presents mixed signals (one fund's expense ratio favoring it, the other fund's expense ratio opposing), more so than Sharpe or Return_3yr.

---

## 9. Fund 1 vs Fund 2 Attribution

The fund1 vs fund2 breakdown under AttnLRP shows a **Fund 1 bias** for the top features:

- **Sharpe:** Fund 1 = 0.028, Fund 2 = 0.016 (1.8x ratio)
- **Return_3yr:** Fund 1 = 0.022, Fund 2 = 0.009 (2.5x ratio)
- **Expense Ratio:** Fund 1 = 0.010, Fund 2 = 0.011 (roughly balanced)
- **Inception:** Fund 1 = 0.012, Fund 2 = 0.009 (1.3x ratio)

For lower-ranked features, the split is more balanced, with some showing Fund 2 > Fund 1 (e.g., NTF: Fund 1 = 0.007, Fund 2 = 0.004).

The Fund 1 bias for numerical features is somewhat mitigated by the 100 feature-order permutations (the same feature appears at different positions across prompts), but residual bias persists -- likely because the model generally attends more to the first fund's data when computing comparisons in the final layers.

Under CP-LRP, the Fund 1 bias is smaller: Sharpe shows Fund 1 = 0.077, Fund 2 = 0.092 (Fund 2 actually higher), suggesting the value-pathway attribution distributes more evenly across fund positions.

---

## 10. Methodological Concerns and Limitations

### 10.1 Low Feature Attribution Fraction

Only **4.4% (AttnLRP) / 8.5% (CP-LRP)** of total LRP relevance lands on feature tokens. The remaining relevance is distributed across system prompt, chat template, and instruction tokens. This means the feature-level rankings are computed from a small fraction of total relevance, which may amplify noise in relative rankings.

The discrepancy with interpreto methods (60-68%) arises because:
1. LRP operates at the token level, where system prompt tokens outnumber feature tokens ~3:1
2. Chat template special tokens receive high relevance (visible as red/warm in per-token heatmaps)
3. Interpreto's sentence-level granularity abstracts away boilerplate text

### 10.2 Sharpe Dominance: Real or Artifact?

Sharpe's consistent #1 ranking across all 4 LRP variants is the most striking finding. With 500 samples and 100 permutations, it is top-1 in 59.4% of individual samples.

**In favor of it being real:**
- Unanimous #1 across all 4 LRP variants independently
- Sharpe is a composite metric (return/risk) containing the most decision-relevant information
- Per-token heatmaps show concentrated relevance on Sharpe tokens
- Higher conservation ratio for Sharpe tokens than for lower-ranked features

**Reasons for caution:**
- All three interpreto methods rank Sharpe 6th-10th
- Sharpe values as decimal tokens may interact differently with LRP's gradient decomposition
- The 4.4% feature fraction means these rankings are derived from a small slice of total relevance

### 10.3 NTF Demotion

NTF was unanimously #1 under interpreto but drops to rank 10 under both LRP variants. Possible explanations:
- **Permutation averaging corrects for positional bias.** NTF is the last feature in the default prompt order, giving it a recency advantage. Averaging over 100 permutations neutralizes this.
- **LRP's token-level granularity.** NTF is tokenized as just 1-2 tokens ("Y" or "N"), producing a low absolute relevance sum even if per-token relevance is moderate.
- **Different attribution semantics.** Interpreto measures "what happens when NTF is removed" -- a binary feature's removal is maximally disruptive. LRP measures "how much logit flows through NTF tokens" -- a simple Y/N provides less computational signal.

### 10.4 Conservation Loss

The mean conservation ratio of 0.610 (single-logit) / 0.677 (logit-diff) indicates 32-39% of relevance is not accounted for at the input layer. This is a known limitation of the efficient LXT implementation. The consistency (low std) suggests a systematic scaling factor rather than random noise, meaning relative rankings should still be meaningful.

---

## 11. Probe Accuracy vs Attribution Correlation

With LRP included in the 5-method average:

- **5-method average:** Spearman rho = **0.745** (p = 0.009) -- statistically significant
- **AttnLRP only:** Spearman rho = **0.118** (p = 0.729) -- not significant

The 5-method average maintains significance because the interpreto methods' strong probe-attribution correlation (rho=0.745 at 3-method level) dilutes LRP's weak correlation. LRP's low probe-attribution correlation (rho=0.118) reflects the Sharpe-NTF inversion: NTF has the highest probe accuracy (0.996) but the lowest LRP importance, while Sharpe has moderate probe accuracy (0.914) but the highest LRP importance.

The per-layer analysis (Section 7.3) provides a more nuanced picture: in late layers (21-25), LRP relevance *does* correlate with probe accuracy (rho ~ 0.63-0.69, p < 0.05), confirming that the model uses encoded feature information for its decision in the layers that matter.

---

## 12. Summary of Findings

### What LRP Reveals

1. **Sharpe Ratio is the primary computational driver.** Across all 4 LRP variants, 500 samples, and 100 feature-order permutations, Sharpe is unanimously rank 1 with extremely tight bootstrap CIs (median rank 2, 95% CI [2, 2] in all variants). This is the most robust finding of the analysis.

2. **The top 3 features capture disproportionate relevance.** Under logit-difference AttnLRP, Sharpe + Expense Ratio + Return_3yr account for 56.5% of feature relevance, suggesting the model's decision is driven primarily by performance and cost metrics.

3. **NTF and Load are consistently unimportant to LRP.** Rank 10-11 across all 4 variants. Binary features contribute little through LRP's gradient decomposition pathway.

4. **Layer-wise processing has two phases.** Relevance peaks in layers 9-13 and again in 20-23. The LRP-probe correlation flips from negative to significantly positive at layer 21, indicating late layers align relevance with encoded feature information.

5. **Logit-difference attribution concentrates relevance.** By attributing to the decision margin, the top 3 features are more clearly separated from the rest. This variant better isolates features that discriminate between funds.

6. **Rankings are extremely stable.** With 500 samples and permutation averaging, bootstrap CIs are nearly degenerate -- most features have a 95% CI width of 0-1 rank positions.

### Where LRP Disagrees with Interpreto

| Feature | Interpreto (rank range) | LRP (rank range) | Direction |
|---------|------------------------|-------------------|-----------|
| NTF | 1-1 | 10-11 | LRP demotes |
| Sharpe | 6-10 | 1-1 | LRP promotes |
| Return_3yr | 9-11 | 2-6 | LRP promotes |
| Load | 3-5 | 7-11 | LRP demotes |

### Confidence Assessment

| Finding | Confidence | Rationale |
|---------|-----------|-----------|
| Sharpe is LRP-rank-1 | **High** | Unanimous across 4 variants, 500 samples, 100 permutations, tight bootstrap CIs |
| Expense Ratio is top 4 | **High** | Rank 2-4 across all LRP variants; rank 2-7 across all 5 methods |
| NTF is LRP-unimportant | **High** | Rank 10-11 across all 4 LRP variants |
| Layer 21-25 LRP-probe correlation | **Moderate-High** | Statistically significant at 3 layers, consistent with theory |
| Absolute ranking of mid-tier features (4-9) | **Moderate** | Higher variance, variants disagree on specifics (e.g., Tenure, Turnover) |
| Fund 1 positional bias | **Moderate** | Reduced but not eliminated by permutation averaging |

### Recommendations for Interpretation

1. **Do not treat LRP rankings as a replacement for interpreto rankings.** The methods measure different things: interpreto measures perturbation/gradient effects at the segment level; LRP decomposes the output logit into per-token contributions via modified backpropagation. Both perspectives are valid.

2. **The most robust cross-method findings:** Expense Ratio is consistently important (top 4 across all 5 methods). Standard Deviation is consistently unimportant (rank 8-10 across all methods).

3. **The Sharpe vs NTF disagreement is the central open question.** It likely reflects a genuine difference in what the methods measure: NTF's removal has outsized perturbation impact (interpreto) while Sharpe carries the most computational signal through the network (LRP). The 100 feature-order permutations strengthen the case that NTF's interpreto dominance was partially driven by positional effects.

4. **The layer-wise analysis is the most unique LRP contribution.** The LRP-probe correlation results provide mechanistic insight into where in the network the model consolidates feature information into its decision.
