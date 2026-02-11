# Linear Probing Results: Interpretation Report (Llama-3.2-3B & Qwen3-4B)

**Condition:** `2_fewshot_cot_temp0` (few-shot chain-of-thought, temperature 0)  
**Data:** 5,000 samples, 70/10/20 train/val/test split  
**Probe:** Logistic regression with 5-fold CV for regularization; binomial test vs 50% chance; 1,000 bootstrap iterations for 95% CI.

---

## 1. Setup Summary

| | Llama-3.2-3B | Qwen3-4B |
|---|:---:|:---:|
| **Layers** | 28 (0–27) | 36 (0–35) |
| **d_model** | 3,072 | 2,560 |
| **Activations** | Last-token residual stream (post-attention) | Same |
| **Features probed** | 11 input-derived + Medalist (target) | Same |

**Input-derived features:** Expense Ratio (lower better), Sharpe (higher), Stdev (lower), 3Y Return (higher), Beta (fund 1 lower), Tenure (longer), Inception (older), Assets (higher), Turnover (lower), Load (No), NTF (Yes).  
**Target:** Medalist (fund 1 rating higher than fund 2).

---

## 2. Headline Comparison: Best-Layer Accuracy

### 2.1 Input feature encoding (average across 11 features)

- **Llama-3.2-3B:** Best-layer accuracy **92.3%** (mean over 11 input features). Inputs are encoded very well; most features exceed 85% at their best layer.
- **Qwen3-4B:** Best-layer accuracy **85.4%** (mean). Still strong but about **7 percentage points lower** than Llama on average.

So under the same prompt and data, **Llama encodes the comparison-relevant input structure more linearly** in its residual stream than Qwen.

### 2.2 Target (Medalist) and the “last mile” gap

- **Llama:** Medalist best-layer accuracy **86.6%** (layer 2). Gap vs input average: **5.5 pp** (92.3% − 86.6%).
- **Qwen:** Medalist best-layer accuracy **74.3%** (layer 22). Gap vs input average: **11.2 pp** (85.4% − 74.3%).

Interpretation:

- The **target** (which fund has the better Medalist rating) is not directly in the prompt; it has to be inferred or correlated with the inputs. Both models encode it above chance and well above 50%, but **below** their encoding of the raw input comparisons.
- The **gap** (inputs vs Medalist) is larger for Qwen (11.2 pp) than for Llama (5.5 pp). So Llama’s internal representation is closer to “already combining inputs toward the Medalist decision” than Qwen’s; Qwen encodes inputs well but the “last mile” to the target is steeper.

---

## 3. Per-Feature Best-Layer Results

### 3.1 Llama-3.2-3B

| Feature | Best layer | Test accuracy | Test AUC |
|--------|------------|---------------|----------|
| NTF (Yes) | 6 | **99.6%** | 1.000 |
| Load (No) | 18 | **99.1%** | 0.998 |
| Expense Ratio (Lower) | 13 | **97.5%** | 0.997 |
| Inception (Older) | 13 | **94.6%** | 0.986 |
| Sharpe (Higher) | 15 | **91.4%** | 0.961 |
| Tenure (Longer) | 14 | **92.1%** | 0.971 |
| Assets (Higher) | 13 | **92.0%** | 0.968 |
| Return 3Y (Higher) | 14 | **90.7%** | 0.954 |
| Turnover (Lower) | 13 | **87.5%** | 0.940 |
| Stdev (Lower) | 15 | **85.8%** | 0.922 |
| Beta (Lower) | 13 | **85.4%** | 0.924 |
| **Medalist (Target)** | **2** | **86.6%** | 0.923 |

Observations:

- **Easiest:** NTF, Load, Expense Ratio — near or above 97%; binary/cost features are extremely well encoded.
- **Hardest inputs:** Stdev, Beta (mid–high 85%); still strong.
- **Medalist** peaks **very early (layer 2)** and then stays in the 80–86% range. So the “decision-relevant” signal for the target appears in shallow layers; deeper layers refine input features but don’t improve Medalist further.

### 3.2 Qwen3-4B

| Feature | Best layer | Test accuracy | Test AUC |
|--------|------------|---------------|----------|
| NTF (Yes) | 11 | **97.2%** | 0.982 |
| Load (No) | 11 | **96.0%** | 0.958 |
| Expense Ratio (Lower) | 19 | **93.0%** | 0.964 |
| Sharpe (Higher) | 23 | **91.3%** | 0.934 |
| Return 3Y (Higher) | 23 | **90.3%** | 0.923 |
| Assets (Higher) | 19 | **84.3%** | 0.892 |
| Inception (Older) | 10 | **81.2%** | 0.870 |
| Tenure (Longer) | 19 | **83.5%** | 0.892 |
| Beta (Lower) | 19 | **74.6%** | 0.811 |
| Turnover (Lower) | 10 | **74.5%** | 0.794 |
| Stdev (Lower) | 21 | **73.7%** | 0.766 |
| **Medalist (Target)** | **22** | **74.3%** | 0.792 |

Observations:

- **Easiest:** NTF, Load, Expense Ratio (93–97%) — same “easy” set as Llama but with lower accuracies.
- **Hardest:** Stdev (73.7%), Turnover (74.5%), Beta (74.6%), and **Medalist (74.3%)**. These sit in the low–mid 70s.
- **Medalist** peaks at **layer 22** (deeper than Llama’s layer 2), and its best accuracy (74.3%) is much lower than Llama’s 86.6%. So Qwen both encodes the target more weakly and uses a deeper layer for it.

---

## 4. Layer-Wise Dynamics

### 4.1 Llama-3.2-3B (28 layers)

- **Early (0–6):** NTF and Load already very high; Medalist peaks at layer 2 (~86.6%) and then declines slightly.
- **Mid (7–15):** Most input features climb to their best: expense ratio, Sharpe, return, tenure, inception, assets, turnover, stdev, beta peak between layers 13–15.
- **Late (16–27):** Input features stay high (mostly 85–97%); Medalist drifts down into the high 79–82% range.

So Llama: **early** encoding of the target; **mid** encoding of detailed inputs; **late** layers don’t add much for the probed target.

### 4.2 Qwen3-4B (36 layers)

- **Early (0–11):** Fast rise for NTF, Load; other features and Medalist still modest (60–70%).
- **Mid (12–23):** Most input features and Medalist improve; best layers for many features and for Medalist lie in 19–23.
- **Late (24–35):** Many accuracies **decline** (e.g. tenure, inception, turnover, return, Sharpe), suggesting the representation is transformed in a way that is less linearly aligned with these comparisons.

So Qwen: **no early Medalist peak**; target and several inputs improve through the middle and then **degrade in deeper layers**, unlike Llama.

---

## 5. Model Comparison Summary

| Aspect | Llama-3.2-3B | Qwen3-4B |
|--------|--------------|----------|
| **Input encoding (avg)** | 92.3% | 85.4% |
| **Medalist (target)** | 86.6% (layer 2) | 74.3% (layer 22) |
| **Input–target gap** | 5.5 pp | 11.2 pp |
| **Weakest input features** | Stdev, Beta (~85%) | Stdev, Turnover, Beta (73–75%) |
| **Strongest inputs** | NTF, Load, Expense (97–99.6%) | NTF, Load, Expense (93–97%) |
| **Medalist layer** | Very early (2) | Mid–deep (22) |
| **Late-layer trend** | Inputs stay high; Medalist slightly lower | Many features and Medalist drop after peak |

**Takeaways:**

1. **Llama** encodes both input comparisons and the Medalist target more strongly and with a smaller “last mile” gap; the target is already well represented by layer 2.
2. **Qwen** encodes inputs well but less strongly than Llama, and encodes Medalist notably worse; the gap between “inputs” and “target” is larger, and the best target signal appears in deeper layers and then degrades in the last layers.
3. **Risk/volatility-related features** (Stdev, Beta) and **Turnover** are relatively harder for both models, and especially for Qwen (low 70s).
4. **Binary/cost features** (NTF, Load, Expense Ratio) are easiest for both and could be used as a sanity check for any change in data or probing setup.

---

## 6. Relation to Plots

- **`accuracy_curves_llama-3.2-3b.png`** / **`accuracy_curves_qwen3-4b.png`:** Layer (x) vs test accuracy (y) per feature. Shows the trends described above (early Medalist peak for Llama; mid peak and late drop for Qwen).
- **`best_layer_comparison.png`:** Top: bar chart of best-layer accuracy per input feature (Llama vs Qwen). Bottom: “Last mile” — average input accuracy vs Medalist accuracy per model, with gap annotated.
- **`heatmap_accuracy_*.png`:** Layer × feature heatmaps; darker = higher accuracy.
- **`significance_heatmaps.png`:** Which layer–feature pairs are statistically significant vs 50% chance.

---

## 7. Conclusions

- **Linear probe accuracy** measures how much of each concept (input comparisons and Medalist) is **linearly present** in the residual stream at each layer; it does not prove the model “uses” that signal for its final answer.
- Under the **same prompt and condition** (few-shot CoT, temp 0):
  - **Llama-3.2-3B** shows stronger linear encoding of both inputs and the Medalist target, with a smaller last-mile gap and an early Medalist peak.
  - **Qwen3-4B** shows good but weaker input encoding and a larger gap to Medalist; the target is best in mid-depth layers and degrades in the last layers.
- These results are consistent with **Llama** being relatively better at lining up its internal state with the “which fund is better (by Medalist)” decision in a linearly recoverable way, and **Qwen** putting more of that structure in deeper, less linearly separable form, or combining inputs in a way that doesn’t preserve a simple linear boundary for Medalist in the probed space.

For downstream use (e.g. using probes as features or interpreting model behavior), Llama’s representations are more “probe-friendly” for both input features and the target under this setup.
