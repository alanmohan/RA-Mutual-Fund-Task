### Linear Probing Report (Mutual Funds)

**Model**: `llama-3.2-3b`  
**Prompt condition**: `2_fewshot_cot_temp0`  
**Results directory**: `llm_eval_mutual_funds/data/probe_results/`  
**Config**: `probe_config_llama-3.2-3b_2_fewshot_cot_temp0.json`

---

### 1) What this probing step measures

This probing run answers: **how much information is linearly recoverable** from the model’s residual-stream activations at each layer, after the model reads the mutual-fund comparison prompt.

You probed:
- **Input-derived comparisons** (e.g., “expense ratio lower for fund 1”)
- **Target**: `medalist_p1_higher` (fund 1’s Medalist rating higher than fund 2’s)

High probe accuracy means the feature is **linearly encoded** in that layer’s representation (not necessarily that the model “uses” it, but it’s available).

---

### 2) Setup & data

From `probe_config_llama-3.2-3b_2_fewshot_cot_temp0.json`:

- **N extracted activations**: 5000
- **Layers**: 28 (0–27)
- **d_model**: 3072
- **Split**: 70/10/20 train/val/test
- **Probe**: logistic regression with validation-set selection of regularization `C`
- **Significance**: binomial test vs 50% chance, 95% CI, 1000 bootstrap iterations for CI

#### Important: per-feature sample counts differ

Each feature label can be `NaN` (missing values, invalid parsing, or Medalist ties). Those rows are dropped **per feature**. So a probe’s effective dataset size depends on the feature.

Examples (from `probe_results_llama-3.2-3b_2_fewshot_cot_temp0.csv`):
- Many features have **1000 test samples** (no missing labels).
- `beta_p1_closer_to_1`: **897 test samples** (missing betas).
- `medalist_p1_higher`: **641 test samples** (ties/invalid Medalist ratings excluded).

---

### 3) Artifact inventory (what to look at)

- **Per layer × feature results**: `probe_results_llama-3.2-3b_2_fewshot_cot_temp0.csv`
- **Accuracy matrix**: `probe_matrix_accuracy_llama-3.2-3b_2_fewshot_cot_temp0.csv`
- **AUC matrix**: `probe_matrix_auc_llama-3.2-3b_2_fewshot_cot_temp0.csv`
- **Best layer per feature**: `probe_best_layers_llama-3.2-3b_2_fewshot_cot_temp0.csv`
- **Feature-level summaries**: `probe_summary_llama-3.2-3b_2_fewshot_cot_temp0.csv`
- **Plots** (in `plots/`):
  - `heatmap_accuracy_llama-3.2-3b.png`
  - `accuracy_curves_llama-3.2-3b.png`
  - `best_layer_comparison.png`
  - `significance_heatmaps.png`

---

### 4) Headline findings

#### 4.1 Inputs are encoded extremely well

For nearly every input comparison feature, there exists a layer where a linear probe achieves **very high accuracy**:
- Several features are near-perfect (`load_p1_no`, `ntf_p1_yes`)
- Many numeric comparisons peak above 0.90 (expense ratio, Sharpe, return, inception date, assets, tenure)

#### 4.2 The target (“Medalist”) is strongly encoded, but weaker than inputs

Best-layer target performance:
- **`medalist_p1_higher`: 0.8565 accuracy (AUC 0.919)** at **layer 3**

Compare that to the “easy” input comparisons:
- The **average best-layer accuracy across the 11 input features** (excluding Medalist) is ~**0.920**.
- That yields a **~6.3 pp “last mile” gap**: the model linearly represents the inputs more cleanly than the final label.

This is exactly what `plots/best_layer_comparison.png` highlights.

#### 4.3 Why probe accuracy (~85%) is higher than end-task accuracy (~70%)

On the same setup (Llama-3.2-3B, few-shot CoT), the **model’s actual choice** (which fund it says to invest in) agrees with Medalist on only **~72%** of 2000 samples (`experiment_summary_llama-3.2-3b.csv`), while a **linear probe** on the model’s activations predicts Medalist with **~85%** accuracy at the best layer. So the representation **contains** the right signal more often than the model **uses** it in its final answer.

**Why the gap?**

1. **Probe vs. behavior**  
   The probe is a post-hoc classifier trained to predict “fund 1 better than fund 2” from a **fixed** representation (e.g. last-token activations at layer 3). It only needs to find a linear decision boundary. So **85%** means: “the information needed to predict the better fund is present and linearly recoverable at that layer.”

2. **What the model actually does**  
   The model’s answer is produced by the **full** forward pass: more layers, the LM head, and (if used) sampling. The path from “representation with 85% signal” to the final token “mutual fund 1” or “mutual fund 2” can:
   - Use a **different** decision rule (e.g. overweight expense ratio vs. Medalist’s mix),
   - Be affected by **decoding** (temperature, sampling),
   - Be biased by **reasoning** (e.g. the CoT may emphasize one factor and underuse another).

So the **same** information that is linearly recoverable at 85% is not fully translated into the model’s stated choice, hence ~70% end-task accuracy. That’s the **“last mile”** in practice: the model has the signal in the middle layers but doesn’t convert it into the correct output as often as the probe would.

---

### 5) Best layer per feature (key table)

From `probe_best_layers_llama-3.2-3b_2_fewshot_cot_temp0.csv`:

| Feature | Best layer | Best test acc | Best test AUC |
|---|---:|---:|---:|
| `ntf_p1_yes` | 9 | **1.000** | **1.000** |
| `load_p1_no` | 14 | **0.994** | **0.999** |
| `expense_ratio_p1_lower` | 13 | **0.981** | **0.997** |
| `inception_p1_older` | 13 | **0.958** | **0.986** |
| `assets_p1_higher` | 12 | **0.921** | **0.966** |
| `sharpe_p1_higher` | 15 | **0.924** | **0.970** |
| `return_3yr_p1_higher` | 16 | **0.911** | **0.959** |
| `tenure_p1_longer` | 13 | **0.906** | **0.962** |
| `turnover_p1_lower` | 13 | **0.881** | **0.945** |
| `stdev_p1_lower` | 13 | **0.854** | **0.930** |
| `beta_p1_closer_to_1` | 4 | **0.789** | **0.862** |
| `medalist_p1_higher` | 3 | **0.856** | **0.919** |

---

### 6) Layer-wise dynamics & mechanistic interpretation

Use `probe_matrix_accuracy_llama-3.2-3b_2_fewshot_cot_temp0.csv` plus the plots:

#### 6.1 Most input features peak mid-network (~12–16)

From `plots/heatmap_accuracy_llama-3.2-3b.png` and `plots/accuracy_curves_llama-3.2-3b.png`:

- Many features start above chance early (often 0.73–0.82 at layer 0–3),
- then climb sharply and **peak around layers 12–16**,
- then remain high (sometimes slightly decaying) in later layers.

Interpretation:
- Early layers: parse the table-like structure and surface tokens.
- Mid layers: form **normalized comparison facts** (“fund 1 wins on X”), which are very linearly readable.
- Late layers: stabilize the answer-generation behavior; some features may be deemphasized or mixed with other objectives.

#### 6.2 Binary flags (Load/NTF) are nearly perfectly linearly readable

`load_p1_no` and `ntf_p1_yes` are close to 1.0 across a wide band of layers.

Interpretation:
- These are low-entropy, token-level facts (often `Y/N`-like), easy to preserve and linearly separate.

#### 6.3 `beta_p1_closer_to_1` is the hardest input feature

It has:
- the lowest ceiling (~0.789 accuracy),
- and peaks very early (layer 4), with weaker performance later.

Interpretation:
- This label is nonlinear: it depends on \(|\beta - 1|\), not just “bigger vs smaller”.
- A single linear readout may struggle unless the model explicitly constructs a “distance-to-1” feature.

---

### 7) Target analysis: Medalist (“the last mile”)

#### 7.1 No label leakage via prompts

Your prompts include the numeric/table fields but do **not** include `Medalist_1`/`Medalist_2`. So the probe isn’t trivially reading label tokens.

#### 7.2 Why Medalist is lower than inputs

Even with near-perfect decodability of inputs, Medalist remains lower because:

- Medalist is partly **qualitative** and may depend on factors not present in the prompt (process/people/portfolio, fees beyond what’s shown, analyst judgement, etc.).
- Effective dataset size is smaller for Medalist due to excluding ties and invalids.
- The true mapping from these specific metrics → Medalist is likely **not deterministic**, so there may be an irreducible error floor.

#### 7.3 Why Medalist peaks early (layer 3)

`medalist_p1_higher` peaks at **layer 3** (0.856), then trends downward in later layers.

Plausible interpretation:
- Early layers preserve a “raw mixture” of input fields whose linear combination correlates with Medalist.
- Mid layers increasingly emphasize explicit *per-metric comparisons* (very probe-friendly), which may not align as directly with the latent “Medalist mapping”.
- So the best linear separator for Medalist occurs early, before representations shift toward structured comparison features.

This is consistent with the “last mile problem” framing: **inputs are available and cleanly encoded, but the target decision is a harder transformation.**

---

### 8) Statistical significance (what it means here)

From `probe_summary_llama-3.2-3b_2_fewshot_cot_temp0.csv`:
- `is_significant_sum = 28` for every feature ⇒ **every layer is significantly above chance**.

From `plots/significance_heatmaps.png`:
- Most features saturate the \(-\log_{10}(p)\) scale (extremely small p-values).
- `beta_p1_closer_to_1` shows relatively weaker (but still significant) regions.

Caveat:
- With large \(n\), p-values become tiny even for modest improvements. For interpretation, **effect size (accuracy/AUC + CI) matters more** than p-values.

---

### 9) Suggested next steps (high ROI)

1. **Probe other prompt conditions** (`zeroshot`, `topp01`, etc.) and compare:
   - Does few-shot CoT increase “input encoding” but not Medalist?
   - Does any condition improve Medalist encodability?

2. **Probe different token positions**
   - Current extraction uses the last-token position by default.
   - Try extracting at:
     - end of each fund block,
     - or right before the “final answer” instruction.
   - This can reveal *where* the model stores per-feature comparisons vs final decision state.

3. **Add a “distance-to-1” beta feature for interpretability**
   - If you want beta to behave like other features, consider adding derived values like \(|\beta-1|\) into the prompt (still without Medalist).

4. **Reduce storage / compute (optional)**
   - If you’re storage-bound, store activations in float16 and/or only a subset of layers (e.g., layers 0–3, 10–16, 27) based on these results.

---

### 10) Plots (quick guide)

- **Accuracy heatmap**: `plots/heatmap_accuracy_llama-3.2-3b.png`  
  Shows a mid-layer band where most features are maximally decodable; Medalist is strong early but doesn’t dominate mid layers.

- **Accuracy curves**: `plots/accuracy_curves_llama-3.2-3b.png`  
  Highlights the mid-layer “rise” for numeric features; Medalist (dashed) peaks early and drifts.

- **Best layer comparison**: `plots/best_layer_comparison.png`  
  Summarizes “inputs vs target” and visualizes the “last mile gap”.

- **Significance heatmap**: `plots/significance_heatmaps.png`  
  Confirms results are far above chance; treat p-values as secondary to effect sizes due to large \(n\).

