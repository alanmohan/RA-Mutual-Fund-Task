# Attribution Methods: A Detailed Technical Explainer

**Context:** Llama-3.2-3B-Instruct, zero-shot mutual fund pairwise comparison  
**Library:** `interpreto` (For-Sight AI)  
**Methods covered:** Integrated Gradients · KernelSHAP · Occlusion

---

## 1. The Core Problem: What Are We Trying to Explain?

The model receives a prompt like this:

```
Task: Compare mutual fund 1 vs mutual fund 2 using only the data below and
decide which fund you would invest in. Do not use any outside information.

Mutual fund 1:
Expense Ratio - Net: 0.0075
3 Year Sharpe Ratio: 0.52
Standard Deviation: 14.3
3 Yr: 8.1
Beta: 0.95
Manager Tenure: 6.2
Inception Date: 1998
Assets (Millions): 4200
Turnover Rates: 0.18
Load (Y/N): No
NTF: Yes

Mutual fund 2:
Expense Ratio - Net: 0.0123
... (same 11 features)
```

The model then generates a response that ultimately contains "mutual fund 1" or "mutual fund 2".

**The attribution question is:** Given that the model chose "mutual fund 1", which parts of the input prompt were most responsible for that decision?

Attribution methods answer this by assigning a scalar **importance score** to each input token (or group of tokens). High score = this part of the input strongly influenced the output. Low score = the model's decision was largely indifferent to this part.

These scores are then **aggregated from token-level to feature-level**: all tokens belonging to the "Expense Ratio" lines (both funds) are summed/averaged to give a single importance score for the expense ratio feature.

---

## 2. How the Output Is Targeted

All three methods need to know *what output* they are explaining. For a generation model, this is non-trivial — the model produces hundreds of tokens, not a single class probability.

In this experiment, `ATTRIB_TARGET_MODE = "minimal"` was used. This means:

1. The model first generates its full response (up to 128 tokens).
2. The last word of that response that contains "1" or "2" is extracted as the target (e.g., `"1"` from "mutual fund 1").
3. Attribution methods then explain only the logit for that single target token, not the full generation.

This dramatically reduces memory cost because the attribution matrix has dimensions `(input_length × 1)` rather than `(input_length × output_length)`.

The explanation this answers is: **"What input features most influenced the model's probability of generating the token `'1'` (or `'2'`) at the final decision position?"**

---

## 3. Integrated Gradients (IG)

### 3.1 The Intuition

Imagine you have a function `f(x)` and you want to know how much each dimension of `x` contributed to the output `f(x)` relative to some baseline `f(x₀)`. The simplest answer is the gradient `∂f/∂x` — but gradients are local and can be misleading if the function is saturated or nonlinear. Integrated Gradients solves this by integrating the gradient along the entire path from the baseline `x₀` to the actual input `x`.

### 3.2 The Mathematics

Given:
- Input embedding matrix **x** ∈ ℝ^(L×d) where L = sequence length, d = embedding dimension
- Baseline embedding **x₀** = zero matrix (or uniform noise)
- Model output (target logit) `F(x)` — the logit for the target token at the decision position

The Integrated Gradients attribution for input token i is:

$$\text{IG}_i(\mathbf{x}) = (\mathbf{x}_i - \mathbf{x}_{0,i}) \times \int_{\alpha=0}^{1} \frac{\partial F(\mathbf{x}_0 + \alpha(\mathbf{x} - \mathbf{x}_0))}{\partial \mathbf{x}_i} \, d\alpha$$

The integral is approximated numerically by evaluating the gradient at `n_perturbations = 10` equally spaced interpolation points along the path from **x₀** to **x**, then taking their average:

$$\text{IG}_i(\mathbf{x}) \approx (\mathbf{x}_i - \mathbf{x}_{0,i}) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F\left(\mathbf{x}_0 + \frac{k}{m}(\mathbf{x} - \mathbf{x}_0)\right)}{\partial \mathbf{x}_i}$$

The result is a d-dimensional vector per token (one value per embedding dimension). These are then reduced to a single scalar per token by taking the L2 norm (or mean absolute value).

### 3.3 What Actually Happens in the Code

For each of the 200 mutual fund prompts:

1. **Tokenize** the chat-formatted prompt → sequence of L ≈ 500–600 tokens
2. **Forward pass (baseline):** Run the model with zero embeddings, get baseline logit `F(x₀)`
3. **10 interpolated forward passes:** At each step k/10 (k = 1..10), set all input embeddings to `x₀ + (k/10) * (x - x₀)` and run a backward pass to compute `∂F/∂xᵢ`
4. **Average gradients** across the 10 steps; multiply element-wise by `(x - x₀)`
5. **Reduce to scalar** per token: take the absolute mean across embedding dimensions
6. **WORD-level aggregation:** Tokens that form the same word (after subword tokenization) are averaged via `GranularityAggregationStrategy.MEAN` to give one score per word
7. **Feature-level aggregation:** For each word, find its character position in the original prompt. Use `build_char_to_feature_map` to determine which feature line it belongs to (and whether it's fund 1 or fund 2). Sum all word scores belonging to the same feature.

The gradient checkpointing (`USE_GRADIENT_CHECKPOINTING = True`) trades speed for memory: instead of storing intermediate activations for all 28 layers simultaneously during the backward pass, it recomputes them on demand, cutting peak VRAM roughly in half.

### 3.4 What the Scores Mean

- **A high IG score for a token/feature** means the gradient of the output logit with respect to that token's embedding was consistently large along the path from baseline to input. In other words, the model's output is sensitive to perturbations of that token — it matters.
- **A low IG score** means the output was relatively insensitive to changes in that token's embedding. The model would produce a similar output even if this part of the input were replaced with noise.
- **The key theoretical guarantee** (Completeness axiom): The sum of all IG scores equals exactly `F(x) - F(x₀)`. This means the scores genuinely explain the difference between the actual output and the baseline output, not just local sensitivity.

### 3.5 Applied to the Mutual Fund Task

When IG ranks `expense_ratio` first and `return_3yr` last, it is saying: **the model's final decision logit is most sensitive to variations in the tokens encoding the expense ratio, and least sensitive to variations in the return_3yr tokens.** This does not necessarily mean expense ratio determines the answer — it means the gradient landscape is steepest there.

An important caveat for autoregressive models: early tokens in the sequence tend to have larger gradient norms because they influence all subsequent attention operations. Since expense ratio appears on line 4 of the prompt (among the first feature lines), some of its high IG ranking may reflect positional priority rather than genuine financial relevance.

---

## 4. KernelSHAP

### 4.1 The Intuition

Shapley values come from cooperative game theory. The question is: given a coalition game where multiple "players" cooperate to produce an output, how do you fairly distribute credit among the players? The Shapley value is the unique credit-allocation satisfying four axioms: efficiency, symmetry, dummy, and linearity.

For attribution, each input token (or word) is a "player", and the "output" is the model's target logit. The Shapley value of token i is the average marginal contribution of token i to the output, averaged over all possible orderings/subsets of the other tokens.

Computing exact Shapley values requires 2^n model evaluations (n = number of tokens), which is computationally infeasible. **KernelSHAP** approximates them efficiently by:
1. Sampling random subsets of tokens
2. Running the model with the rest masked out (replaced by a baseline/reference)
3. Fitting a weighted linear regression to estimate each token's marginal contribution

### 4.2 The Mathematics

For a prompt with n words, KernelSHAP samples S random binary masks **z** ∈ {0,1}^n. For each mask:
- Tokens where `zⱼ = 1` are kept; tokens where `zⱼ = 0` are replaced with a baseline (e.g., the `[MASK]` token or zero embedding)
- The model is run to get the masked output `F(z)`

The Shapley value for token i is estimated by solving the weighted least-squares problem:

$$\hat{\phi} = \arg\min_{\phi} \sum_{\mathbf{z} \in S} \pi(\mathbf{z}) \left( F(\mathbf{z}) - \phi_0 - \sum_{j=1}^{n} \phi_j z_j \right)^2$$

where the SHAP kernel weight is:

$$\pi(\mathbf{z}) = \frac{(n-1)}{\binom{n}{|\mathbf{z}|} \cdot |\mathbf{z}| \cdot (n - |\mathbf{z}|)}$$

This weighting ensures that small and large coalitions (subsets) are sampled more, giving higher-quality estimates for boundary effects.

### 4.3 What Actually Happens in the Code

For each of the 200 prompts:

1. **Tokenize and word-segment** the prompt (Granularity.WORD gives n ≈ 200–250 words)
2. **Sample 64 random binary masks** (SHAP_N_PERTURBATIONS = 64). Each mask specifies which words to keep and which to replace with the baseline
3. **64 forward passes:** For each mask, substitute masked-out words with the baseline and run the model to get the target logit. These are full model forward passes through all 28 layers at ~3B parameters — expensive.
4. **Weighted linear regression** over the 64 (mask, logit) pairs to estimate each word's SHAP value
5. **WORD-level and feature-level aggregation:** Same as IG — word scores are mapped to character positions and summed per feature

The inference mode is `InferenceModes.LOGITS`: the model does not generate new tokens; it evaluates the logit for the target token at the end of a single forward pass. This is more memory-efficient than a full generation for each mask.

### 4.4 What the Scores Mean

- **A high KernelSHAP score for a feature** means that including the tokens of that feature in the input (versus masking them out) consistently increases the target logit (the model's confidence in the correct answer). This feature's presence helps the model make the right choice.
- **A low KernelSHAP score** means the feature's presence or absence does not reliably shift the model's output. The model is essentially indifferent to that feature.
- **Unlike IG,** SHAP values satisfy the Efficiency axiom exactly (when exact): they sum to `F(x) - F(baseline)`. With sampling approximation, this holds only approximately.
- **The scores are in logit units** (raw, unbounded). A score of 1e9 does not mean "very important in practice" — it means "this word's presence shifted the logit by ~1e9 units relative to baseline." These large magnitudes arise because the difference between full-token and zero-token embeddings is enormous in absolute terms.

### 4.5 Applied to the Mutual Fund Task

With only 64 perturbations for a ~250-word prompt, the Shapley estimates are significantly undersampled. Reliable KernelSHAP typically requires hundreds to thousands of samples. The standard deviation-to-mean ratio exceeding 10:1 in our results is a direct consequence of this: individual samples produce wildly different Shapley values because the randomly masked subsets give insufficient coverage of the combinatorial space.

When KernelSHAP ranks `inception` first instead of `expense_ratio` (as IG does), it is observing that **masking out inception date tokens caused larger logit drops** than masking expense ratio tokens. One plausible explanation: inception date is expressed as a long numeric string (e.g., "1998-05-12"), which when masked changes the prompt structure noticeably and destabilizes the model's sequential processing. This is a methodological artifact, not evidence that inception date is financially more important.

---

## 5. Occlusion

### 5.1 The Intuition

Occlusion is the simplest perturbation-based method conceptually: systematically remove (occlude) one segment of the input at a time, run the model without it, and measure how much the output changes. A segment that, when removed, causes a large drop in the target logit was important. A segment whose removal barely changes the output was unimportant.

This is the machine learning equivalent of asking: "If I cover up this part of the image/text, does the model still make the same prediction?" — originally popularized by Zeiler & Fergus (2014) for convolutional neural networks.

### 5.2 The Mathematics

For a prompt with n segments (sentences, words, or part-sentences) **s₁, s₂, ..., sₙ**:

$$\text{Occ}(s_i) = F(\mathbf{x}) - F(\mathbf{x}_{-i})$$

where **x₋ᵢ** is the prompt with segment sᵢ removed (replaced by a mask token or simply deleted), and `F(·)` is the target logit.

No gradients are computed. This is a purely black-box method — the model is treated as a function that takes text and returns a logit. The attribution for each segment is just the difference in output when that segment is present versus absent.

### 5.3 Granularity and the Period-Appended Prompts

Granularity controls what constitutes one "segment" to occlude. Three granularities are available in `interpreto`:
- `WORD`: each word is one segment (~250 segments per prompt → ~250 forward passes)
- `SENTENCE`: split on `.`, `?`, `!` → each sentence is one segment
- `PART_SENTENCE`: split on `:`, `,`, `;`, `.`, `!`, `?` → finer than sentence

The original feature lines in the prompt do **not** end with periods:
```
Expense Ratio - Net: 0.0075
3 Year Sharpe Ratio: 0.52
```

Without terminal periods, `SENTENCE` granularity would lump multiple feature lines into one large sentence segment, making it impossible to attribute importance to individual features.

**The modification applied:** Before running occlusion, `_add_periods_to_feature_lines()` appends a period to every line starting with a known feature label:
```
Expense Ratio - Net: 0.0075.
3 Year Sharpe Ratio: 0.52.
```

Now `SENTENCE` granularity splits on `.` and produces **one sentence-segment per feature line per fund** — a clean 1:1 mapping between segments and the 11 × 2 = 22 feature instances in the prompt. Total segments ≈ 25–30 (22 feature lines + a few structural sentences), yielding ≈ 25–30 forward passes per sample.

### 5.4 What Actually Happens in the Code

For each of the 200 prompts:

1. **Apply `_add_periods_to_feature_lines()`** to create `occ_chat_prompts[i]` with periods added
2. **Segment** the period-appended prompt at sentence boundaries → ~25–30 segments
3. **25–30 forward passes:** For each segment sᵢ, replace it with the mask token and run the model to obtain `F(x₋ᵢ)`
4. **Compute occlusion score:** `Occ(sᵢ) = |F(x) - F(x₋ᵢ)|` (absolute difference; `extract_words_and_scores` takes absolute value after nanmean across generated token dimension)
5. **Feature-level aggregation (`aggregate_sentence_scores_by_feature`):** For each sentence-segment, find its character position in `occ_chat_prompts[i]`. Use `build_char_to_feature_map` to look up which feature and which fund it belongs to. Sum scores for fund 1 and fund 2 separately, then compute totals.

The target token extraction uses `OCC_USE_SINGLE_TOKEN_TARGET = True`: only `"1"` or `"2"` is used as the target, minimizing the generated-token dimension and keeping VRAM manageable for the many forward passes.

### 5.5 What the Scores Mean

- **A high occlusion score for a feature** means: when the sentence containing that feature's value is removed from the prompt, the model's confidence in the correct answer drops significantly. The model *needed* that sentence to make its decision.
- **A low occlusion score** means: removing that feature's sentence barely changes the output. The model can make the same decision without it — either because the remaining features are sufficient, or because the model never attended to it.
- **Unlike IG and SHAP**, occlusion measures a global, non-local effect: the impact of completely removing a feature, not the sensitivity to small perturbations. It answers "is this feature necessary?" rather than "is this feature a gradient bottleneck?"
- **The scores are in logit units** representing the logit drop when a segment is removed. Larger absolute values mean the removal was more disruptive.

### 5.6 Applied to the Mutual Fund Task

With sentence granularity and period-appended prompts, each occlusion trial removes exactly one feature line for one fund. For example:
- Trial 1: Remove `"Expense Ratio - Net: 0.0075."` from fund 1 → run model → record logit change
- Trial 2: Remove `"Expense Ratio - Net: 0.0123."` from fund 2 → run model → record logit change

The feature-level score for `expense_ratio` is the sum of these two changes. Features whose removal causes larger total logit disruption are ranked higher.

The key limitation here is that removing one feature from a 22-feature prompt leaves 21 features intact. If the model can substitute from other correlated features (e.g., Sharpe ratio and standard deviation both capture risk), removing one may not cause a large drop. This is the **feature redundancy problem** inherent to leave-one-out occlusion.

---

## 6. How Scores Are Aggregated from Tokens to Features

All three methods produce token-level (or word-level, or sentence-level) scores. The experiment aggregates these to feature-level using the following pipeline:

### Step 1: `build_char_to_feature_map(prompt_text)`

This function creates a character-level index of the prompt. For each character position in the string, it stores a tuple `(feature_short_name, fund_number)` or `None` if the character does not belong to a feature line.

Example snippet of what the map encodes:
```
Position 240–270: "Expense Ratio - Net: 0.0075"
  → every character in [240, 270] maps to ("expense_ratio", 1)

Position 480–510: "Expense Ratio - Net: 0.0123" (fund 2 block)
  → every character in [480, 510] maps to ("expense_ratio", 2)
```

Fund 1 vs fund 2 separation is determined by finding the character position of the string "Mutual fund 2:" and treating everything after it as fund 2.

### Step 2: Token/word/segment → character position lookup

For each scored unit (word, segment), the code searches for that string in the prompt text using `str.find()`, using a rolling `search_start` pointer so that duplicate values (e.g., two features both having value "0.5") are matched to their correct positions in order.

### Step 3: Feature score accumulation

For each word/segment matched to `(feature_short_name, fund_num)`, its absolute attribution score is added to `result[feature_short_name][fund1]` or `result[feature_short_name][fund2]`. The total score per feature is `fund1_score + fund2_score`.

This aggregation treats importance symmetrically across both funds: a feature is considered important if its tokens in *either* fund's block receive high attribution. This is reasonable because the model must compare both fund 1's and fund 2's values to make a decision.

---

## 7. Interpreting the Results: A Practical Guide

### 7.1 Reading an Attribution Table

Take the IG table as an example:

| Rank | Feature | Mean Attribution | Normalized Attribution |
|---:|---|---:|---:|
| 1 | expense_ratio | 0.0073 | 1.000 |
| 11 | return_3yr | 0.0024 | 0.327 |

This means:
- Across 200 samples, the average total attribution score assigned to expense ratio tokens was 0.0073 (in gradient-magnitude × embedding-delta units)
- Normalized to the maximum feature score, expense ratio is 1.0 and return_3yr is 0.33
- The 3:1 ratio is actually quite flat — this is not a dominant vs negligible situation. The model is not strongly selective.

### 7.2 When Two Methods Agree

If both IG and KernelSHAP rank a feature similarly (e.g., `expense_ratio` is #1 and #2 respectively), this convergent evidence is meaningful. Two fundamentally different measurement approaches (gradient sensitivity vs. masking-based marginal contribution) agree that the model's output is most sensitive to this feature's tokens. That is a stronger claim than either method alone.

### 7.3 When Methods Disagree

Method disagreement is not a failure — it is informative about what different questions are being asked:

| Question | Best method to answer it |
|---|---|
| What tokens does the model's gradient flow through most? | Integrated Gradients |
| What is the marginal contribution of each feature to the final decision? | KernelSHAP |
| What features are necessary for the model to maintain its answer? | Occlusion |

A feature could score high on IG (gradient-heavy path) but low on occlusion (other features can compensate) — and both would be correct, just answering different questions.

### 7.4 Absolute Magnitudes vs Rankings

- **IG scores (~0.002–0.007):** These are in gradient-magnitude × embedding-delta units. The absolute values are not interpretable; only relative rankings matter.
- **KernelSHAP scores (~1e9):** These are logit differences. The extreme magnitudes arise from the large L2 norm difference between full-token embeddings and zero-embeddings used as the baseline. Only the ranking is meaningful. The very high standard deviations (10–15× the mean) indicate substantial estimation noise from using only 64 perturbations.
- **Occlusion scores (~0.7–3.0):** These are absolute logit changes when a sentence is removed. More grounded in meaning — a score of 3.0 means removing that feature sentence changed the decision logit by ~3.0 units on average. Still context-dependent (depends on the scale of logits for this model).

### 7.5 Fund 1 vs Fund 2 Breakdown

The aggregation stores separate scores for fund 1 tokens and fund 2 tokens. Asymmetries reveal prompt ordering effects:
- Higher fund 1 attribution: model may be more influenced by the first fund presented (primacy effect)
- Higher fund 2 attribution: model may be more influenced by what it read most recently before generating (recency effect, especially in occlusion where last-seen tokens are proximal to the generation)

Both effects are well-documented in LLM evaluation literature and are separate from the genuine financial reasoning being probed.

### 7.6 What Attribution Analysis Cannot Tell You

Attribution methods explain the model's computation on a given prompt — they do not:
- Tell you whether the model's reasoning is correct or financially sound
- Prove that a feature "caused" the model to choose fund 1 (correlation is measured, not causation)
- Generalize across prompts automatically — a feature important for one pair may be irrelevant for another
- Distinguish between the model genuinely using financial knowledge vs. positional/lexical shortcuts

The complement to attribution analysis is the **linear probing experiment** already conducted. Probes measure what information is encoded in the model's internal representations, while attribution measures what information influences the output decision. The lack of correlation between the two (Spearman ρ = 0.28, p = 0.40) in this experiment is itself an important finding: the model encodes all features well (high probe accuracy for all features) but does not rely on them equally for the final decision.

---

## 8. Summary: Method Comparison at a Glance

| Property | Integrated Gradients | KernelSHAP | Occlusion |
|---|---|---|---|
| **Type** | Gradient-based | Perturbation-based (Shapley) | Perturbation-based (leave-one-out) |
| **Granularity** | Word | Word | Sentence |
| **Model forward passes per sample** | 10 (+ 1 baseline) | 64 | ~25–30 |
| **What is perturbed** | Embedding values (interpolation) | Token presence (masking) | Sentence removal |
| **Question answered** | Gradient sensitivity | Marginal contribution | Necessity |
| **Theoretical guarantee** | Completeness axiom | Shapley axioms (approx.) | None (heuristic) |
| **Memory cost** | High (backprop required) | Moderate (forward-only) | Low (forward-only, few segments) |
| **Positional bias risk** | High (gradient norms decay with position in autoregressive LMs) | Moderate | Low |
| **Feature interaction captured** | No (local gradient) | Partial (marginal over random coalitions) | No (leave-one-out) |
| **Score interpretability** | Relative only | Relative only | Relative (logit delta) |
| **Reliability with current settings** | Good (10 steps, stable) | Moderate (64 perturbs — undersampled) | Good (clean sentence-level segments) |
