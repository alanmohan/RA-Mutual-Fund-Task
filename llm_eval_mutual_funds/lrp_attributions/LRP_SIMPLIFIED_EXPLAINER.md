# LRP for Mutual Fund Interpretability: A Simplified Guide

## What LRP Does

Layer-wise Relevance Propagation (LRP) takes the model's output -- say, a logit of 5.3 for "mutual fund 2" -- and traces it backward through every layer of the network, asking: **how much of that 5.3 came from each input token?**

The result is a relevance score for every token in the prompt. Tokens from the Sharpe Ratio line might account for 1.2, Expense Ratio tokens for 0.8, system prompt tokens for 2.1, and so on. In the ideal case, all scores sum back to 5.3. This is the **conservation property** -- relevance is neither created nor destroyed, only redistributed.

This makes LRP fundamentally different from the interpreto methods:

| Method | What it measures |
|--------|-----------------|
| Integrated Gradients | How much does each feature contribute along an interpolation path from a blank input to the real input? |
| KernelSHAP | What is the average marginal contribution of each feature across all possible feature combinations? |
| Occlusion | How much does the output change when each feature is removed? |
| **LRP** | **How much of the actual output logit was produced by each token, traced through the network's computation?** |

The interpreto methods all involve some form of "what if this feature were absent?" LRP does not. It decomposes the computation as-is, without hypothetical removals. This is why the methods can disagree -- they answer different questions.

---

## The Two LRP Variants

We ran two variants from the LXT library, both based on the AttnLRP paper (Achtibat et al., ICML 2024). They differ in how they handle the **attention mechanism** -- specifically, the softmax that converts raw attention scores into attention weights.

### AttnLRP

AttnLRP lets relevance flow through **all three pathways** in attention:

```
             +---> Query (Q) ---+
             |                  |
Input token --+---> Key (K) -----+--> Attention Weights (softmax) --+--> Output
             |                                                      |
             +---> Value (V) --------------------------------------+
```

- **Query path (receives 1/4 of relevance):** Captures "what the model was searching for." If the decision token's query specifically looked for risk-adjusted return data, the Sharpe token gets credit through this path.
- **Key path (receives 1/4 of relevance):** Captures "what each token advertised about itself." If the Sharpe token's key representation signaled "I contain performance information," it gets credit here.
- **Value path (receives 1/2 of relevance):** Captures "what information was actually passed forward." The Sharpe token's actual numerical content flows through this path.

The 1/4, 1/4, 1/2 split comes from the mathematical structure of attention: there are two sequential multiplications (Q*K and then scores*V), and at each multiplication both inputs share credit equally.

**What AttnLRP tells you:** The full picture -- how much each token contributed through every pathway, including *why* the model attended to it (Q, K paths) and *what* it extracted from it (V path).

### CP-LRP (Conservative Propagation)

CP-LRP simplifies things by **blocking** relevance through the query and key paths entirely:

```
             +---> Query (Q) ---+
             |        [BLOCKED] |
Input token --+---> Key (K) -----+--> Attention Weights (softmax) --+--> Output
             |        [BLOCKED]                                     |
             +---> Value (V) [ALL RELEVANCE] -----------------------+
```

- **Query path:** Blocked. No relevance flows through.
- **Key path:** Blocked. No relevance flows through.
- **Value path (receives all relevance):** The only pathway carrying information.

CP-LRP also blocks the gating path in the MLP (Llama uses a gated MLP where one path decides *whether* to pass information and another decides *what* to pass -- CP-LRP only credits the "what" path).

**What CP-LRP tells you:** Which tokens' **content** mattered, ignoring *why* the model chose to attend to them. It answers: "what information flowed through the network?" rather than "what information did the model seek out?"

### When They Disagree

The two variants agreed that **Sharpe is #1** but disagreed on mid-tier features:

| Feature | AttnLRP Rank | CP-LRP Rank | Interpretation |
|---------|-------------|-------------|----------------|
| Return_3yr | 2 | 6 | Return_3yr's high AttnLRP rank comes partly from the Q/K pathway (the model actively searches for performance data). Under CP-LRP, which ignores the search mechanism, its content alone is less distinctive. |
| Tenure | 6 | 3 | Tenure's date/duration content carries substantial information through the value path (CP-LRP picks this up), but the model's attention mechanism doesn't specifically seek it out (AttnLRP ranks it lower). |
| Turnover | 11 | 5 | Turnover has rich numerical content that flows through the value path, but the model doesn't actively attend to it via the Q/K pathway. |

Their rank correlation is rho = 0.618 (p = 0.043, statistically significant), confirming they capture related but distinct aspects of the model's computation.

---

## Single-Logit vs Logit-Difference Attribution

The two LRP variants above (AttnLRP and CP-LRP) differ in *how relevance flows through the network*. There is a second, orthogonal choice: *what output to attribute to*. We ran both options, giving us four total configurations.

### Single-Logit Attribution (the standard approach)

In single-logit attribution, we trace backward from the model's chosen answer. If the model picks "mutual fund 2," we take the logit for token "2" at the last position -- say it's 5.3 -- and decompose that 5.3 into per-token contributions.

```
Prompt tokens  ──►  Model  ──►  logit("2") = 5.3
                                      │
                         backward: decompose 5.3
                                      │
                         relevance per token (sums ≈ 5.3)
```

This answers: **"How much did each token contribute to the model's confidence in its chosen answer?"**

The problem is that a high logit for "2" doesn't necessarily mean the model strongly *prefers* fund 2. Maybe the model assigns logit 5.3 to "2" and logit 5.1 to "1" -- barely a preference. Or maybe it assigns 5.3 to "2" and 2.0 to "1" -- a strong preference. Single-logit attribution can't distinguish these cases. It decomposes the raw confidence, not the decision.

### Logit-Difference Attribution (the decision-margin approach)

Logit-difference attribution traces backward from the *gap between the two options* instead of the chosen option alone. If the model assigns logit 5.3 to "2" and logit 4.1 to "1," the difference is 1.2. We decompose that 1.2 into per-token contributions.

```
Prompt tokens  ──►  Model  ──►  logit("2") = 5.3
                                 logit("1") = 4.1
                                      │
                         backward: decompose (5.3 - 4.1) = 1.2
                                      │
                         relevance per token (sums ≈ 1.2)
```

This answers: **"How much did each token contribute to the model *preferring* its chosen answer over the alternative?"**

### Why the difference matters

Consider an analogy. You're choosing between two restaurants. Single-logit is like asking "why do you like Restaurant A?" -- you'd mention the food, the ambiance, the location. Logit-difference is like asking "why do you *prefer* Restaurant A over Restaurant B?" -- now you'd focus on what actually distinguishes them: maybe the food quality is similar, but A is closer and cheaper.

For mutual fund comparison, this distinction matters because:

- **Features that both funds share similar values for** will produce high single-logit relevance (they contribute to the absolute confidence) but low logit-difference relevance (they don't help distinguish the funds).
- **Features where the two funds differ substantially** will produce high logit-difference relevance because they're what actually drives the preference.

In practice, logit-difference works via standard backpropagation because gradients are linear: calling `.backward()` on `logit_chosen - logit_rejected` is mathematically equivalent to computing the gradient of `logit_chosen` minus the gradient of `logit_rejected`. The LRP rules activate during this backward pass just as they do for single-logit.

### What changed in the rankings

The logit-difference variant concentrates relevance more sharply on the top features:

| Feature | AttnLRP (single) | AttnLRP (logit-diff) | Interpretation |
|---------|-------------------|---------------------|----------------|
| sharpe | 20.8% (rank 1) | 21.0% (rank 1) | Stable at #1. Sharpe drives both absolute confidence and the decision margin. |
| expense_ratio | 10.7% (rank 3) | 18.7% (rank 2) | **Big jump.** Expense Ratio contributes more to *distinguishing* the funds than its absolute relevance suggests. The two funds often differ meaningfully on cost. |
| return_3yr | 14.9% (rank 2) | 16.8% (rank 3) | Stays high. 3-year return is a strong discriminator. |
| inception | 10.5% (rank 4) | 8.1% (rank 4) | Drops slightly. Inception dates contribute to absolute confidence but less to fund discrimination. |
| tenure | 6.9% (rank 6) | 4.8% (rank 9) | **Drops.** Tenure contributes to the model's overall reasoning but doesn't strongly distinguish the two funds. |

The top 3 under logit-diff (Sharpe + Expense Ratio + Return_3yr) account for **56.5%** of feature relevance, compared to 46.4% under single-logit. By focusing on the decision margin, the method more clearly separates the features that actually drive fund preference from those that contribute to general reasoning.

The single-logit vs logit-diff correlation (AttnLRP) is rho = 0.745 (p = 0.009) -- correlated but not identical, confirming they measure related but distinct aspects.

---

## How LRP Was Applied to the Mutual Fund Task

### Setup

- **Model:** Llama-3.2-3B-Instruct, bfloat16, no quantization (prioritizing attribution quality)
- **Samples:** 500 non-tie mutual fund pairs (seed=42)
- **Feature-order permutations:** 100 deterministic diverse permutations to control for positional bias
- **Decimal truncation:** 4 decimal places to eliminate tokenization artifacts
- **Prompts:** Zero-shot prompts with system prompt, formatted via the chat template
- **Hardware:** NVIDIA A100 40GB

### The Two-Pass Process

**Pass 1 -- Generate the model's response:**
The model is run in eval mode to generate its answer ("mutual fund 1" or "mutual fund 2") for each sample. The decision token ("1" or "2") is extracted. Each sample is assigned one of 100 feature-order permutations, so the 11 features appear in different positions across different prompts.

**Pass 2 -- Trace the decision backward (single-logit):**
For each sample, the prompt is fed through the model again, but this time:
1. The input embeddings are set to track gradients
2. The model is run with LRP-patched layers (modified backward pass)
3. The logit for the specific decision token at the last position is selected
4. `.backward()` is called -- the LRP rules automatically activate during backpropagation
5. The per-token relevance is extracted as `(input_embeddings * input_embeddings.grad).sum(-1)`

**Pass 3 -- Trace the decision margin backward (logit-difference):**
Same as Pass 2, except step 3 selects `logit_chosen - logit_rejected` instead of `logit_chosen` alone. Everything else (LRP rules, relevance extraction, aggregation) is identical.

This produces one relevance score per input token for each attribution target.

### Token-to-Feature Aggregation

The prompt has ~591 tokens, but only ~120-140 of those are mutual fund feature tokens. The rest are system prompt, chat template markers, and instruction text.

To get feature-level scores, each token is mapped to its corresponding feature (e.g., "0.12" maps to "Sharpe Ratio for Fund 1") using the tokenizer's character offset mapping. Token-level relevance scores are summed within each feature. The same per-sample normalization as the interpreto notebook is applied (each sample's feature scores sum to 1).

### Layer-Wise Analysis (LRP-Unique)

Beyond input-level attribution, LRP can extract relevance at every intermediate layer. For each of the 28 transformer layers, forward hooks capture the layer's output and its gradient. Per-layer relevance is computed identically: `(layer_output * layer_output.grad).sum(-1)`.

This produces a 28-layer x 11-feature matrix showing where in the network each feature is processed. This analysis was cross-referenced with linear probe accuracy (from the probing experiments) to test whether layers that encode feature information also assign high relevance to those features.

---

## What the Results Show

### Feature Rankings (All Four LRP Variants)

| Feature | AttnLRP | CP-LRP | LogitDiff AttnLRP | LogitDiff CP-LRP | IG | SHAP | Occlusion |
|---------|---------|--------|-------------------|------------------|----|------|-----------|
| sharpe | **1** | **1** | **1** | **1** | 9 | 10 | 6 |
| expense_ratio | 3 | 4 | **2** | 3 | 2 | 4 | 7 |
| return_3yr | 2 | 6 | 3 | 6 | 11 | 9 | 11 |
| inception | 4 | 2 | 4 | 4 | 6 | 7 | 5 |
| ntf | 10 | 10 | 10 | 11 | **1** | **1** | **1** |

All four LRP variants unanimously rank **Sharpe #1**. All three interpreto methods unanimously rank **NTF #1**. This is the central disagreement.

Both perspectives have merit:
- **LRP's Sharpe dominance** means the model's internal computation routes more logit-value through Sharpe tokens than any other feature. Sharpe is a composite metric (return/risk) that directly maps to the comparison task.
- **Interpreto's NTF dominance** means removing or perturbing NTF changes the output more than removing other features. NTF is a simple binary flag that may serve as an easy heuristic for the model.

These are not contradictory. A feature can contribute heavily to the output (high LRP) while being partially redundant with other features (lower perturbation impact). Sharpe may carry the most information through the network, but if you remove it, the model can partially reconstruct the comparison from Return_3yr and Std Dev. NTF, being a simple binary signal, cannot be reconstructed from other features, so its removal has outsized impact.

The logit-difference variants sharpen the picture: under logit-diff AttnLRP, the top 3 (Sharpe, Expense Ratio, Return_3yr) capture 56.5% of feature relevance -- these are the features that actually drive the model's *preference* between funds, not just its overall confidence.

### Layer-Wise Findings

The LRP-probe correlation flips from negative (early layers) to significantly positive (layers 21-25, rho ~ 0.62-0.69, p < 0.05). This means:
- In early layers, the model is "processing" features (high gradient activity) without yet encoding the comparison direction
- In late layers (21-25), features that the model successfully encoded (high probe accuracy) also receive more LRP relevance -- the model uses what it knows to make its decision

### What All Methods Agree On

Despite the disagreements, some findings are robust across all methods:
- **Expense Ratio is consistently important** (rank 2-4 across all LRP variants; rank 2-7 across all methods)
- **Standard Deviation is consistently unimportant** (rank 7-10 across all methods)
- **Return_3yr is consistently low** for interpreto methods but elevated for LRP, especially under logit-difference attribution
