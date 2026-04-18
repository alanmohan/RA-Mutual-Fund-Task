# Layer-wise Relevance Propagation: Theory and Application to Mutual Fund Interpretability

## Overview

This document explains the theoretical foundations of Layer-wise Relevance Propagation (LRP) as implemented in the LXT library and applied to interpreting Llama-3.2-3B-Instruct's mutual fund comparison decisions. It covers the core LRP framework, the specific challenges of applying LRP to transformers, and how AttnLRP and CP-LRP address those challenges. Each concept is presented with both its mathematical formulation (as published in the referenced papers) and an intuitive explanation.

### References

All theoretical content in this document is grounded in the following published works:

1. **Bach et al. (2015):** Sebastian Bach, Alexander Binder, Gregoire Montavon, Frederick Klauschen, Klaus-Robert Mueller, and Wojciech Samek. "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation." *PLOS ONE*, July 10, 2015. DOI: 10.1371/journal.pone.0130140. -- The foundational LRP paper.

2. **Achtibat et al. (2024):** Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain, Thomas Wiegand, Sebastian Lapuschkin, and Wojciech Samek. "AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers." In *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR Volume 235, Pages 135-168, 2024. arXiv: 2402.05602. -- The AttnLRP paper that adapts LRP for transformer architectures.

3. **Arras et al. (2025):** Leila Arras et al. "Close Look at Decomposition-based XAI-Methods for Transformer Language Models." arXiv: 2502.15886. -- Introduces the efficient Input x Gradient reformulation used in the `lxt.efficient` module.

4. **Ali et al. (2022):** "XAI for Transformers: Better Explanations through Conservative Propagation." -- The CP-LRP method that routes relevance only through the value path.

---

## Part I: Foundational LRP Theory

### 1. The Core Idea

**Mathematical formulation (Bach et al., 2015):**

Given a neural network that produces output f(x) for input x, LRP decomposes f(x) into a sum of input-level relevance scores:

```
f(x) = R_1 + R_2 + ... + R_n
```

where R_i is the relevance of the i-th input token. Each R_i can be positive (supports the prediction) or negative (opposes it). The key constraint is that these scores must sum to the model's actual output value.

**Intuitive explanation:**

Imagine the model outputs a logit of 5.3 for "mutual fund 2." LRP asks: *where did that 5.3 come from?* It traces the score backward through every layer of the network, distributing it among the inputs. At the end, you might find that the Sharpe Ratio tokens contributed +1.2, the Expense Ratio tokens contributed +0.8, the NTF tokens contributed +0.5, and so on -- all adding up to 5.3 (in the ideal case).

This is fundamentally different from gradient-based methods. Gradients answer the question "which input would change the output the most if perturbed?" -- they measure *sensitivity*. LRP answers "which input *did* contribute to this specific output?" -- it measures *contribution*. Sensitivity and contribution are related but not identical. A feature can have high sensitivity (large gradient) but low actual contribution if its current value happens to be near a neutral baseline.

### 2. Layer-by-Layer Propagation

LRP works backward through the network, one layer at a time. At each layer, it redistributes the relevance from the layer's outputs to its inputs using local redistribution rules.

**The conservation property:**

At every layer l:

```
sum_i R_i^(l-1) = sum_j R_j^(l)
```

Total relevance entering a layer must equal total relevance leaving it. This is the mathematical guarantee that no relevance is created or destroyed -- only redistributed.

**Intuitive explanation:**

Think of the network as a system of pipes carrying water. The model's output logit is the total amount of water (e.g., 5.3 liters). LRP traces this water backward through the pipe system. At every junction (layer), the total water entering must equal the total water leaving -- no water appears or disappears. By the time you reach the inputs, you know exactly how much water (relevance) came from each source.

This conservation property is why LRP is called a "faithful" attribution method. If you sum up all the attributions, you get back the model's actual prediction. Methods like vanilla gradients or SHAP do not guarantee this.

### 3. The Epsilon-LRP Rule (Linear Layers)

**Mathematical formulation (Equation 8 in Achtibat et al., 2024; Equation 16 in Bach et al., 2015):**

For a linear layer computing z_j = sum_i (W_ji * x_i) + b_j, the epsilon-LRP rule distributes relevance as:

```
R_i^(l-1) = sum_j [ (W_ji * x_i) / (z_j + epsilon) ] * R_j^(l)
```

where epsilon is a small stabilization constant to prevent division by zero.

**Intuitive explanation:**

Consider the model's QKV projection layer, which takes hidden state tokens and produces query, key, and value vectors. Each output neuron z_j is a weighted sum of inputs. The epsilon rule says: *distribute the relevance of z_j back to the inputs in proportion to how much each input contributed to z_j's value.* If input x_i contributed a large weighted activation W_ji * x_i to z_j, it receives a proportionally large share of z_j's relevance.

In the mutual fund context: when the model's attention layer computes a query vector from the token "0.12" (a Sharpe Ratio value), the epsilon rule determines how much of the query's relevance flows back to that specific token based on how much the token's embedding contributed to the query vector's value.

This rule applies to all linear transformations in the model: the QKV projections, the output projection in attention, all MLP layers, and the final language model head.

---

## Part II: The Transformer Challenge

### 4. Why Standard LRP Breaks on Transformers

Standard LRP (Bach et al., 2015) was designed for feedforward networks with ReLU activations. Transformers introduce four operations that standard LRP rules cannot handle correctly:

| Operation | Where It Occurs | Why It's Problematic |
|-----------|----------------|---------------------|
| **Softmax** | Attention weights | Non-linear, f(0) != 0, outputs sum to 1 |
| **Bilinear multiplication** | Q * K^T in attention | Two learned inputs multiplied together (not linear) |
| **Layer normalization** | RMSNorm after attention and MLP | Division by a function of the input (non-linear) |
| **Gated multiplication** | SwiGLU MLP (gate * up) | Element-wise product of two learned paths |

AttnLRP (Achtibat et al., 2024) provides a principled propagation rule for each of these, derived within the Deep Taylor Decomposition framework. This is what makes it "the first method to faithfully and holistically attribute not only input but also latent representations of transformer models."

### 5. The Softmax Problem (The Hardest Part)

**Why softmax is uniquely difficult:**

For most activation functions (ReLU, GELU, SiLU), f(0) = 0. This means the "zero baseline" -- what the network would output if an input were absent -- is simply zero. The identity rule (Section 7 below) exploits this: relevance passes through unchanged because removing the input would remove its effect entirely.

Softmax violates this. If you set all inputs to zero, softmax returns 1/N (uniform distribution), not zero. This means there is a "virtual bias" -- even with no meaningful input, softmax produces a non-trivial output. Standard LRP rules ignore this virtual bias and produce incorrect decompositions.

**AttnLRP's solution -- Proposition 3.1 (Deep Taylor Decomposition for Softmax):**

```
R_i^(l-1) = x_i * ( R_i^(l) - s_i * sum_j R_j^(l) )
```

where s_i = softmax(x)_i is the i-th softmax output.

The correction term `s_i * sum_j R_j^(l)` subtracts the "virtual bias" contribution. Each input's relevance is reduced by the fraction of total relevance that would be assigned even with a zero input, weighted by the softmax output.

**Intuitive explanation:**

When the model computes attention weights (e.g., how much the decision token attends to the Sharpe Ratio token), softmax converts raw attention scores into a probability distribution. Even if the Sharpe token had zero attention score, it would still receive some baseline attention (1/N from uniform softmax). The correction term in Proposition 3.1 removes this baseline, so the relevance only reflects the *above-baseline* attention that the Sharpe token actually earned.

In the mutual fund task: suppose the model attends to 11 feature tokens. Even with no meaningful content, each would receive 1/11 = 9.1% of attention. If the Sharpe token receives 25% attention, only the 15.9% above baseline (25% - 9.1%) represents genuine decision-relevant attention. The Deep Taylor Decomposition for softmax mathematically implements this correction.

### 6. The Bilinear Multiplication Problem (Q * K^T)

**The problem:**

In standard feedforward networks, each layer has one learned input (the hidden state) and one fixed parameter (the weight matrix). But in attention, the Q * K^T computation multiplies two learned inputs (the query from one position and the key from another). Standard LRP's epsilon rule assumes a single input, so it cannot correctly distribute relevance between Q and K.

**AttnLRP's solution -- Proposition 3.3 (Combined Epsilon + Uniform Rule for Matrix Multiplication):**

For the bilinear operation O = A * B (where both A and B are learned tensors):

```
R_A = A * (R_out / (2 * O + epsilon)) * B^T
R_B = B * A^T * (R_out / (2 * O + epsilon))
```

The factor of 2 in the denominator reflects the bilinear nature: there are two contributing inputs, so each receives half the relevance (the uniform rule, Proposition 3.2) combined with the epsilon rule for the linear contribution of each.

**Intuitive explanation:**

When the model computes attention scores via Q * K^T, both the query (representing "what information am I looking for?") and the key (representing "what information do I contain?") contribute to the resulting attention score. It's like a handshake -- both parties participate equally. The combined rule distributes relevance 50/50 between the query path and the key path, and within each path, distributes proportionally to how much each element contributed (the epsilon rule).

In the mutual fund task: when the decision token's query attends strongly to the Sharpe Ratio token's key, the bilinear rule splits the credit evenly between:
- The query ("the model was looking for risk-adjusted performance information")
- The key ("the Sharpe Ratio token advertised itself as containing that information")

### 7. The Identity Rule for Non-linearities and Normalization

**Mathematical formulation (Equation 9, Proposition 3.4 in Achtibat et al., 2024):**

For element-wise non-linear functions (SiLU, GELU, ReLU) and normalization layers (RMSNorm, LayerNorm):

```
R_i^(l-1) = R_i^(l)
```

Relevance passes through unchanged.

**Mathematical justification:**

For activation functions: since f(0) = 0 (for SiLU, GELU, ReLU), the Deep Taylor Decomposition at reference point 0 yields a first-order term that is equivalent to the identity propagation rule (Equation 9 in the paper). Intuitively, removing the input would remove the output, so the input gets full credit.

For normalization: Proposition 3.4 proves that normalization operations (RMSNorm, LayerNorm) can be treated with the identity rule because:
1. The normalization is applied element-wise to each token independently
2. The variance/standard deviation term can be treated as a constant (it doesn't depend on any single element in isolation)
3. This makes the operation effectively a linear rescaling, for which the identity rule is appropriate

The implementation handles this by "detaching the standard deviation from the gradient graph" -- treating it as a fixed scalar rather than a function of the input.

**Intuitive explanation:**

These layers transform values without mixing information across tokens. SiLU activation squashes/rescales each value independently. RMSNorm rescales the entire vector to unit norm. Neither operation introduces new information or combines information from different tokens. Therefore, the relevance simply passes through: if a token's hidden state had relevance R before the activation, it has the same relevance R after.

In the mutual fund task: when the Sharpe Ratio token's hidden representation passes through SiLU activation in the MLP, its relevance is unchanged. The activation modifies the *value* of the representation but doesn't change *how much* it contributes to the final decision.

### 8. The Uniform Rule for Gated Multiplication

**Mathematical formulation (Equation 7, Proposition 3.2 in Achtibat et al., 2024):**

For element-wise multiplication of N inputs (like the gated MLP: output = gate(x) * up(x)):

```
R_i^(l-1) = (1/N) * R^(l)
```

Each input receives an equal share of the relevance.

**Mathematical justification:**

This is derived from the Shapley value decomposition at a zero baseline. For the product of two quantities a * b, the Shapley values at baseline (0, 0) are:

```
phi_a = a*b / 2,    phi_b = a*b / 2
```

Each factor receives half the credit for the product, regardless of their individual magnitudes.

**Intuitive explanation:**

Llama-3.2 uses a SwiGLU MLP architecture: the output is computed as `SiLU(gate(x)) * up(x)`. Two independent paths process the same input and their results are multiplied together. The gating path decides *whether* to pass information; the up-projection path decides *what* information to pass. Since both are necessary for the output (if either is zero, the product is zero), they share credit equally -- 50/50.

In the mutual fund task: when the model's MLP processes the Sharpe Ratio token, both the gating path ("this information is relevant, let it through") and the value path ("this is a high Sharpe Ratio") receive equal credit for the resulting MLP output.

---

## Part III: AttnLRP vs CP-LRP

### 9. AttnLRP (Attention-aware LRP)

**What it does:** AttnLRP applies the mathematically derived rules from Propositions 3.1-3.4 to every component of the transformer. Relevance flows through *all* pathways: the query path, the key path, the value path, and through the softmax function.

**How it handles attention (from the LXT library implementation):**

In the efficient implementation, the attention computation uses pre-applied gradient correction factors:

```python
query = divide_gradient(query, 4)   # Q receives 1/4 of attention relevance
key   = divide_gradient(key, 4)     # K receives 1/4 of attention relevance
value = divide_gradient(value, 2)   # V receives 1/2 of attention relevance
```

**Why Q/4, K/4, V/2:**

The attention computation involves two sequential bilinear operations:
1. `scores = Q @ K^T` -- bilinear, so Q and K each get 1/2
2. `output = softmax(scores) @ V` -- bilinear, so the scores path and V each get 1/2

Combining these:
- V participates in one bilinear operation: 1/2
- Q participates in both: 1/2 (from Q@K) * 1/2 (from scores@V) = 1/4
- K participates in both: 1/2 (from Q@K) * 1/2 (from scores@V) = 1/4
- Total: 1/4 + 1/4 + 1/2 = 1 (conservation holds)

These factors are pre-applied to the Q, K, V inputs rather than at each intermediate step. This is an implementation optimization described in the LXT documentation (`extending.rst`): since modern attention implementations (FlashAttention, SDPA) are opaque (you cannot access intermediate attention weight tensors), the factors are applied at the inputs where they are accessible, achieving the same mathematical result.

**How it handles the gated MLP:**

```python
gate_out = self.gate_proj(x)
gate_out = identity_rule_implicit(self.act_fn, gate_out)  # Eq. 9: identity rule on SiLU
weighted = gate_out * self.up_proj(x)
weighted = divide_gradient(weighted, 2)  # Eq. 7: uniform rule (50/50 split)
return self.down_proj(weighted)
```

The activation function gets the identity rule (relevance passes through); the element-wise multiplication gets the uniform rule (50/50 split between gate and up paths).

**Strengths:**
- Theoretically principled: every rule is derived from the Deep Taylor Decomposition framework
- Relevance flows through all pathways, capturing the full computational graph
- The paper reports it "surpasses alternative methods in terms of faithfulness" (measured by perturbation curves on LLaMA 2, Mixtral 8x7b, Flan-T5, and ViT)

**Weaknesses:**
- The softmax rule (Proposition 3.1) relies on a first-order Taylor approximation, which is exact only locally
- Pre-applying gradient correction factors (Q/4, K/4, V/2) is an efficient approximation that may not perfectly match the explicit mathematical implementation
- Conservation is approximate in the efficient implementation (~60% in our experiments)

### 10. CP-LRP (Conservative Propagation LRP)

**What it does:** CP-LRP takes a more aggressive approach: it completely blocks gradient flow through the softmax function by stopping gradients at the query and key tensors. All relevance is routed exclusively through the value path.

**How it handles attention (from the LXT library implementation):**

```python
query = stop_gradient(query)   # No relevance flows through Q
key   = stop_gradient(key)     # No relevance flows through K
value = value                  # V receives ALL attention relevance
```

**How it handles the gated MLP:**

```python
gate_out = stop_gradient(self.gate_proj(x))  # Gate path completely blocked
gate_out = self.act_fn(gate_out)             # Activation applied but no gradient
weighted = gate_out * self.up_proj(x)        # Only up_proj path carries relevance
return self.down_proj(weighted)
```

**Intuitive explanation:**

CP-LRP simplifies the question. Instead of asking "how much did each token contribute through the full attention mechanism?" it asks "how much relevance was carried by the value vectors that the attention mechanism selected?"

Think of attention as a librarian (the softmax over Q*K^T) selecting books (the V vectors) from a shelf. AttnLRP gives credit to both the librarian's selection process AND the content of the selected books. CP-LRP ignores the librarian entirely and only credits the books -- it measures *what information was passed through attention* but not *why that information was selected*.

In the mutual fund task:
- **AttnLRP** might show high Sharpe relevance because (a) the model's attention mechanism specifically searched for risk-adjusted return data (Q*K pathway) AND (b) the Sharpe values themselves were informative (V pathway).
- **CP-LRP** would only capture (b) -- whether the Sharpe value tokens carried useful information through the value path, regardless of how attention selected them.

**Strengths:**
- Simpler and more numerically stable (avoids the softmax approximation entirely)
- Guarantees strict conservation by avoiding problematic non-linearities
- May produce cleaner attributions for features whose importance comes from their content rather than their attention patterns

**Weaknesses:**
- Discards information flowing through the attention weighting mechanism
- The ICML 2024 paper reports AttnLRP outperforms CP-LRP by 46% on faithfulness metrics (top-1 accuracy: 2.50 vs 1.72 on Mixtral 8x7b)
- Produces higher-variance attributions because the stabilizing signal from the softmax pathway is removed

### 11. Direct Comparison

| Property | AttnLRP | CP-LRP |
|----------|---------|--------|
| Relevance through Q | Yes (1/4 share) | Blocked |
| Relevance through K | Yes (1/4 share) | Blocked |
| Relevance through V | Yes (1/2 share) | All relevance |
| Relevance through softmax | Yes (Deep Taylor Decomposition) | Blocked |
| Relevance through MLP gate | Yes (1/2 share via uniform rule) | Blocked |
| Conservation guarantee | Approximate (theoretical exact, empirical ~60%) | Strict (by construction) |
| Faithfulness (ICML 2024 eval) | Higher | Lower (46% less on Mixtral) |
| Variance of attributions | Lower | Higher |

---

## Part IV: The Efficient Implementation (Input x Gradient Framework)

### 12. How LXT Computes LRP Efficiently

The explicit mathematical implementation of LRP (implementing each rule as a custom autograd function) is slow because it requires storing intermediate activations and computing custom backward passes for every operation. Arras et al. (2025) showed that LRP can be reformulated as a modified backpropagation pass using the Input x Gradient framework.

**The key insight:**

For a chain of operations x1 -> x2 -> x3 -> z:

```
x2 = W1 * x1              (linear)
x3 = SiLU(x2)             (activation)
z  = W2 * x3              (linear)
```

The explicit LRP computation is:

```
R^(x3) = x3 * W2^T * [R^(z) / (z + epsilon)]           (epsilon rule)
R^(x2) = R^(x3)                                         (identity rule)
R^(x1) = x1 * W1^T * [R^(x2) / (x2 + epsilon)]         (epsilon rule)
```

Substituting and simplifying:

```
R^(x1) = x1 * [ W1^T * (SiLU(x2) / (x2 + epsilon)) * W2^T * (R^(z) / (z + epsilon)) ]
```

The terms `W1^T` and `W2^T` are exactly the Jacobians that PyTorch computes during a standard backward pass. The only non-standard term is `SiLU(x2) / (x2 + epsilon)` -- the ratio of the activation output to its input.

This means: **if we modify only the activation function's backward pass** (to return the ratio instead of the standard gradient), and run a normal PyTorch backward pass, the result of `input * input.grad` gives us the LRP relevance.

**What the monkey patch does:**

LXT's `monkey_patch()` replaces the forward functions of three types of layers:
1. **Activation functions** (SiLU, GELU): Modified to implement the identity rule via the ratio trick
2. **Normalization layers** (RMSNorm): Modified to detach the standard deviation from the gradient graph
3. **Attention mechanism**: Modified to pre-apply gradient correction factors (Q/4, K/4, V/2 for AttnLRP, or stop gradients on Q/K for CP-LRP)

All linear layers (QKV projections, MLPs, LM head) use standard PyTorch backpropagation unchanged, because the epsilon-LRP rule for linear layers is mathematically equivalent to the standard gradient computation when combined with the Input x Gradient formulation.

**The final computation:**

```python
input_embeds = model.get_input_embeddings()(input_ids)
input_embeds.requires_grad_()
output_logits = model(inputs_embeds=input_embeds, use_cache=False).logits
target_logit = output_logits[0, -1, target_token_id]
target_logit.backward()

relevance = (input_embeds * input_embeds.grad).sum(-1)  # Per-token relevance
```

The multiplication `input_embeds * input_embeds.grad` implements the Input x Gradient formulation. The `.sum(-1)` collapses the embedding dimension to get a single relevance score per token.

**Computational cost:** One forward pass + one backward pass. This is O(1) relative to the number of layers and has the same cost as computing standard gradients. The O(sqrt(N)) memory claim comes from gradient checkpointing, which recomputes intermediate activations during the backward pass rather than storing them all.

---

## Part V: Application to Mutual Fund Interpretability

### 13. What LRP Tells Us About the Model's Decision Process

When applied to the mutual fund comparison task, LRP answers the question: **"When the model output the logit for 'mutual fund 2,' how much of that logit value came from each input token?"**

This is distinct from what the other attribution methods answer:

| Method | Question Answered |
|--------|------------------|
| **Integrated Gradients** | "How much does each feature contribute along the interpolation path from a zero baseline to the actual input?" |
| **KernelSHAP** | "What is the average marginal contribution of each feature across all possible feature coalitions?" |
| **Occlusion** | "How much does the output change when each feature is removed?" |
| **LRP (AttnLRP)** | "How much of the output logit was produced by each input token, traced through all layers?" |
| **LRP (CP-LRP)** | "How much of the output logit was carried by each token through the value/residual pathway?" |

These are all valid but different questions. The disagreements between methods in our results (e.g., Sharpe at rank 1 for LRP vs rank 9 for IG) reflect these fundamental differences in what is being measured, not necessarily errors in any method.

### 14. Why LRP Might See Sharpe Differently

The Sharpe Ratio's dominance under LRP (24.4% AttnLRP, 16.8% CP-LRP) contrasted with its low ranking under interpreto methods deserves theoretical explanation:

**The LRP perspective:** LRP traces how the logit value is built up through the network. Sharpe is a composite metric (return divided by risk) that maps directly onto the model's decision criterion ("which fund is better?"). If the model's internal representations encode Sharpe as a strong decision signal -- i.e., the neurons activated by Sharpe tokens carry large magnitudes through the network -- then LRP will assign it high relevance because those large activations directly contribute large terms to the final logit via the epsilon rule.

**Why interpreto methods might disagree:**
- **IG** integrates gradients along a path from zero to the actual input. If Sharpe's contribution is non-linear along this path (e.g., it matters a lot once you have it, but the gradient at intermediate interpolation points is small), IG may underestimate it.
- **Occlusion** removes a feature and measures the change. If removing Sharpe only slightly changes the output (because the model can partially reconstruct the comparison from other features like Return_3yr and Std Dev), Occlusion will rank it lower even though the model heavily uses it when present.
- **SHAP** averages over feature coalitions. Similar to Occlusion, if other features are partially redundant with Sharpe, SHAP will distribute credit among them.

LRP, by contrast, does not involve any counterfactual (what if this feature were absent?). It decomposes the actual computation as-is. This makes it sensitive to the *magnitude of internal activations* rather than the *counterfactual impact of removal*.

### 15. What the Layer-Wise Analysis Reveals

LRP's unique ability to extract per-layer relevance provides mechanistic insight into how the model processes mutual fund features:

**The bimodal relevance profile (peaks at layers 9-13 and 20-23):**

This suggests two processing phases:
1. **Mid-network phase (layers 5-13):** The model encodes and compares feature values. Sharpe, Expense Ratio, and Inception show peak relevance here. This is where the model likely performs the numerical comparisons ("Fund 1's Sharpe is 0.12, Fund 2's is 0.10").
2. **Late-network phase (layers 18-25):** The model consolidates comparisons into a decision. The positive correlation between LRP relevance and probe accuracy at these layers (rho = 0.63-0.69, p < 0.05) means: features that the model has successfully encoded (high probe accuracy) also receive more relevance in the decision-making layers. The model uses what it knows.

**Layer 27 (final transformer layer) has zero relevance:** All feature relevance drops to zero at the last layer, meaning the final transformer layer's role is not feature-specific processing but rather projecting the aggregated representation into the vocabulary space via the LM head.

### 16. What Signed Relevance Reveals

LRP uniquely provides signed relevance, where positive values support the decision and negative values oppose it. In our experiments:
- All features are overwhelmingly positive (80-98%)
- Sharpe has the highest positive fraction (97.6%)
- Turnover has the most negative relevance (20.2%)

This means the model's decision is driven primarily by *evidence for* the chosen fund rather than *evidence against* the other fund. Features that support the winning fund contribute positively; features that favor the losing fund (e.g., it has better turnover) contribute small negative amounts that are overridden by the positive evidence.

---

## Part VI: Interpreting the Disagreements

### 17. The Feature Attribution Fraction Problem

The most important caveat for interpreting LRP results in this task is the low feature attribution fraction: only 4.5% (AttnLRP) / 7.0% (CP-LRP) of total relevance lands on mutual fund feature tokens. The remaining 93-95.5% goes to system prompt tokens, chat template tokens, and instruction text.

**Why this happens (theory):** LRP distributes relevance to *every* token based on its contribution to the output logit. The system prompt tokens ("You are an objective financial analyst...") and instruction tokens ("Compare mutual fund 1 vs mutual fund 2...") establish the task context that is essential for the model to produce a meaningful decision. Without them, the model would not know to compare funds. LRP correctly identifies these tokens as highly relevant to the output -- they are literally part of the causal chain that produces the logit.

**Why interpreto methods don't have this problem:** Interpreto methods work at the sentence/segment level and typically treat the system prompt as a fixed context. IG, SHAP, and Occlusion perturb or integrate over the *feature segments* while keeping the instruction/system tokens constant. This implicitly attributes zero importance to the instruction tokens and normalizes across feature tokens only.

**Practical consequence:** LRP's feature rankings are derived from a small fraction of total relevance. The relative ordering within that 4.5% is meaningful (Sharpe's fraction of feature relevance is robust), but the absolute magnitudes should not be directly compared to interpreto scores. The per-sample normalization (making each sample's feature scores sum to 1) addresses this by putting all methods on the same scale for comparison.

### 18. Conservation Loss in Practice

Our experiments show a conservation ratio of 0.604 (std: 0.015), meaning approximately 60% of the target logit is recovered at the input layer. The theoretical guarantee is 100% conservation.

**Why the gap exists:**
1. **The efficient implementation uses Input x Gradient**, which is an approximation of the explicit mathematical LRP rules. Arras et al. (2025) show this is exact for linear layers and activation functions, but the pre-applied gradient correction factors (Q/4, K/4, V/2) for attention are an efficient approximation.
2. **Gradient checkpointing** recomputes intermediate activations during the backward pass. This recomputation may introduce small numerical differences in the modified gradient flow.
3. **bfloat16 precision** introduces rounding errors that accumulate across 28 layers.

**Impact on results:** The very low standard deviation (0.015) indicates the loss is systematic and consistent, not random. This means it acts as a uniform scaling factor: all relevance scores are attenuated by roughly the same factor. Relative rankings between features are preserved even if absolute magnitudes are reduced.

---

## Summary

LRP, as implemented in the LXT library through AttnLRP and CP-LRP, offers a fundamentally different lens on model interpretability compared to perturbation-based methods (Occlusion, SHAP) and path-integration methods (Integrated Gradients). It decomposes the model's actual output into per-token contributions by propagating relevance backward through modified backpropagation rules that are mathematically derived from the Deep Taylor Decomposition framework.

**AttnLRP** preserves relevance flow through all pathways (Q, K, V, softmax, gated MLP) using Propositions 3.1-3.4 from the ICML 2024 paper. It produces more stable attributions and achieves higher faithfulness scores but relies on approximations (especially for softmax).

**CP-LRP** blocks relevance through the attention weighting mechanism (softmax, Q, K) and the MLP gate, routing everything through the value and up-projection paths. It is simpler and strictly conservative but discards information about *why* the model attended to certain tokens.

The disagreements between LRP and interpreto methods in the mutual fund task are not errors but reflections of fundamentally different questions being asked. LRP measures *contribution to the output logit through the computational graph*; interpreto methods measure *perturbation impact on the output*. Both perspectives are valuable, and the most reliable findings are those where multiple methods converge.
