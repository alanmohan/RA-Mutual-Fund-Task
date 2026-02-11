# Mechanistic Interpretability of LLMs on Mutual Fund Comparison

**Why LLMs struggle to rank mutual funds using structured financial features**

---

## Project Overview

This project mirrors the `llm_eval_housing` pipeline, but for **mutual fund comparisons**.  
We evaluate LLM accuracy on a pairwise fund ranking task, then use **activation extraction** and **linear probing** to analyze what information the model encodes vs. where it fails.

**Key difference:** Ground truth is based on **Morningstar Medalist ratings**, not 3-year rank.

---

## Dataset

- **Source**: `input_csvs/mutual_funds_pairs_no_date.csv`
- **Unit of analysis**: Pair of mutual funds from the same Morningstar category

### Ground Truth (Target)
We use **Medalist ratings** as ground truth:

`Gold > Silver > Bronze > Neutral > Negative`

If the two funds have the same Medalist rating, the pair is marked as a tie and excluded from accuracy metrics by default.

---

## Pipeline

```
Stage 1: Prompted Model Evaluation
Stage 2: Activation Extraction (last-token residual stream)
Stage 3: Linear Probing (feature comparisons + target)
Stage 4: Causal Tracing (planned)
```

---

## Models Evaluated

Same models as the housing project:

- **Llama-3.2-3B-Instruct**
- **Qwen3-4B-Instruct-2507**

---

## Running the Experiment

### 1) Prompted accuracy evaluation

```bash
python main.py
```

### 2) Extract activations (for probing)

```bash
python linear_probing/extract_activations.py \
  --model llama-3.2-3b \
  --condition 2_fewshot_cot_temp0
```

### 3) Linear probing

```bash
python linear_probing/probe.py \
  --model llama-3.2-3b \
  --condition 2_fewshot_cot_temp0
```

### 4) Generate plots

```bash
python linear_probing/analyze_lp_results.py
```

---

## Repository Structure

```
llm_eval_mutual_funds/
├── README.md
├── config.py
├── data_loader.py
├── prompts.py
├── experiment.py
├── inference.py
├── results.py
├── utils.py
├── checkpoint.py
├── main.py
├── data/
│   ├── activations/
│   ├── probe_results/
│   └── results/
└── linear_probing/
    ├── lp_config.py
    ├── lp_utils.py
    ├── extract_activations.py
    ├── probe.py
    └── analyze_lp_results.py
```

---

## Notes

- The prompt strategies and evaluation structure closely mirror the housing experiment.
- Feature probing uses fund-level comparisons (e.g., expense ratio lower, Sharpe higher, etc.).
- The target label is **Medalist-based**, not 3-year rank.

