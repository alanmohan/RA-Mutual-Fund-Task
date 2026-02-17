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
Stage 4: Nonlinear Probing (optional; MLP probes + control tasks)
Stage 5: Causal Tracing (planned)
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

### 4) Generate plots (linear probing)

```bash
python linear_probing/analyze_lp_results.py
```

### 5) Nonlinear probing (optional)

Nonlinear probing uses **MLP probes** on the same activations and splits as linear probing. It runs only on **configurable layers** (see below) and includes **control tasks** (shuffled labels) to check selectivity.

**Prerequisites:** Activations must already be extracted (step 2). Linear probes can be run either separately (step 3) or as part of the combined pipeline below.

**Config** (`nonlinear_probing/nlp_config.py`):

- **`NONLINEAR_PROBE_LAYERS`** – `None` = probe all layers; or a list, e.g. `[0, 5, 10, 15, 20, 25]`, to probe only those layers (faster). This does **not** affect linear probes (they always run on all layers).
- **`NONLINEAR_PROBE_HIDDEN`**, **`NONLINEAR_PROBE_MAX_EPOCHS`**, **`NONLINEAR_PROBE_EARLY_STOPPING_PATIENCE`** – MLP architecture and training.
- **`CONTROL_TASK_SEED`** – Seed for shuffled-label control tasks (reproducibility).

**Run the combined pipeline** (linear probes + nonlinear probes + control tasks + comparison plots):

From the `llm_eval_mutual_funds` directory:

```bash
python nonlinear_probing/run_linear_and_nonlinear.py \
  --model llama-3.2-3b \
  --condition 2_fewshot_cot_temp0
```

Optional arguments:

- **`--output-dir`** – Where to write results (default: `data/probe_results`). Linear results go here; nonlinear and control go to `output_dir/nonlinear/`; plots to `output_dir/plots/`.
- **`--features`** – Subset of features to probe (default: all).

**Outputs:**

- **Linear:** Same as step 3 (`probe_results_*.csv`, `probe_matrix_accuracy_*.csv`, etc.) in the output directory.
- **Nonlinear:** `nonlinear/probe_nonlinear_results_*.csv`, `probe_nonlinear_matrix_accuracy_*.csv`.
- **Control:** `nonlinear/probe_nonlinear_control_results_*.csv`, `probe_nonlinear_control_matrix_accuracy_*.csv`.
- **Plots** (in `plots/`):
  - `linear_vs_nonlinear_by_feature_*.png` – For each feature: linear vs nonlinear vs control (best-layer accuracy).
  - `linear_vs_nonlinear_by_layer_*.png` – Mean accuracy by layer (linear, nonlinear, control).
  - `nonlinear_best_by_feature_*.png` – Nonlinear vs control per feature (selectivity check).

**Interpreting control tasks:** Control uses the same activations with **shuffled labels**. If control accuracy stays near chance (~50%) while the real probe is high, the probe is **selective**. If control accuracy is high and close to the real probe, the MLP may be overfitting or the representation may not be selective for that feature.

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
│   │   ├── nonlinear/      # nonlinear + control results
│   │   └── plots/          # linear vs nonlinear vs control plots
│   └── results/
├── linear_probing/
│   ├── lp_config.py
│   ├── lp_utils.py
│   ├── extract_activations.py
│   ├── probe.py
│   └── analyze_lp_results.py
└── nonlinear_probing/
    ├── nlp_config.py           # layers, MLP settings, control seed
    ├── nonlinear_probe.py      # MLP probe + control task logic
    ├── plot_linear_vs_nonlinear.py
    └── run_linear_and_nonlinear.py   # pipeline: linear + nonlinear + control + plots
```

---

## Notes

- The prompt strategies and evaluation structure closely mirror the housing experiment.
- Feature probing uses fund-level comparisons (e.g., expense ratio lower, Sharpe higher, etc.).
- The target label is **Medalist-based**, not 3-year rank.

