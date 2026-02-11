# LLM Mutual Fund Comparison Evaluation Project

## Project Overview

This project evaluates the reliability and performance of Large Language Models (LLMs) in comparing mutual funds and predicting which fund will deliver higher returns. The goal is to assess how well LLMs can make objective, data-driven investment decisions using structured financial features.

### Research Questions

1. Can LLMs effectively compare mutual funds using only provided financial features?
2. How do different prompting strategies (zero-shot vs single-shot) affect performance?
3. How do different hyperparameters (temperature vs top_p) influence predictions?
4. Is it easier for LLMs to predict short-term returns or determine overall fund quality?

## Dataset

### Source Files

- **Primary Dataset**: `input_csvs/mutual_funds_pairs_no_date.csv`
- **Alternative Dataset**: `input_csvs/mutual_funds_pairs.csv` (includes date information in some fields)
- **Reference Documentation**: `Help Me Read This Table.pdf` (explains all feature definitions)

### Dataset Structure

Each row contains a pair of mutual funds to be compared. The dataset includes:

- **Identifiers**: `Morningstar Category`, `Name_1`, `Name_2`, `Medalist_1`, `Medalist_2`
- **Ground Truth**: `3-year Rank_1`, `3-year Rank_2` (used to determine `lowest_rank`)
- **Features for Fund 1**: All financial metrics with suffix `_1`
- **Features for Fund 2**: All financial metrics with suffix `_2`
- **Metadata**: `interestingness` score

### Features Used

For each fund in a pair, the following 12 features are provided to the LLM:

1. **Expense Ratio - Net**: Total annual operating expense (after waivers/reimbursements)
2. **3 Year Sharpe Ratio**: Risk-adjusted performance measure (higher is better)
3. **Standard Deviation**: Historical volatility measure (lower is better for same returns)
4. **3 Yr**: 3-year return percentage
5. **Beta**: Sensitivity to market movements (1.0 = market average)
6. **Manager Tenure**: Years the current manager has served
7. **Inception Date**: When the fund was first opened
8. **Assets (Millions)**: Total assets under management
9. **Turnover Rates**: Trading activity measure
10. **Load (Y/N)**: Whether the fund charges a sales charge
11. **NTF**: Whether the fund charges a transaction fee (No Transaction Fee)

### Ground Truth Calculation

The ground truth label (`lowest_rank`) is determined by comparing the **3-year Rank** values:
- **3-year Rank** represents a fund's performance percentile within its Morningstar Category (lower rank = better performance)
- If `3-year Rank_1` ≤ `3-year Rank_2`, then `lowest_rank = "mutual fund 1"`
- Otherwise, `lowest_rank = "mutual fund 2"`

## Task Design

### Two-Part Question

Each fund pair is evaluated with two separate questions:

1. **Better Fund**: "Which mutual fund is better overall?" → Output: `better_fund`
2. **Next Month Return**: "Which mutual fund will likely deliver a higher return in the next month?" → Output: `next_month`

Both questions must be answered with either `"mutual fund 1"` or `"mutual fund 2"`.

### Prompting Strategies

#### Zero-Shot Prompting
- No examples provided
- Model relies on its general knowledge and reasoning
- Templates: `prompts/zero_shot_prompt_template.txt` and `prompts/zero_shot_next_month_prompt_template.txt`

#### Single-Shot Prompting
- Includes one worked example showing step-by-step reasoning
- Demonstrates how to compare funds and arrive at a conclusion
- Templates: `prompts/single_shot_prompt_template.txt` and `prompts/single_shot_next_month_prompt_template.txt`

### System Prompt

The system prompt (`prompts/system_prompt.txt`) provides:
- Role definition (mutual fund analyst)
- Brief descriptions of all features
- Instructions to use only provided values
- Output format requirements

## Experimental Design

### Experiments

Four experiments are run, combining:
- **Prompt Type**: Zero-shot or Single-shot
- **Hyperparameters**: Temperature=0.0 or Top-p=0.1

| Experiment ID | Prompt Type | Hyperparameters |
|---------------|-------------|-----------------|
| `zero_shot_temp0` | Zero-shot | Temperature = 0.0 |
| `zero_shot_top_p_0_1` | Zero-shot | Top-p = 0.1 |
| `single_shot_temp0` | Single-shot | Temperature = 0.0 |
| `single_shot_top_p_0_1` | Single-shot | Top-p = 0.1 |

### Sampling Strategy

- All experiments use the **same randomly sampled set** of `n` rows (for fair comparison)
- Default: `n=10`, `seed=42`
- Each row index is preserved in the output for traceability

### Batch Processing

- Each experiment is split into **2 chunks** (default) to avoid token limit issues
- Chunk size is automatically calculated: `chunk_size = (n + 1) // 2`
- Each chunk is submitted as a separate batch job
- Results are merged into a single CSV per experiment

## Implementation

### Core Scripts

#### 1. `batch_compare.py` - Main Batch Processing Script

**Purpose**: Run all 4 experiments using OpenAI batch API for cost efficiency.

**Key Features**:
- Samples `n` rows once and reuses across all experiments
- Creates batch jobs with 2 prompts per row (better_fund + next_month)
- Handles chunking automatically (2 chunks per experiment)
- Includes queue awareness to avoid token limit errors
- Supports resume functionality with `--skip` flag

**Usage**:
```bash
# Basic usage
python batch_compare.py --n 1000 --seed 42

# Resume from existing CSVs
python batch_compare.py --skip --output-dir batch_experiments

# Custom chunk size
python batch_compare.py --n 1000 --chunk-size 200

# Wait for batch capacity
python batch_compare.py --n 1000 --wait-capacity
```

**Arguments**:
- `--csv`: Path to input CSV (default: `input_csvs/mutual_funds_pairs_no_date.csv`)
- `--n`: Number of pairs to sample (default: 10)
- `--seed`: Random seed for sampling (default: 42)
- `--system-prompt`: Path to system prompt (default: `prompts/system_prompt.txt`)
- `--user-prompt-better-zero`: Zero-shot better fund template
- `--user-prompt-next-month-zero`: Zero-shot next month template
- `--user-prompt-better-single`: Single-shot better fund template
- `--user-prompt-next-month-single`: Single-shot next month template
- `--output-dir`: Output directory (default: `batch_experiments`)
- `--chunk-size`: Rows per chunk (default: None, auto-calculates for 2 chunks)
- `--wait-capacity`: Wait if too many batches are in progress
- `--model`: Model name (default: `gpt-5.2`)
- `--skip`: Resume from existing CSV indices instead of random sampling

**Output Files** (per experiment):
- `{exp_id}_chunk_{n}_input.jsonl`: Batch input file
- `{exp_id}_chunk_{n}_output.jsonl`: Batch output file
- `{exp_id}.csv`: Final merged results

**CSV Output Format**:
- `index`: Original row index from dataset
- `name_1`, `name_2`: Fund names
- `better_fund`: Model's answer to "which fund is better"
- `next_month`: Model's answer to "which fund will have higher next-month return"
- `lowest_rank`: Ground truth (fund with lower 3-year Rank)

#### 2. `direct_compare.py` - Direct API Testing Script

**Purpose**: Quick testing with direct API calls (not batch processing).

**Key Features**:
- Uses first `n` rows (no random sampling)
- Makes direct API calls (faster for small tests)
- Stores results in CSV format

**Usage**:
```bash
python direct_compare.py --n 10
```

**Output**: `direct_output.csv` with same format as batch outputs

#### 3. `check_batches.py` - Batch Status Monitor

**Purpose**: Check status of in-progress batch jobs.

**Usage**:
```bash
python check_batches.py --limit 50
```

**Output**: Lists all active batches with:
- Batch ID and status
- Progress (completed/total/failed)
- Elapsed time
- Errors (if any)

#### 4. `eval_scripts/evaluate_batch_outputs.py` - Results Evaluator

**Purpose**: Evaluate experiment results against ground truth.

**Usage**:
```bash
python eval_scripts/evaluate_batch_outputs.py --dir batch_experiments --output batch_experiments/eval_summary.csv
```

**Output**:
- Prints accuracy metrics to console
- Saves summary CSV with:
  - `file`: Experiment CSV filename
  - `rows`: Number of rows evaluated
  - `better_match_pct`: % where `better_fund` matches `lowest_rank`
  - `next_month_match_pct`: % where `next_month` matches `lowest_rank`

### File Structure

```
RA Mutual Fund Task/
├── README.md                          # This file
├── batch_compare.py                   # Main batch processing script
├── direct_compare.py                  # Direct API testing script
├── check_batches.py                   # Batch status monitor
├── input_csvs/                        # Dataset files
│   ├── mutual_funds_pairs_no_date.csv
│   ├── mutual_funds_pairs_no_date_old.csv
│   └── mutual_funds_pairs.csv
├── prompts/                           # Prompt templates
│   ├── system_prompt.txt
│   ├── zero_shot_prompt_template.txt
│   ├── zero_shot_next_month_prompt_template.txt
│   ├── single_shot_prompt_template.txt
│   └── single_shot_next_month_prompt_template.txt
├── batch_experiments/                 # Experiment outputs
│   ├── {exp_id}.csv                  # Final results per experiment
│   ├── {exp_id}_chunk_{n}_input.jsonl
│   ├── {exp_id}_chunk_{n}_output.jsonl
│   ├── eval_summary.csv              # Evaluation summary
│   ├── evaluation_report.md          # Detailed evaluation report
│   └── confusion_matrices/           # Confusion matrix analysis
│       ├── {exp_id}_confusion_matrix.csv
│       └── results_summary.md
└── eval_scripts/                      # Evaluation utilities
    ├── evaluate_batch_outputs.py       # Main evaluation script
    ├── evaluate_all_batch_experiments.py
    ├── evaluate_medalist.py
    └── analyze_guess_results.py
```

**Note**: Additional files in the project root (not shown above) include:
- `Help Me Read This Table.pdf` - Feature definitions reference
- `Mutual Fund Evaluation Description.pdf` - Project description
- `LLM_MF_Evalutaion.ipynb` - Jupyter notebook for analysis
- `direct_output.csv` / `direct_output_results.md` - Direct API test results
- `guess_fund_name.py` / `guess_fund_name_output.csv` - Additional experiments
- `create_results_summary.py` - Summary generation utility
- `batch_experiments_old/` - Archive of old experiment results
- `batch_experiments_test/` - Test experiment outputs
- `llm_eval_housing/` - Related evaluation project

## Setup

### Prerequisites

```bash
pip install openai pandas python-dotenv
```

### Environment Variables

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

### Model Requirements

- Model: `gpt-5.2` (or compatible OpenAI model)
- API: OpenAI Batch API (`/v1/responses` endpoint)

## Running Experiments

### Step 1: Run Batch Experiments

```bash
python batch_compare.py --n 1000 --seed 42
```

This will:
1. Sample 1000 rows from the dataset
2. Run all 4 experiments
3. Create 2 chunks per experiment (500 rows each)
4. Submit batch jobs to OpenAI
5. Poll for completion
6. Download and merge results
7. Generate CSV files

### Step 2: Evaluate Results

```bash
python eval_scripts/evaluate_batch_outputs.py --dir batch_experiments
```

This generates:
- Console output with accuracy metrics
- `batch_experiments/eval_summary.csv` with summary statistics

### Step 3: Review Report

Check `batch_experiments/evaluation_report.md` for detailed analysis.

## Resuming Failed Experiments

If an experiment fails partway through:

1. **Check batch status**:
   ```bash
   python check_batches.py
   ```

2. **Resume from existing CSVs**:
   ```bash
   python batch_compare.py --skip --output-dir batch_experiments
   ```
   
   This will:
   - Read indices from existing CSV files
   - Filter dataset to only those indices
   - Re-run all experiments with the same data

## Output Format

### Experiment CSV (`{exp_id}.csv`)

| Column | Description |
|--------|-------------|
| `index` | Original row index from dataset |
| `name_1` | Name of first mutual fund |
| `name_2` | Name of second mutual fund |
| `better_fund` | Model prediction: "mutual fund 1" or "mutual fund 2" |
| `next_month` | Model prediction: "mutual fund 1" or "mutual fund 2" |
| `lowest_rank` | Ground truth: "mutual fund 1" or "mutual fund 2" |

### Evaluation Summary (`eval_summary.csv`)

| Column | Description |
|--------|-------------|
| `file` | Experiment CSV filename |
| `rows` | Number of rows evaluated |
| `better_match_pct` | Accuracy for better_fund predictions |
| `next_month_match_pct` | Accuracy for next_month predictions |

## Key Implementation Details

### Prompt Construction

1. **System Prompt**: Loaded once, used for all requests
2. **User Prompt**: Built per-row using template formatting
3. **Two Requests Per Row**: One for "better fund", one for "next month"
4. **Custom IDs**: Format `{exp_id}|{prompt_type}|{index}` for tracking

### Batch Request Structure

```json
{
  "custom_id": "zero_shot_temp0|better|123",
  "method": "POST",
  "url": "/v1/responses",
  "body": {
    "model": "gpt-5.2",
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "temperature": 0.0,
    "max_output_tokens": 16
  }
}
```

### Chunking Logic

- Default: 2 chunks per experiment
- Calculation: `chunk_size = (n + 1) // 2`
- Each chunk is a separate batch job
- Results are merged by parsing `custom_id` fields

### Error Handling

- **Token Limit Errors**: Script detects and reports, suggests reducing `n` or `chunk-size`
- **Timeout Errors**: Retries with exponential backoff
- **Failed Batches**: Error details saved to `{exp_id}_batch_info.json`

### Queue Management

- `--wait-capacity` flag checks for in-progress batches before submitting
- Prevents hitting organization token limits
- Uses `check_batches.py` logic internally

## Evaluation Methodology

### Metrics

1. **Better Fund Match %**: Percentage where `better_fund == lowest_rank`
2. **Next Month Match %**: Percentage where `next_month == lowest_rank`

### Ground Truth

- Based on **3-year Rank** comparison
- Lower rank = better historical performance
- Assumes historical performance is predictive

### Limitations

- Ground truth is based on historical performance, not future returns
- "Next month" predictions are evaluated against historical rank (may not reflect actual future returns)
- Model outputs are constrained to "mutual fund 1" or "mutual fund 2" (no confidence scores)

## Extending the Project

### Adding New Experiments

Edit `batch_compare.py`, add to `experiments` list:
```python
{
    "id": "new_experiment_id",
    "prompt_type": "zero",  # or "single"
    "hyperparams": {"temperature": 0.5},  # or {"top_p": 0.9}
}
```

### Adding New Features

1. Update CSV reading to include new columns
2. Update `build_user_prompt()` to format new features
3. Update prompt templates to include feature descriptions
4. Update system prompt if needed

### Adding New Prompt Types

1. Create new template files in `prompts/`
2. Add prompt loading logic in `batch_compare.py`
3. Add new experiment configuration

### Custom Evaluation Metrics

Modify `eval_scripts/evaluate_batch_outputs.py` to:
- Add new comparison logic
- Calculate additional statistics
- Generate custom reports

## Troubleshooting

### Batch Jobs Failing

**Error**: `token_limit_exceeded`
- **Solution**: Reduce `--n` or `--chunk-size`, or use `--wait-capacity`

**Error**: `Request timed out`
- **Solution**: Script includes retries; check network connection

### Missing Output Files

- Check `batch_experiments/{exp_id}_batch_info.json` for error details
- Verify batch completed: `python check_batches.py`
- Check for `{exp_id}_errors.jsonl` files

### Resume Not Working

- Ensure CSV files exist in output directory
- Verify CSV files contain `index` column
- Check that indices exist in the original dataset

## Results Interpretation

### Typical Findings

1. **Zero-shot vs Single-shot**: Zero-shot often performs better (allows model flexibility)
2. **Next Month vs Better Fund**: Next month predictions typically more accurate (simpler task)
3. **Hyperparameters**: Minimal impact (model is consistent)

### Best Practices

- Use zero-shot prompting for better performance
- Temperature=0.0 for deterministic results
- Evaluate both metrics separately (they measure different things)

## References

- **Feature Definitions**: See `Help Me Read This Table.pdf`
- **OpenAI Batch API**: https://platform.openai.com/docs/guides/batch
- **Model Documentation**: https://platform.openai.com/docs/models

## License

[Add license information if applicable]

## Contact

[Add contact information if applicable]
