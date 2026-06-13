# SQLJudge: Benchmarking and Learning Generative Reward Models for Execution-Free Text-to-SQL Candidate Selection

SQLJudge-Bench evaluates the **judgment quality of Text2SQL reward models**. Given a user question, a database schema, and two candidate responses (Response A / Response B), the model under evaluation acts as a *judge* and decides which response is better. The framework quantifies a model's judging ability along three dimensions:

| Metric | Full name | Meaning | Direction |
|--------|-----------|---------|-----------|
| **SA**  | Strict Accuracy | Under position-symmetric evaluation, whether the model can reliably judge the *better* response as better | Higher is better |
| **PI**  | Preference Instability | The fraction of cases where swapping the A/B positions flips the verdict or violates antisymmetry (position bias) | Lower is better |
| **GLC** | Global Logical Consistency | The degree to which judgments over a multi-candidate chain violate the *weak total order* axiom (transitivity violation, TOV) | Lower is better |

The benchmark covers two scenarios:

- **SQL preference** (`pair_sql.jsonl`): judge which of two SQL queries better matches the question's intent.
- **Refusal preference** (`pair_reject.jsonl`): when a question is unanswerable or the schema is insufficient, judge whether a *reasonable refusal* or a *forced hallucinated answer* is better.

---

## Repository Structure

```
sqljudge-bench/
├── evaluate.py          # Main script: queries the judge model pair-by-pair and computes SA / PI / GLC
├── post_process.py      # Post-processing: re-runs records that failed due to API errors
├── requirements.txt     # Python dependencies
├── .env.example         # API configuration (API_KEY / BASE_URL) — fill in yourself
├── eval_data/           # Evaluation datasets
│   ├── sample_pair_sql.jsonl      # Sample SQL preference pairs (99 pairs)
│   └── sample_pair_reject.jsonl   # Sample refusal preference pairs (99 pairs)
└── eval_results/        # Output directory (create it yourself, see below)
```

> **Note on data file names.** `evaluate.py` expects the data files to be named `pair_sql.jsonl` and `pair_reject.jsonl`. This repository sample subsets named `sample_pair_sql.jsonl` and `sample_pair_reject.jsonl`. 

---

## Data Format

Data is stored as JSONL (one JSON object per line). All records share a common set of top-level fields; the difference lies in the `pair` sub-object.

### Top-level fields (shared by both files)

| Field | Description |
|-------|-------------|
| `source` | Data source |
| `db_id` | Database identifier |
| `question` | User's natural-language question |
| `evidence` | External knowledge / hints related to the question |
| `gold_sql` | Reference (gold) SQL |
| `database_schema` | Database schema text |
| `scenario_id` | Scenario identifier (used for per-scenario metric breakdowns) |
| `_sql_features` | SQL feature tags (list) |
| `_complexity` | Complexity tag |
| `_intent` | Question-intent tag |
| `pair` | The preference-pair content (see below) |

### `pair` field in `pair_sql.jsonl`

`pair.type` takes one of three values, each with its own fields:

- **`pos_neg_pref`** / **`homogenization`**: contain `positive_sql` (the better response) and `negative_sql` (the worse response).
- **`transitivity_chain`**: contains multiple candidate SQLs (`sql_e` / `sql_s` / `sql_m` / `sql_x`, etc.) plus a `pairs` list (each element has `pair_id` / `sql_a` / `sql_b` / `expected_preference`), used to compute GLC.

### `pair` field in `pair_reject.jsonl`

| Field | Description |
|-------|-------------|
| `type` | Refusal scenario type (`truthfulness_refusal` / `hallucination` / `quality_alignment`) |
| `reject_schema` | Schema used for this scenario (falls back to the top-level `database_schema` when absent) |
| `positive_rej` | The better response (typically a reasonable refusal or a correct answer) |
| `negative_rej` | The worse response |
| `positive_type` / `negative_type` | Type tags for each of the two responses |
| `recall_rate` / `precision_rate` | Recall / precision metrics relevant to the scenario |
| `missing_info` / `failure_type` | Auxiliary fields describing how the scenario was constructed |

---

## Quick Start

### 1. Install dependencies

Python 3.10+ is recommended (the code uses built-in generic syntax such as `tuple[str, str]` and `list[dict]`).

```bash
pip install -r requirements.txt
```

Dependencies: `openai`, `python-dotenv`, `pandas`, `openpyxl`, `tqdm`.

### 2. Configure the API

Edit the `.env` file in the project root and fill in the OpenAI-compatible endpoint of the model under evaluation (the judge):

```dotenv
API_KEY=sk-your-api-key
BASE_URL=https://your-openai-compatible-endpoint/v1
```

- `API_KEY`: required.
- `BASE_URL`: optional. Leave it empty to use the official OpenAI API; set it to point at a self-hosted or third-party OpenAI-compatible service (e.g. vLLM, a local inference gateway).

### 3. Run the evaluation

```bash
# Create the output directory first (the script does not create it for you)
mkdir eval_results

python evaluate.py --model <model-name> --data_dir ./eval_data --output_dir ./eval_results
```

`evaluate.py` command-line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | Name of the model under evaluation; passed as the `model` field to the API and used as the output filename prefix |
| `--data_dir` | `./eval_data` | Directory containing the datasets |
| `--output_dir` | `./eval_results` | Directory for results (**must be created beforehand**) |
| `--batch_size` | `8` | Number of concurrent requests per batch |
| `--batch_delay` | `2.0` | Sleep seconds between batches (to avoid rate limiting) |

Example:

```bash
python evaluate.py --model gpt-4o-mini --batch_size 16 --batch_delay 1.0
```

> **Resumable runs.** The script reads any existing result files and automatically skips records that are already done. After an interruption, simply re-run the same command to resume from where it stopped — there is no need to delete existing results.

### 4. Inspect the results

When evaluation finishes, the following files are written to `--output_dir` (where `<model>` is the value of `--model`):

| File | Contents |
|------|----------|
| `<model>_sql_results.jsonl` | Raw judge output and parsed verdicts (forward + backward) for each SQL preference pair |
| `<model>_reject_results.jsonl` | Forward + backward judge results for each refusal preference pair |
| `<model>_metrics.xlsx` | Metric summary: `Summary` sheet (overall SA / PI / GLC) + `By_Scenario` sheet (per-scenario breakdown) |

---

## How It Works

### Position-symmetric evaluation (for SA and PI)

For each preference pair, the framework makes two judge calls:

1. **Forward**: place the better response at position A and the worse one at position B → the model is expected to choose **A**.
2. **Backward**: swap them — worse at A, better at B → the model is expected to choose **B**.

From these, the metrics are defined as:

- **Strict Accuracy (SA)**: `forward == A` **and** `backward == B`, i.e. the model judges correctly regardless of position.
- **Preference Instability (PI)**: the verdict violates antisymmetry after the swap. The three *stable* outcomes are `(A, B)`, `(B, A)`, and `(TIE, TIE)`; anything else counts as unstable.

All calls use `temperature=0`, and verdicts are parsed via the regex `<answer>A</answer>` / `<answer>B</answer>`.

### Transitivity / weak-order violation TOV (for GLC)

For `transitivity_chain` records, there are several pairwise comparisons among the candidate SQLs. The framework enumerates all *weak total orders* of the candidate set (ordered arrangements that allow ties), finds the ordering that *least conflicts* with the model's actual judgments, and the number of conflicts is that chain's TOV. GLC is the average TOV across all chains — the smaller the value, the more transitive and self-consistent the model's judgments are.

---

## Error Reprocessing (optional)

If some records fail during evaluation due to API timeouts or rate limiting (their `*_parsed` fields are `ERROR`), you can use `post_process.py` to re-run only the failed items instead of re-evaluating everything:

```bash
# Re-run failed items in the SQL results
python post_process.py \
  --file ./eval_results/<model>_sql_results.jsonl \
  --type sql \
  --eval_data ./eval_data/pair_sql.jsonl \
  --model <model-name>

# Re-run failed items in the refusal results
python post_process.py \
  --file ./eval_results/<model>_reject_results.jsonl \
  --type reject \
  --eval_data ./eval_data/pair_reject.jsonl \
  --model <model-name>
```

Arguments: `--file` (the result file to fix), `--type` (`sql` or `reject`), `--eval_data` (the corresponding original dataset, used to restore schema/question/evidence context), `--model`, plus optional `--batch_size` / `--batch_delay`. The script updates the result file in place and reports whether any errors remain afterward (you can run it repeatedly until none are left). Once fixed, re-run `evaluate.py` (resumable) to recompute the metrics.

---

## Notes

- **The output directory must be created manually.** `evaluate.py` does not create `--output_dir`; run `mkdir` first, otherwise writing results will fail.
- **The model under evaluation is the judge.** This framework evaluates a model's ability to *act as a judge*; `--model` points at the model being tested.
- **OpenAI-compatible endpoints.** The implementation uses the `openai` Python SDK, so any service compatible with the OpenAI Chat Completions API can be plugged in via `BASE_URL`.
- **Concurrency and rate limiting.** If you encounter many `ERROR`s, lower `--batch_size`, raise `--batch_delay`, and then use `post_process.py` to backfill the failed items.
