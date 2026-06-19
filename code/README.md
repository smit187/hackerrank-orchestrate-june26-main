# HackerRank Orchestrate Solution

This folder contains the runnable evidence-review agent, evaluation harness,
prompt assets, and a one-command workflow for creating submission artifacts.

## Fastest Path

Run the full local workflow:

```bash
python code/run_all.py --make-zip
```

That command:

1. evaluates the pipeline on `dataset/sample_claims.csv`
2. generates `output.csv` from `dataset/claims.csv`
3. validates row count, schema order, and required cells
4. mirrors the operational report into `code/evaluation/`
5. creates `code.zip`

## Run Predictions

```bash
python code/main.py --input dataset/claims.csv --output output.csv
```

By default, the pipeline uses a VLM only when `OPENAI_API_KEY` is present. To
force the deterministic local fallback:

```bash
python code/main.py --use-vlm never
```

## Evaluate on Sample Data

```bash
python code/evaluation/main.py
```

To evaluate an existing predictions file:

```bash
python code/evaluation/main.py --predictions path/to/predictions.csv
```

## Architecture

- `main.py` hydrates CSV context, optionally calls a vision model, applies strict
  guardrails, and writes `output.csv` with the exact required schema.
- `prompts.py` stores the deterministic multimodal prompt template and taxonomy.
- `run_all.py` gives a single repeatable workflow for evaluation, prediction,
  output validation, and zip packaging.
- `evaluation/main.py` computes Macro F1 for `issue_type`, `object_part`, and
  `claim_status`, plus exact-match accuracy and per-label confusion summaries.

Secrets are read from environment variables only. No API keys or tokens should
be committed.

## Real-Life Applications

This architecture is directly useful outside the hackathon for:

- insurance vehicle-damage triage
- laptop warranty intake and repair prioritization
- logistics package-damage review
- marketplace seller/buyer dispute evidence checks
- manual-review queue routing for risky or unclear claims

The same pattern works in production because the model does only visual
judgment. Deterministic Python code handles user-history risk, evidence-rule
hydration, schema validation, retry behavior, caching, and safe fallbacks.

## Submission Checklist

- `output.csv` exists at the repository root.
- `output.csv` has one row per `dataset/claims.csv` row.
- `output.csv` columns match `problem_statement.md` exactly.
- `code.zip` includes `code/` and `evaluation/` assets.
- Chat transcript comes from `C:\Users\DTAdmin\hackerrank_orchestrate\log.txt`.
