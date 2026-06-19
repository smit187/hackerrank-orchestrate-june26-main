# Submission Manifest

This project follows the source contract from `problem_statement.md`,
`README.md`, and `AGENTS.md`.

## Files To Submit

- `code.zip`: runnable code, prompt files, README, and evaluation assets.
- `output.csv`: predictions for every row in `dataset/claims.csv`.
- `chat_transcript`: `C:\Users\DTAdmin\hackerrank_orchestrate\log.txt`.

## Runnable Entry Points

- Main pipeline: `python code/main.py --input dataset/claims.csv --output output.csv`
- Evaluation: `python code/evaluation/main.py`
- Full workflow: `python code/run_all.py --make-zip`

## Included Code Files

- `code/main.py`: context hydration, optional async VLM processing, strict
  schema guardrail, prompt sanitization, confidence scoring, and output writer.
- `code/prompts.py`: VLM prompt template and allowed taxonomy.
- `code/run_all.py`: one-command evaluation, prediction, validation, and
  optional `code.zip` creation.
- `code/evaluation/main.py`: sample-set evaluator.
- `code/evaluation/evaluation_report.md`: operational report included inside
  the code package.
- `code/README.md`: usage instructions.

## Real-World Fit

The solution maps directly to an insurance, warranty, logistics, or device
protection review queue. It can triage image-backed claims, add user-history
risk context, reduce duplicate image review with hashing, and safely fall back
to manual review when evidence is unclear.
