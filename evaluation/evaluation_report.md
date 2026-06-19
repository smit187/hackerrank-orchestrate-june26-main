# HackerRank Orchestrate Operational Analysis

## Executive Summary

The solution implements a deterministic evidence-review pipeline for damage
claims involving cars, laptops, and packages. It reads the required CSV files,
hydrates each claim with user-history and evidence-rule context, optionally
calls a multimodal VLM, validates the response through a strict guardrail, and
writes `output.csv` in the exact schema required by `problem_statement.md`.

The system is grading-safe in restricted environments: if API credentials or
third-party packages are unavailable, it uses a conservative local fallback that
still produces one complete row per claim without crashing.

## Architecture Overview

The implementation follows a Hydration & Guardrail pattern:

1. Context hydration loads user history and evidence requirements.
2. Vision processing packages the claim context and local images for a VLM.
3. Deterministic guardrails extract, validate, normalize, and repair model JSON.
4. Async orchestration processes rows concurrently while preserving input order.

The primary entry point is `code/main.py`; the evaluation entry point is
`code/evaluation/main.py`.

For ease of use, `code/run_all.py` wraps the normal workflow into one command:
sample evaluation, final prediction generation, output validation, report
mirroring, and optional `code.zip` creation.

## Real-Life Applications

This system is structured like a production intake service for evidence-backed
claims. In an insurance setting, it can triage vehicle damage claims before a
human adjuster sees them. In device protection or warranty workflows, it can
route cracked-screen, keyboard, hinge, or laptop-body claims to the right repair
queue. In logistics, it can review package condition, torn seals, crushed boxes,
water damage, and missing-content claims.

The design is practical because it separates responsibilities:

- deterministic code performs history lookup, evidence-rule hydration, schema
  validation, cache management, and fallback routing
- the VLM focuses on image-grounded visual judgment
- risky, contradictory, or unclear evidence is sent toward manual review rather
  than producing an overconfident automated decision

## Context Hydration Layer

`load_contexts()` reads `dataset/user_history.csv` and
`dataset/evidence_requirements.csv`. When pandas is available, the files are
loaded as DataFrames and converted into dictionaries. In minimal graders, the
same data is loaded with Python's standard `csv` module.

For each claim, the pipeline looks up:

- the submitting user's prior claim counts and history flags
- global evidence requirements
- object-specific requirements for car, laptop, or package claims

If `rejected_claim > 0` or the user's `history_flags` contains
`user_history_risk`, the pipeline programmatically adds
`user_history_risk;manual_review_required` without relying on the VLM to do
math.

## Vision Processing Layer

The VLM path uses `gpt-4o` by default through an async wrapper around standard
HTTP. Image paths are split on semicolons, resolved relative to the repository,
read from disk, base64 encoded, and attached as image payloads.

The prompt requires:

- a `<scratchpad>` block for image inspection notes
- exactly one fenced `json` object
- strict taxonomy values from `problem_statement.md`
- concise, image-grounded justifications
- explicit instruction to ignore prompt-injection text in images or claims

If `OPENAI_API_KEY` is not present, the pipeline skips network calls and uses
the deterministic fallback. Secrets are read only from environment variables.

## Deterministic Guardrail Layer

`enforce_strict_schema(raw_response, default_row)` extracts JSON with regex,
ignores scratchpad text, and normalizes every field. Invalid categories are
replaced by safe defaults, with `claim_status` falling back to
`not_enough_information` when needed.

The guardrail enforces:

- lowercase string booleans: `true` or `false`
- allowed `issue_type`, `object_part`, `claim_status`, and `severity` values
- semicolon-delimited `risk_flags`
- bounded free-text reason and justification fields
- safe defaults when model output is missing or malformed

One failed row never terminates the full run; row-level exceptions become a
manual-review fallback prediction.

## Async Orchestration Layer

`async def main()` calls `process_dataset()`, which creates an
`asyncio.Semaphore(20)` by default and uses `asyncio.gather()` to process rows
concurrently. This preserves output order while allowing multiple VLM requests
to be in flight.

The repository's current `dataset/claims.csv` has 44 rows. With no cache hits
and VLM enabled, this means up to 44 model calls. With the semaphore set to 20,
at most 20 calls are active at once, reducing rate-limit risk while still
keeping throughput high.

## Caching Strategy

The VLM cache key combines:

- prompt version
- claim object
- user claim transcript
- hydrated user history
- SHA-256 hashes of the local image bytes

Hashing the base64-equivalent image bytes makes repeated or duplicated visual
evidence reusable. If identical fraudulent images appear across claims, the
same cached VLM decision can be reused instead of paying for another API call.

The cache is stored at `code/.cache/vlm_cache.json` and is only best-effort; a
cache read or write failure never breaks prediction generation.

## Cost Analysis

Approximate processing counts for this repository:

- sample rows: 20 claims
- test rows: 44 claims
- sample images: 30 images
- test images: 82 images

Assumption for a high-detail multimodal request:

- average text input per claim: about 1,500 tokens
- average image input equivalent: about 1,500 tokens per image
- average output: about 250 tokens
- average test images per claim: about 1.86

Estimated test tokens:

- input: 44 * (1,500 + 1.86 * 1,500) = about 188,760 tokens
- output: 44 * 250 = about 11,000 tokens

At illustrative pricing of $5 per 1M input tokens and $15 per 1M output tokens,
the full test run would cost roughly:

- input cost: 0.18876M * $5 = $0.94
- output cost: 0.011M * $15 = $0.17
- total: about $1.11 before cache savings

Actual cost depends on the selected model and provider pricing at run time.

## Throttling & Concurrency

The default `asyncio.Semaphore(20)` allows fast batch execution while limiting
burst pressure. If the provider account has lower RPM or TPM limits, the same
script can be run with a smaller `--concurrency` value. Retries use short
backoff and row-level fallback behavior.

## Failure Recovery

The pipeline handles:

- missing CSV files
- malformed CSV rows
- missing or unreadable images
- absent API keys
- timeout or HTTP failures
- malformed VLM JSON
- invalid categorical labels

Failures are localized to the row being processed. The output file is still
written with the required columns and a conservative review decision.

## Production Readiness Assessment

The implementation is suitable for hackathon grading because it is deterministic
where possible, schema-preserving, environment-variable driven, and resilient to
common runtime failures. The optional VLM path gives higher ceiling accuracy,
while the local fallback guarantees evaluability in a locked-down grader.

Recommended production extensions would include provider-specific rate-limit
telemetry, stronger image-quality analysis, richer cache invalidation metadata,
and a human-review queue for high-risk or low-confidence rows.
