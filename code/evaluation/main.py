"""Evaluation harness for the HackerRank Orchestrate evidence-review agent.

The evaluator is intentionally dependency-light: it can run in a bare Python
environment, yet it imports the production pipeline when no predictions file is
provided. It scores the categorical fields emphasized by the challenge and
prints per-label summaries that are easy to inspect during prompt iteration.

The production pipeline includes claim text sanitization, prompt-injection
protections, image-quality heuristics, confidence scoring, and conservative
manual-review escalation for ambiguous cases.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


TARGET_COLUMNS = ("issue_type", "object_part", "claim_status")

ALLOWED_VALUES: Mapping[str, set[str]] = {
    "claim_status": {"supported", "contradicted", "not_enough_information"},
    "issue_type": {
        "dent",
        "scratch",
        "crack",
        "glass_shatter",
        "broken_part",
        "missing_part",
        "torn_packaging",
        "crushed_packaging",
        "water_damage",
        "stain",
        "none",
        "unknown",
    },
    "object_part": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into dictionaries, returning an empty list for bad input."""
    if not path.exists():
        print(f"WARNING: CSV not found: {path}", file=sys.stderr)
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except csv.Error as exc:
        print(f"WARNING: malformed CSV {path}: {exc}", file=sys.stderr)
        return []
    except OSError as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return []


def align_predictions(
    expected_rows: Sequence[Mapping[str, str]],
    prediction_rows: Sequence[Mapping[str, str]],
) -> List[Mapping[str, str]]:
    """Align predictions to expected rows and pad missing outputs safely."""
    aligned: List[Mapping[str, str]] = []
    for index, _expected in enumerate(expected_rows):
        if index < len(prediction_rows):
            aligned.append(prediction_rows[index])
        else:
            aligned.append({})
    return aligned


def normalize_label(value: object, column: str) -> str:
    """Normalize labels and mark invalid or missing values explicitly."""
    text = str(value or "").strip().lower()
    if not text:
        return "__missing__"
    if column in ALLOWED_VALUES and text not in ALLOWED_VALUES[column]:
        return "__invalid__"
    return text


def confusion_matrix(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> Dict[str, Counter[str]]:
    """Return expected-label to predicted-label counts."""
    matrix: Dict[str, Counter[str]] = defaultdict(Counter)
    for gold, pred in zip(expected, predicted):
        matrix[gold][pred] += 1
    return matrix


def f1_for_label(expected: Sequence[str], predicted: Sequence[str], label: str) -> float:
    """Compute one-vs-rest F1 for a single label."""
    tp = sum(1 for gold, pred in zip(expected, predicted) if gold == label and pred == label)
    fp = sum(1 for gold, pred in zip(expected, predicted) if gold != label and pred == label)
    fn = sum(1 for gold, pred in zip(expected, predicted) if gold == label and pred != label)
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(expected: Sequence[str], predicted: Sequence[str]) -> float:
    """Compute macro F1 across labels present in either expected or predicted."""
    labels = sorted(set(expected) | set(predicted))
    if not labels:
        return 0.0
    return sum(f1_for_label(expected, predicted, label) for label in labels) / len(labels)


def exact_match_accuracy(
    expected_rows: Sequence[Mapping[str, str]],
    prediction_rows: Sequence[Mapping[str, str]],
    columns: Sequence[str] = TARGET_COLUMNS,
) -> float:
    """Compute exact-match accuracy across the selected target columns."""
    if not expected_rows:
        return 0.0
    matches = 0
    for expected, predicted in zip(expected_rows, prediction_rows):
        if all(
            normalize_label(expected.get(column), column)
            == normalize_label(predicted.get(column), column)
            for column in columns
        ):
            matches += 1
    return matches / len(expected_rows)


def per_label_summary(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> List[Tuple[str, int, int, int, float]]:
    """Return support, predicted count, correct count, and F1 by label."""
    labels = sorted(set(expected) | set(predicted))
    rows: List[Tuple[str, int, int, int, float]] = []
    for label in labels:
        support = sum(1 for value in expected if value == label)
        predicted_count = sum(1 for value in predicted if value == label)
        correct = sum(1 for gold, pred in zip(expected, predicted) if gold == label and pred == label)
        rows.append((label, support, predicted_count, correct, f1_for_label(expected, predicted, label)))
    return rows


def print_column_report(column: str, expected_rows: Sequence[Mapping[str, str]], predicted_rows: Sequence[Mapping[str, str]]) -> None:
    """Print F1, per-label summary, and a compact confusion matrix."""
    expected = [normalize_label(row.get(column), column) for row in expected_rows]
    predicted = [normalize_label(row.get(column), column) for row in predicted_rows]
    print(f"\n{column}")
    print("-" * len(column))
    print(f"Macro F1: {macro_f1(expected, predicted):.4f}")
    print("Per-label summary:")
    print("  label | support | predicted | correct | f1")
    for label, support, predicted_count, correct, label_f1 in per_label_summary(expected, predicted):
        print(f"  {label} | {support} | {predicted_count} | {correct} | {label_f1:.4f}")
    print("Confusion matrix (expected -> predicted counts):")
    for gold, predictions in sorted(confusion_matrix(expected, predicted).items()):
        rendered = ", ".join(f"{pred}:{count}" for pred, count in sorted(predictions.items()))
        print(f"  {gold} -> {rendered}")


async def generate_sample_predictions(sample_path: Path) -> List[Dict[str, str]]:
    """Run the production pipeline on sample rows without requiring VLM access."""
    try:
        from main import process_dataset
    except ImportError as exc:
        print(f"WARNING: could not import production pipeline: {exc}", file=sys.stderr)
        return []
    return await process_dataset(input_csv=sample_path, output_csv=None, use_vlm="never")


async def evaluate(sample_path: Path, predictions_path: Path | None) -> int:
    """Evaluate a predictions file, or generate predictions from the pipeline."""
    expected_rows = read_csv_rows(sample_path)
    if predictions_path:
        prediction_rows = read_csv_rows(predictions_path)
    else:
        prediction_rows = await generate_sample_predictions(sample_path)

    aligned_predictions = align_predictions(expected_rows, prediction_rows)
    missing = max(0, len(expected_rows) - len(prediction_rows))
    extra = max(0, len(prediction_rows) - len(expected_rows))

    print("HackerRank Orchestrate Evaluation Report")
    print("=======================================")
    print(f"Expected rows: {len(expected_rows)}")
    print(f"Prediction rows: {len(prediction_rows)}")
    print(f"Missing prediction rows: {missing}")
    print(f"Extra prediction rows ignored: {extra}")
    print(f"Exact Match Accuracy ({', '.join(TARGET_COLUMNS)}): {exact_match_accuracy(expected_rows, aligned_predictions):.4f}")

    for column in TARGET_COLUMNS:
        print_column_report(column, expected_rows, aligned_predictions)

    invalid_counts: MutableMapping[str, int] = Counter()
    for row in aligned_predictions:
        for column in TARGET_COLUMNS:
            if normalize_label(row.get(column), column) in {"__missing__", "__invalid__"}:
                invalid_counts[column] += 1
    if invalid_counts:
        print("\nInvalid or missing prediction counts:")
        for column, count in sorted(invalid_counts.items()):
            print(f"  {column}: {count}")
    return 0 if expected_rows else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate evidence-review predictions on sample_claims.csv.")
    parser.add_argument(
        "--sample",
        type=Path,
        default=REPO_ROOT / "dataset" / "sample_claims.csv",
        help="Path to labeled sample_claims.csv.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional predictions CSV. If omitted, code/main.py is run on the sample data.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    return asyncio.run(evaluate(args.sample, args.predictions))


if __name__ == "__main__":
    raise SystemExit(main())
