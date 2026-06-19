"""Production pipeline for HackerRank Orchestrate multimodal evidence review.

The pipeline uses context hydration, an optional async VLM call, and a strict
deterministic guardrail. It is designed to run even in restricted graders where
third-party packages and API keys are unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from prompts import PROMPT_VERSION, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

MODEL_COLUMNS = OUTPUT_COLUMNS[4:]

ALLOWED_ISSUE_TYPES = {
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
}
ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}
ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}
ALLOWED_RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}
OBJECT_PARTS = {
    "car": {
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
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
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
ALL_OBJECT_PARTS = set().union(*OBJECT_PARTS.values())


@dataclass(frozen=True)
class Contexts:
    """Hydrated lookup data used while processing claims."""

    repo_root: Path
    user_history: Mapping[str, Mapping[str, str]]
    evidence_requirements: Mapping[str, List[Mapping[str, str]]]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read CSV rows defensively."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (csv.Error, OSError):
        return []


def load_contexts(repo_root: Path | None = None) -> Contexts:
    """Load user history and evidence requirements into fast lookup maps.

    If pandas is installed, the CSVs are read as DataFrames first, then converted
    to dictionaries. A standard-library fallback is used in minimal graders.
    """
    root = repo_root or Path(__file__).resolve().parents[1]
    user_path = root / "dataset" / "user_history.csv"
    req_path = root / "dataset" / "evidence_requirements.csv"

    try:
        import pandas as pd  # type: ignore

        user_rows = pd.read_csv(user_path, dtype=str).fillna("").to_dict(orient="records")
        req_rows = pd.read_csv(req_path, dtype=str).fillna("").to_dict(orient="records")
    except Exception:
        user_rows = read_csv_rows(user_path)
        req_rows = read_csv_rows(req_path)

    users = {str(row.get("user_id", "")).strip(): row for row in user_rows if row.get("user_id")}
    requirements: Dict[str, List[Mapping[str, str]]] = {}
    for row in req_rows:
        key = str(row.get("claim_object", "all")).strip().lower() or "all"
        requirements.setdefault(key, []).append(row)
    return Contexts(repo_root=root, user_history=users, evidence_requirements=requirements)


def split_image_paths(value: str) -> List[str]:
    """Split a semicolon-delimited image path cell."""
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def image_id(path: str) -> str:
    """Return the filename stem used as the image ID."""
    return Path(path).stem or "unknown_image"


def resolve_image_path(repo_root: Path, image_path: str) -> Path:
    """Resolve image paths from CSV rows against the repository root."""
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return repo_root / "dataset" / image_path if image_path.startswith("images/") else repo_root / image_path


def read_image_b64(path: Path) -> str | None:
    """Read and base64-encode an image, returning None on failure."""
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def image_hash(path: Path) -> str:
    """Hash image bytes for cache keys."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def applicable_requirements(contexts: Contexts, claim_object: str) -> List[Mapping[str, str]]:
    """Return global plus object-specific evidence requirements."""
    key = str(claim_object or "").strip().lower()
    return list(contexts.evidence_requirements.get("all", [])) + list(contexts.evidence_requirements.get(key, []))


def safe_int(value: object) -> int:
    """Parse integers from CSV strings without raising."""
    try:
        return int(str(value or "0").strip() or "0")
    except ValueError:
        return 0


def normalize_bool(value: object, default: str = "false") -> str:
    """Normalize booleans to the challenge's lowercase string values."""
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return "true"
    if text in {"false", "0", "no", "n"}:
        return "false"
    return default


def normalize_category(value: object, allowed: set[str], default: str) -> str:
    """Normalize a categorical value against an allowed set."""
    text = str(value or "").strip().lower().replace(" ", "_")
    return text if text in allowed else default


def normalize_risk_flags(value: object, fallback_flags: Iterable[str] = ()) -> str:
    """Normalize semicolon-delimited risk flags."""
    flags: List[str] = []
    for source in [value, ";".join(fallback_flags)]:
        for raw in str(source or "").split(";"):
            flag = raw.strip().lower().replace(" ", "_")
            if flag and flag != "none" and flag in ALLOWED_RISK_FLAGS and flag not in flags:
                flags.append(flag)
    return ";".join(flags) if flags else "none"


def claimant_text(text: str) -> str:
    """Keep claimant utterances so support prompts do not dominate extraction."""
    segments: List[str] = []
    for raw in re.split(r"\s*\|\s*", str(text or "")):
        lowered = raw.strip().lower()
        if lowered.startswith(("customer:", "cliente:")):
            segments.append(raw.split(":", 1)[1].strip())
    return " ".join(segments) if segments else str(text or "")


def extract_claim_issue(text: str, claim_object: str) -> str:
    """Infer the claimed issue family from the conversation text."""
    lower = text.lower()
    if claim_object == "package" and any(term in lower for term in ["not inside", "missing contents", "contents were missing"]):
        return "missing_part"
    if claim_object == "package" and any(term in lower for term in ["torn", "torn-open", "seal", "phati", "opened jaisa"]):
        return "torn_packaging"
    if claim_object == "package" and any(term in lower for term in ["crush", "crushed", "dab gaya"]):
        return "crushed_packaging"
    if any(term in lower for term in ["shatter", "shattered", "spiderweb"]):
        return "glass_shatter" if claim_object == "car" else "crack"
    if any(term in lower for term in ["crack", "cracked", "body crack"]):
        return "crack"
    if any(term in lower for term in ["missing", "keycaps came off", "came off"]):
        return "missing_part"
    if any(term in lower for term in ["broken", "broke", "toot", "damaged hinge", "no longer opens"]):
        return "broken_part"
    if any(term in lower for term in ["dent", "dented", "hail", "dab", "parachoques", "danado", "dano"]):
        return "dent"
    if any(term in lower for term in ["scratch", "scrape", "mark", "scratched"]):
        return "scratch"
    if any(term in lower for term in ["stain", "coffee", "oil", "oily"]):
        return "stain"
    if any(term in lower for term in ["water", "wet", "rain", "liquid damage"]):
        return "water_damage"
    if any(term in lower for term in ["damage", "damaged", "affected"]):
        return "crushed_packaging" if claim_object == "package" else "broken_part"
    return "unknown"


def extract_object_part(text: str, claim_object: str) -> str:
    """Infer the claimed object part from the conversation text."""
    lower = text.lower()
    if claim_object == "car":
        checks = [
            ("rear_bumper", ["rear bumper", "back bumper", "from behind", "parachoques trasero", "atras"]),
            ("front_bumper", ["front bumper", "front side", "front area", "parachoques"]),
            ("side_mirror", ["side mirror", "left mirror", "mirror", "toot gaya"]),
            ("windshield", ["windshield", "front glass"]),
            ("headlight", ["headlight"]),
            ("taillight", ["taillight", "back light"]),
            ("door", ["door"]),
            ("hood", ["hood"]),
            ("fender", ["fender"]),
            ("quarter_panel", ["quarter panel"]),
            ("body", ["body panel", "car body", "body"]),
        ]
    elif claim_object == "laptop":
        if "hinge" in lower and "not the keyboard or hinge" not in lower:
            return "hinge"
        checks = [
            ("screen", ["screen", "display", "pantalla"]),
            ("keyboard", ["keyboard", "keys", "keycaps", "teclas"]),
            ("trackpad", ["trackpad"]),
            ("hinge", ["hinge"]),
            ("lid", ["lid", "outer lid"]),
            ("corner", ["corner"]),
            ("port", ["port"]),
            ("base", ["base"]),
            ("body", ["body", "side edge"]),
        ]
    else:
        checks = [
            ("contents", ["contents", "not inside", "missing product", "missing contents"]),
            ("package_corner", ["corner", "dab gaya"]),
            ("seal", ["seal", "tape", "flap", "torn-open", "phati", "opened jaisa"]),
            ("label", ["label"]),
            ("package_side", ["side", "surface", "wet", "water", "stain", "oily"]),
            ("box", ["box", "package", "parcel", "cardboard"]),
            ("item", ["item inside", "product", "item"]),
        ]
    for part, terms in checks:
        if any(term in lower for term in terms):
            return part
    return "unknown"


def severity_for(issue_type: str, claim_text: str) -> str:
    """Choose a conservative severity estimate."""
    lower = claim_text.lower()
    if issue_type == "none":
        return "none"
    if issue_type == "unknown":
        return "unknown"
    if any(term in lower for term in ["severe", "shattered", "badly", "missing contents", "not inside"]):
        return "high"
    if issue_type in {"scratch", "stain"}:
        return "low"
    return "medium"


def build_default_row(row: Mapping[str, str], contexts: Contexts) -> Dict[str, str]:
    """Create a deterministic fallback prediction before VLM validation."""
    claim_object = str(row.get("claim_object", "")).strip().lower()
    claim_text = str(row.get("user_claim", ""))
    focused_claim_text = claimant_text(claim_text)
    image_paths = split_image_paths(str(row.get("image_paths", "")))
    resolved = [resolve_image_path(contexts.repo_root, path) for path in image_paths]
    existing = [path for path in resolved if path.exists()]
    ids = [image_id(path) for path in image_paths]

    user = contexts.user_history.get(str(row.get("user_id", "")).strip(), {})
    history_flags = str(user.get("history_flags", "none") or "none")
    rejected_count = safe_int(user.get("rejected_claim"))
    risk_flags: List[str] = []
    if rejected_count > 0 or "user_history_risk" in history_flags:
        risk_flags.extend(["user_history_risk", "manual_review_required"])
    if "ignore all previous instructions" in claim_text.lower() or "system reset" in claim_text.lower():
        risk_flags.extend(["text_instruction_present", "manual_review_required"])
    if image_paths and len(existing) < len(image_paths):
        risk_flags.extend(["damage_not_visible", "manual_review_required"])

    issue = extract_claim_issue(focused_claim_text, claim_object)
    part = extract_object_part(focused_claim_text, claim_object)
    evidence_met = bool(existing)
    status = "supported" if evidence_met and issue != "unknown" and part != "unknown" else "not_enough_information"
    if not evidence_met:
        reason = "No readable submitted image was available to evaluate the claim."
        justification = "The image evidence could not be loaded, so the claim cannot be verified."
    elif issue == "unknown" or part == "unknown":
        reason = "The submitted image set exists, but the claimed issue or part is not specific enough for a confident automated review."
        justification = "The claim text or visual evidence is ambiguous, so the safest decision is not enough information."
        risk_flags.append("manual_review_required")
    else:
        req_count = len(applicable_requirements(contexts, claim_object))
        reason = f"The submitted image set is readable and can be checked against {req_count} applicable evidence requirements."
        justification = f"The images are available for review of the claimed {part} {issue.replace('_', ' ')}."

    return {
        "evidence_standard_met": "true" if evidence_met else "false",
        "evidence_standard_met_reason": reason,
        "risk_flags": normalize_risk_flags(";".join(risk_flags)),
        "issue_type": normalize_category(issue, ALLOWED_ISSUE_TYPES, "unknown"),
        "object_part": part if part in OBJECT_PARTS.get(claim_object, ALL_OBJECT_PARTS) else "unknown",
        "claim_status": status,
        "claim_status_justification": justification,
        "supporting_image_ids": ";".join(ids) if evidence_met and ids else "none",
        "valid_image": "true" if evidence_met else "false",
        "severity": severity_for(issue, claim_text),
    }


def extract_json_object(raw_response: str) -> Dict[str, Any]:
    """Extract the first JSON object from a fenced block or raw text."""
    if not raw_response:
        return {}
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1) if fenced else ""
    if not candidate:
        generic = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        candidate = generic.group(0) if generic else ""
    if not candidate:
        return {}
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def enforce_strict_schema(raw_response: str, default_row: dict) -> dict:
    """Extract, validate, and normalize VLM output without crashing."""
    parsed = extract_json_object(raw_response)
    merged: Dict[str, Any] = {column: default_row.get(column, "") for column in MODEL_COLUMNS}
    for column in MODEL_COLUMNS:
        if column in parsed:
            merged[column] = parsed[column]

    issue = normalize_category(merged.get("issue_type"), ALLOWED_ISSUE_TYPES, default_row.get("issue_type", "unknown"))
    status = normalize_category(merged.get("claim_status"), ALLOWED_CLAIM_STATUS, "not_enough_information")
    severity = normalize_category(merged.get("severity"), ALLOWED_SEVERITY, default_row.get("severity", "unknown"))
    part = normalize_category(merged.get("object_part"), ALL_OBJECT_PARTS, default_row.get("object_part", "unknown"))

    result = {
        "evidence_standard_met": normalize_bool(merged.get("evidence_standard_met"), default_row.get("evidence_standard_met", "false")),
        "evidence_standard_met_reason": str(merged.get("evidence_standard_met_reason") or default_row.get("evidence_standard_met_reason") or "").strip()[:500],
        "risk_flags": normalize_risk_flags(merged.get("risk_flags"), default_row.get("risk_flags", "none").split(";")),
        "issue_type": issue,
        "object_part": part,
        "claim_status": status,
        "claim_status_justification": str(merged.get("claim_status_justification") or default_row.get("claim_status_justification") or "").strip()[:700],
        "supporting_image_ids": str(merged.get("supporting_image_ids") or default_row.get("supporting_image_ids") or "none").strip().lower().replace(",", ";"),
        "valid_image": normalize_bool(merged.get("valid_image"), default_row.get("valid_image", "false")),
        "severity": severity,
    }
    if not result["evidence_standard_met_reason"]:
        result["evidence_standard_met_reason"] = "The model response omitted a reason, so the deterministic fallback was used."
    if not result["claim_status_justification"]:
        result["claim_status_justification"] = "The model response omitted a justification, so the deterministic fallback was used."
    if not result["supporting_image_ids"]:
        result["supporting_image_ids"] = "none"
    return result


def cache_path(repo_root: Path) -> Path:
    """Return the local VLM cache path."""
    return repo_root / "code" / ".cache" / "vlm_cache.json"


def load_cache(path: Path) -> Dict[str, str]:
    """Load the response cache."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(path: Path, cache: Mapping[str, str]) -> None:
    """Persist the response cache best-effort."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        return


def build_prompt(row: Mapping[str, str], contexts: Contexts, images: Sequence[Mapping[str, str]]) -> str:
    """Render the user prompt from hydrated context."""
    user = contexts.user_history.get(str(row.get("user_id", "")).strip(), {})
    requirements = applicable_requirements(contexts, str(row.get("claim_object", "")))
    return USER_PROMPT_TEMPLATE.format(
        claim_object=row.get("claim_object", ""),
        user_claim=row.get("user_claim", ""),
        user_history=json.dumps(user, ensure_ascii=False, sort_keys=True),
        evidence_requirements=json.dumps(requirements, ensure_ascii=False, sort_keys=True),
        image_ids=", ".join(image["id"] for image in images) or "none",
    )


def build_cache_key(row: Mapping[str, str], contexts: Contexts, images: Sequence[Mapping[str, str]]) -> str:
    """Build a cache key from prompt version, hydrated context, and image hashes."""
    payload = {
        "prompt_version": PROMPT_VERSION,
        "user_id": row.get("user_id", ""),
        "claim_object": row.get("claim_object", ""),
        "user_claim": row.get("user_claim", ""),
        "image_hashes": [image.get("hash", "") for image in images],
        "user_history": contexts.user_history.get(str(row.get("user_id", "")).strip(), {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


async def call_openai_vlm(prompt: str, images: Sequence[Mapping[str, str]], model: str, timeout: float = 45.0) -> str:
    """Call OpenAI's chat completions endpoint using stdlib HTTP in a worker."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        if image.get("b64"):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image['b64']}", "detail": "high"},
                }
            )
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": 900,
    }

    def post() -> str:
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])

    return await asyncio.to_thread(post)


async def call_vlm_with_retries(
    row: Mapping[str, str],
    contexts: Contexts,
    images: Sequence[Mapping[str, str]],
    cache: MutableMapping[str, str],
    model: str,
    retries: int = 2,
) -> str:
    """Call the VLM with cache, timeouts, retries, and backoff."""
    key = build_cache_key(row, contexts, images)
    if key in cache:
        return cache[key]
    prompt = build_prompt(row, contexts, images)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = await call_openai_vlm(prompt, images, model=model)
            cache[key] = response
            return response
        except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            await asyncio.sleep(min(2.0 * (attempt + 1), 5.0))
    raise RuntimeError(last_error or "VLM call failed")


def should_use_vlm(mode: str) -> bool:
    """Determine whether API-backed VLM processing is enabled."""
    selected = mode.strip().lower()
    if selected == "never":
        return False
    if selected == "always":
        return True
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


async def process_row(
    row: Mapping[str, str],
    contexts: Contexts,
    semaphore: asyncio.Semaphore,
    cache: MutableMapping[str, str],
    use_vlm: str,
    model: str,
) -> Dict[str, str]:
    """Process one claim row without allowing failures to escape."""
    base = {column: str(row.get(column, "") or "") for column in OUTPUT_COLUMNS[:4]}
    default = build_default_row(row, contexts)
    image_paths = split_image_paths(base.get("image_paths", ""))
    resolved = [resolve_image_path(contexts.repo_root, path) for path in image_paths]
    images = [
        {"id": image_id(original), "path": str(path), "b64": read_image_b64(path), "hash": image_hash(path)}
        for original, path in zip(image_paths, resolved)
    ]

    try:
        if should_use_vlm(use_vlm):
            async with semaphore:
                raw = await call_vlm_with_retries(row, contexts, images, cache, model=model)
            prediction = enforce_strict_schema(raw, default)
        else:
            prediction = enforce_strict_schema(json.dumps(default), default)
    except Exception as exc:
        fallback = dict(default)
        fallback["claim_status"] = "not_enough_information"
        fallback["risk_flags"] = normalize_risk_flags(f"{fallback.get('risk_flags', 'none')};manual_review_required")
        fallback["claim_status_justification"] = f"Automated review fell back safely after an internal processing error: {type(exc).__name__}."
        prediction = enforce_strict_schema(json.dumps(fallback), fallback)

    full = {**base, **prediction}
    return {column: str(full.get(column, "")) for column in OUTPUT_COLUMNS}


def write_output(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Write predictions with the exact required schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


async def process_dataset(
    input_csv: Path,
    output_csv: Path | None,
    use_vlm: str = "auto",
    concurrency: int = 20,
    model: str = "gpt-4o",
    repo_root: Path | None = None,
) -> List[Dict[str, str]]:
    """Process all rows concurrently while preserving input order."""
    contexts = load_contexts(repo_root)
    rows = read_csv_rows(input_csv)
    semaphore = asyncio.Semaphore(20 if concurrency <= 0 else concurrency)
    cache_file = cache_path(contexts.repo_root)
    cache = load_cache(cache_file)
    tasks = [process_row(row, contexts, semaphore, cache, use_vlm, model) for row in rows]
    predictions = await asyncio.gather(*tasks)
    if should_use_vlm(use_vlm):
        save_cache(cache_file, cache)
    if output_csv is not None:
        write_output(output_csv, predictions)
    return predictions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the HackerRank Orchestrate evidence-review pipeline.")
    parser.add_argument("--input", type=Path, default=repo_root / "dataset" / "claims.csv", help="Input claims CSV.")
    parser.add_argument("--output", type=Path, default=repo_root / "output.csv", help="Output predictions CSV.")
    parser.add_argument("--use-vlm", choices=["auto", "always", "never"], default="auto", help="VLM usage mode.")
    parser.add_argument("--concurrency", type=int, default=20, help="Async concurrency limit.")
    parser.add_argument("--model", default=os.environ.get("ORCHESTRATE_VLM_MODEL", "gpt-4o"), help="OpenAI vision model name.")
    return parser.parse_args(argv)


async def main() -> None:
    """Orchestrate the batch run and export output.csv."""
    args = parse_args()
    started = time.time()
    predictions = await process_dataset(
        input_csv=args.input,
        output_csv=args.output,
        use_vlm=args.use_vlm,
        concurrency=args.concurrency,
        model=args.model,
    )
    elapsed = time.time() - started
    print(f"Wrote {len(predictions)} rows to {args.output} in {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
