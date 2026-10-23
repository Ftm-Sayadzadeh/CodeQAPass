"""Prepare, run, and summarize the blinded v4 automatic judge pass."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from pathlib import Path

from codecompass.llm import LLMRequest
from codecompass.providers import ProviderConfig, create_llm_provider

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "reports/evaluation/swe_qa_v4/frozen"
INPUT = FROZEN / "judge_input_blinded.jsonl"
KEY = FROZEN / "judge_key.json"
SCORES = FROZEN / "judge_scores.jsonl"
RETRY = FROZEN / "judge_scores_retry.jsonl"
RETRY2 = FROZEN / "judge_scores_retry2.jsonl"
SUMMARY = FROZEN / "judge_summary.json"
SEED = 20260930
MODEL = "gemini-2.5-flash"
MAX_TOKENS = 2000
TEMPERATURE = 0.0
ARMS = ("a0", "b0_k5", "a1")
CITATION_RE = re.compile(r"[\w./-]+\.py:\d+(?:-\d+)?")
SCORE_KEYS = ("correctness", "completeness", "relevance", "clarity", "reasoning")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def official_prompt(question: str, reference: str, candidate: str) -> str:
    source = (ROOT / "reports/evaluation/swe_qa_v2/official_llm_as_a_judge.py").read_text(encoding="utf-8")
    template = re.search(r"prompt = f\"\"\"(.*?)\"\"\"", source, flags=re.S)
    if not template:
        raise RuntimeError("official judge prompt not found")
    text = template.group(1).replace("{{", "{").replace("}}", "}")
    values = {"question": question, "reference": reference, "candidate": candidate}
    return re.sub(r"\{(question|reference|candidate)\}", lambda match: values[match.group(1)], text)


def prepare() -> int:
    if INPUT.exists() or KEY.exists():
        raise SystemExit("judge input already exists; frozen inputs are immutable")
    arms = {
        arm: read_jsonl(FROZEN / f"main_{arm}_combined.jsonl")
        for arm in ARMS
    }
    records = []
    for arm in ARMS:
        for row in arms[arm]:
            if row["status"] not in {"usable", "no_evidence"}:
                continue
            records.append({"arm": arm, "question_id": row["question_id"], "repo": row["repo"],
                            "question": row["question"], "reference_answer": row.get("reference_answer", ""),
                            "candidate": CITATION_RE.sub("", row.get("answer") or "")})
    random.Random(SEED).shuffle(records)
    prompts = []
    key = {}
    for index, row in enumerate(records, 1):
        blind_id = f"J{index:03d}"
        prompts.append({"blind_id": blind_id, "question": row["question"],
                        "reference_answer": row["reference_answer"], "candidate": row["candidate"]})
        key[blind_id] = {"question_id": row["question_id"], "repo": row["repo"], "arm": row["arm"]}
    INPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts), encoding="utf-8")
    KEY.write_text(json.dumps({"seed": SEED, "records": len(records), "mapping": key}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                      "key_sha256": hashlib.sha256(KEY.read_bytes()).hexdigest()}, indent=2))
    return 0


def parse_scores(text: str) -> dict[str, int]:
    value = text.strip()
    if value.startswith("```json"):
        value = value[7:]
    if value.endswith("```"):
        value = value[:-3]
    parsed = json.loads(value.strip())
    if any(not isinstance(parsed.get(key), int) or not 1 <= parsed[key] <= 20 for key in SCORE_KEYS):
        raise ValueError("judge score outside 1..20")
    return {key: parsed[key] for key in SCORE_KEYS}


def run_pass(output: Path, only_ids: set[str] | None = None) -> int:
    rows = read_jsonl(INPUT)
    existing = set()
    for path in (SCORES, RETRY, RETRY2):
        if path.exists():
            existing.update(row["blind_id"] for row in read_jsonl(path))
    output_existing = {row["blind_id"] for row in read_jsonl(output)} if output.exists() else set()
    todo = [row for row in rows if (only_ids is None and row["blind_id"] not in existing) or
            (only_ids is not None and row["blind_id"] in only_ids and row["blind_id"] not in output_existing)]
    if not todo:
        print("no judge records to run")
        return 0
    provider = create_llm_provider(ProviderConfig(provider="openai_compatible", base_url=os.environ["CODECOMPASS_COMPARE_BASE_URL"],
        api_key=os.environ["CODECOMPASS_COMPARE_API_KEY"], llm_model=MODEL, timeout_seconds=180))
    with output.open("a", encoding="utf-8") as target:
        for row in todo:
            result = {"blind_id": row["blind_id"], "judge_model": MODEL, "temperature": TEMPERATURE,
                      "max_tokens": MAX_TOKENS}
            prompt = official_prompt(row["question"], row["reference_answer"], row["candidate"])
            for attempt in range(1, 5):
                try:
                    response = provider.generate(LLMRequest(prompt=prompt, temperature=TEMPERATURE,
                                                            max_tokens=MAX_TOKENS, response_format="json"))
                    scores = parse_scores(response.text)
                    result.update({"status": "usable", "scores": scores, "mean": sum(scores.values()) / 5,
                                   "attempts": attempt, "finish_reason": response.finish_reason})
                    break
                except Exception as error:  # preserve provider/parser failures as unavailable
                    result.update({"status": "unavailable", "error_type": type(error).__name__,
                                   "error": str(error)[:300], "attempts": attempt})
                    if attempt < 4:
                        time.sleep(2 ** (attempt - 1))
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            print(f"{row['blind_id']} {result['status']}", flush=True)
    return 0


def wilcoxon_p(values: list[float]) -> float | None:
    nonzero = [value for value in values if value]
    if len(nonzero) < 20:
        return None
    order = sorted(range(len(nonzero)), key=lambda index: abs(nonzero[index]))
    ranks = [0.0] * len(nonzero)
    ties = 0.0
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and abs(nonzero[order[end + 1]]) == abs(nonzero[order[index]]):
            end += 1
        for position in range(index, end + 1):
            ranks[order[position]] = (index + end) / 2 + 1
        size = end - index + 1
        ties += size**3 - size
        index = end + 1
    w_plus = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    variance = len(nonzero) * (len(nonzero) + 1) * (2 * len(nonzero) + 1) / 24 - ties / 48
    z = (abs(w_plus - len(nonzero) * (len(nonzero) + 1) / 4) - 0.5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


def bootstrap_stratified(groups: dict[str, list[float]], n: int = 20_000) -> list[float]:
    rng = random.Random(SEED)
    values = []
    for _ in range(n):
        sample = [rng.choice(group) for group in groups.values() for _ in group]
        values.append(sum(sample) / len(sample))
    values.sort()
    return [values[int(0.025 * n)], values[int(0.975 * n) - 1]]


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(((name, value) for name, value in p_values.items() if value is not None), key=lambda item: item[1])
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    for rank, (name, value) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - rank) * value))
        adjusted[name] = running
    return adjusted


def summarize() -> int:
    if not KEY.exists():
        raise SystemExit("judge key missing; run prepare first")
    key = json.loads(KEY.read_text(encoding="utf-8"))["mapping"]
    latest = {}
    for path in (SCORES, RETRY, RETRY2):
        if path.exists():
            for row in read_jsonl(path):
                latest[row["blind_id"]] = row
    scored = {}
    for blind_id, row in latest.items():
        if row.get("status") == "usable":
            info = key[blind_id]
            scored[(info["question_id"], info["arm"])] = {**row, **info}
    comparisons = (("a1", "a0"), ("a1", "b0_k5"), ("b0_k5", "a0"))
    unavailable_rows = [row for row in latest.values() if row.get("status") != "usable"]
    error_categories: dict[str, int] = {}
    for row in unavailable_rows:
        error = str(row.get("error", "")).lower()
        if "429" in error or "rate limit" in error or "quota" in error:
            category = "provider_rate_limit"
        elif "timeout" in error:
            category = "timeout"
        elif "parse" in error or "json" in error:
            category = "parse_failure"
        else:
            category = str(row.get("error_type", "unknown"))
        error_categories[category] = error_categories.get(category, 0) + 1
    summary = {"judge_model": MODEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
               "bootstrap": {"resamples": 20_000, "seed": SEED}, "records": {"total": len(latest),
               "usable": sum(row.get("status") == "usable" for row in latest.values()),
               "unavailable": len(unavailable_rows),
               "unavailable_by_category": error_categories}, "comparisons": {}}
    p_values = {}
    for left, right in comparisons:
        groups: dict[str, list[float]] = {}
        diffs = []
        for question_id in sorted({question for question, arm in scored if arm == left} & {question for question, arm in scored if arm == right}):
            a = scored[(question_id, left)]
            b = scored[(question_id, right)]
            groups.setdefault(a["repo"], []).append(a["mean"] - b["mean"])
            diffs.append(a["mean"] - b["mean"])
        name = f"{left}_minus_{right}"
        p = wilcoxon_p(diffs)
        p_values[name] = p
        by_repo = {repo: {"n": len(values), "difference_mean": statistics.mean(values)}
                   for repo, values in groups.items()}
        ci = bootstrap_stratified(groups) if diffs else None
        comparison = {"n_pairs": len(diffs), "difference_mean": statistics.mean(diffs) if diffs else None,
            "bootstrap_ci95": ci, "wilcoxon_p": p, "by_repo": by_repo}
        if left == "a1" and right in {"a0", "b0_k5"}:
            margin = 1.5
            lower = ci[0] if ci else None
            comparison["non_inferiority"] = {"margin": margin, "lower_ci95": lower,
                "criterion": "lower_ci95 > -margin", "passed": bool(lower is not None and lower > -margin)}
        summary["comparisons"][name] = comparison
    summary["holm_adjusted_p"] = holm(p_values)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "judge", "retry", "retry2", "summary"))
    args = parser.parse_args()
    load_dotenv()
    if args.phase == "prepare":
        return prepare()
    if args.phase == "judge":
        return run_pass(SCORES)
    if args.phase == "retry":
        unresolved = {row["blind_id"] for row in read_jsonl(SCORES) if row.get("status") != "usable"} if SCORES.exists() else set()
        return run_pass(RETRY, unresolved)
    if args.phase == "retry2":
        latest = {}
        for path in (SCORES, RETRY):
            if path.exists():
                latest.update({row["blind_id"]: row for row in read_jsonl(path)})
        unresolved = {blind_id for blind_id, row in latest.items() if row.get("status") != "usable"}
        return run_pass(RETRY2, unresolved)
    return summarize()


if __name__ == "__main__":
    raise SystemExit(main())
