"""B0 answer generation, judging and paired comparison for SWE-QA v2.

Reuses the frozen A1 prompt, system prompt, generation settings and judge prompt; only the context differs. Raw files
are never overwritten (`open_new`); retries are separate `*_retry.jsonl` records. Existing A0/A1/A2 and judge records are
only read.

    python scripts/evaluation/swe_qa_b0_answers.py generate [--dry-run]   # b0/answers_b0_k5.jsonl, answers_b0_k10.jsonl
    python scripts/evaluation/swe_qa_b0_answers.py retry                  # separate retry records for unavailable answers
    python scripts/evaluation/swe_qa_b0_answers.py judge [--dry-run]      # b0/judge_scores_b0.jsonl (append, resumable)
    python scripts/evaluation/swe_qa_b0_answers.py judge-retry-network    # b0/judge_scores_b0_retry.jsonl (network/HTTP failures only)
    python scripts/evaluation/swe_qa_b0_answers.py summary                # b0/judge_summary_b0.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_swe_qa_v2 as analyze  # noqa: E402
import run_swe_qa_v2 as run  # noqa: E402
from codecompass.llm import LLMRequest  # noqa: E402
from codecompass.providers import ProviderConfig, create_llm_provider  # noqa: E402

B0 = run.OUT / "b0"
CONFIGS = ("b0_k5", "b0_k10")
CRITERIA = ("correctness", "completeness", "relevance", "clarity", "reasoning")
SEED = 20260930
REQUIRED_REASONING_EFFORT = "low"  # the A0/A1/A2 arms were generated with this setting (manifest.json)


def require_frozen_generation_settings() -> None:
    """Stop before any generation call unless the frozen reasoning effort is set; without it answers are not comparable to A1."""
    actual = os.environ.get("CODECOMPASS_LLM_REASONING_EFFORT")
    if actual != REQUIRED_REASONING_EFFORT:
        raise SystemExit(f"CODECOMPASS_LLM_REASONING_EFFORT must be '{REQUIRED_REASONING_EFFORT}' (got {actual!r}); "
                         "the A1 arm was generated with it, so B0 answers generated without it are not comparable")


def answers_path(config: str, retry: bool = False) -> Path:
    return B0 / f"answers_{config}{'_retry' if retry else ''}.jsonl"


def merged_answers(config: str) -> dict[str, dict]:
    """Original records with each unavailable one replaced by its separate retry record (raw files untouched)."""
    records = {r["question_id"]: r for r in run.read_jsonl(answers_path(config))}
    if answers_path(config, True).exists():
        for r in run.read_jsonl(answers_path(config, True)):
            records[r["question_id"]] = {**r, "retry_of_unavailable": True}
    return records


def b0_contexts() -> dict[str, dict]:
    return {r["question_id"]: r for r in run.read_jsonl(B0 / "contexts.jsonl")}


def answer_record(provider, row: dict, entry: dict) -> dict:
    record = run.call(provider, run.a1_prompt(row["question"], entry["context"]))
    record.update({"question_id": row["question_id"], "repo": row["repo"], "question": row["question"],
                   "reference_answer": row.get("answer"), "context_sha256": entry["context_sha256"],
                   "context_chunk_ids": entry["chunk_ids"], "citations": []})  # B0 attaches no citations (protocol section 3)
    return record


def generate(retry: bool, dry_run: bool) -> None:
    rows = sum((run.load_rows(path, repo) for repo, (path, *_rest) in run.REPOS.items()), [])
    contexts = b0_contexts()
    if dry_run:
        for row in rows[:2]:
            for config in CONFIGS:
                prompt = run.a1_prompt(row["question"], contexts[row["question_id"]]["configs"][config]["context"])
                print(f"--- {row['question_id']} {config}: prompt {len(prompt)} chars\n{prompt[:300]}...\n")
        return
    require_frozen_generation_settings()
    providers = {repo: run.llm_from_env() for repo in run.REPOS}
    for config in CONFIGS:
        if retry:
            todo = [r for r in run.read_jsonl(answers_path(config)) if r.get("status") == "unavailable"]
            target_path = answers_path(config, True)
        else:
            todo, target_path = rows, answers_path(config)
        with run.open_new(target_path) as target:
            for item in todo:
                row = next(r for r in rows if r["question_id"] == item["question_id"])
                record = answer_record(providers[row["repo"]], row, contexts[row["question_id"]]["configs"][config])
                target.write(json.dumps(record, ensure_ascii=False) + "\n"); target.flush()
                print(f"{config} {row['question_id']} {record['status']}", flush=True)


def judge(dry_run: bool) -> None:
    records = [{"arm": config, **row} for config in CONFIGS for row in merged_answers(config).values() if row["status"] == "usable"]
    random.Random(SEED).shuffle(records)
    path = B0 / "judge_scores_b0.jsonl"
    if dry_run:
        print(len(records), "usable B0 answers to judge")
        for row in records[:1]:
            print(analyze.official_prompt(row["question"], row["reference_answer"], analyze.CITATION_RE.sub("", row["answer"]))[:600])
        return
    provider = judge_provider()
    completed = {r["blind_id"] for r in run.read_jsonl(path) if r.get("status") == "usable"} if path.exists() else set()
    with path.open("a", encoding="utf-8") as target:
        for i, row in enumerate(records, 1):
            blind_id = f"B{i:03d}"
            if blind_id in completed:
                continue
            result = judge_one(provider, blind_id, row)
            target.write(json.dumps(result, ensure_ascii=False) + "\n"); target.flush()
            print(f"{blind_id} {result.get('status')}", flush=True)


def judge_provider():
    return create_llm_provider(ProviderConfig(provider="openai_compatible", base_url=os.environ["CODECOMPASS_COMPARE_BASE_URL"],
        api_key=os.environ["CODECOMPASS_COMPARE_API_KEY"], llm_model="gemini-2.5-flash", timeout_seconds=180))


def judge_one(provider, blind_id: str, row: dict) -> dict:
    """Score one answer with the original judge configuration (gemini-2.5-flash, official prompt, temperature 0, max_tokens 500)."""
    prompt = analyze.official_prompt(row["question"], row["reference_answer"], analyze.CITATION_RE.sub("", row["answer"]))
    result = {"blind_id": blind_id, "question_id": row["question_id"], "arm": row["arm"], "judge_model": "gemini-2.5-flash"}
    for attempt in range(1, 5):
        try:
            response = provider.generate(LLMRequest(prompt=prompt, temperature=0, max_tokens=500, response_format="json"))
            scores = json.loads(response.text)
            if all(isinstance(scores.get(k), int) and 1 <= scores[k] <= 20 for k in CRITERIA):
                result.update({"status": "usable", "scores": scores, "mean": sum(scores.values()) / 5, "attempts": attempt})
                break
        except Exception as exc:
            result.update({"status": "unavailable", "error": str(exc)[:300], "attempts": attempt})
        time.sleep(2 ** (attempt - 1))
    return result


def failure_type(record: dict) -> str:
    """Coarse failure class of an unavailable judge record: truncated/invalid JSON, HTTP 429, timeout, or connection."""
    error = record.get("error") or ""
    if "429" in error:
        return "http_429"
    if "timed out" in error:
        return "timeout"
    if "request failed" in error:
        return "connection"
    return "invalid_or_truncated_json"


def all_judged_records() -> list[dict]:
    """Main judge pass followed by the network-retry pass; a later record supersedes an earlier one per blind_id."""
    records: list[dict] = []
    for name in ("judge_scores_b0.jsonl", "judge_scores_b0_retry.jsonl"):
        if (B0 / name).exists():
            records += run.read_jsonl(B0 / name)
    return records


def judge_retry_network() -> None:
    """Retry once, in a separate file, only records whose latest failure is a connection, timeout or HTTP error."""
    records = [{"arm": config, **row} for config in CONFIGS for row in merged_answers(config).values() if row["status"] == "usable"]
    random.Random(SEED).shuffle(records)
    latest: dict[str, dict] = {}
    for record in all_judged_records():
        latest[record["blind_id"]] = record
    todo = {bid: r for bid, r in latest.items() if r.get("status") != "usable" and failure_type(r) in ("connection", "http_429", "timeout")}
    print(len(todo), "records to retry")
    provider = judge_provider()
    with open(B0 / "judge_scores_b0_retry.jsonl", "a", encoding="utf-8") as target:
        for i, row in enumerate(records, 1):
            blind_id = f"B{i:03d}"
            if blind_id not in todo:
                continue
            result = judge_one(provider, blind_id, row)
            result["retry_of_network_failure"] = True
            target.write(json.dumps(result, ensure_ascii=False) + "\n"); target.flush()
            print(f"{blind_id} {result.get('status')}", flush=True)


def latest_usable(records: list[dict]) -> dict[tuple[str, str], dict]:
    latest = {}
    for row in records:
        latest[row["blind_id"]] = row
    return {(r["question_id"], r["arm"]): r for r in latest.values() if r.get("status") == "usable"}


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    ordered = sorted((p, name) for name, p in p_values.items() if p is not None)
    adjusted, running = {name: None for name in p_values}, 0.0
    for rank, (p, name) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def compare(scores: dict, arm_a: str, arm_b: str, metric: str) -> dict:
    """Paired by question over questions with a usable judged answer in both arms; difference = arm_a - arm_b."""
    questions = sorted(q for q, arm in scores if arm == arm_a and (q, arm_b) in scores)
    def value(q: str, arm: str) -> float:
        row = scores[(q, arm)]
        return row["mean"] if metric == "mean" else row["scores"][metric]
    diffs = [value(q, arm_a) - value(q, arm_b) for q in questions]
    rng = random.Random(SEED)
    boot = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(20000))
    return {"n_pairs": len(questions), "mean_a": sum(value(q, arm_a) for q in questions) / len(questions),
            "mean_b": sum(value(q, arm_b) for q in questions) / len(questions), "mean_difference": sum(diffs) / len(diffs),
            "bootstrap_95": [boot[500], boot[19499]], "a_better": sum(d > 0 for d in diffs), "a_worse": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs), "wilcoxon": analyze.wilcoxon_signed_rank(diffs)}


def summary() -> None:
    scores = latest_usable(run.read_jsonl(run.OUT / "judge_scores.jsonl")) | latest_usable(all_judged_records())
    primary = (("a1_codeqapass", "a0_neutral"), ("a1_codeqapass", "b0_k5"), ("b0_k5", "a0_neutral"))
    secondary = (("a1_codeqapass", "b0_k10"), ("b0_k10", "a0_neutral"), ("b0_k10", "b0_k5"))
    out: dict = {"judge_model": "gemini-2.5-flash", "bootstrap": {"resamples": 20000, "seed": SEED},
                 "difference": "first arm minus second arm", "primary_endpoint": "mean of five criteria",
                 "primary": {}, "secondary": {},
                 "note": "automatic judge, not human quality scores; A0-neutral and A1 scores come from judge_scores.jsonl, "
                         "B0 scores from b0/judge_scores_b0.jsonl (same prompt and settings, scored in a separate pass)"}
    for a, b in primary:
        out["primary"][f"{a}_minus_{b}"] = compare(scores, a, b, "mean")
    holm_p = holm({name: v["wilcoxon"]["p_value"] for name, v in out["primary"].items()})
    for name, adjusted in holm_p.items():
        out["primary"][name]["holm_adjusted_p"] = adjusted
    for a, b in secondary:
        out["secondary"][f"{a}_minus_{b}"] = {m: compare(scores, a, b, m) for m in CRITERIA + ("mean",)}
    for a, b in primary[1:]:
        out["secondary"][f"{a}_minus_{b}"] = {m: compare(scores, a, b, m) for m in CRITERIA}
    latest_b0: dict[str, dict] = {}
    for record in all_judged_records():
        latest_b0[record["blind_id"]] = record
    unresolved = [r for r in latest_b0.values() if r.get("status") != "usable"]
    out["b0_judging_status"] = {"judged_items": len(latest_b0), "usable": len(latest_b0) - len(unresolved), "unresolved": len(unresolved),
                                "unresolved_by_failure_type": dict(collections.Counter(failure_type(r) for r in unresolved)),
                                "complete": not unresolved}
    (B0 / "judge_summary_b0.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, v in out["primary"].items():
        print(name, v["n_pairs"], round(v["mean_difference"], 3), [round(x, 2) for x in v["bootstrap_95"]],
              v["wilcoxon"]["p_value"], v["holm_adjusted_p"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("generate", "retry", "judge", "judge-retry-network", "summary"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.phase in ("generate", "retry"):
        generate(retry=args.phase == "retry", dry_run=args.dry_run)
    elif args.phase == "judge":
        judge(args.dry_run)
    elif args.phase == "judge-retry-network":
        judge_retry_network()
    else:
        summary()
