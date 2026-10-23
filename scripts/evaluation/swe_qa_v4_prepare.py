"""Validate and freeze the four-repository SWE-QA v4 answer inputs.

Raw generation files are append-only.  This command creates derived, sorted
files for the primary analysis and a manifest proving that every arm contains
the same 192 questions and repository snapshots.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/evaluation/swe_qa_v4"
PRIMARY = REPORTS / "main_avalai_gemini2"
SUPPLEMENT = REPORTS / "main_avalai_gemini2_flask_requests"
FROZEN = REPORTS / "frozen"
REPOS = ("flask", "requests", "pytest", "pylint")
EXPECTED = {repo: 48 for repo in REPOS}
EXPECTED_COMMITS = {
    "flask": "85c5d93cbd049c4bd0679c36fd1ddcae8c37b642",
    "requests": "46e939b5525d9c72b677340985582b04b128478a",
    "pytest": "5989efe3efec31a9d14f550d89903b6311dfc828",
    "pylint": "44740e5e5cdd2b80321a4f0ed4254213d058b9f6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sort_key(row: dict) -> tuple[int, int]:
    repo = REPOS.index(row["repo"])
    return repo, int(row["question_id"].rsplit("-", 1)[1])


def source_for(arm: str) -> tuple[Path, Path]:
    return (
        (PRIMARY / f"main_{arm}.jsonl", SUPPLEMENT / f"main_{arm}.jsonl")
        if arm != "a0"
        else (PRIMARY / "main_a0.jsonl", SUPPLEMENT / "main_a0.jsonl")
    )


def validate_arm(arm: str) -> tuple[list[dict], list[str]]:
    paths = source_for(arm)
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing raw input for {arm}: {path}")
        rows.extend(read_rows(path))
    ids = [row.get("question_id") for row in rows]
    if len(rows) != 192 or len(set(ids)) != 192:
        raise SystemExit(f"{arm}: expected 192 unique rows, found {len(rows)}")
    counts = {repo: sum(row.get("repo") == repo for row in rows) for repo in REPOS}
    if counts != EXPECTED:
        raise SystemExit(f"{arm}: repository counts differ: {counts}")
    for row in rows:
        repo = row.get("repo")
        if row.get("snapshot_commit") != EXPECTED_COMMITS[repo]:
            raise SystemExit(f"{arm}/{row.get('question_id')}: snapshot mismatch")
        if row.get("status") not in {"usable", "no_evidence", "truncated", "unavailable"}:
            raise SystemExit(f"{arm}/{row.get('question_id')}: invalid status {row.get('status')!r}")
    return sorted(rows, key=sort_key), [str(path) for path in paths]


def main() -> int:
    FROZEN.mkdir(parents=True, exist_ok=True)
    arms: dict[str, list[dict]] = {}
    sources: dict[str, list[str]] = {}
    for arm in ("a0", "b0_k5", "a1"):
        arms[arm], source_paths = validate_arm(arm)
        sources[arm] = source_paths

    reference = {
        row["question_id"]: (row["repo"], row["question"], row.get("reference_answer"))
        for row in arms["a0"]
    }
    for arm, rows in arms.items():
        for row in rows:
            expected = reference[row["question_id"]]
            actual = (row["repo"], row["question"], row.get("reference_answer"))
            if actual != expected:
                raise SystemExit(f"question/reference mismatch for {row['question_id']} in {arm}")

    output_hashes: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for arm, rows in arms.items():
        path = FROZEN / f"main_{arm}_combined.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        output_hashes[path.name] = sha256(path)
        counts[arm] = {}
        for row in rows:
            counts[arm][row["status"]] = counts[arm].get(row["status"], 0) + 1

    manifest = {
        "protocol": "swe_qa_v4",
        "purpose": "derived frozen primary-answer inputs; raw records remain unchanged",
        "questions": 192,
        "repositories": list(REPOS),
        "snapshots": EXPECTED_COMMITS,
        "arms": counts,
        "raw_sources": {arm: [{"path": path, "sha256": sha256(Path(path))} for path in paths] for arm, paths in sources.items()},
        "derived_outputs": output_hashes,
        "v2_records_pooled": False,
        "notes": [
            "All three primary arms were generated under the v4 frozen settings.",
            "No v2/v3 answer record is pooled into this manifest.",
            "A1 no_evidence records remain explicit and are not scored as zero.",
        ],
    }
    manifest_path = FROZEN / "PRIMARY_INPUT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "counts": counts, "outputs": output_hashes}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
