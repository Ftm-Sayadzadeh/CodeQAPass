"""Build the non-scored, blinded v4 Task-1 human-review package."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "reports/evaluation/swe_qa_v4/SAMPLE_FREEZE.json"
OUT = ROOT / "reports/evaluation/swe_qa_v4/human_review_v4"
ARMS = ("a0", "a1", "b0_k5")
SOURCE = {
    "a0": ROOT / "reports/evaluation/swe_qa_v4/frozen/main_a0_combined.jsonl",
    "a1": ROOT / "reports/evaluation/swe_qa_v4/frozen/main_a1_combined.jsonl",
    "b0_k5": ROOT / "reports/evaluation/swe_qa_v4/frozen/main_b0_k5_combined.jsonl",
}
PATH_LINE = re.compile(r"(?<!\w)(?:[\w./-]+\.py:\d+(?:-\d+)?)(?!\w)")


def read_jsonl(path: Path) -> dict[str, dict]:
    return {row["question_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    question_ids = freeze["human_review_sample"]["ids_in_order"]
    records = {arm: read_jsonl(path) for arm, path in SOURCE.items()}
    rows = []
    for qid in question_ids:
        for arm in ARMS:
            row = records[arm][qid]
            status = row.get("status", "unavailable")
            answer = row.get("answer", "") if status == "usable" else ""
            flag = "ok" if status == "usable" and row.get("finish_reason") == "stop" else status
            rows.append({
                "item_id": secrets.token_hex(8),
                "question_id": qid,
                "question": row.get("question", ""),
                "reference_answer": row.get("reference_answer", ""),
                "answer": PATH_LINE.sub("[source citation]", answer),
                "status_flag": flag,
                "correctness": "",
                "completeness": "",
                "relevance": "",
                "groundedness": "",
                "refusal": "",
                "comment": "",
            })
    secrets.SystemRandom().shuffle(rows)
    path = OUT / "task1_items.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "INSTRUCTIONS.md").write_text(
        "# SWE-QA v4 Task 1\n\nUse HUMAN_REVIEW_RUBRIC.md from the parent directory. "
        "Score only the four 0–10 criteria; do not infer unavailable outputs as zero.\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol": "swe_qa_v4",
        "task1_rows": len(rows),
        "questions": len(question_ids),
        "arms": list(ARMS),
        "question_ids_sha256": hashlib.sha256("\n".join(question_ids).encode()).hexdigest(),
        "task1_sha256": digest(path),
        "a2_task2_status": "pending: v4 A2 post-hoc model citations have not been generated",
        "blinding_note": "Task-1 arm labels are omitted; item IDs are random. No plaintext arm key is written.",
    }
    (OUT / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
