"""Build the blinded A1-vs-A2 citation-support sheet after successful A2 generation."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "reports/evaluation/swe_qa_v4/SAMPLE_FREEZE.json"
A1 = ROOT / "reports/evaluation/swe_qa_v4/frozen/main_a1_combined.jsonl"
A2 = ROOT / "reports/evaluation/swe_qa_v4/frozen/a2_model_citations_retry.jsonl"
OUT = ROOT / "reports/evaluation/swe_qa_v4/human_review_v4"
REPOS = {name: ROOT / "data/benchmarks/swe_qa_v4/repos" / name for name in ("flask", "requests", "pytest", "pylint")}
PATH_LINE = re.compile(r"(?<!\w)(?:[\w./-]+\.py:\d+(?:-\d+)?)(?!\w)")


def rows(path: Path) -> dict[str, dict]:
    return {r["question_id"]: r for r in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()) if r.get("status") == "usable"}


def answer_without_citations(text: str) -> str:
    return PATH_LINE.sub("[source citation]", text or "")


def main() -> int:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    qids = freeze["human_review_sample"]["ids_in_order"]
    a1, a2 = rows(A1), rows(A2)
    if any(qid not in a1 or qid not in a2 for qid in qids):
        raise SystemExit("Task 2 requires usable A1 and A2 records for all 30 frozen questions")
    task1 = []
    task2 = []
    cited_code = []
    for qid in qids:
        base = a1[qid]; repo = base["repo"]
        labels = ["X", "Y"] if secrets.randbelow(2) == 0 else ["Y", "X"]
        for label, arm in zip(labels, ("a1", "a2_model_citations")):
            source = base if arm == "a1" else a2[qid]
            answer = answer_without_citations(base.get("answer", ""))
            for citation in source.get("citations", []):
                citation_id = secrets.token_hex(8)
                path = REPOS[repo] / citation["source_file"]
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
                start, end = int(citation["start_line"]), min(len(lines), int(citation["end_line"]))
                cited_range = f"{citation['source_file']}:{start}-{end}"
                task2.append({"item_id": secrets.token_hex(8), "question_id": qid, "question": base["question"], "answer_being_attributed": answer, "system_label": label, "citation_id": citation_id, "cited_range": cited_range, "support": "", "missing_key_source": ""})
                cited_code.append({"citation_id": citation_id, "cited_range": cited_range, "code": "\n".join(lines[start - 1:end])})
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / "task2_items.csv", OUT / "task2_cited_code.csv"):
        if path.exists() and path.stat().st_size:
            raise SystemExit(f"refusing to overwrite {path}")
    with (OUT / "task2_items.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task2[0])); writer.writeheader(); writer.writerows(task2)
    with (OUT / "task2_cited_code.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cited_code[0])); writer.writeheader(); writer.writerows(cited_code)
    manifest_path = OUT / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"task2_rows": len(task2), "task2_questions": len(qids), "task2_sha256": hashlib.sha256((OUT / "task2_items.csv").read_bytes()).hexdigest(), "task2_code_sha256": hashlib.sha256((OUT / "task2_cited_code.csv").read_bytes()).hexdigest(), "a2_task2_status": "ready_for_human_scoring"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
