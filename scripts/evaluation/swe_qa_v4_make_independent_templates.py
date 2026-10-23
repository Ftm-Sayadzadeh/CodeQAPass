"""Create blank independent-rater copies without touching the author's scored CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "reports/evaluation/swe_qa_v4/human_review_v4"


def copy_blank(source: str, target: str, blank_fields: tuple[str, ...]) -> None:
    src = PACKAGE / source
    dst = PACKAGE / target
    if dst.exists() and dst.stat().st_size:
        raise SystemExit(f"refusing to overwrite existing template: {dst}")
    with src.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    for row in rows:
        for field in blank_fields:
            if field in row:
                row[field] = ""
    with dst.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{target}: {len(rows)} rows")


def main() -> int:
    copy_blank("task1_items.csv", "task1_items_independent.csv",
               ("correctness", "completeness", "relevance", "groundedness", "refusal", "comment"))
    copy_blank("task2_items.csv", "task2_items_independent.csv",
               ("support", "missing_key_source"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
