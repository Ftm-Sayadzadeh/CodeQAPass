"""Tests for the SWE-QA v7 preflight helpers (no model, network or index access)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v7_preflight as pre  # noqa: E402


def chunks(*files: str) -> list[dict]:
    return [{"file": f} for f in files]


def test_study2_subset_is_deterministic_and_balanced() -> None:
    first, second = pre.study2_subset(), pre.study2_subset()
    assert first == second
    assert set(first) == set(pre.REPOS) and all(len(v) == pre.PER_REPO_SUBSET and len(set(v)) == pre.PER_REPO_SUBSET for v in first.values())
    assert all(qid.startswith(repo + "-") and 1 <= int(qid.rsplit("-", 1)[1]) <= 48 for repo, v in first.items() for qid in v)


def test_retrieval_row_uses_v4_file_depth_and_chunk_depth_conventions() -> None:
    ranking = chunks("a.py", "a.py", "b.py", "c.py", "d.py", "e.py", "f.py")  # chunks 1-7 over files a, b, c, d, e, f
    row = pre.retrieval_row(ranking, {"e.py", "z.py"})
    assert row["recall5"] == 0.5 and row["hit5"] == 1.0  # e is the 5th distinct file, within the first 5 files
    assert row["recall5_chunks"] == 0.0 and row["hit5_chunks"] == 0.0  # but only the 6th chunk
    assert row["mrr10"] == 1 / 5 and row["recall10"] == 0.5
    assert pre.retrieval_row(chunks("x.py"), {"a.py"})["mrr10"] == 0.0


def test_bootstrap_is_reproducible_and_paired_difference_counts_wins() -> None:
    groups = {"r1": [0.1, 0.2, 0.3], "r2": [0.4, 0.5]}
    assert pre.bootstrap(groups, 200) == pre.bootstrap(groups, 200)
    a = [{"question_id": "r1-1", "repo": "r1", "m": 1.0}, {"question_id": "r1-2", "repo": "r1", "m": 0.0}]
    b = [{"question_id": "r1-1", "repo": "r1", "m": 0.5}, {"question_id": "r1-2", "repo": "r1", "m": 0.5}]
    result = pre.paired(a, b, "m")
    assert result["better"] == 1 and result["worse"] == 1 and result["mean_difference"] == 0.0


def test_write_once_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    pre.write_once(target, "{}")
    try:
        pre.write_once(target, "{}")
    except SystemExit:
        return
    raise AssertionError("second write must be refused")
