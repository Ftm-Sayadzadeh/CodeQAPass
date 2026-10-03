"""Focused tests for the SWE-QA v2 evaluation helpers (no network, no providers)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import analyze_swe_qa_v2 as analyze  # noqa: E402
import run_swe_qa_v2 as run  # noqa: E402


def test_official_prompt_keeps_braces_in_inserted_text() -> None:
    prompt = analyze.official_prompt("Q {reference}", "ref {{x}}", "cand {{ y }} {question}")
    assert "Q {reference}" in prompt and "ref {{x}}" in prompt and "cand {{ y }} {question}" in prompt
    assert '"correctness": <1-20>' in prompt


def test_open_new_refuses_to_overwrite_raw_record(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    with run.open_new(path) as handle:
        handle.write("{}\n")
    with pytest.raises(SystemExit):
        run.open_new(path)


def test_a0_neutral_prompt_differs_from_context_only_system_prompt() -> None:
    assert "only the supplied code context" in run.SYSTEM
    assert "only the supplied code context" not in run.A0_NEUTRAL_SYSTEM
    assert run.a0_neutral_prompt("q", "flask").endswith(run.a0_prompt("q"))


def test_wilcoxon_matches_known_result_and_handles_small_samples() -> None:
    assert analyze.wilcoxon_signed_rank([1.0] * 10)["p_value"] is None
    all_positive = analyze.wilcoxon_signed_rank([float(i) for i in range(1, 31)])
    assert all_positive["p_value"] < 0.001
    balanced = analyze.wilcoxon_signed_rank([float(i) * (1 if i % 2 else -1) for i in range(1, 31)])
    assert balanced["p_value"] > 0.5


def test_gold_files_and_ranking_metrics() -> None:
    import swe_qa_retrieval_metrics as metrics

    files = ["src/flask/app.py", "src/flask/json/provider.py", "tests/test_app.py"]
    answer = "See `src/flask/json/provider.py` and flask/app.py, plus tests/test_app.py and nothing/missing.py."
    assert metrics.gold_files(answer, files) == ["src/flask/app.py", "src/flask/json/provider.py", "tests/test_app.py"]
    result = metrics.question_metrics(["a.py", "src/flask/app.py", "b.py"], {"src/flask/app.py", "c.py"})
    assert result["hit@1"] == 0.0 and result["hit@3"] == 1.0
    assert result["recall@3"] == 0.5 and abs(result["precision@3"] - 1 / 3) < 1e-9 and result["mrr@10"] == 0.5


def test_b0_chunking_and_ranking() -> None:
    import numpy as np
    import swe_qa_b0 as b0

    source = '"""m"""\n\n@deco\ndef top(a):\n    return a\n\n\nclass Box:\n    """A box."""\n\n    def open(self):\n        return 1\n'
    chunks = b0.chunk_source("pkg/mod.py", source)
    by_symbol = {c["qualified"]: c for c in chunks}
    assert set(by_symbol) == {"top", "Box", "Box.open"}
    assert by_symbol["top"]["start_line"] == 3 and "return a" in by_symbol["top"]["text"]
    assert by_symbol["Box"]["text"] == "class_name: Box\nclass_docstring: A box."
    assert b0.chunk_source("bad.py", "def (") == []
    matrix = b0.normalize(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    ranked = b0.top_chunks(b0.normalize(np.array([[1.0, 0.1]], dtype=np.float32))[0], matrix, 2)
    assert [i for i, _ in ranked] == [0, 2]
