"""Tests for the SWE-QA v7 post-hoc extras (in-memory data only)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v7_posthoc_extras as extras  # noqa: E402

ARMS = extras.RETRIEVAL_ARMS


def test_common_answered_uses_only_questions_no_arm_refused() -> None:
    scores = {(q, arm): float(i + 1) for q in ("q1", "q2", "q3") for i, arm in enumerate(ARMS)}
    refused = {arm: set() for arm in ARMS}
    refused["a1_v4"] = {"q2"}
    result = extras.common_answered(scores, refused, ["q1", "q2", "q3"])
    assert result["questions"] == 2 and result["means"]["b0_k5"] == 1.0 and result["means"]["a1_v6e"] == 4.0


def test_latency_split_subtracts_embedding_and_skips_failed_and_warmup() -> None:
    rows = [
        {"arm": "b0_k5", "status": "ok", "warmup": False, "question_id": "q1", "retrieval_assembly_seconds": 2.0, "embedding_seconds": 1.5},
        {"arm": "b0_k5", "status": "ok", "warmup": False, "question_id": "q1", "retrieval_assembly_seconds": 4.0, "embedding_seconds": 3.0},
        {"arm": "b0_k5", "status": "failed", "warmup": False, "question_id": "q1", "retrieval_assembly_seconds": 99.0, "embedding_seconds": 1.0},
        {"arm": "b0_k5", "status": "ok", "warmup": True, "question_id": "q2", "retrieval_assembly_seconds": 99.0, "embedding_seconds": 1.0},
    ]
    result = extras.latency_split(rows)["b0_k5"]
    assert result["questions"] == 1 and result["embedding_seconds"] == 2.25 and result["local_retrieval_assembly_seconds"] == 0.75


def test_cost_per_good_answer_scales_to_1e4_units() -> None:
    cost = {arm: 0.0003 for arm in ARMS}
    rate = {arm: 0.3 for arm in ARMS}
    assert extras.cost_per_good_answer(cost, rate)["a1_v5a"] == 10.0
