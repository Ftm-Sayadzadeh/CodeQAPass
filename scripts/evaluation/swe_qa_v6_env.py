"""Point the frozen v4/v5 evaluation helpers at the SWE-QA v6 data tree (fresh repositories) without editing them.

`configure()` rebinds module-level paths and commits of `swe_qa_v4_runner` and `swe_qa_v5_dev_context_gate`; nothing under
`reports/evaluation/swe_qa_v4/`, `reports/evaluation/swe_qa_v5/` or the v4 data tree is read or written afterwards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v4_runner as v4  # noqa: E402
import swe_qa_v5_dev_context_gate as gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/benchmarks/swe_qa_v6"
ARTIFACT_ROOT = ROOT / "data/benchmarks/swe_qa_v6_avalai_gemini2"
OUT = ROOT / "reports/evaluation/swe_qa_v6"
REPOS = ("streamlink", "sqlfluff")  # prespecified substitute (not used unless the chunk cap is exceeded): xarray


def configure() -> None:
    v4.DATA = DATA
    v4.COMMITS.update(json.loads((DATA / "commits.json").read_text(encoding="utf-8")))
    gate.INDEX_ROOT = ARTIFACT_ROOT / "index"
    gate.B0_ROOT = ARTIFACT_ROOT / "b0"
    gate.DATASET = DATA / "dataset"
