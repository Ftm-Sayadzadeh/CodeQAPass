"""Build the A1 and B0 indexes of the SWE-QA v6 repositories (embedding calls; resumable, cached on disk).

    python scripts/evaluation/swe_qa_v6_index.py a1 --repo streamlink
    python scripts/evaluation/swe_qa_v6_index.py b0 --repo streamlink

A1 runs the unchanged official indexing CLI through the v4 caching driver; B0 uses the v4 `index_b0`. Both refuse to overwrite an existing index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v6_env as env  # noqa: E402

env.configure()
import swe_qa_b0 as b0lib  # noqa: E402
import swe_qa_v4_index_a1 as a1  # noqa: E402
import swe_qa_v4_judge as judge  # noqa: E402
import swe_qa_v4_runner as v4  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("a1", "b0"))
    parser.add_argument("--repo", required=True, choices=env.REPOS)
    parser.add_argument("--pace-seconds", type=float, default=a1.DEFAULT_PACE_SECONDS)
    args = parser.parse_args()
    judge.load_dotenv()
    os.environ.setdefault("CODECOMPASS_LLM_REASONING_EFFORT", "low")
    v4.require_frozen_settings()
    environment = dict(os.environ)
    v4.ensure_embedding_run_identity(env.ARTIFACT_ROOT, environment)
    if args.kind == "a1":
        os.environ.setdefault("CODECOMPASS_API_KEY", environment["CODECOMPASS_GEMINI2_EMBEDDING_API_KEY"])
        code, wrappers = a1.run(a1.build_argv(args.repo, environment, args.pace_seconds, env.ARTIFACT_ROOT, 32),
                                env.ARTIFACT_ROOT / "embedding_cache", args.pace_seconds)
        for wrapper in wrappers:
            print("EMBEDDING_CACHE_STATS " + json.dumps(wrapper.stats))
        return code
    provider = b0lib.embedding_provider()
    embed = lambda texts: v4.embed_with_rate_limit_backoff(lambda t: b0lib.embed_with_retry(provider, t), texts)  # noqa: E731
    print(args.repo, v4.index_b0(args.repo, embed, artifact_root=env.ARTIFACT_ROOT), "chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
