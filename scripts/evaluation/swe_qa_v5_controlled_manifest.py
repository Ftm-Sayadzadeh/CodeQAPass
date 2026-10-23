"""Create the fixed 24-question manifest for a controlled multi-query test."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = {
    "flask-1": ["How does the ASCII-only encoding parameter affect JSON serialization?", "How are Unicode characters represented in the serialized JSON output?"],
    "flask-3": ["How are static assets registered and served by the modular component?", "How is caching behavior kept consistent with the main application framework?"],
    "flask-5": ["How is the subdomain routing configuration established?", "How are the route and request context created in the test?"],
    "requests-2": ["How does test_pragmas detect encoding declarations in HTTP content?", "How are extracted encodings normalized?"],
    "requests-5": ["How does BOM-based encoding detection identify UTF variants?", "How does raw byte-pattern detection identify UTF variants?"],
    "pytest-1": ["What does pytester.runpytest_subprocess() return?", "What does the .ret attribute mean for pytest exit codes?"],
    "pytest-5": ["How does TestRequestScopeAccess interact with the fixture system and parametrization?", "How does Pytester validate scoped request attributes?"],
    "pylint-2": ["How does the test class initialization access the class name attribute?", "What does Python's object model imply about that class-name access?"],
}

def main() -> None:
    rows = []
    for repo in ("flask", "requests", "pytest", "pylint"):
        path = ROOT / "data/benchmarks/swe_qa_v4/dataset" / f"{repo}.jsonl"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()[:6], 1):
            qid = f"{repo}-{number}"
            row = json.loads(line)
            rows.append({"question_id": qid, "repo": repo, "question": row["question"], "subqueries": MANUAL.get(qid, [row["question"]]), "decomposed": qid in MANUAL})
    out = ROOT / "reports/evaluation/swe_qa_v5_dev/controlled_multquery"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({"sample_size": len(rows), "questions": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_size": len(rows), "decomposed": sum(row["decomposed"] for row in rows), "path": str(out / "manifest.json")}, ensure_ascii=False))

if __name__ == "__main__":
    main()
