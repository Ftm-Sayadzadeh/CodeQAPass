"""Build the SWE-QA v7 Study 4 reviewer packages (protocol section 7): 30 questions x 4 arms, two raters, blinded and shuffled with OS entropy.

    python scripts/evaluation/swe_qa_v7_human_package.py

Writes `reports/evaluation/swe_qa_v7/human_review_v7/` once (refuses to overwrite). Rater-visible files carry no arm name. The item-to-arm key is encrypted with a
random passphrase (OpenSSL AES-256-CBC, PBKDF2); the plaintext key is never written to disk, and the passphrase file sits in `custodian/`, outside the rater folders.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import secrets
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v4_runner as v4  # noqa: E402
import swe_qa_v5_dev_context_gate as gate  # noqa: E402
import swe_qa_v7_runner as runner  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V7 = ROOT / "reports/evaluation/swe_qa_v7"
OUT = V7 / "human_review_v7"
ARMS = ("a0", "b0_k5", "a1_v4", "a1_v5a")
CITED_ARMS = ("a1_v4", "a1_v5a")
RATERS = ("author", "independent")
FORBIDDEN = re.compile(r"\b(a0|a1|a2|b0|a1_v4|a1_v5a|a1_v6e|v5a|v6e|CodeCompass|CodeQAPass|baseline)\b", re.IGNORECASE)
TASK1_COLUMNS = ["item_id", "question_id", "question", "reference_answer", "answer", "status_flag", "correctness", "completeness", "relevance", "groundedness", "refusal", "comment"]
TASK2_COLUMNS = ["item_id", "question_id", "question", "answer_being_attributed", "system_label", "citation_id", "cited_range", "support", "missing_key_source"]

INSTRUCTIONS_EN = """# SWE-QA v7 human review (Task 1 and Task 2)

You rate answers of code question-answering systems about four Python repositories. You score independently: do not use another rater's scores, the automatic judge or a language model to produce scores.

## What you see
Task 1 (`task1_items.csv`): 120 rows = 30 questions x 4 anonymous answers, shuffled. Each row has the question, the reference answer (from the benchmark) and one answer with all citation strings removed.
You do not see which system wrote an answer. You may open the repository at the snapshot commit to verify claims (commits below).

| repository | snapshot commit |
|---|---|
{commits}

## Task 1: answer quality (integers 0-10 for each of four criteria)
| Criterion | 0 | 5 | 10 |
|---|---|---|---|
| **Correctness** (are the stated facts right?) | The main claims contradict the reference answer or the code, or are invented. | About half of the key facts are right and some are wrong. | All key facts agree with the reference answer and the code; no errors. |
| **Completeness** (are the points of the reference covered?) | None of the main points of the reference answer appear. | About half of the main points appear; important ones are missing. | All main points of the reference answer appear. |
| **Relevance** (does it answer this question?) | Off topic or answers a different question. | Partly addresses the question, with substantial generic or unrelated content. | Directly answers the question; no filler. |
| **Groundedness** (is every statement verifiable in the repository?) | Names files, symbols or behaviour that do not exist or cannot be verified. | Some statements are verifiable and some are not. | Every stated file, symbol and behaviour can be verified in the repository. |

Rules:
- Judge against the reference answer and the repository, not against how confident or fluent the text sounds. Do not reward length.
- Groundedness is judged against the repository, whether or not the system had a context.
- Do not give an overall score; the mean of the four criteria is computed afterwards.
- If an answer says there is not enough evidence, score it on the four criteria like any other answer and write `yes` in the `refusal` column (otherwise `no`).
- Rows with `status_flag` other than `ok` get no scores (leave the four score cells empty); they are counted separately and never treated as zero.

## Task 2: citation support (`task2_items.csv`, code in `task2_cited_code.csv`)
For each question two answers (labels X and Y) are shown with the list of sources the system cited for its own answer; one row per citation. For each citation choose one `support` value:
- `yes`: the cited range supports a claim in the answer;
- `partial`: related, but supports only part of a claim or a minor point;
- `no`: unrelated, or does not support any claim;
- `not_applicable`: the answer makes no claim a source could support (for example a refusal); these rows leave the precision denominators.
Use `missing_key_source` once per question and label (put it on the first row of that question and label): `not_applicable`, `source_identified` (a source the answer needed is cited) or `source_missing` (a needed source is not cited). It is diagnostic only.

## Submitting
Edit only your own folder's files, keep the column names and the item order, save as CSV (UTF-8), and do not look for any key. Record the dates you scored (role only, no name).
"""

INSTRUCTIONS_FA = """# دستورالعمل بازبینی انسانی SWE-QA نسخه‌ی v7

شما پاسخ‌های سیستم‌های پرسش‌وپاسخ کد را درباره‌ی چهار repository پایتون ارزیابی می‌کنید. ارزیابی را کاملاً مستقل انجام دهید و از امتیاز داور دیگر، automatic judge یا مدل زبانی برای تولید امتیاز استفاده نکنید.

## چه چیزی می‌بینید
**Task 1** (`task1_items.csv`): ۱۲۰ ردیف = ۳۰ سؤال × ۴ پاسخ ناشناس، به‌صورت تصادفی مخلوط‌شده. هر ردیف شامل سؤال، پاسخ مرجع benchmark و یک پاسخ (بدون citation) است. نمی‌دانید هر پاسخ از کدام سیستم است. برای بررسی ادعاها می‌توانید repository را در commit زیر باز کنید:

| repository | commit |
|---|---|
{commits}

## Task 1: کیفیت پاسخ (برای هر یک از چهار معیار، عدد صحیح ۰ تا ۱۰)
- **صحت (correctness):** آیا واقعیت‌های بیان‌شده درست‌اند؟ ۰ = ادعاهای اصلی با مرجع یا کد در تناقض‌اند یا ساختگی‌اند؛ ۵ = حدود نیمی از واقعیت‌های کلیدی درست و بخشی غلط؛ ۱۰ = همه‌ی واقعیت‌های کلیدی با مرجع و کد می‌خوانند و خطایی نیست.
- **کامل بودن (completeness):** آیا نکته‌های مرجع پوشش داده شده‌اند؟ ۰ = هیچ‌کدام از نکته‌های اصلی نیست؛ ۵ = حدود نیمی از آن‌ها هست و موارد مهم جا افتاده؛ ۱۰ = همه‌ی نکته‌های اصلی هست.
- **ارتباط (relevance):** آیا به همین سؤال جواب می‌دهد؟ ۰ = بی‌ربط یا پاسخ سؤال دیگر؛ ۵ = تا حدی پاسخ می‌دهد و محتوای کلی/نامرتبط زیاد دارد؛ ۱۰ = مستقیم پاسخ می‌دهد و حشو ندارد.
- **پایه‌داری (groundedness):** آیا هر گزاره در repository قابل تأیید است؟ ۰ = فایل، نماد یا رفتاری را نام می‌برد که وجود ندارد یا قابل تأیید نیست؛ ۵ = بخشی قابل تأیید و بخشی نه؛ ۱۰ = هر فایل، نماد و رفتار ذکرشده قابل تأیید است.

قواعد:
- پاسخ را با مرجع و کد بسنجید، نه با اعتمادبه‌نفس یا روانی متن. طول را پاداش ندهید.
- پایه‌داری را نسبت به repository بسنجید، چه سیستم context داشته باشد چه نه.
- امتیاز کلی ندهید؛ میانگین چهار معیار بعداً خودکار محاسبه می‌شود.
- اگر پاسخ می‌گوید شواهد کافی نیست، آن را مثل بقیه روی چهار معیار نمره بدهید و در ستون `refusal` بنویسید `yes` (وگرنه `no`).
- ردیف‌هایی که `status_flag` آن‌ها `ok` نیست نمره نمی‌گیرند (چهار سلول خالی بماند). این‌ها جدا شمرده می‌شوند و هرگز صفر حساب نمی‌شوند.

## Task 2: پشتیبانی citation (`task2_items.csv` و کد در `task2_cited_code.csv`)
برای هر سؤال دو پاسخ (برچسب X و Y) همراه فهرست منابعی که سیستم برای همان پاسخ آورده نشان داده می‌شود؛ هر citation یک ردیف است. برای هر citation یکی از مقادیر `support` را بنویسید:
- `yes`: بازه‌ی ذکرشده یکی از ادعاهای پاسخ را پشتیبانی می‌کند.
- `partial`: مرتبط است ولی فقط بخشی از یک ادعا یا نکته‌ی فرعی را پشتیبانی می‌کند.
- `no`: نامرتبط است یا هیچ ادعایی را پشتیبانی نمی‌کند.
- `not_applicable`: پاسخ ادعایی ندارد که منبعی بتواند پشتیبانی کند (مثلاً امتناع)؛ این ردیف‌ها از مخرج دقت کنار می‌روند.
ستون `missing_key_source` را برای هر سؤال و هر برچسب فقط یک بار (روی اولین ردیف همان سؤال و برچسب) پر کنید: `not_applicable`، `source_identified` (منبعی که پاسخ لازم داشت ذکر شده) یا `source_missing` (منبع لازم ذکر نشده). این ستون فقط تشخیصی است.

## تحویل
فقط فایل‌های پوشه‌ی خودتان را ویرایش کنید، نام ستون‌ها و ترتیب ردیف‌ها را حفظ کنید، به‌صورت CSV (UTF-8) ذخیره کنید و دنبال هیچ کلیدی نگردید. تاریخ‌هایی را که امتیاز داده‌اید ثبت کنید (فقط نقش، بدون نام).
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_answers() -> dict[str, dict[str, dict]]:
    v4frozen = {"a0": "main_a0_combined.jsonl", "b0_k5": "main_b0_k5_combined.jsonl", "a1_v4": "main_a1_combined.jsonl"}
    out: dict[str, dict[str, dict]] = {arm: {r["question_id"]: r for r in runner.read_jsonl(gate.V4 / name)} for arm, name in v4frozen.items()}
    out["a1_v5a"] = runner.latest_generation("a1_v5a")
    return out


def status_flag(record: dict) -> str:
    if record.get("status") == "no_evidence":
        return "ok"
    if record.get("status") != "usable" or record.get("finish_reason") not in (None, "stop") or not (record.get("answer") or "").strip():
        return "truncated" if record.get("finish_reason") == "length" else "unavailable"
    return "ok"


def build_rater(rater: str, ids: list[str], rows: dict, answers: dict, contexts: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    rng = secrets.SystemRandom()
    pairs = [(q, arm) for q in ids for arm in ARMS]
    rng.shuffle(pairs)
    used: set[str] = set()

    def new_id() -> str:
        while True:
            value = secrets.token_hex(8)
            if value not in used:
                used.add(value)
                return value

    task1, key1 = [], {}
    for q, arm in pairs:
        record = answers[arm][q]
        item = new_id()
        task1.append({"item_id": item, "question_id": q, "question": rows[q]["question"], "reference_answer": rows[q].get("answer") or "",
                      "answer": runner.strip_citations(record.get("answer")), "status_flag": status_flag(record),
                      **{c: "" for c in ("correctness", "completeness", "relevance", "groundedness", "refusal", "comment")}})
        key1[item] = {"question_id": q, "arm": arm}
    task2, code_rows, key2 = [], [], {"labels": {}, "citations": {}}
    order = list(ids)
    rng.shuffle(order)
    for q in order:
        labels = list(CITED_ARMS)
        rng.shuffle(labels)
        key2["labels"][q] = {"X": labels[0], "Y": labels[1]}
        for label, arm in zip(("X", "Y"), labels):
            record = answers[arm][q]
            blocks = {b["chunk_id"]: b for b in contexts[(q, arm)]["blocks"]}
            for citation in record.get("citations") or []:
                cid = new_id()
                block = blocks.get(citation["chunk_id"])
                span = f'{citation["source_file"]}:{citation["start_line"]}-{citation["end_line"]}'
                task2.append({"item_id": new_id(), "question_id": q, "question": rows[q]["question"], "answer_being_attributed": runner.strip_citations(record.get("answer")),
                              "system_label": label, "citation_id": cid, "cited_range": span, "support": "", "missing_key_source": ""})
                code_rows.append({"citation_id": cid, "cited_range": span, "code": block["code"] if block else ""})
                key2["citations"][cid] = {"question_id": q, "arm": arm, "label": label}
    return task1, task2, code_rows, {"task1": key1, "task2": key2}


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if (OUT / "package_manifest.json").exists():
        raise SystemExit(f"refusing to overwrite the finished package in {OUT}")  # an interrupted, unfinished build (no manifest) may be redone
    runner.require_freeze()
    ids = json.loads((runner.PRE / "subsets.json").read_text(encoding="utf-8"))["human_review_sample"]
    assert len(ids) == 30
    rows = runner.question_rows()
    answers = read_answers()
    contexts = runner.study1_contexts()
    commits = "\n".join(f"| {repo} | `{v4.COMMITS[repo]}` |" for repo in runner.REPOS)
    (OUT / "custodian").mkdir(parents=True, exist_ok=True)
    key = {}
    manifest: dict = {"questions": 30, "arms": list(ARMS), "raters": list(RATERS), "files": {}, "known_leaks": {}}
    for rater in RATERS:
        folder = OUT / rater
        folder.mkdir(exist_ok=True)
        task1, task2, code_rows, rater_key = build_rater(rater, ids, rows, answers, contexts)
        key[rater] = rater_key
        write_csv(folder / "task1_items.csv", TASK1_COLUMNS, task1)
        write_csv(folder / "task2_items.csv", TASK2_COLUMNS, task2)
        write_csv(folder / "task2_cited_code.csv", ["citation_id", "cited_range", "code"], code_rows)
        (folder / "INSTRUCTIONS.md").write_text(INSTRUCTIONS_EN.format(commits=commits), encoding="utf-8")
        (folder / "INSTRUCTIONS_FA.md").write_text(INSTRUCTIONS_FA.format(commits=commits), encoding="utf-8")
        # blinding and count checks (they must pass before the package is sent)
        visible = "\n".join(path.read_text(encoding="utf-8-sig") for path in folder.iterdir() if path.suffix in {".csv", ".md"} and path.name != "task2_cited_code.csv")
        header_hits = [c for c in TASK1_COLUMNS + TASK2_COLUMNS if FORBIDDEN.search(c)]
        assert len(task1) == 120 and len({r["question_id"] for r in task1}) == 30 and all(sum(1 for r in task1 if r["question_id"] == q) == 4 for q in ids)
        assert not header_hits, header_hits
        text_hits = sorted({m.group(0) for m in FORBIDDEN.finditer("\n".join(r.get("answer", r.get("answer_being_attributed", "")) + " " + r["question"] for r in task1 + task2))})
        manifest["files"][rater] = {name: sha256_bytes((folder / name).read_bytes()) for name in ("task1_items.csv", "task2_items.csv", "task2_cited_code.csv", "INSTRUCTIONS.md", "INSTRUCTIONS_FA.md")}
        manifest["files"][rater]["task1_rows"] = len(task1)
        manifest["files"][rater]["task2_rows"] = len(task2)
        manifest["files"][rater]["status_flags"] = {s: sum(1 for r in task1 if r["status_flag"] == s) for s in sorted({r["status_flag"] for r in task1})}
        manifest["files"][rater]["arm_name_matches_inside_question_or_answer_text"] = text_hits
        assert len(visible) > 0
    # known leaks, measured per arm (not edited out)
    for arm in ARMS:
        texts = [runner.strip_citations(answers[arm][q].get("answer")) for q in ids]
        manifest["known_leaks"][arm] = {"median_answer_chars": sorted(len(t) for t in texts)[len(texts) // 2], "answers_mentioning_the_context": sum(1 for t in texts if re.search(r"\bthe (provided )?context\b", t, re.IGNORECASE))}
    # encrypted key: plaintext only travels through the pipe, never through a file
    passphrase = secrets.token_urlsafe(24)
    (OUT / "custodian" / "passphrase.txt").write_text(passphrase + "\n", encoding="utf-8")
    cipher = subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-pass", f"file:{OUT / 'custodian' / 'passphrase.txt'}"], input=json.dumps(key).encode("utf-8"),
                            capture_output=True, check=True).stdout
    (OUT / "custodian" / "key.enc").write_bytes(cipher)
    manifest["key_ciphertext_sha256"] = sha256_bytes(cipher)
    (OUT / "custodian" / "README.md").write_text(
        "# Key custody\n\n`key.enc` is the encrypted item-to-arm key (OpenSSL AES-256-CBC, PBKDF2); `passphrase.txt` opens it. A rater must not hold the passphrase: hand `passphrase.txt` to a person who is not a rater (for example the supervisor), "
        "delete the local copy, and keep `key.enc` anywhere. The custodian releases the passphrase only after both score files are submitted and validated.\n\nOpen with: `openssl enc -d -aes-256-cbc -pbkdf2 -in key.enc -pass file:passphrase.txt`\n", encoding="utf-8")
    (OUT / "package_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"task1_rows_per_rater": 120, "task2_rows": {r: manifest["files"][r]["task2_rows"] for r in RATERS}, "status_flags": {r: manifest["files"][r]["status_flags"] for r in RATERS},
                      "arm_name_matches": {r: manifest["files"][r]["arm_name_matches_inside_question_or_answer_text"] for r in RATERS}, "known_leaks": manifest["known_leaks"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
