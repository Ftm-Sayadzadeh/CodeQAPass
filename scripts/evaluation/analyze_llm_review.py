"""Analyze the blinded LLM-reviewer scores against the key (run only after scoring is finished)."""
import json, csv, random, math, statistics as S, collections as C
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "reports/evaluation/swe_qa_v2"
RUNS = {"codex": D / "reviewer_runs/codex", "claude": D / "reviewer_runs/claude"}
key = json.load(open(D / "human_sample_key.json"))


def load(p):
    t = Path(p).read_text(encoding="utf-8-sig")
    return json.loads(t)


def unwrap(d, name):
    if isinstance(d, dict):
        for k in ("items", "scores", "results", name):
            if k in d and isinstance(d[k], list):
                return d[k]
    return d


csv1 = {r["item_id"]: r for r in csv.DictReader(open(D / "human_sample_blinded.csv", encoding="utf-8", newline=""))}
csv2 = {r["item_id"]: r for r in csv.DictReader(open(D / "human_citation_sample_blinded.csv", encoding="utf-8", newline=""))}
T1, T2 = {}, {}
for name, p in RUNS.items():
    a = unwrap(load(p / "task1_scores.json"), "task1")
    b = unwrap(load(p / "task2_scores.json"), "task2")
    print(name, type(a).__name__, len(a), type(b).__name__, len(b))
    ids1 = [x["item_id"] for x in a]
    ids2 = [x["item_id"] for x in b]
    assert len(ids1) == len(set(ids1)) == 60 and set(ids1) == set(csv1), (name, "task1 ids")
    assert len(ids2) == len(set(ids2)) == 60 and set(ids2) == set(csv2), (name, "task2 ids")
    for x in a:
        for k in ("correctness", "completeness", "relevance", "groundedness", "score_0_10"):
            assert isinstance(x[k], int) and 0 <= x[k] <= 10, (name, x["item_id"], k, x[k])
        assert x["hallucination"] in ("none", "mild", "explicit")
    bad = 0
    for x in b:
        n = int(csv2[x["item_id"]]["n_citations"])
        if x["n_supporting_citations"] + x["n_unsupported_citations"] != n:
            bad += 1
            print(" mismatch", name, x["item_id"], x["n_supporting_citations"], x["n_unsupported_citations"], n)
        assert x["missing_key_source_0_or_1"] in (0, 1)
    print(name, "task2 sum mismatches:", bad)
    T1[name] = {x["item_id"]: x for x in a}
    T2[name] = {x["item_id"]: x for x in b}


def arm(section, i, xy):
    k = key[section][i]
    return k["system_x"] if xy == "X" else k["system_y"]


DIMS = ["correctness", "completeness", "relevance", "groundedness", "score_0_10"]


def boot(diffs, seed=20260930, n=20000):
    rng = random.Random(seed)
    b = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return [b[int(.025 * n)], b[int(.975 * n) - 1]]


def wilcoxon(diffs):
    nz = [d for d in diffs if d]
    n = len(nz)
    if n < 10:
        return None
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    tie = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        tie += t ** 3 - t
        i = j + 1
    w = sum(r for r, d in zip(ranks, nz) if d > 0)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - tie / 48
    z = (abs(w - mu) - .5) / math.sqrt(var)
    return math.erfc(z / math.sqrt(2))


def pearson(x, y):
    mx, my = S.mean(x), S.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    return pearson(rank(x), rank(y))


def qwk(x, y, lo=0, hi=10):
    n = len(x)
    k = hi - lo + 1
    O = [[0] * k for _ in range(k)]
    for a, b in zip(x, y):
        O[a - lo][b - lo] += 1
    hx = [sum(r) for r in O]
    hy = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * O[i][j]
            den += w * hx[i] * hy[j] / n
    return 1 - num / den if den else float("nan")


out = {"reviewers": {"codex": "GPT-5 (Codex), version not exposed (user reports 5.6 sol medium)", "claude": "claude-sonnet-5-5 (medium)"}}
# ---- Task 1
res1 = {}
q1 = {}
for name in RUNS:
    per = {}
    for iid, x in T1[name].items():
        i, xy = iid[:-1], iid[-1]
        a = arm("answer_sample", i, xy)
        per.setdefault(i, {})[a] = x
    q1[name] = per
for name in list(RUNS) + ["mean_of_both"]:
    res1[name] = {}
    for d in DIMS:
        diffs, a0, a1 = [], [], []
        for i in key["answer_sample"]:
            if name == "mean_of_both":
                v0 = S.mean(q1[r][i]["A0N"][d] for r in RUNS)
                v1 = S.mean(q1[r][i]["A1"][d] for r in RUNS)
            else:
                v0, v1 = q1[name][i]["A0N"][d], q1[name][i]["A1"][d]
            a0.append(v0); a1.append(v1); diffs.append(v1 - v0)
        res1[name][d] = {"a0n": S.mean(a0), "a1": S.mean(a1), "diff": S.mean(diffs), "boot95": boot(diffs),
                         "a1_better": sum(x > 0 for x in diffs), "a1_worse": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs),
                         "wilcoxon_p": wilcoxon(diffs)}
out["task1_quality_A1_minus_A0N"] = res1
# hallucination by arm
hall = {}
for name in RUNS:
    c = C.Counter()
    for i, per in q1[name].items():
        for a, x in per.items():
            c[(a, x["hallucination"])] += 1
    hall[name] = {f"{a}:{h}": v for (a, h), v in sorted(c.items())}
out["hallucination_counts"] = hall
# agreement task1 (per row, blind to arm)
ids = sorted(csv1)
ag = {}
for d in DIMS:
    x = [T1["codex"][i][d] for i in ids]
    y = [T1["claude"][i][d] for i in ids]
    ag[d] = {"pearson": pearson(x, y), "spearman": spearman(x, y), "qwk": qwk(x, y),
             "exact": sum(a == b for a, b in zip(x, y)) / 60, "within1": sum(abs(a - b) <= 1 for a, b in zip(x, y)) / 60,
             "mean_abs_diff": S.mean(abs(a - b) for a, b in zip(x, y)), "mean_codex": S.mean(x), "mean_claude": S.mean(y)}
out["task1_agreement"] = ag
h = [T1["codex"][i]["hallucination"] == T1["claude"][i]["hallucination"] for i in ids]
out["hallucination_label_exact_agreement"] = sum(h) / 60
# pairwise preference agreement: which arm each reviewer prefers on score_0_10
pref = {}
for name in RUNS:
    pref[name] = {i: (q1[name][i]["A1"]["score_0_10"] > q1[name][i]["A0N"]["score_0_10"]) - (q1[name][i]["A1"]["score_0_10"] < q1[name][i]["A0N"]["score_0_10"]) for i in key["answer_sample"]}
out["preference_agreement_score_0_10"] = sum(pref["codex"][i] == pref["claude"][i] for i in pref["codex"]) / 30
# by repo (mean of both)
rep = {}
for repo in ("flask", "requests"):
    d_ = [S.mean(q1[r][i]["A1"]["score_0_10"] - q1[r][i]["A0N"]["score_0_10"] for r in RUNS) for i in key["answer_sample"] if key["answer_sample"][i]["question_id"].startswith(repo)]
    rep[repo] = {"n": len(d_), "mean_diff_A1_minus_A0N": S.mean(d_)}
out["by_repo_score_0_10"] = rep
# position bias: X vs Y
posb = {}
for name in RUNS:
    xs = [T1[name][i]["score_0_10"] for i in ids if i.endswith("X")]
    ys = [T1[name][i]["score_0_10"] for i in ids if i.endswith("Y")]
    posb[name] = {"mean_X": S.mean(xs), "mean_Y": S.mean(ys)}
out["position_bias_score_0_10"] = posb
# ---- Task 2
res2 = {}
for name in RUNS:
    tot = {"A1": C.Counter(), "A2": C.Counter()}
    perq = {"A1": [], "A2": []}
    for iid, x in T2[name].items():
        i, xy = iid[:-1], iid[-1]
        a = arm("citation_sample", i, xy)
        n = int(csv2[iid]["n_citations"])
        tot[a]["items"] += 1
        tot[a]["citations"] += n
        tot[a]["supporting"] += x["n_supporting_citations"]
        tot[a]["unsupported"] += x["n_unsupported_citations"]
        tot[a]["missing_key"] += x["missing_key_source_0_or_1"]
        perq[a].append((iid, x["n_supporting_citations"] / n if n else None, x["missing_key_source_0_or_1"]))
    res2[name] = {a: {**dict(t), "precision": t["supporting"] / t["citations"], "missing_key_rate": t["missing_key"] / t["items"]} for a, t in tot.items()}
out["task2_citation_by_reviewer"] = res2
# paired precision diff A2-A1 per question (mean of both reviewers)
pq = {}
for i in key["citation_sample"]:
    row = {}
    for name in RUNS:
        for xy in "XY":
            iid = f"{i}{xy}"
            a = arm("citation_sample", i, xy)
            n = int(csv2[iid]["n_citations"])
            row.setdefault(a, []).append((T2[name][iid]["n_supporting_citations"] / n, T2[name][iid]["missing_key_source_0_or_1"]))
    pq[i] = {a: (S.mean(p for p, _ in v), S.mean(m for _, m in v)) for a, v in row.items()}
dp = [pq[i]["A2"][0] - pq[i]["A1"][0] for i in pq]
dm = [pq[i]["A2"][1] - pq[i]["A1"][1] for i in pq]
out["task2_paired_A2_minus_A1"] = {"precision_diff": S.mean(dp), "precision_boot95": boot(dp), "precision_wilcoxon_p": wilcoxon(dp),
                                   "missing_key_diff": S.mean(dm), "missing_boot95": boot(dm), "missing_wilcoxon_p": wilcoxon(dm)}
# agreement on task2 counts
x = [T2["codex"][i]["n_supporting_citations"] for i in sorted(csv2)]
y = [T2["claude"][i]["n_supporting_citations"] for i in sorted(csv2)]
xm = [T2["codex"][i]["missing_key_source_0_or_1"] for i in sorted(csv2)]
ym = [T2["claude"][i]["missing_key_source_0_or_1"] for i in sorted(csv2)]
po = sum(a == b for a, b in zip(xm, ym)) / 60
p1, p2 = S.mean(xm), S.mean(ym)
pe = p1 * p2 + (1 - p1) * (1 - p2)
out["task2_agreement"] = {"supporting_count_exact": sum(a == b for a, b in zip(x, y)) / 60, "supporting_count_spearman": spearman(x, y),
                          "missing_key_exact": po, "missing_key_cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else None}
# ---- Compare with Gemini judge
js = json.load(open(D / "judge_summary.json"))
judge = {}
for r in map(json.loads, open(D / "judge_scores.jsonl", encoding="utf-8")):
    if r["status"] == "usable":
        judge[(r["question_id"], r["arm"])] = r
jd, hd = [], []
for i, k in key["answer_sample"].items():
    q = k["question_id"]
    jd.append(judge[(q, "a1_codeqapass")]["mean"] - judge[(q, "a0_neutral")]["mean"])
    hd.append(S.mean(q1[r][i]["A1"]["score_0_10"] - q1[r][i]["A0N"]["score_0_10"] for r in RUNS))
out["judge_vs_llm_reviewers_on_30"] = {"spearman_of_paired_diffs": spearman(jd, hd), "pearson_of_paired_diffs": pearson(jd, hd),
                                       "sign_agreement": sum((a > 0) == (b > 0) for a, b in zip(jd, hd) if a and b) / max(1, sum(1 for a, b in zip(jd, hd) if a and b))}
# stated cue-check: does either reviewer score correlate with answer length
lens = {iid: len(csv1[iid]["answer"]) for iid in csv1}
out["length_score_spearman"] = {name: spearman([lens[i] for i in ids], [T1[name][i]["score_0_10"] for i in ids]) for name in RUNS}
# task2 identical-answers claim
out["task2_identical_ranges_pairs"] = sum(csv2[f"{i}X"]["cited_ranges"] == csv2[f"{i}Y"]["cited_ranges"] for i in key["citation_sample"])
(D / "llm_review_summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=lambda o: round(o, 3) if isinstance(o, float) else str(o)))
