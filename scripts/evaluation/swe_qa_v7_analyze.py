"""SWE-QA v7 analysis (protocol sections 6, 8, 9): quality, reliability, efficiency. Reads only the frozen inputs and the append-only run records.

    python scripts/evaluation/swe_qa_v7_analyze.py

Writes `reports/evaluation/swe_qa_v7/main/analysis_v7.json` once and refuses to overwrite it. Unavailable records are excluded pairwise and counted, never scored as zero.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v4_judge as v4judge  # noqa: E402
import swe_qa_v5_confirm as c5  # noqa: E402
import swe_qa_v5_dev_context_gate as gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V7 = ROOT / "reports/evaluation/swe_qa_v7"
OUT = V7 / "main"
PRE = V7 / "preflight_final"
V4F = gate.V4
REPOS = gate.REPOS
ARMS = ("a0", "b0_k5", "a1_v4", "a1_v5a", "a1_v6e")
NEW_ARMS = ("a1_v5a", "a1_v6e")
CRITERIA = v4judge.SCORE_KEYS
SEED = 20260930
RESAMPLES = 20_000
NI_MARGIN = 1.5
GOOD = 15.0
PRIMARY = {"H1": ("a1_v5a", "a1_v4"), "H2": ("a1_v5a", "b0_k5")}
SECONDARY = {"H3": ("a1_v6e", "a1_v5a"), "H4": ("a1_v6e", "b0_k5")}
HEDGE_RE = re.compile(r"not enough retrieved evidence|not enough evidence|insufficient|cannot be (determined|established)|not (shown|visible|included|provided) in (the )?(provided|retrieved)", re.IGNORECASE)
SKIP = gate.IDENT_SKIP | {"print", "return", "class", "def", "import", "from", "lambda", "yield", "async", "await"}
EFFICIENCY_PAIRS = (("a1_v5a", "b0_k5"), ("a1_v5a", "a1_v4"), ("a1_v5a", "a0"), ("a1_v6e", "a1_v5a"), ("a1_v6e", "b0_k5"))
STAGES = ("total_seconds", "retrieval_assembly_seconds", "generation_seconds")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


# ---------------------------------------------------------------- statistics
def bootstrap_interval(groups: dict[str, list[float]], statistic=stats.mean, level: float = 0.95, resamples: int = RESAMPLES) -> list[float]:
    """Stratified (by repository) bootstrap percentile interval of `statistic` over the pooled values."""
    rng = random.Random(SEED)
    draws = sorted(statistic([x for g in groups.values() for x in rng.choices(g, k=len(g))]) for _ in range(resamples))
    tail = (1 - level) / 2
    return [round(draws[int(tail * resamples)], 4), round(draws[int((1 - tail) * resamples) - 1], 4)]


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    return v4judge.holm(p_values)


def sign_test_p(deltas: list[float]) -> float | None:
    nonzero = [d for d in deltas if d]
    n = len(nonzero)
    if not n:
        return None
    k = sum(d > 0 for d in nonzero)
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar_exact_p(only_a: int, only_b: int) -> float | None:
    n = only_a + only_b
    if not n:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n)


def wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(centre - half, 4), round(centre + half, 4)]


def spearman(a: list[float], b: list[float]) -> float | None:
    return c5.spearman(a, b)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low, high = int(math.floor(position)), int(math.ceil(position))
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 4)


def paired_summary(pairs: list[tuple[str, str, float, float]]) -> dict:
    """pairs: (question_id, repo, value_a, value_b). Mean paired difference with both intervals, Wilcoxon, effect size, win/tie/loss and sign test."""
    groups: dict[str, list[float]] = {}
    for _, repo, a, b in pairs:
        groups.setdefault(repo, []).append(a - b)
    deltas = [d for g in groups.values() for d in g]
    if len(deltas) < 2:
        return {"pairs": len(deltas), "status": "insufficient"}
    sd = stats.stdev(deltas)
    return {"pairs": len(deltas), "mean_difference": round(stats.mean(deltas), 4), "ci95": bootstrap_interval(groups), "ci97_5": bootstrap_interval(groups, level=0.975),
            "wilcoxon_p": v4judge.wilcoxon_p(deltas), "sign_test_p": sign_test_p(deltas), "standardized_effect_dz": round(stats.mean(deltas) / sd, 4) if sd else None,
            "wins": sum(d > 0 for d in deltas), "ties": sum(d == 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
            "by_repo": {repo: {"n": len(g), "mean_difference": round(stats.mean(g), 4), "ci95": bootstrap_interval({repo: g}) if len(g) > 1 else None} for repo, g in sorted(groups.items())}}


def decide(contrast: dict, margin: float | None = None) -> dict:
    """Decision basis fixed in the protocol: Holm-adjusted Wilcoxon p below 0.05 and a 95% interval excluding 0; non-inferiority by the 97.5% lower limit."""
    out: dict = {}
    if contrast.get("mean_difference") is None:
        return {"superiority_supported": None}
    adjusted = contrast.get("holm_adjusted_p")
    out["superiority_supported"] = bool(adjusted is not None and adjusted < 0.05 and contrast["ci95"][0] > 0)
    if margin is not None:
        out["non_inferiority_margin"] = -margin
        out["non_inferior"] = contrast["ci97_5"][0] > -margin
    return out


# ---------------------------------------------------------------- data
def question_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for repo in REPOS:
        for number, row in enumerate(read_jsonl(gate.DATASET / f"{repo}.jsonl"), 1):
            rows[f"{repo}-{number}"] = {"question_id": f"{repo}-{number}", "repo": repo, **row}
    return rows


def answers() -> dict[str, dict[str, dict]]:
    """arm -> question id -> answer record (new arms from the v7 generation files; old arms from the v4 frozen files)."""
    out: dict[str, dict[str, dict]] = {a: {} for a in ARMS}
    old = {"a0": "main_a0_combined.jsonl", "b0_k5": "main_b0_k5_combined.jsonl", "a1_v4": "main_a1_combined.jsonl"}
    for arm, name in old.items():
        out[arm] = {r["question_id"]: r for r in read_jsonl(V4F / name)}
    for arm in NEW_ARMS:
        for retry in (False, True):
            for r in read_jsonl(OUT / f"study1_generation_{arm}{'_retry' if retry else ''}.jsonl"):
                if not retry or r.get("status") != "unavailable":
                    out[arm][r["question_id"]] = r
    return out


def new_batch_scores() -> tuple[dict[tuple[str, str], dict], dict]:
    key = json.loads((OUT / "judge_key.json").read_text(encoding="utf-8"))["mapping"]
    passes: dict[tuple[str, str], dict[int, dict]] = {}
    spend = 0.0
    counts = {1: 0, 2: 0}
    for number in (1, 2):
        latest = {r["blind_id"]: r for r in read_jsonl(OUT / f"judge_pass{number}.jsonl") if r.get("status") == "usable"}
        counts[number] = len(latest)
        for blind, r in latest.items():
            info = key[blind]
            passes.setdefault((info["question_id"], info["arm"]), {})[number] = r
            spend += (r.get("usage") or {}).get("provider_cost_unit") or 0.0
    units = {unit: {"mean": stats.mean(p["mean"] for p in ps.values()), "passes": len(ps), **{k: stats.mean(p["scores"][k] for p in ps.values()) for k in CRITERIA}} for unit, ps in passes.items()}
    return units, {"usable_records_pass1": counts[1], "usable_records_pass2": counts[2], "judge_spend_provider_cost_unit": round(spend, 6)}


def v4_frozen_scores() -> dict[tuple[str, str], float]:
    key = json.loads((V4F / "judge_key.json").read_text(encoding="utf-8"))["mapping"]
    arm_name = {"a1": "a1_v4"}
    latest: dict[str, dict] = {}
    for name in ("judge_scores.jsonl", "judge_scores_retry.jsonl", "judge_scores_retry2.jsonl"):
        for r in read_jsonl(V4F / name):
            if r.get("status") == "usable":
                latest[r["blind_id"]] = r
    return {(key[b]["question_id"], arm_name.get(key[b]["arm"], key[b]["arm"])): r["mean"] for b, r in latest.items() if b in key}


def unsupported_share(answer: str, context_text: str, question: str) -> tuple[int, int]:
    names = {n.lower() for token in re.findall(r"`([^`\n]+)`", answer or "") for n in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", token.replace("()", "")) if len(n) >= 4 and n.lower() not in SKIP}
    haystack = (context_text + "\n" + question).lower()
    return sum(1 for n in names if n not in haystack), len(names)


def is_hedged(record: dict) -> bool:
    return not c5.is_refusal(record) and bool(HEDGE_RE.search(record.get("answer") or ""))


# ---------------------------------------------------------------- quality analysis
def pairs_for(units: dict, rows: dict, left: str, right: str, metric: str = "mean") -> list[tuple[str, str, float, float]]:
    return [(q, rows[q]["repo"], units[(q, left)][metric], units[(q, right)][metric]) for q in sorted(rows) if (q, left) in units and (q, right) in units]


def quality_analysis(units: dict, rows: dict, frozen: dict, recs: dict) -> dict:
    result: dict = {"arms": {}, "primary": {}, "secondary": {}, "exploratory": {}, "criteria": {}, "sensitivity_v4_frozen_scores": {}}
    for arm in ARMS:
        scored = [q for q in rows if (q, arm) in units]
        values = [units[(q, arm)]["mean"] for q in scored]
        result["arms"][arm] = {"judged": len(scored), "mean": round(stats.mean(values), 3) if values else None,
                               "by_repo": {repo: round(stats.mean(units[(q, arm)]["mean"] for q in scored if rows[q]["repo"] == repo), 3) for repo in REPOS if any(rows[q]["repo"] == repo for q in scored)},
                               "criteria": {k: round(stats.mean(units[(q, arm)][k] for q in scored), 3) for k in CRITERIA} if scored else None,
                               "statuses": {s: sum(1 for r in recs[arm].values() if r["status"] == s) for s in sorted({r["status"] for r in recs[arm].values()})}}
    for family, hypotheses in (("primary", PRIMARY), ("secondary", SECONDARY)):
        contrasts = {h: paired_summary(pairs_for(units, rows, a, b)) for h, (a, b) in hypotheses.items()}
        adjusted = holm({h: c.get("wilcoxon_p") for h, c in contrasts.items()})
        for h, c in contrasts.items():
            c["holm_adjusted_p"] = adjusted[h]
            c["comparison"] = " - ".join(hypotheses[h])
            c["decision"] = decide(c, NI_MARGIN if h == "H2" else None)
            result[family][h] = c
    listed = {tuple(v) for v in {**PRIMARY, **SECONDARY}.values()}
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            pair = (a, b) if (a, b) in listed else ((b, a) if (b, a) in listed else None)
            if pair is None:
                result["exploratory"][f"{a}_minus_{b}"] = paired_summary(pairs_for(units, rows, a, b))
    for h, (a, b) in {**PRIMARY, **SECONDARY}.items():
        result["criteria"][h] = {k: paired_summary(pairs_for(units, rows, a, b, k)) for k in CRITERIA}
    # sensitivity: primary contrasts with the v4 frozen scores for the old arms (new arms keep their new-batch scores)
    mixed = {(q, arm): {"mean": frozen[(q, arm)]} for (q, arm) in frozen if arm in ("a0", "b0_k5", "a1_v4")}
    mixed.update({(q, arm): v for (q, arm), v in units.items() if arm in NEW_ARMS})
    for h, (a, b) in PRIMARY.items():
        result["sensitivity_v4_frozen_scores"][h] = paired_summary(pairs_for(mixed, rows, a, b))
    return result


def reliability_analysis(units: dict, frozen: dict, rows: dict) -> dict:
    key = json.loads((OUT / "judge_key.json").read_text(encoding="utf-8"))["mapping"]
    p1 = {r["blind_id"]: r["mean"] for r in read_jsonl(OUT / "judge_pass1.jsonl") if r.get("status") == "usable"}
    p2 = {r["blind_id"]: r["mean"] for r in read_jsonl(OUT / "judge_pass2.jsonl") if r.get("status") == "usable"}
    common = sorted(set(p1) & set(p2))
    out = {"pass_agreement": {"n": len(common), "spearman": spearman([p1[b] for b in common], [p2[b] for b in common]),
                              "mean_abs_difference": round(stats.mean(abs(p1[b] - p2[b]) for b in common), 4) if common else None}}
    stability = [(q, rows[q]["repo"], units[(q, arm)]["mean"], frozen[(q, arm)]) for (q, arm) in frozen if (q, arm) in units and arm in ("a0", "b0_k5", "a1_v4")]
    out["v4_batch_vs_new_batch_old_answers"] = {**paired_summary(stability), "spearman": spearman([x[2] for x in stability], [x[3] for x in stability]) if stability else None}
    return out


def behaviour_analysis(units: dict, rows: dict, recs: dict, contexts: dict) -> dict:
    out: dict = {"arms": {}, "pairwise": {}, "length": {}}
    for arm in ARMS:
        rs = recs[arm]
        n = len(rs)
        refusals = sum(c5.is_refusal(r) for r in rs.values())
        hedged = sum(is_hedged(r) for r in rs.values())
        good = [q for q in rows if (q, arm) in units]
        goods = sum(units[(q, arm)]["mean"] >= GOOD for q in good)
        entry = {"answers": n, "refusal": {"count": refusals, "rate": round(refusals / n, 4) if n else None, "wilson95": wilson(refusals, n)},
                 "hedged": {"count": hedged, "rate": round(hedged / n, 4) if n else None, "wilson95": wilson(hedged, n)},
                 "good_answer": {"count": goods, "rate": round(goods / len(good), 4) if good else None, "wilson95": wilson(goods, len(good))},
                 "answer_chars_median": percentile([len(r.get("answer") or "") for r in rs.values()], 0.5)}
        if arm != "a0":
            unsupported = total = 0
            for q, r in rs.items():
                blocks = contexts.get((q, arm), {}).get("blocks", [])
                u, t = unsupported_share(r.get("answer") or "", "\n".join(b["code"] for b in blocks), rows[q]["question"])
                unsupported, total = unsupported + u, total + t
            entry["unsupported_identifier_share"] = {"unsupported": unsupported, "checked": total, "share": round(unsupported / total, 4) if total else None}
        out["arms"][arm] = entry
        lengths = [(len(rs[q].get("answer") or ""), units[(q, arm)]["mean"]) for q in good if q in rs]
        out["length"][arm] = {"spearman_length_score": spearman([x[0] for x in lengths], [x[1] for x in lengths]) if lengths else None}
    for h, (a, b) in {**PRIMARY, **SECONDARY}.items():
        for name, flag in (("refusal", c5.is_refusal), ("good_answer", None)):
            only_a = only_b = 0
            for q in rows:
                if q in recs[a] and q in recs[b] and ((q, a) in units if flag is None else True) and ((q, b) in units if flag is None else True):
                    fa = c5.is_refusal(recs[a][q]) if flag else units[(q, a)]["mean"] >= GOOD
                    fb = c5.is_refusal(recs[b][q]) if flag else units[(q, b)]["mean"] >= GOOD
                    only_a += fa and not fb
                    only_b += fb and not fa
            out["pairwise"][f"{h}_{name}"] = {"comparison": f"{a} vs {b}", "only_first": only_a, "only_second": only_b, "exact_mcnemar_p": mcnemar_exact_p(only_a, only_b)}
        # length-tercile sensitivity of the primary contrasts
        pairs = [(q, rows[q]["repo"], units[(q, a)]["mean"], units[(q, b)]["mean"], (len(recs[a][q].get("answer") or "") + len(recs[b][q].get("answer") or "")) / 2)
                 for q in sorted(rows) if (q, a) in units and (q, b) in units and q in recs[a] and q in recs[b]]
        if len(pairs) >= 9:
            order = sorted(pairs, key=lambda p: p[4])
            third = len(order) // 3
            out["length"][f"{h}_terciles"] = {name: paired_summary([(p[0], p[1], p[2], p[3]) for p in chunk]) for name, chunk in (("short", order[:third]), ("middle", order[third:2 * third]), ("long", order[2 * third:]))}
    return out


# ---------------------------------------------------------------- efficiency analysis
def list_cost(usage: dict, embedding: dict, prices: dict, cache_adjusted: bool) -> float | None:
    if not usage.get("usage_available") or not embedding.get("usage_available", True):
        return None
    glm, emb = prices["pricing"]["glm-5.3-flash"], prices["pricing"]["gemini-embedding-2"]
    prompt, cached = usage["prompt_tokens"], usage["cached_tokens"]
    input_cost = ((prompt - cached) * glm["input"] + cached * glm["cached_input"]) if cache_adjusted else prompt * glm["input"]
    embed_tokens = embedding.get("prompt_tokens") or 0
    return (input_cost + usage["completion_tokens"] * glm["output"] + embed_tokens * emb["input"]) / 1e6


def efficiency_analysis(rows: dict, prices: dict) -> dict:
    records = [r for r in read_jsonl(OUT / "study2_timing.jsonl") if not r.get("warmup")]
    ok = [r for r in records if r.get("status") == "ok"]
    out: dict = {"attempts": {a: {"ok": sum(1 for r in ok if r["arm"] == a), "failed": sum(1 for r in records if r["arm"] == a and r.get("status") != "ok")} for a in ARMS}, "arms": {}, "contrasts": {}}
    metrics = {s: (lambda r, s=s: r.get(s)) for s in STAGES}
    metrics.update({"prompt_tokens": lambda r: r["llm_usage"]["prompt_tokens"], "completion_tokens": lambda r: r["llm_usage"]["completion_tokens"],
                    "total_tokens": lambda r: (r["llm_usage"]["prompt_tokens"] + r["llm_usage"]["completion_tokens"]) if r["llm_usage"]["usage_available"] else None,
                    "list_cost": lambda r: list_cost(r["llm_usage"], r["embedding_usage"], prices, False), "cache_adjusted_cost": lambda r: list_cost(r["llm_usage"], r["embedding_usage"], prices, True),
                    "provider_cost_unit": lambda r: (r["llm_usage"]["provider_cost_unit"] + (r["embedding_usage"].get("provider_cost_unit") or 0.0)) if r["llm_usage"]["usage_available"] else None})
    medians: dict[str, dict[tuple[str, str], float]] = {m: {} for m in metrics}
    for arm in ARMS:
        arm_rows = [r for r in ok if r["arm"] == arm]
        entry = {"context_matches_study1": {"match": sum(1 for r in arm_rows if r.get("context_matches_study1")), "mismatch": sum(1 for r in arm_rows if r.get("context_matches_study1") is False), "not_applicable": sum(1 for r in arm_rows if r.get("context_matches_study1") is None)},
                 "cached_token_share": None}
        prompt_total = sum(r["llm_usage"]["prompt_tokens"] or 0 for r in arm_rows if r["llm_usage"]["usage_available"])
        cached_total = sum(r["llm_usage"]["cached_tokens"] or 0 for r in arm_rows if r["llm_usage"]["usage_available"])
        entry["cached_token_share"] = round(cached_total / prompt_total, 4) if prompt_total else None
        for metric, getter in metrics.items():
            if arm == "a0" and metric == "retrieval_assembly_seconds":
                entry[metric] = "not applicable"
                continue
            per_attempt = [getter(r) for r in arm_rows if getter(r) is not None]
            by_question: dict[str, list[float]] = {}
            for r in arm_rows:
                value = getter(r)
                if value is not None:
                    by_question.setdefault(r["question_id"], []).append(value)
            per_question = {q: stats.median(v) for q, v in by_question.items()}
            for q, v in per_question.items():
                medians[metric][(q, arm)] = v
            entry[metric] = {"attempts": len(per_attempt), "median_attempts": percentile(per_attempt, 0.5), "p95_attempts": percentile(per_attempt, 0.95),
                             "questions": len(per_question), "median_of_question_medians": percentile(list(per_question.values()), 0.5), "p95_of_question_medians": percentile(list(per_question.values()), 0.95),
                             "mean_of_question_medians": round(stats.mean(per_question.values()), 6) if per_question else None}
        out["arms"][arm] = entry
    for a, b in EFFICIENCY_PAIRS:
        out["contrasts"][f"{a}_over_{b}"] = {}
        for metric in metrics:
            if "a0" in (a, b) and metric == "retrieval_assembly_seconds":
                out["contrasts"][f"{a}_over_{b}"][metric] = "not applicable"
                continue
            groups: dict[str, list[float]] = {}
            for (q, arm), va in medians[metric].items():
                if arm == a and (q, b) in medians[metric] and medians[metric][(q, b)] > 0:
                    groups.setdefault(rows[q]["repo"], []).append(va / medians[metric][(q, b)])
            ratios = [x for g in groups.values() for x in g]
            if len(ratios) < 3:
                out["contrasts"][f"{a}_over_{b}"][metric] = {"pairs": len(ratios), "status": "insufficient"}
                continue
            interval = bootstrap_interval(groups, statistic=stats.median)
            out["contrasts"][f"{a}_over_{b}"][metric] = {"pairs": len(ratios), "median_ratio": round(stats.median(ratios), 4), "ci95": interval,
                                                         "claim": ("lower" if interval[1] < 1 else "higher" if interval[0] > 1 else "no detectable difference")}
    return out


def quality_adjusted_cost(efficiency: dict, quality: dict, behaviour: dict) -> dict:
    out = {}
    for arm in ARMS:
        cost = ((efficiency["arms"].get(arm) or {}).get("list_cost") or {}).get("mean_of_question_medians")
        mean_score = quality["arms"][arm]["mean"]
        good = behaviour["arms"][arm]["good_answer"]["rate"]
        out[arm] = {"list_cost_per_answer": cost, "mean_score": mean_score, "cost_per_score_point": round(cost / mean_score, 8) if cost and mean_score else None,
                    "good_answer_rate": good, "cost_per_good_answer": round(cost / good, 8) if cost and good else None}
    return out


def main() -> int:
    target = OUT / "analysis_v7.json"
    if target.exists():
        raise SystemExit("analysis_v7.json already exists; it is written once")
    rows = question_rows()
    recs = answers()
    units, judge_counts = new_batch_scores()
    frozen = v4_frozen_scores()
    contexts = {(r["question_id"], r["arm"]): r for r in read_jsonl(PRE / "contexts_study1.jsonl")}
    prices = json.loads((V7 / "prices.json").read_text(encoding="utf-8"))
    quality = quality_analysis(units, rows, frozen, recs)
    behaviour = behaviour_analysis(units, rows, recs, contexts)
    efficiency = efficiency_analysis(rows, prices)
    result = {"protocol": "swe_qa_v7", "questions": len(rows), "judge": judge_counts, "quality": quality, "reliability": reliability_analysis(units, frozen, rows), "behaviour": behaviour,
              "efficiency": efficiency, "quality_adjusted_cost": quality_adjusted_cost(efficiency, quality, behaviour),
              "retrieval_and_delivered_evidence": "see preflight_final/study3_offline.json", "price_table": prices,
              "evaluation_spend": {"judge_provider_cost_unit": judge_counts["judge_spend_provider_cost_unit"]}}
    target.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"primary": {h: {k: c.get(k) for k in ("comparison", "mean_difference", "ci95", "holm_adjusted_p", "decision")} for h, c in quality["primary"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
