"""Evaluate a fixed configuration; no threshold fitting on the final test set."""
from pathlib import Path
import json


def summarize(rows):
    positives = [r for r in rows if r["expected"] == "self"]
    negatives = [r for r in rows if r["expected"] == "other"]
    if not positives or not negatives:
        raise ValueError("评测必须同时包含本人和他人样本")
    return {
        "trials": len(rows), "self_trials": len(positives), "other_trials": len(negatives),
        "false_accept_rate": sum(r["identity"] == "self" for r in negatives) / len(negatives),
        "self_miss_rate_including_unknown": sum(r["identity"] != "self" for r in positives) / len(positives),
        "decision_coverage": sum(r["identity"] != "unknown" for r in rows) / len(rows),
        "unknown_rate": sum(r["identity"] == "unknown" for r in rows) / len(rows),
    }


def evaluate(service, manifest):
    manifest = Path(manifest).resolve()
    rows = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        trial = json.loads(line)
        if trial.get("expected") not in ("self", "other"):
            raise ValueError(f"第 {number} 行 expected 必须为 self 或 other")
        audio = Path(trial["audio"])
        if not audio.is_absolute():
            audio = manifest.parent / audio
        # Fail the run rather than silently dropping difficult/broken samples.
        result = service.verify(trial["profile_id"], audio)
        rows.append({"trial": number, "audio": trial["audio"], "expected": trial["expected"],
                     "condition": trial.get("condition", "unspecified"), **result})
    overall = summarize(rows)
    by_condition = {}
    for condition in sorted({r["condition"] for r in rows}):
        group = [r for r in rows if r["condition"] == condition]
        if {r["expected"] for r in group} == {"self", "other"}:
            by_condition[condition] = summarize(group)
        else:
            by_condition[condition] = {"trials": len(group), "note": "此分组缺少本人或他人样本，未计算双类指标"}
    return {"summary": overall, "by_condition": by_condition, "results": rows,
            "notes": ["固定阈值评测，不在测试集上拟合阈值。", "未计算置信区间或证明商用性能；配对样本可能相关。"]}
