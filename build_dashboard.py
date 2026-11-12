#!/usr/bin/env python3
"""
从评测结果 JSONL 文件生成 Dashboard 所需的数据文件。
输出到 dashboard/ 目录：
  - summary.json     : 所有模型的汇总数据（含分类、原子能力、level 维度）
  - all_results.json : 全部题目详细数据（用于错题分析页）

运行方式:
  python3 build_dashboard.py
"""
import json, glob, os, sys
from collections import defaultdict
from pathlib import Path

from release_results import (
    BENCHMARK_VERSION,
    EVALUATION_CONTRACT_VERSION,
    load_compatible_summary,
    validate_detail_file,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DASHBOARD_DIR = ROOT / "dashboard"
EVAL_DIR = ROOT / "benchmark"


def load_config():
    import yaml

    with open(ROOT / "models.yaml") as f:
        return yaml.safe_load(f)


def load_probes():
    """Load all probes from the probes directory."""
    probes_dir = EVAL_DIR / "probes"
    probes_by_id = {}
    for f in sorted(probes_dir.glob("*.jsonl")):
        with f.open() as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("//"): continue
                try:
                    p = json.loads(ln)
                    probes_by_id[p["probe_id"]] = p
                except (json.JSONDecodeError, KeyError):
                    pass
    return probes_by_id


def find_latest_result(tag, expected_model=None):
    """Find the latest paired, publishable result for this full suite."""
    summary_files = sorted(glob.glob(str(RESULTS_DIR / f"summary_{tag}_*.json")))
    for summary_path in reversed(summary_files):
        summary, _ = load_compatible_summary(summary_path)
        if summary is None:
            continue
        if expected_model is not None:
            meta = summary["meta"]
            candidate = meta["evaluation_signature_payload"]["candidate"]
            if (meta.get("model_name") != expected_model
                    or candidate.get("model") != expected_model):
                continue
        timestamp = Path(summary_path).stem.rsplit("_", 1)[-1]
        detail_path = RESULTS_DIR / f"results_{tag}_{timestamp}.jsonl"
        detail = validate_detail_file(
            detail_path,
            expected_signature=summary["meta"]["evaluation_signature"],
            expected_contract_sha256=summary["meta"]["evaluation_contract_sha256"],
            expected_runtime_sha256=summary["meta"]["evaluator_runtime_sha256"],
        )
        if not detail.compatible or detail.retryable_probe_ids:
            continue
        if len(detail.completed_probe_ids) != summary.get("n_total"):
            continue
        if len(detail.excluded_probe_ids) != summary.get("n_excluded"):
            continue
        return str(summary_path), str(detail_path)
    return None, None


def compute_atom_stats(detail_path):
    """Compute per-atom (canonical_id) pass rates by level."""
    atoms = defaultdict(lambda: {"L1": [], "L2": [], "L3": [], "all": []})
    with open(detail_path) as fh:
        for ln in fh:
            r = json.loads(ln)
            if r.get("excluded_from_score"): continue
            aid = r["canonical_id"]
            atoms[aid][r["level"]].append(r["passed"])
            atoms[aid]["all"].append(r["passed"])
    
    result = {}
    for aid, levels in atoms.items():
        cat = aid.split(".")[0]
        result[aid] = {
            "category": cat,
            "n": len(levels["all"]),
            "pass_rate": _rate(levels["all"]),
            "L1": _rate(levels["L1"]),
            "L2": _rate(levels["L2"]),
            "L3": _rate(levels["L3"]),
            "n_L1": len(levels["L1"]),
            "n_L2": len(levels["L2"]),
            "n_L3": len(levels["L3"]),
        }
    return result


def compute_level_stats(detail_path):
    """Compute overall level pass rates."""
    levels = defaultdict(lambda: {"pass": 0, "total": 0})
    with open(detail_path) as fh:
        for ln in fh:
            r = json.loads(ln)
            if r.get("excluded_from_score"): continue
            levels[r["level"]]["total"] += 1
            if r["passed"]: levels[r["level"]]["pass"] += 1
    return {
        lv: round(d["pass"] / d["total"], 4) if d["total"] else None
        for lv, d in levels.items()
    }


def _rate(bools):
    return round(sum(1 for b in bools if b) / len(bools), 4) if bools else None


def build_summary(config, probes_by_id):
    """Build the summary.json with all model data."""
    models_data = []
    
    for m in config["models"]:
        tag = m["tag"]
        summary_path, detail_path = find_latest_result(tag, m["id"])
        if not summary_path:
            print(f"  ⚠️  No results found for {tag}, skipping")
            continue
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        # Compute atom-level stats
        atom_stats = compute_atom_stats(detail_path)
        level_stats = compute_level_stats(detail_path)
        
        models_data.append({
            "id": m["id"],
            "tag": tag,
            "description": m.get("description", ""),
            "group": m.get("group", "other"),
            "overall": {
                "pass_rate": summary["overall_pass_rate"],
                "n_pass": summary["n_pass"],
                "n_eff": summary["n_eff"],
                "n_excluded": summary["n_excluded"],
                "n_total": summary["n_total"],
            },
            "levels": level_stats,
            "categories": summary.get("by_category", {}),
            "atoms": atom_stats,
            "meta": summary.get("meta", {}),
        })
        print(f"  ✅ {tag}: loaded ({summary['n_pass']}/{summary['n_eff']} = {summary['overall_pass_rate']*100:.1f}%)")
    
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "models": models_data,
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_all_results(config, probes_by_id):
    """Build all_results.json with per-probe detail for all models."""
    all_records = []
    
    for m in config["models"]:
        tag = m["tag"]
        _, detail_path = find_latest_result(tag, m["id"])
        if not detail_path: continue
        
        with open(detail_path) as fh:
            for ln in fh:
                r = json.loads(ln)
                if r.get("excluded_from_score"): continue
                probe = probes_by_id.get(r["probe_id"], {})
                all_records.append({
                    "model": m["id"],
                    "probe_id": r["probe_id"],
                    "canonical_id": r["canonical_id"],
                    "category": r["category"],
                    "level": r["level"],
                    "passed": r["passed"],
                    "prompt": probe.get("prompt", "") or "",
                    "ground_truth": str(probe.get("ground_truth", "") or ""),
                    "expected": str(
                        probe.get("scorer", {}).get("expected", "")
                        or probe.get("scorer", {}).get("pattern", "")
                        or probe.get("scorer", {}).get("keywords", "")
                        or ""
                    ),
                    "scorer_type": probe.get("scorer", {}).get("type", ""),
                    "response": r.get("response", "") or "",
                    "score_extracted": str(r.get("score_extracted", "") or ""),
                    "score_strategy": r.get("score_strategy", ""),
                    "score_note": r.get("score_note", ""),
                    "finish_reason": r.get("finish_reason", ""),
                    "completion_tokens": r.get("completion_tokens", 0),
                })
    
    return all_records


def main():
    if not os.environ.get("ACE_JUDGE_BASE_URL"):
        raise SystemExit("Set ACE_JUDGE_BASE_URL for result selection.")

    print("=" * 60)
    print(f"📊 Building ACE {BENCHMARK_VERSION} Dashboard Data")
    print("=" * 60)
    
    config = load_config()
    DASHBOARD_DIR.mkdir(exist_ok=True, parents=True)
    
    print("\n📂 Loading probes...")
    probes_by_id = load_probes()
    print(f"  Loaded {len(probes_by_id)} probes")
    
    print("\n📈 Building summary data...")
    summary = build_summary(config, probes_by_id)
    summary_path = DASHBOARD_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2,
                  default=lambda x: round(x, 4) if isinstance(x, float) else x)
    print(f"  → {summary_path} ({os.path.getsize(summary_path) / 1024:.0f} KB)")
    
    print("\n📋 Building detailed results data...")
    all_results = build_all_results(config, probes_by_id)
    results_path = DASHBOARD_DIR / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False)
    print(f"  → {results_path} ({os.path.getsize(results_path) / 1024 / 1024:.1f} MB)")
    print(f"  Total records: {len(all_results)}")
    
    print("\n✅ Dashboard data generation complete!")
    print("   Use summary.json and all_results.json with your analysis tools")


if __name__ == "__main__":
    main()
