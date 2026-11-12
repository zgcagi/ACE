"""Static validation for all judge_rubric probe contracts."""
import json
import sys
from collections import Counter
from pathlib import Path

ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))
from eval.judge import PROMPT_VERSION
PROBES = ACE / "probes"

rows = []
for path in sorted(PROBES.glob("*.jsonl")):
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.lstrip().startswith("{"):
            continue
        probe = json.loads(raw)
        if probe.get("scorer", {}).get("type") == "judge_rubric":
            rows.append((path.name, line_no, probe))

errors = []
rubric_shapes = Counter()
for filename, line_no, probe in rows:
    scorer = probe["scorer"]
    rubric = scorer.get("rubric")
    rubric_shapes[type(rubric).__name__] += 1
    if not probe.get("prompt"):
        errors.append((probe["probe_id"], "missing prompt"))
    if not isinstance(rubric, dict):
        errors.append((probe["probe_id"], "rubric must be canonical object"))
    else:
        required_keys = {"version", "pass_criteria", "fail_criteria", "pass_rule"}
        if set(rubric) != required_keys:
            errors.append((probe["probe_id"], f"rubric keys={sorted(rubric)}"))
        if rubric.get("version") != "1":
            errors.append((probe["probe_id"], "rubric version must be 1"))
        if not isinstance(rubric.get("pass_criteria"), list) or not rubric.get("pass_criteria"):
            errors.append((probe["probe_id"], "pass_criteria must be nonempty list"))
        if not isinstance(rubric.get("fail_criteria"), list) or not rubric.get("fail_criteria"):
            errors.append((probe["probe_id"], "fail_criteria must be nonempty list"))
        if rubric.get("pass_rule") != "all pass criteria and no fail criteria":
            errors.append((probe["probe_id"], "unexpected pass_rule"))
    try:
        json.dumps(rubric, ensure_ascii=False)
    except Exception as exc:
        errors.append((probe["probe_id"], f"rubric not JSON serializable: {exc}"))

if errors:
    raise AssertionError(errors[:20])

print(json.dumps({
    "judge_rubric_probes": len(rows),
    "rubric_shapes": rubric_shapes,
    "errors": 0,
    "judge_protocol": "openai_compatible",
    "judge_model": "deepseek-v4-flash",
    "judge_prompt_version": PROMPT_VERSION,
}, ensure_ascii=False, indent=2))
