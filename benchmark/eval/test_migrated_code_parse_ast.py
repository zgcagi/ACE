"""Positive and negative integration tests for the 11 migrated static probes."""
import json
import sys
from pathlib import Path

ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))
from eval.dispatch import load_policy, score_probe

POLICY = load_policy(ACE / "policy/atom_intents.yaml")
PROBES = ACE / "probes/code.jsonl"

BAD = {
    "code.layout_composition.L1.003": ".row { display:block; flex-direction:row; align-items:center; gap:12px; }",
    "code.import_resolution.L1.002": "# import json\nfrom datetime import datetime",
    "code.import_resolution.L2.004": "import hashlib\nimport pathlib",
    "code.api_idiomatic_use.L1.003": "r = requests.post('https://api.example.com/users')",
    "code.api_idiomatic_use.L2.001": "df.groupby('country').sum().sort_values('sales')\ndef top5_by_total(df): return df.head(5)",
    "code.api_idiomatic_use.L2.005": "import numpy as np\ndef save_predictions(arr,path): np.save(path,arr)",
    "code.api_idiomatic_use.L3.002": "from datetime import datetime\nfrom pydantic import BaseModel, EmailStr\nclass User(BaseModel):\n id: str\n email: EmailStr\n signup_at: datetime",
    "code.config_artifact_authoring.L1.003": '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":1}},"required":["name","age"]}',
    "code.native_module_edit.L3.004": '[dependencies]\naho-corasick = { version = "1", features = [] }',
    "code.cross_language_translation.L1.004": "def capitalize_words(s): return s",
    "code.cross_language_translation.L2.003": "def uniq_sorted(xs): return xs",
}

rows = {}
for raw in PROBES.read_text().splitlines():
    if raw.lstrip().startswith("{"):
        probe = json.loads(raw)
        if probe["scorer"]["type"] == "code_parse_ast":
            rows[probe["probe_id"]] = probe

assert set(BAD) <= set(rows), set(BAD) - set(rows)
for probe_id, bad_response in BAD.items():
    probe = rows[probe_id]
    positive, positive_denom = score_probe(probe, probe["ground_truth"], POLICY)
    negative, negative_denom = score_probe(probe, bad_response, POLICY)
    assert positive.passed and positive_denom == 1, (probe_id, positive.to_dict())
    assert not negative.passed and negative_denom == 1, (probe_id, negative.to_dict())

print(json.dumps({"covered": len(BAD), "positive_passed": len(BAD), "negative_rejected": len(BAD),
                  "all_code_parse_ast_probes": len(rows)}))
