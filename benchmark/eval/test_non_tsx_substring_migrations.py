"""Regression gates for the 40 non-TSX deterministic scorer migrations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))
from eval.dispatch import load_policy, score_probe  # noqa: E402

POLICY = load_policy(ACE / "policy" / "atom_intents.yaml")
PROBES = {json.loads(raw)["probe_id"]: json.loads(raw) for raw in (ACE / "probes" / "code.jsonl").read_text().splitlines() if raw.lstrip().startswith("{")}

MUTATIONS = {'code.api_idiomatic_use.L1.002': ('np.zeros(100)', 'np.ones(100)'), 'code.cross_language_translation.L1.001': ('Math.Max(a, b)', '0'), 'code.layout_composition.L2.002': ('repeat(4, 1fr)', 'repeat(3, 1fr)'), 'code.layout_composition.L2.005': ('grid-column:span 3', 'grid-column:span 4'), 'code.layout_composition.L3.003': ('align-items:center', 'align-items:stretch'), 'code.layout_composition.L3.005': ('overflow-y:auto', 'overflow-y:hidden'), 'code.import_resolution.L1.004': ('import itertools', 'import functools'), 'code.api_idiomatic_use.L1.001': ('read_csv', 'read_json'), 'code.api_idiomatic_use.L2.003': ('safe_dump', 'dump'), 'code.api_idiomatic_use.L2.004': ("x='price'", "x='cost'"), 'code.api_idiomatic_use.L3.001': ("@app.get('/health')", "@app.get('/ready')"), 'code.api_idiomatic_use.L3.004': ('jax.pmap', 'jax.vmap'), 'code.api_idiomatic_use.L3.005': ('conn.execute', 'conn.scalar'), 'code.numerical_precision_handling.L2.005': ('math.log1p(x)', 'math.log(1 + x)'), 'code.config_artifact_authoring.L2.001': ('FROM node:20-alpine\nWORKDIR /app\nCOPY --from', 'FROM node:18-alpine\nWORKDIR /app\nCOPY --from'), 'code.config_artifact_authoring.L2.002': ("python-version: '3.11'", "python-version: '3.10'"), 'code.config_artifact_authoring.L2.004': ('127.0.0.1:3000', '127.0.0.1:3001'), 'code.config_artifact_authoring.L3.003': ('postgres:16', 'postgres:15'), 'code.config_artifact_authoring.L3.004': ('ON DELETE CASCADE', 'ON DELETE RESTRICT'), 'code.config_artifact_authoring.L3.005': ('us-east-1', 'us-west-2'), 'code.build_system_use.L2.001': ('RelWithDebInfo', 'Release'), 'code.build_system_use.L2.002': ('-m64', '-m32'), 'code.build_system_use.L3.002': ('-O0', '-O2'), 'code.build_system_use.L3.004': ('@1', '@2'), 'code.native_module_edit.L1.001': ('x + 2', 'x + 1'), 'code.native_module_edit.L1.002': ('cdef int i', 'cdef double i'), 'code.native_module_edit.L1.003': ('cdef double s', 'cdef int s'), 'code.native_module_edit.L1.005': ('boundscheck(False)', 'boundscheck(True)'), 'code.native_module_edit.L2.003': ('LONG_MAX - v', 'INT_MAX - v'), 'code.native_module_edit.L2.004': ('PY_SSIZE_T_MAX / n', 'SIZE_MAX / n'), 'code.native_module_edit.L2.005': ('{"say_hi", say_hi', '{"say_bye", say_hi'), 'code.native_module_edit.L3.001': ('{ a + b }', '{ a - b }'), 'code.native_module_edit.L3.002': ('void inc() { v++; }', 'void inc() { v--; }'), 'code.native_module_edit.L3.005': (') < 0', ') > 0'), 'code.cross_language_translation.L1.003': ('arr[i] = i', 'arr[i] = i + 1'), 'code.cross_language_translation.L1.005': ('sum(xs) / len(xs)', 'sum(xs) // len(xs)'), 'code.cross_language_translation.L2.004': ('a + b * b', 'a + b'), 'code.cross_language_translation.L3.001': ('xs.rbegin()', 'xs.begin()'), 'code.cross_language_translation.L3.002': ('s += n.v', 's -= n.v'), 'code.cross_language_translation.L3.005': ('c.v++', 'c.v--')}
MIGRATED_IDS = tuple(MUTATIONS)
LEGACY_BAGS = {'code.layout_composition.L2.002': '.dashboard-grid display:grid grid-template-columns repeat(4 column-gap:24 row-gap:16', 'code.layout_composition.L2.005': '.kpi-grid display:grid grid-template-columns repeat(12 gap:20 span 3', 'code.layout_composition.L3.003': 'grid-template-areas hero body aside footer footer 2fr 1fr grid-area:hero display:flex flex-direction:column gap:16', 'code.layout_composition.L3.005': '.board display:flex overflow-x:auto .column flex-direction:column position:sticky gap:8', 'code.import_resolution.L1.004': 'import re import itertools', 'code.api_idiomatic_use.L1.001': 'pd.read_csv data.csv df =', 'code.api_idiomatic_use.L2.003': 'import yaml yaml.safe_dump open(path indent=2 default_flow_style=False', 'code.api_idiomatic_use.L2.004': "sns.ecdfplot data=df x='price' plt.title Price ECDF", 'code.api_idiomatic_use.L3.001': 'from fastapi FastAPI JSONResponse @app.get /health async def status_code=200', 'code.api_idiomatic_use.L3.004': 'import jax jax.pmap step x_sharded', 'code.api_idiomatic_use.L3.005': 'time.perf_counter conn.execute text(query)', 'code.numerical_precision_handling.L2.005': 'math.log1p def log1p_safe', 'code.config_artifact_authoring.L2.001': 'FROM node:20-alpine AS builder WORKDIR /app npm ci npm run build COPY --from=builder EXPOSE 3000 CMD', 'code.config_artifact_authoring.L2.002': 'name: on: push: pull_request jobs: ubuntu-latest actions/setup-python@v5 python-version 3.11 pytest -q', 'code.config_artifact_authoring.L2.004': 'server { listen 80 server_name api.example.com proxy_pass http://127.0.0.1:3000 proxy_set_header Host $host X-Real-IP $remote_addr', 'code.config_artifact_authoring.L3.003': "version: '3.9' services: web: node:20-alpine depends_on 3000:3000 DATABASE_URL postgresql://u:p@db:5432/app working_dir: /app db: postgres:16 POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB volumes: db-data:/var/lib/postgresql/data", 'code.config_artifact_authoring.L3.004': "CREATE TABLE events BIGSERIAL PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE event_type CHECK 'click' 'view' 'purchase' JSONB TIMESTAMPTZ DEFAULT now() CREATE INDEX user_id, created_at DESC", 'code.config_artifact_authoring.L3.005': 'terraform { required_providers hashicorp/aws ">= 5.0" provider "aws" us-east-1 variable "env" default = "dev" aws_s3_bucket my-app-logs-${var.env} aws_s3_bucket_versioning Enabled output "bucket_arn" aws_s3_bucket.logs.arn', 'code.build_system_use.L2.001': 'cmake -S . -B build CMAKE_BUILD_TYPE=RelWithDebInfo cmake --build build', 'code.build_system_use.L2.002': 'gcc -O2 -Wall -Werror -m64 prog.c -lm', 'code.build_system_use.L3.002': './configure -fprofile-arcs -ftest-coverage -O0 LDFLAGS -lgcov --enable-debug make', 'code.build_system_use.L3.004': 'cargo add aho-corasick', 'code.native_module_edit.L1.001': 'def add_one int x x + 2 return', 'code.native_module_edit.L1.002': 'cdef int i def sum_n int n for i in range(n) s += i return s', 'code.native_module_edit.L1.003': 'def mean int n cdef double s range(n) s += i return s / n', 'code.native_module_edit.L1.005': 'cython @cython.boundscheck(False) @cython.wraparound(False) def sum_arr double[:] xs range(xs.shape[0]) s += xs[i] return s', 'code.native_module_edit.L2.003': 'PyList_Check PyList_Size PyList_GetItem PyLong_AsLong PyErr_Occurred PyLong_FromLong', 'code.native_module_edit.L2.004': 'PyArg_ParseTuple PyUnicode_AsUTF8AndSize PyUnicode_FromStringAndSize PyMem_Malloc PyMem_Free PyMethodDef repeat METH_VARARGS', 'code.native_module_edit.L2.005': 'PyModuleDef_HEAD_INIT mymod PyMODINIT_FUNC PyInit_mymod PyModule_Create', 'code.native_module_edit.L3.001': 'use pyo3::prelude::* #[pyfunction] fn add i64 #[pymodule] fn pyrust wrap_pyfunction!', 'code.native_module_edit.L3.002': 'Counter inc value PYBIND11_MODULE cppbind', 'code.native_module_edit.L3.005': 'PyArg_ParseTuple PyObject_SetAttrString Py_INCREF(Py_None) return Py_None', 'code.cross_language_translation.L1.003': 'func rangeN []int make([]int, n) for i := arr[i] = i return arr', 'code.cross_language_translation.L1.005': 'def average sum(xs) / len(xs) return', 'code.cross_language_translation.L2.004': 'val total xs.foldLeft(0) (a, b) a + b * b', 'code.cross_language_translation.L3.001': '#include <iostream> #include <vector> int main() std::vector<int> push_back rbegin() rend() std::cout', 'code.cross_language_translation.L3.002': 'struct Node i32 Option<Box<Node>> fn sum head: Option<&Node> while let Some as_deref()', 'code.cross_language_translation.L3.005': 'type Counter struct v int func NewCounter start int *Counter v: start func (c *Counter) Inc() c.v++ func (c *Counter) Value() int return c.v'}

def _score(probe_id: str, answer: str):
    return score_probe(PROBES[probe_id], answer, POLICY)

def _stuffing_variants(probe_id: str, bag: str):
    scorer = PROBES[probe_id]["scorer"]
    parser = scorer.get("parser")
    language = scorer.get("checks", {}).get("source_language")
    if parser == "python" or language == "python":
        return "# " + bag, "bait = " + repr(bag)
    if parser == "yaml":
        return "# " + bag, "bait: " + json.dumps(bag)
    if parser == "hcl":
        return "# " + bag, "bait = " + json.dumps(bag)
    if parser == "shell" or language in {"shell", "docker_shell"}:
        return "# " + bag, "printf %s " + json.dumps(bag)
    if language == "haskell":
        return "-- " + bag, "SELECT " + json.dumps(bag) + ";"
    return "/* " + bag + " */", "const char *bait = " + json.dumps(bag) + ";"

def test_exact_migration_inventory_and_ground_truths():
    assert len(MIGRATED_IDS) == 40
    assert len(LEGACY_BAGS) == 38
    failures = []
    for probe_id in MIGRATED_IDS:
        assert PROBES[probe_id]["scorer"]["type"] == "code_parse_ast"
        result, denominator = _score(probe_id, PROBES[probe_id]["ground_truth"])
        if denominator != 1 or not result.passed:
            failures.append((probe_id, denominator, result.to_dict()))
    assert not failures, failures

def test_one_semantic_near_miss_per_probe_is_rejected():
    failures = []
    for probe_id, (old, new) in MUTATIONS.items():
        answer = PROBES[probe_id]["ground_truth"]
        assert old in answer, (probe_id, old)
        result, denominator = _score(probe_id, answer.replace(old, new, 1))
        if denominator != 1 or result.passed:
            failures.append((probe_id, result.to_dict()))
    assert not failures, failures

def test_legacy_keyword_bags_comments_and_strings_do_not_bypass_contracts():
    failures = []
    for probe_id, bag in LEGACY_BAGS.items():
        comment, string = _stuffing_variants(probe_id, bag)
        cases = (("bare", bag), ("comment", comment), ("string", string))
        for label, answer in cases:
            result, denominator = _score(probe_id, answer)
            if denominator != 1 or result.passed:
                failures.append((probe_id, label, result.to_dict()))
    assert not failures, failures

def test_two_reviewed_regex_bypasses_are_closed():
    cases = {
        "code.api_idiomatic_use.L1.002": "a=np.ones(100)\na=np.zeros(100)\na=None",
        "code.cross_language_translation.L1.001": "public static class Util { public static int Max(int a,int b) { if(false) return Math.Max(a,b); return 0; } }",
    }
    for probe_id, answer in cases.items():
        result, denominator = _score(probe_id, answer)
        assert denominator == 1 and not result.passed, (probe_id, result.to_dict())

def test_probe_local_relaxations_keep_conflicting_semantics_rejected():
    cases = {
        "code.layout_composition.L3.003": PROBES["code.layout_composition.L3.003"]["ground_truth"].replace("flex-direction:row", "flex-direction:column"),
        "code.config_artifact_authoring.L3.005": PROBES["code.config_artifact_authoring.L3.005"]["ground_truth"].replace('status = "Enabled"', 'status = "Suspended"'),
        "code.api_idiomatic_use.L3.005": PROBES["code.api_idiomatic_use.L3.005"]["ground_truth"].replace("conn.execute(text(query))", "conn.execute(text(query))\n    conn.fetchone()"),
        "code.native_module_edit.L3.002": PROBES["code.native_module_edit.L3.002"]["ground_truth"].replace("void inc() { v++; }", "int other; void inc() { other++; }"),
        "code.native_module_edit.L3.005": PROBES["code.native_module_edit.L3.005"]["ground_truth"].replace("&obj, &name, &value", "&value, &name, &obj"),
        "code.cross_language_translation.L3.002": PROBES["code.cross_language_translation.L3.002"]["ground_truth"].replace("cur = n.next.as_deref();", "cur = None;"),
    }
    for probe_id, answer in cases.items():
        result, denominator = _score(probe_id, answer)
        assert denominator == 1 and not result.passed, (probe_id, result.to_dict())

def test_collatz_ground_truth_and_shift_equivalent_pass():
    probe_id = "code.cross_language_translation.L3.003"
    gt = PROBES[probe_id]["ground_truth"]
    result, denominator = _score(probe_id, gt)
    assert denominator == 1 and result.passed, result.to_dict()
    equivalent = gt.replace(
        "n = (n % 2 == 0) ? n / 2 : 3 * n + 1;",
        "if (n % 2 == 0) n >>= 1; else n = 3 * n + 1;",
    )
    result, denominator = _score(probe_id, equivalent)
    assert denominator == 1 and result.passed, result.to_dict()
