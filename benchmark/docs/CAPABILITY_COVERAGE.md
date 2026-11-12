# ACE capability coverage

ACE 2.6.4 defines 187 atoms across 19 capability categories. Each atom has
a canonical ID and a scoring intent. This guide explains the mapping from
selected capability dimensions to ACE atoms. The 20 atoms listed here account
for 60 probes in the full suite.

## Dimensions considered for new atoms

An atom describes a distinct capability that can be tested in isolation.
Some dimensions extend an existing atom along an input axis.

| # | Early dimension | Current mapping | Rationale |
|---|---|---|---|
| 7  | 排序 (sort)            | **ADD** as `arith.numeric_sort` | Basic, distinct from `lang.lexicographic_order` (alphabet) |
| 8  | 第K大 (kth largest)    | **ADD** as `arith.kth_largest`  | Different cognitive op from full sort (selection) |
| 10 | 进制转换 (base conv)   | **ADD** as `arith.base_conversion` | Symbolic encoding distinct from modular arith |
| 16 | 量词推理 (Chinese 量词) | **ADD** as `lang.classifier_concord` | Measures how models match Chinese classifiers to nouns. |
| 19 | 编码解码 (Caesar/etc)  | **ADD** as `lang.cipher_encoding` | Symbolic transformation distinct from translation |
| 23 | 博弈推理 (game theory) | **ADD** as `logic.game_theory` | Distinct from causal — strategy/equilibrium |
| 29 | 第N字符提取 (s[k])     | **ADD** as `lang.nth_character_extract` | Tokenization-aware; LLMs are notoriously bad at this |
| 30 | 首字母拼读 (pinyin)    | **SKIP** | Too narrowly Chinese-specific. Skipping. |
| 31 | 字符频率 (char freq)   | **ADD** as `lang.character_frequency` | Tokenization stress test |
| 32 | 字符串反转 (reverse)   | **ADD** as `lang.string_reverse` | Classic LLM weakness, distinct from token ops |
| 33 | 查表映射 (lookup)      | **ADD** as `read.table_lookup` | Distinct from `formal_notation_parse` — pure key→value |
| 36 | 规则归纳 (rule induce) | **ADD** as `logic.rule_induction` | Inductive reasoning — distinct from deductive |
| 39 | 价格合理性 (price OK?) | **MERGE** into `world.naive_physics` (extend with economics axis) | Same "real-world plausibility" capability; just expand the atom |
| 41 | 网络缩写 (yyds/awsl)   | **SKIP** | Too culturally specific. A "current slang knowledge" atom would go stale. Skipping. |
| 42 | 版本号比较 (semver)    | **ADD** as `arith.version_compare` | Practical, deterministic, distinct |
| 47 | 的地得辨析 (Chinese 的) | **ADD** as `lang.chinese_de_particle` | Chinese-specific but a clean atomic test |
| 48 | 谎言推理 (knights)     | **ADD** as `logic.liar_puzzle` | Classic but distinct from deductive_chain (involves modeling truth-state of speakers) |
| 49 | 同构判断 (isomorphism) | **ADD** as `logic.isomorphism_check` | Abstract structural reasoning |

**Result**: 15 ADD, 1 MERGE, 2 SKIP.

## Dimensions related to existing atoms

| # | Early dimension and related atom | Current mapping |
|---|---|---|
| 2  | 多实体属性记忆 → `read.identify_query` | **ADD** `read.multi_entity_attribute_memory` — ACE only tests single-fact retrieval; the n-entity-by-k-attribute matrix is a real distinct capability |
| 6  | 条件计数 → `logic.constraint_satisfaction` | **MERGE** — extend the existing atom with a counting axis instead of new atom |
| 9  | 单位换算 → `output.unit_correct_final` | **ADD** `arith.unit_conversion` — output.unit_correct_final tests "remembers to attach a unit"; this tests "perform the conversion arithmetic" |
| 18 | 多重否定 → `lang.polarity_clash` | **ADD** `lang.multi_negation` — stacked negation (不可能不) is a real distinct capability from polarity-clash detection |
| 37 | 错别字识别 → `lang.lexical_edit_recognition` | **MERGE** — same atom, just add Chinese-typo axis |
| 38 | 品牌大小写 → `instr.case_uniformity` | **ADD** `know.brand_canonical_case` — `instr.case_uniformity` tests "follow user's case rule"; this tests "recall canonical case (iPhone, eBay)" — a knowledge atom |
| 40 | 时长合理性 → `world.naive_physics` | **MERGE** — extend the existing atom |
| 43 | 格式有效性 → `instr.format_compliance` | **ADD** `know.format_validity_check` — instr tests user-imposed format; this tests "is this a valid email/IP/URL?" — pattern recognition |

**Result**: 5 ADD, 3 MERGE.

## Canonical atom IDs

The 20 atoms use the following IDs in ACE 2.6.4.

```
arith.numeric_sort
arith.kth_largest
arith.base_conversion
arith.unit_conversion
arith.version_compare

lang.classifier_concord
lang.cipher_encoding
lang.nth_character_extract
lang.character_frequency
lang.string_reverse
lang.chinese_de_particle
lang.multi_negation

logic.game_theory
logic.rule_induction
logic.liar_puzzle
logic.isomorphism_check

read.table_lookup
read.multi_entity_attribute_memory

know.brand_canonical_case
know.format_validity_check
```

Three existing atoms also gained input axes.

- `world.naive_physics` += economics/duration plausibility axes (D39, D40)
- `lang.lexical_edit_recognition` += Chinese-typo axis (D37)
- `logic.constraint_satisfaction` += counting axis (D6)

## Dimensions omitted from the taxonomy

- D30 首字母拼读 (Chinese pinyin) covers a narrow language task.
- D41 网络缩写 (internet abbreviations) reflects slang that changes over time.

## Atom definitions

Each atom definition includes the following fields.

- canonical_id
- short description (capability under test)
- expected scoring intent (in `policy/atom_intents.yaml`)
- 1 sentence example probe-shape
- relation to nearest existing atom (so future authors don't duplicate)

[Atom definitions](../merged/zatom_added_atoms.yaml) · [Probe records](../probes/zatom_added.jsonl)
