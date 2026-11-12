# ACE 2.6.4 原子能力评测

[English](README.md) · [可视化教程](examples/tutorial.zh-CN.md)

**ACE** 全称 **Atom Capability Evaluation**，即原子能力评测。ACE 将语言模型的能力拆分为可单独评测的原子能力（atom），提供逐题判定，以及按原子能力、类别和难度汇总的结果。

## 设计理念

- **减少能力之间的干扰。** 例如，模型解出了数学题，却在答案外多写了推导。只要该题的评分规则允许提取答案，就按提取出的答案计分。考查指令遵循或依赖特定输出结构的题目会直接检查这些要求。
- **用多道题覆盖不同输入和难度。** 每项能力使用多道题（probe），覆盖不同输入和 L1–L3 三档难度，以观察模型表现如何随任务难度变化。
- **让评分有据可查。** 能用规则判定的题目优先采用确定性评分。需要语义判断的题目为裁判模型提供明确的通过和失败准则。

相关模型项目见 [ZGCM-1](https://github.com/zgcagi/ZGCM-1)。

## 评测概览

| 指标 | 数量 |
| --- | ---: |
| 原子能力 | 187 |
| 能力类别 | 19 |
| 题目总数 | 2,565 |
| 公开版全量评测计分题数 | 2,489 |
| 由大模型裁判评分的启用题数 | 348 |
| 排除计分的题数 | 76 |

题库涵盖算术、数学、逻辑、因果推理、语言、阅读、知识、世界理解、长上下文、对话、真实性、元认知、指令遵循、输出格式、代码、规划、工具使用、智能体行为和多模态任务。76 道排除题包含缺少媒体文件的 60 道多模态题、15 道安全专项题和 1 道外部引文题。安全专项题和外部引文题以排除记录的形式保留。评测程序会记录这些题目的排除状态，并跳过对待测模型和裁判的请求。

ACE 2.6.4 是首个公开版本，版本号记录在 [`benchmark/VERSION`](benchmark/VERSION)。

## 安装与验证

在仓库根目录运行以下命令。评测程序依赖 POSIX 的进程和资源控制机制，全量评测请使用 Linux。离线测试已在 macOS 上用 Python 3.12 验证。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python benchmark/eval/run_eval.py --version
python benchmark/eval/run_eval.py --help
python benchmark/eval/validate_judge_rubrics.py
```

版本命令应输出 `ACE 2.6.4`。评分准则校验可以离线运行，预期校验 348 份准则，错误数为 0。安装测试依赖后，可在仓库根目录运行离线回归测试。

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 小规模试跑

将 `OPENAI_API_KEY` 设为待测模型接口的密钥，并替换示例中的接口地址和模型 ID。以下命令向待测模型发送 5 道算术题，关闭裁判调用，并将诊断结果写入 `results/smoke`。

```bash
export OPENAI_API_KEY='your-candidate-key'

python benchmark/eval/run_eval.py \
  --base-url https://candidate.example.com/v1 \
  --model-name candidate-model \
  --filter arith --limit 5 \
  --concurrency 2 --total-api-concurrency 2 \
  --no-llm-judge \
  --out-dir results/smoke \
  --tag candidate-smoke
```

## 运行全量评测

待测模型接口需兼容 OpenAI Chat Completions API。将 `ACE_JUDGE_BASE_URL` 设为参考裁判的 OpenAI 兼容接口地址，并通过 `ACE_JUDGE_API_KEY` 提供密钥。全量计分要求使用 `deepseek-v4-flash`，裁判输出预算为 4,096 tokens。对使用裁判的 348 道题，裁判会收到任务、评分准则、待测模型回答及相关参考信息。

**全量评测会执行部分代码题中模型生成的 Python。** 请在可销毁的隔离 Linux 虚拟机中运行，使用专门的非特权账户，限制网络访问和评测凭据的权限，并避免挂载私有数据。进程超时和资源限制只提供进程控制，执行隔离依赖运行环境。

```bash
export OPENAI_API_KEY='your-candidate-key'
export ACE_JUDGE_BASE_URL='https://judge.example.com/v1'
export ACE_JUDGE_API_KEY='your-judge-key'

python benchmark/eval/run_eval.py \
  --base-url https://candidate.example.com/v1 \
  --model-name candidate-model \
  --judge-model deepseek-v4-flash \
  --judge-max-tokens 4096 \
  --concurrency 16 --judge-concurrency 16 --total-api-concurrency 16 \
  --out-dir results \
  --tag candidate-model-v1
```

单模型诊断可以用 `--judge-base-url` 覆盖裁判接口，或用 `--judge-model` 选择其他模型。要获得全量计分资格，实际使用的接口必须与 `ACE_JUDGE_BASE_URL` 一致。使用不同裁判接口的结果会记录不同的评测签名。

示例为待测模型和裁判各配置 16 个并发工作线程，两者共享 16 个在途请求的上限。这三项参数的默认值均为 200，请按服务商配额调整。待测模型的默认输出预算为 16,384 tokens，默认温度为 0。接口拒收温度字段时可加 `--omit-temperature`。接口拒收 system 角色时可加 `--no-system-prompt`，将系统提示移入用户消息。

评测程序默认在第二阶段重试因服务调用或执行故障而未完成的题目。每个阶段内，每道题最多发起一次待测模型请求和一次裁判尝试。若只需重试裁判，程序会复用待测模型的原始回答。若裁判接口拒绝 JSON 响应模式，客户端可能在同一次尝试中追加一次省略该选项的兼容请求。使用 `--no-retry-failed` 可关闭第二阶段。待测模型或裁判的请求故障仍未解决时，程序可能在写出结果后以非零状态退出。重新运行前，请查看汇总文件中的暂缓计分题目。

## 计分与输出文件

评测程序根据题目声明的评分规则和原子能力策略选择评分器。数值、文本、选项、关键词、结构化数据和代码可以使用确定性校验。部分语义题使用大模型裁判，按明确的通过和失败准则给出二元判定。指令遵循题会检查题目声明的输出形式约束。其余类别在评分规则允许提取答案时，可接受对答案内容无影响的包装。 [原子能力策略](benchmark/policy/atom_intents.yaml) · [评分规范](benchmark/spec/scoring.md)

总分为 `n_pass / n_eff`。`n_eff` 包含所有已完成判定的通过题和失败题。固定排除的题目，以及因服务调用或执行故障而暂缓计分的题目，均从分母中扣除。正式全量结果需覆盖全部 2,565 道题，满足 `n_eff = 2489` 和 `n_excluded = 76`。

| 文件 | 内容 |
| --- | --- |
| `results_<tag>_<timestamp>.jsonl` | 逐题判定、评分策略、回答摘录及适用时的裁判证据。重试可能为同一题目追加记录，读取时以每个题目 ID 的最后一条为准。 |
| `summary_<tag>_<timestamp>.json` | 题目计数、总通过率、类别和原子能力分项、运行标识及暂缓计分的题目 ID。 |

报告正式全量分数前，请核对以下字段。

```text
meta.benchmark_version     = "2.6.4"
meta.official_full_suite   = true
meta.official_scope_issues = []
meta.score_status          = "complete"
meta.publishable           = true
```

`meta.publishable` 标记的是分数是否符合发布口径。逐题输出可能包含模型回答和推理内容，分享前请先检查。只跑部分题目、修改题库或策略、调整回答保存上限、使用不同的裁判配置，或存在尚未解决的暂缓计分题目时，结果会标记为诊断结果。诊断运行请使用单独的标签和输出目录。

### 断点续跑与复现

评测程序默认启用断点续跑。保持标签、输出目录、评测文件、待测模型设置和裁判配置一致，即可续跑兼容的未完成任务。程序通过 `evaluation_signature` 比较题库、策略、评测器源码摘要，以及待测模型和裁判的设置。使用 `--no-resume` 可开启一次新的运行。调整重试并发或超时不会改变签名。启用或禁用第二阶段会改变签名中记录的请求上限。

为便于复现，请记录待测模型 ID 和快照、待测模型和裁判的接口地址、生成参数、运行日期，以及汇总中的 `evaluation_signature`、`evaluation_contract_sha256` 和 `evaluator_runtime_sha256`。这些字段标识评测输入和设置。托管模型在多次运行中仍可能返回不同的输出。

## 批量评测与离线分析

评测多个模型时，将 `models.example.yaml` 复制为 `models.yaml`，填写待测模型的接口地址、API 密钥和模型 ID。将填好的配置保存在本地。裁判接口和密钥沿用全量示例中的 `ACE_JUDGE_BASE_URL` 和 `ACE_JUDGE_API_KEY`。批量入口使用该接口以及参考裁判模型和输出预算。查询状态或导出结果时，请使用同一个 `ACE_JUDGE_BASE_URL`。 [配置示例](models.example.yaml)

```bash
cp models.example.yaml models.yaml
# 先编辑 models.yaml，再运行以下命令。
python run_all.py --status
python run_all.py --eval-only --model candidate-model
python run_all.py --build-only
```

批量运行将结果写入 `results/`。最后一条命令会按配置的模型 ID 和标签选择最新的兼容全量结果，导出 `dashboard/summary.json` 和 `dashboard/all_results.json`。要将单模型命令的结果纳入导出，运行时使用 `--out-dir results`，并让模型 ID 和标签与批量配置一致。已有结果可以离线审查，无需重新调用模型。 [离线失败审计指南](benchmark/docs/POST_RUN_FAILURE_AUDIT.md)

## 前端与结果对比

ACE 前端提供排行榜、模型筛选、类别热力图、能力雷达图和并排对比表。从仓库根目录运行 `python3 -m http.server 8000 --bind 127.0.0.1`，然后打开 `http://127.0.0.1:8000/frontend/`。样例模式直接读取三份 ACE 2.6.4 存档汇总结果，无需 API 密钥。样例沿用 2,503 道计分题的历史口径，公开版新评测按 2,489 道题计分。 [可视化教程](examples/tutorial.zh-CN.md) · [样例结果](examples/results/)

## 读取或扩展题库

题库使用 UTF-8 编码的 JSONL。解析记录时，请跳过空行和以 `//` 开头的注释行。`probe_id` 是题目的唯一标识，`canonical_id` 标识原子能力，`category` 和 `level` 用于按类别和难度分组，`axis_values` 记录题目的变化维度。模型输入由 `prompt` 和可选的 `system_prompt` 组成。`ground_truth` 和 `scorer` 定义参考答案和评分规则。标记为 `enabled: false` 的题目保留在题库中，并排除计分。 [题库](benchmark/probes/) · [出题指南](benchmark/spec/authoring_guide.md)

| 位置 | 用途 |
| --- | --- |
| [`benchmark/probes/`](benchmark/probes/) | 题目、参考答案和逐题评分规则 |
| [`benchmark/policy/`](benchmark/policy/) 和 [`benchmark/merged/`](benchmark/merged/) | 原子能力评分策略和补充能力定义 |
| [`benchmark/eval/`](benchmark/eval/) | 评测入口、评分器、裁判、校验工具和回归测试 |
| [`benchmark/spec/`](benchmark/spec/) 和 [`benchmark/docs/`](benchmark/docs/) | 评分规范、出题指南、能力覆盖和离线审计说明 |
| [`run_all.py`](run_all.py) 和 [`build_dashboard.py`](build_dashboard.py) | 批量评测和本地结果导出 |

静态代码评分只检查支持范围内的指定语法或行为。若将 ACE 用作独立测试集，请将题目和答案排除在训练数据之外。

## 许可证

本仓库发布的所有 probe 和题目均由 ACE 团队自建。评测器代码和完整题库均采用 MIT 许可证，题库授权涵盖题目、参考答案和逐题评分规则。 [代码许可证](LICENSE) · [题库许可证](benchmark/probes/LICENSE)
