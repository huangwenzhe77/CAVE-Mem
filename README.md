# CAVE-Mem

当前 v2 实现按照 proposal-2 构建**动作条件化、风险约束失败记忆干预框架**。学习对象是选择 `Silent / Remind` 的控制器；任务执行 LLM、文本编码器、记忆库和检索器保持冻结。

**当前状态（2026-11-12）：76 项离线测试通过。** 已接通源验证、采集、训练、策略选择和评估；Terminal-Bench 2.0 的 Docker/API 接口已实现，尚未执行真实容器、模型服务或官方任务实验。测试通过不代表研究假设已得到验证。

## 方法与代码对应

主流程：**源失败轨迹 → 修正假设 → 独立回放确认 → 固定记忆库 → 缓存未经提醒的 a0 → 检索一条候选 → 构造 v/ρ/e 特征 → Silent/Remind → 终局评价 → 训练控制器。**

| 变量 | 含义 |
| --- | --- |
| `v ∈ {0,1,?}` | 源证据是否支持该修正；只有 v=1 的记忆进入主干预候选集 |
| `ρ ∈ {0,1,?}` | 机制在目标当前可观测状态下是否适用 |
| `e ∈ {0,1,?}` | 当前缓存动作 a0 是否暴露于该机制；不能用相似度代替 |
| `τ(x)` / `h(x)` | 提醒相对沉默的条件增量效用 / 指定配对协议下的成功变失败风险 |
| `π(1\|x)` | 控制器选择 Remind 的概率 |

检索使用当前公开上下文和固定窗口的执行历史，在整个冻结库中按词法或冻结向量相似度选择 top-1。`ρ/e` 在主策略中是软特征，`hard_vre` 才使用三者全为1的硬规则。Silent 直接执行 a0；Remind 允许一次新提案，提醒仅出现在该次请求中。复本共用 a0，之后两臂使用相同续跑预算。

源 Eq.6 使用筛选与独立确认队列，支持改善才建立有效证据；机制解释本身仍是假设。终局效用为 `R=Y−λC·C−λT·T`，配对标签为 `D=R1−R0`、`H=1{Y0=1,Y1=0}`。小型 NumPy MLP 拟合收益、风险与成本，再训练独立二元 policy；单臂 Eq.8 使用分组 cross-fitting 的 μ0/μ1，不从单臂日志虚构配对 H。

| 文件（位于 `src/cave_mem_v2/`） | 职责 | Proposal 对应 |
| --- | --- | --- |
| `contracts.py`, `config.py` | 公开/私有边界、记忆合同、效用、预算和配置 | §2，Eq.1–3 |
| `source.py`, `memory.py` | 修正候选、独立确认、来源校验、不可变库和固定检索 | §3.1–3.2，Eq.6 |
| `features.py` | 冻结编码、公开谓词、v/ρ/e、未知掩码 | §2.1、§3.2，Eq.2 |
| `providers.py`, `runtime.py` | 模型调用、缓存a0、瞬时提醒、一次干预、成本计量 | §2.2、§3.3，Eq.3–5 |
| `environments.py`, `alfworld.py` | 环境合同、factorial测试环境、ALFWorld单臂扩展 | §3.3、§5.1 |
| `terminal.py` | 固定任务资产、Docker恢复、独立官方评分 | §3.3、§5.1 |
| `data.py` | 采集前依赖分组、五集合、标签重算和权重 | §3.3、§5.3 |
| `learning.py` | 收益/风险模型、校准、SFT warm start、KL与约束策略 | §4.1–4.3，Eq.7–14 |
| `evaluation.py`, `baselines.py` | 配对/DR/部署评估、组级统计、规则与监督基线 | §4.2–4.3、§5.2–5.3 |
| `workflow.py`, `terminal_workflow.py`, `cli.py` | 离线及真实任务工作流、命令入口 | Algorithm 1、§6 |

## 安装与离线验证

使用 Python 3.11/3.12，在项目根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,dev]'
python -m unittest discover -s tests -v
cave2 doctor
cave2 pipeline --config configs/development.toml --output runs/demo-paired
```

Windows PowerShell 使用 `py -3.12 -m venv .venv` 和 `.venv\Scripts\Activate.ps1`，其余 Python/cave2 命令相同。单臂工程验证在pipeline命令后加 `--kind single`，并换用新输出目录。

默认pipeline使用人工factorial环境、scripted actor和结构化特征，仅验证程序链路。安装 `.[encoder]` 并配置固定revision才会加载真实文本编码器；所有真实模型请求都需要显式 `--real-actor`。输出目录需为空，修改模型、协议、规则或清单应建立新run。

## 首轮 benchmark：Terminal-Bench 2.0

采用离线文件任务子集和 `stateless-offline-filesystem-v1` 协议：每个shell动作从不可变镜像启动新容器，固定cwd/env、关闭actor网络；只持久化文件系统、公开历史和预算。`cd`、shell变量及后台进程不跨动作保留。官方测试在独立评分容器运行，actor不挂载tests/solution。

这是 **Terminal-Bench 2.0 task subset / filesystem-boundary protocol**，与Harbor的持久shell条件有区别，不能直接当作官方排行榜成绩。所有对照使用相同协议；恢复认证只覆盖审核过的任务状态，OS时钟/RNG及远端模型seed不因此获得完全耦合。

[terminal-pilot.toml](configs/terminal-pilot.toml) 提供六个待审核候选：源任务 `fix-git`，目标任务 `log-summary-date-ranges`、`openssl-selfsigned-cert`、`large-scale-text-editing`、`sqlite-db-truncate`、`regex-log`。这仅用于探索pilot，不能保证有可迁移的有效记忆，也不够五集合训练。正式研究需扩充独立任务、来源与留出组，不能把复本当成独立任务。

<details>
<summary><strong>展开 Linux 部署、任务审核与完整运行步骤</strong></summary>

### 1. 部署依赖与冻结模型

需要Linux、可用Docker daemon、Git和OpenAI-compatible模型服务。门控及默认编码器可在CPU上运行，LLM可使用远程GPU服务。先安装适合服务器的Docker和vLLM，再执行：

```bash
python -m pip install -e '.[train,encoder,dev]'
docker version
```

初始actor建议使用Qwen2.5-7B-Instruct，编码器使用BGE-small-en-v1.5；配置已固定编码器revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`。在独立推理环境启动模型：

```bash
export CAVE_MODEL_API_KEY='replace-with-your-server-token'
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --served-model-name qwen2.5-7b-a09a35458c70 \
  --dtype auto --max-model-len 32768 \
  --host 127.0.0.1 --port 8000 --api-key "$CAVE_MODEL_API_KEY"
```

实验进程设置同名环境变量；远程服务修改配置的 `base_url`。密钥不写入配置。显存需求随精度和上下文变化，7B BF16建议从24 GiB级GPU评估；量化模型需登记为不同actor。输出上限及预算应提前固定；保存权重、镜像及Python/Docker/vLLM版本。seed请求不代表服务保证确定性。

### 2. 固定资产并诊断恢复

下面以pilot为例。正式研究复制另一份配置并扩充任务，随后以新的 `runs/tb-study` 路径重新执行准备、审核和来源构建。

```bash
cp configs/terminal-pilot.toml configs/my-terminal.toml
git clone https://github.com/laude-institute/terminal-bench-2 ../terminal-bench-2
git -C ../terminal-bench-2 checkout --detach 2fd12b88aafdd04a52c298e3940bcb189f9766d6
python scripts/prepare_terminal_bench.py \
  --repository ../terminal-bench-2 --config configs/my-terminal.toml \
  --output runs/tb-pilot/manifest
cave2 terminal-preflight --config configs/my-terminal.toml \
  --manifest runs/tb-pilot/manifest/manifest.json --output runs/tb-pilot/preflight.json
python scripts/diagnose_terminal_restore.py \
  --manifest runs/tb-pilot/manifest/manifest.json --output runs/tb-pilot/restore-diagnostics
```

准备脚本锁定Git commit与资产哈希；诊断检验文件保留和分支隔离，**不会自动批准任务范围**。结合任务指令、环境、评分协议及真实smoke，审核无外部状态/持续服务、文件边界语义和时钟/RNG假设；有依据时填写诊断文件的 `reviewer`、`review_notes`、`reviewed_constraints`，保存到manifest对应的 `audits/<task-id>.json`。然后重跑preflight检查 `paired_primary_eligible`。不能把诊断中的false批量改成true。

源Eq.6也要求合法恢复；目标采用single不能绕过来源验证，不能恢复时也不能生成配对H。六任务配置的 `training_cohort_feasible=false` 是预期结果，仍可运行自然轨迹pilot。Docker镜像是恢复材料，运行期间不要全局prune。

### 3. 自然轨迹与源记忆

```bash
cave2 terminal-pilot --config configs/my-terminal.toml \
  --manifest runs/tb-pilot/manifest/manifest.json \
  --output runs/tb-pilot/source-pilot --role source --real-actor
cave2 terminal-pilot --config configs/my-terminal.toml \
  --manifest runs/tb-pilot/manifest/manifest.json \
  --output runs/tb-pilot/target-pilot --role target --real-actor
```

检查 `pilot.json`、`traces/`。在源 `candidates.json` 中保留生成的schema/pilot_hash、trace_id、step_index和真实problem_action，填写可执行 `correction`、机制假设 `mechanism`、`positive_conditions`/`exclusion_conditions`，审查后设 `approved=true`。每个源检查点固定一个修正，再执行：

```bash
cave2 terminal-source --config configs/my-terminal.toml \
  --manifest runs/tb-pilot/manifest/manifest.json \
  --pilot runs/tb-pilot/source-pilot --candidates runs/tb-pilot/source-pilot/candidates.json \
  --output runs/tb-pilot/sources --real-actor
```

默认每个候选4对筛选、32对独立确认；不支持改善则停止或排除，不把手工approved当成v=1。用于调配置的探索任务不能继续作为未触碰的最终test。

`public_fact_rules`/`exposure_rules` 需在正式配置/manifest冻结前登记。状态事实要求精确诊断动作、成功完整输出及审核依据，默认下一次动作失效；跨动作保留需登记允许动作及依据。它只解释actor已有公开观察，不隐式执行探针。缺事实或机制规则时ρ/e为 `?`。格式见 [事实审计测试](tests/test_terminal_fact_audit.py) 和 [环境测试](tests/test_terminal_environment.py)。默认pilot没有任务专属规则；检验H1的ρ=1,e=0需要在探索阶段建立并审核这些规则。

### 4. 正式训练与评估

扩充并审核正式清单后，按上一步建立与正式配置匹配的来源bank，再运行：

```bash
cave2 terminal-pipeline --config configs/my-terminal.toml \
  --manifest runs/tb-study/manifest/manifest.json --sources runs/tb-study/sources \
  --output runs/tb-study/full-pipeline --kind paired --real-actor
```

它串联采集、数据导出、训练、selection实际rollout、audit/test评估、冻结策略和no-memory执行，以及六种同组件规则基线。需要分步运行时使用以下入口，完整参数见 `cave2 <命令> --help`：

| 阶段 | 命令 | 主要输出 |
| --- | --- | --- |
| 采集 | `terminal-collect` | 冻结a0/特征、split-manifest、配对或随机单臂记录 |
| 数据 | `dataset` | train/calibration/selection/audit/test、quality及排除原因 |
| 学习 | `train` | outcome、supervised、direct/DR、约束候选与policy |
| 选择 | `terminal-select` | 仅使用selection实际执行选型，不可行时Silent回退 |
| 离线评估 | `evaluate` | audit/test的配对或DR估计 |
| 实际评估 | `terminal-rollout` | 冻结策略或指定基线的实际执行及成本 |

基线参数：`--no-memory` 真正跳过检索/门控；`--baseline` 支持 `silent`、`always_remind`、`source_only`、`applicability`、`exposure`、`hard_vre`、`relevance`。silent仍有共同检索开销；relevance另设 `--relevance-threshold`，阈值只能在验证阶段选择。

</details>

## 实验约束与解释

划分先于目标结局，同检查点复本整体分组。五个集合工程上各至少两个独立组，单臂训练至少三个且各折各臂需有支持，这只是运行下限。`source_disjoint=false` 表示固定共享bank下的目标泛化；来源/机制/模板留出需新建实验，登记 `source_disjoint=true`、语义轴及足够独立来源，分组不足不能静默放宽。

默认成本单位为模型调用次数，λ=0.01、δ=0.05、B=2.0只是示例。tokens、工具/编码器/gate调用及耗时另报，未知费用保留null，不当作美元或完整硬件预算。配对与单臂估计分开；后者不宣称识别了配对伤害。校准和bootstrap是经验诊断，不是形式安全保证。

尚未完成：真实模型结论与两档actor迁移、完整来源/机制留出及噪声库实验、PMA/TwinCheck等原系统完整复现、τ²-Bench适配，以及可选Inspect的两阶段探索/VOI（Eq.15）。轨迹差异d保留输入接口和unknown掩码，尚无通用估计器。ALFWorld单臂示例见 [alfworld_single.py](examples/alfworld_single.py)，未认证精确配对恢复。

## 工程参考与验证

实现不依赖旧项目运行。参考过 CAVE-Mem 的 `source/alfworld.py`、`source/replay.py`、`memory/provenance.py`、`memory/retrieval.py`、`data/splits.py`：借鉴失败步骤定位、独立回放、证据校验及连通分组。参考 MemRL 的 `providers/embedding.py`、`providers/llm.py`、`service/retrievers.py` 和trace模块：借鉴真实编码器封装、检索接口与调用计量。MemRL的可变memory-Q不等同于本项目的τ(x)，未加入主策略；旧工程的证据规则和暴露代理也没有直接当成正确结论。

参考提交：CAVE-Mem `591b276e55880023365913b60003a9c6cc2352e6`；MemRL `c1b322ca43de36ddf64c6712f89d0095bfc35ce0`。本项目新写Terminal任务适配与恢复范围审核，没有继承任何通用恢复认证。

2026-11-12验证：`python -m unittest discover -s tests -v` **76项通过**，编译及CLI检查通过。包含来源pilot到训练/选型/六基线的注入后端闭环、公开事实来源与失效、私有verifier隔离、基础设施错误、恢复文件泄漏、共享bank范围及数据泄漏检查。真实Docker、API、编码器权重、官方任务均未执行/加载。

## 仓库与运行产物

核心提交内容为 `src/`、`configs/`、`scripts/`、`tests/`、`pyproject.toml`、本README和`.gitignore`；`examples/`为可选扩展示例。`requirements-verified.txt`是历史开发环境记录，不是Linux依赖锁。

`runs/`、`.venv/`、模型权重、缓存与`.env`已由`.gitignore`排除。runs中的JSON分别保存检查点、来源证据、采集记录、数据集、模型或报告，结构相似不表示内容重复。`private/`、原始轨迹和verifier报告属于审计材料，不能直接作为训练特征。本地历史文档保留在忽略的runs归档中，公开仓库以本README为统一说明。
