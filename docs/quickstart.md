# 从看懂到复现：第一次使用指南

v0.7.0 的推荐入口是 **[自然语言全参数调优](open-tuning.md)**。新模式用一份 runtime 接线加一段需求，未明确限制的参数由 Agent 搜索；下文原 `agent-demo` 与三文件配置保留为**受控模式复现路径**，不会自动改成开放搜索。

新模式每次提出候选后，还须通过 MCP 或 A2A→MCP 核查 vLLM-Ascend 文档与实现代码，由大模型审查本轮具体改动。证据不足或兼容性仍有疑问时不执行该候选；审查通过后仍需真实测量和独立复核。

已经下载仓库并准备好 Linux / Python 3.10+ 时，可以先运行新的无卡演示：

```bash
python3 -m ascend_poc autotune-demo --output runs/my-open-demo
python3 -m ascend_poc autotune-report --output runs/my-open-demo
```

没有环境或代码时，先完成下面第 2 节，再按 [新模式指南](open-tuning.md) 操作。新演示同样使用合成指标，不代表真实昇腾性能。

以下受控模式像一名有实验记录本的调优助手：你先给它客户的“及格线”和允许尝试的设置，它反复测量、分析、查资料，再尝试下一组设置。**只有复测达标且确实优于当前方案的配置，才会被保存为新冠军。**

不熟悉推理参数时，先读 [参数与策略通俗图解](tuning-explained.md)。下面的第一部分不需要昇腾卡，也不需要买大模型 API；可以先看懂一次完整实验，再决定如何接入客户环境。

## 1. 先选择你的目标

| 你现在想做什么 | 使用哪条路径 | 能证明什么 |
|---|---|---|
| 用文字需求开始开放调优 | [新模式指南](open-tuning.md)，或本页开头的 `autotune-demo` | 演示验证程序链路；真实结果需接入现场环境测量 |
| 看懂原受控 Agent 怎么调优，展示给同事 | 第 2–4 节：受控模式本地演示 | 程序的工具调用、测量、复核与报告链路可运行 |
| 用原受控模式完成真实 PoC | 第 6–9 节：固定契约与获准策略 | 需要真实模型、负载、性能与质量评测；成绩由实测决定 |
| 审核为什么某个方案胜出 | 查看报告及实验文件 | 是否达标、是否复测、提升依据和失败原因 |

演示中的模型回答、研究内容和性能数字都是预设的合成数据。本地演示成功，不代表真实大模型会提出好方案，也不代表昇腾硬件已得到加速。

## 2. 准备一个能运行 Python 的 Linux 终端

**Linux 用户：**打开终端。控制器需要 Python 3.10 或更新版本。

**Windows 用户：**在 WSL 2 的 Ubuntu 终端中执行本指南的 Bash 命令。尚未安装时，按 [Microsoft 的 WSL 安装说明](https://learn.microsoft.com/en-us/windows/wsl/install)准备：在管理员 PowerShell 中执行 `wsl --install`，按提示重启，再打开 Ubuntu。后面的 `python3`、`cd` 等命令在 Ubuntu 中执行。WSL 路径用于体验 CPU 演示；真实 NPU 测试应在已验证的 Linux 昇腾环境内进行。本项目本次验证环境为 Linux，未声称完成 Windows/WSL 或 macOS 实机验证。

先检查 Python：

```bash
python3 --version
```

输出应为 `Python 3.10.x` 或更新版本。如果找不到 Python，Ubuntu/Debian 可以先安装：

```bash
sudo apt update
sudo apt install python3 git
```

安装后再次检查版本；如果仍低于 3.10，请由环境维护者升级 Python 或提供符合要求的 Linux 环境。

### 获取代码：选一种方式即可

**方式 A：使用 Git 克隆。**

```bash
git clone https://github.com/xushuwei281/ascend-poc-autotuner.git
cd ascend-poc-autotuner
```

当前仓库公开，使用上述 HTTPS 地址通常无需登录。若克隆失败，先在浏览器核对仓库地址与网络访问；不要把 API key 或 GitHub token 写进克隆地址。

**方式 B：不熟悉 Git。**打开 [仓库首页](https://github.com/xushuwei281/ascend-poc-autotuner)，选择绿色 **Code → Download ZIP**，解压后在该文件夹打开 Linux 终端。ZIP 下载的目录通常叫 `ascend-poc-autotuner-main`；进入实际解压目录，不要直接复制一个不存在的路径。

两种方式最终都应位于**仓库根目录**，也就是同时含有 `README.md`、`ascend_poc/`、`examples/` 的文件夹。检查：

```bash
ls README.md ascend_poc examples
python3 -m ascend_poc --help
```

这里直接运行源码，无需 `pip install`，也无需下载模型。若你想安装命令行入口，见 [环境与安装](guide.md#1-环境与安装)。

## 3. 复制两条命令，跑一次完整演示

在仓库根目录执行：

```bash
python3 -m ascend_poc agent-demo --output runs/my-first-demo
python3 -m ascend_poc agent-report --output runs/my-first-demo
```

第一条会启动本机模拟服务，演示 Agent 查资料、尝试候选、复核、晋升，再保留一个失败案例；第二条根据保存的证据重新生成报告。运行结束时终端会显示 `agent_report.md` 的完整路径。演示会占用本地临时端口，不连接外部模型 API，也不产生外部 API 费用。

**再次运行时换一个新目录**，例如 `runs/my-second-demo`。程序拒绝覆盖已有实验，避免把两次运行的记录混在一起。

打开结果有两种方法：

- 在文件管理器进入 `runs/my-first-demo/`，用文本编辑器打开 `agent_report.md`。支持 Markdown 的编辑器可显示表格和标题。
- 只使用终端时，执行下面的命令直接看内容：

```bash
cat runs/my-first-demo/agent_report.md
```

在 WSL 中，也可以从 Ubuntu 终端输入 `explorer.exe .`，在 Windows 文件管理器打开当前仓库，再进入上述报告目录。

## 4. 把报告当作一张实验成绩单

| 报告或文件中的词 | 通俗含义 | 应该怎么判断 |
|---|---|---|
| `experiment` | 试一次新方案 | 先看改了什么、为什么改 |
| `observed_feasible` | 探索时看起来达标 | 还需要独立复核 |
| `accepted` | 通过本次完整独立验收 | 还要看是否真的优于参照 |
| `improved` | 改进幅度通过已设的判定规则 | 与 `accepted` 等其他条件共同决定能否晋升 |
| `generation` | 参照方案更新到第几代 | 次数不等于加速倍数，也不代表模型变聪明 |
| `infeasible_early` | 已测部分不及格，提前停止探索 | 未测部分不会补成零，也不算完整通过 |
| `skipped_insufficient_budget` | 估算剩余时间不够完整复核 | 尚未确认，不能当作验收通过 |
| `champion` | 已通过晋升条件的配置 | 看对应证据，不能只看最高吞吐数字 |

这个内置演示安排了一个可晋升候选和一个延迟不达标候选。因此，正常完成后应该能看到：**一组方案得到保留，另一组即使吞吐更高也不能取代它。** 这是演示的预设情节，不能推广为真实 PoC 一定会成功。

| 输出文件 | 非技术读者先看什么 |
|---|---|
| `agent_report.md` | 总结、实验结论、停止原因与冠军谱系 |
| `champion.json` | 最终保留的策略名称、参数及证据身份 |
| `champion-contract.json` | 交给工程师复现冠军所需的契约配置 |
| `experiments/` | 工程师核对每次原始测量、复核和失败原因的位置 |
| `events.jsonl` | 按顺序记录 Agent 做过的动作，供审计排障 |

**真实任务里，没有冠军文件也可能是合理结论。** 初始方案可能已经很好，候选可能不达标，也可能预算不足。程序结束和“找到新冠军”是两件事；不能用大模型的一句“成功了”代替测量。

### 可选：体验更严格、也更耗时的复核

普通 `aba` 可以理解为“旧方案 → 新方案 → 再测旧方案”，检查新方案是否只是碰巧赶上机器状态较好。`aba_ci` 会完整重复多组，再计算改进的置信下界，同时检查机器漂移；它不是只挑最好的一组。

```bash
python3 -m ascend_poc agent-demo --verification aba_ci --output runs/my-ci-demo
python3 -m ascend_poc agent-report --output runs/my-ci-demo
```

该演示仍使用合成数据。统计方法有适用假设；在真实机器上增加测量次数，也不自动消除环境干扰。设置与成本见 [独立复测说明](guide.md#6-预算硬门槛与独立复测)。

## 5. 第一次运行遇到问题

| 看到的现象 | 可以先做什么 |
|---|---|
| `python3: command not found` | 安装 Python，再确认版本至少为 3.10 |
| `No module named ascend_poc` | 回到含 `README.md` 和 `ascend_poc/` 的仓库根目录 |
| GitHub 404、克隆失败 | 核对仓库地址与网络访问；网页可访问时也可下载 ZIP |
| `requires a fresh, empty output directory` | 换新的 `--output` 路径；不要删除旧证据来掩盖错误 |
| 运行后不知道看哪里 | 找终端打印的报告路径，或执行上面的 `cat` 命令 |
| 演示报端口或进程错误 | 使用允许本地回环连接和创建子进程的 Linux 环境；检查是否被沙箱或系统策略限制 |
| `agent-report` 报完整性错误 | 保留原输出，核对是否改动过配置、代码或记录；不要手动改成绩文件 |

## 6. 受控模式接真实昇腾环境：先把工作分清

本节至第 9 节继续使用旧 `agent/workflow` 的固定契约与获准策略。开放 `autotune` 的接入方式见 [运行接线](open-tuning.md#2-客户需求与运行接线分开)，不要求预先批准每个候选 profile。跑通演示后，真实 PoC 还需要客户现场接入。建议按下表分工；一个人可以承担多个角色。

| 角色 | 需要准备的内容 | 完成标志 |
|---|---|---|
| 客户／售前负责人 | 模型与精度、输入输出长度、并发或请求速率、缓存政策、质量与延迟及格线、卡数与测试预算 | 各候选共用一份不变的验收表 |
| 昇腾环境工程师 | 卡型与拓扑、驱动、CANN、torch-npu、vLLM-Ascend、模型与 tokenizer；确认版本匹配 | 手工启动服务并完成基线测量 |
| 评测／Agent 接入工程师 | 客户性能脚本、质量脚本、指标 JSON、API 配置、获准策略和预算 | 三层配置校验通过，固定测量链路跑通 |

需要提前拿到以下材料：

- 固定版本的模型、tokenizer、数据集与部署软件信息。
- 可用的昇腾卡、专用服务端口，以及实际设备编号。
- 已手工跑通的性能脚本与质量脚本；两者输出格式见 [测试适配器](guide.md#2-接入客户-poc)。
- 一个支持所需 Chat Completions 行为的大模型 API 地址、实际模型 ID 和 API key。
- 能访问 DeepWiki MCP 的网络；如选择 A2A，按 [研究连接说明](guide.md#4-策略与研究配置参考)先启动本地桥。
- 现场允许尝试的策略清单及验证记录，包括每个开关是否被当前软硬件组合支持。

仓库不会安装昇腾软件栈、下载客户模型，或代替客户编写正确性评测。模板中的 `/opt/customer/run_performance.py`、`run_quality.py` 是待提供的文件，不是仓库内现成脚本。

## 7. 三个配置文件，各管一件事

本节仅适用于旧 `agent/workflow` 受控模式。新的 `autotune` 使用 runtime + 自然语言需求，见 [运行接线](open-tuning.md#2-客户需求与运行接线分开)。

| 文件 | 想成什么 | 需要填写什么 |
|---|---|---|
| `customer-contract.json` | 客户的验收合同 | 服务启动方式、固定测试、模型和设备背景、指标门槛、可调参数、大模型 API |
| `customer-workflow.json` | 允许使用的方案菜单 | 引用合同、获准策略、兼容性证据、重复次数、预算和复核规则 |
| `customer-agent.json` | 调优助手的任务单 | 引用菜单、文字目标、工具／实验／API 次数上限、总时间与经验文件路径 |

Agent 的 `goal` 是给规划模型看的文字任务说明；正式评分来自契约的 `objective`，及格线来自 `requirements`。只修改 `goal`，不会改变程序的评分规则。

在仓库根目录复制三份模板。以下操作只准备文件，不会启动 NPU 服务：

```bash
cp examples/customer-template.json customer-contract.json
cp examples/ascend-workflow.json customer-workflow.json
cp examples/inference-rsi.json customer-agent.json
python3 - <<'PY'
import json
from pathlib import Path

workflow_path = Path("customer-workflow.json")
workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
workflow["contract"] = "customer-contract.json"
workflow_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

agent_path = Path("customer-agent.json")
agent = json.loads(agent_path.read_text(encoding="utf-8"))
agent["workflow"] = "customer-workflow.json"
agent["memory_file"] = "runs/customer-memory.json"
agent["goal"] = "保持客户负载和质量约束，满足全部门槛后提高最坏场景吞吐。"
agent_path.write_text(json.dumps(agent, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
```

重复复制会覆盖这三份准备文件；已经编辑过的客户配置不要再次覆盖。引用路径相对**引用它的配置文件所在目录**；测试命令的工作目录则由之后的 `--workdir` 指定。

### 先编辑这些位置，再运行真实实验

| 文件中的位置 | 必须核对的内容 |
|---|---|
| 契约 `context`、`resource_id` | 真实卡型、卡数、模型身份；同组卡使用一致资源锁标识 |
| 契约 `server.command`、`server.env` | 模型路径变量、设备编号、TP 卡数、端口、上下文长度与实际支持的开关 |
| 契约 `scenarios` | 替换性能／质量脚本路径，并固定数据、负载、预热及缓存政策 |
| 契约 `requirements`、`objective` | 改成客户认可的门槛与目标；模板里的数字只是示例 |
| 契约 `llm` | 填实际 API 前缀和模型 ID；不要把 key 填进 JSON |
| 工作流 `profiles` | 基线写入现场验证记录；其他策略验证后才设为启用且 `verified` |
| 工作流与 Agent 的 `budget`／`settings` | 留出模型加载、重复测量和完整最终复核时间；两层预算都会限制执行 |

工作流中的 profile 可以替换整份服务配置。若启用模板中的 `eager`、`prefix_cache` 等方案，要逐个检查其设备、端口、模型和固定参数是否与客户基线一致，不能只改契约后假定其他方案也自动更新。

**模板里基线的 `verified` 是为了展示配置结构，其说明明确是占位文本；这不是现场兼容性证明。** Agent 不会替你核实所有硬件证据。没有验证过的策略继续保持关闭。

## 8. 先校验，再启动真正的 Agent

### A. 只检查配置，不跑卡、不调用模型 API

```bash
python3 -m ascend_poc doctor
python3 -m ascend_poc validate --config customer-contract.json
python3 -m ascend_poc workflow-validate --config customer-workflow.json --estimate
python3 -m ascend_poc agent-validate --config customer-agent.json
```

`Valid ...` 只表示配置的结构与内部逻辑通过，不代表硬件兼容、API 鉴权成功或客户验收通过。`doctor` 显示可获得的软件与 NPU 信息；它不是完整环境认证。成本估算中的 `null` 表示尚无可用估计，不是零耗时。

### B. 在已准备好的昇腾 Linux 主机／容器中启动

下面的 `MODEL_PATH` **必须替换为实际模型目录**。在 Bash 中执行 `read` 后粘贴 API key，按回车；输入不回显，也不会把 key 明文写入该条命令的历史记录。

```bash
export MODEL_PATH=/absolute/path/to/approved-model
read -rsp 'Planner API key: ' POC_LLM_API_KEY
export POC_LLM_API_KEY
python3 -m ascend_poc agent \
  --config customer-agent.json \
  --output runs/customer-poc-001 --workdir "$PWD"
```

该命令会调用真实大模型 API，启动／停止配置中的推理服务并使用 NPU；API 调用和硬件占用按现场规则计费。程序不自动读取 `.env`，也不自动给模板补齐路径。服务端口应空闲，服务和包装脚本应在前台运行，由控制器管理其生命周期。

先使用可容纳完整复核的小规模接线任务；正式验收再按已约定协议设置预算，并使用新的输出目录。不要在同一次任务中改客户门槛、缩短回答或只保留表现最好的请求来提高分数。

真实 `agent` 的退出码可以帮助接入人员判断：`0` 表示有当前有效的晋升冠军，`2` 表示没有有效晋升，`1` 表示配置／完整性／执行异常，`130` 表示中断。基线已经达标但没找到提升时，仍可能返回 `2`。

## 9. 看结果、恢复任务与交付

生成或重新核对报告：

```bash
python3 -m ascend_poc agent-report --output runs/customer-poc-001
```

中断后，只有**原配置、原代码／环境和原证据仍有效**时，才恢复同一次任务：

```bash
python3 -m ascend_poc agent \
  --config customer-agent.json \
  --output runs/customer-poc-001 --workdir "$PWD" --resume
```

恢复会沿用原始截止时间，暂停期间时间仍在流逝，不会重新获得一份预算。修改配置、环境、代码、验收数据或验证模式后，请开新输出目录。若原任务使用 `--ignore-memory`，恢复也需保持该开关。详见 [恢复与报告](rsi.md#6-预算恢复与报告)。

给客户交付时，至少一起提供：

1. 客户验收口径及软硬件、模型、数据版本说明。
2. 运行报告、性能和质量结果，以及失败／未确认情况。
3. 若产生冠军：`champion.json`、`champion-contract.json` 与其对应的完整复核证据。
4. 工程师可复跑的命令、环境准备方式、配置和测试适配器。

冠军契约仍引用现场环境、模型和客户脚本，单独复制一个 JSON 到陌生机器并不能复现。需要报告相对最初基线的最终加速时，另外按同口径比较“最初基线”和“最终冠军”；不要将各代提升百分比直接连乘成实测成绩。

客户配置、API key、业务数据和运行日志留在获准的客户环境；代码仓库用于共享工具与说明。继续深入可以读 [Agent 使用指南](rsi.md) 和 [完整测试手册](guide.md)。
