"""Reports only measured results; final confirmation is separate from exploration."""
from __future__ import annotations

import json
from pathlib import Path
from .storage import atomic_json


def rank_key(record: dict, direction: str) -> tuple:
    evaluation = record.get("evaluation") or {}
    value = evaluation.get("objective")
    success = record.get("status") == "completed"
    feasible = success and evaluation.get("feasible", False)
    objective = float(value) if value is not None else (float("-inf") if direction == "maximize" else float("inf"))
    oriented = objective if direction == "maximize" else -objective
    violation = evaluation.get("violation")
    violation = float(violation) if violation is not None else float("inf")
    return (int(feasible), int(success), -violation if not feasible else 0, oriented)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_report(config: dict, state: dict, output: Path) -> dict | None:
    trials = state.get("trials", [])
    direction = config["objective"]["direction"]
    verified = [t for t in trials if t.get("phase") == "confirmation" and t.get("status") == "completed" and t.get("evaluation", {}).get("feasible")]
    best = max(verified, key=lambda t: rank_key(t, direction)) if verified else None
    if best:
        atomic_json(output / "best.json", {"status": "confirmed", "trial_id": best["id"], "parameters": best["parameters"], "evaluation": best["evaluation"], "config_hash": state["config_hash"]})
    elif (output / "best.json").exists():
        (output / "best.json").unlink()
    lines = [f"# {config['name']} — PoC 实验报告", "", "本报告仅覆盖配置中声明的测试与门槛。演示模式中的指标是合成数据。", "",
             f"- 状态：{state.get('status', 'unknown')}",
             f"- 配置 SHA256：`{state['config_hash']}`",
             f"- 目标：{config['objective']['direction']} `{config['objective']['metric']}`；场景聚合 `{config['objective']['scenario_aggregation']}`",
             f"- 已使用预算：{state.get('elapsed_s', 0):.1f} 秒",
             f"- 已确认通过的最佳配置：{best['id'] if best else '无；不能将探索候选视为最终验收通过'}", "",
             "同一候选必须通过全部场景、全部重复测量的要求。P99 按每次测试独立检查，未将多次 P99 平均后冒充合并 P99。", "",
             "| Trial | 阶段 | 状态 | 全部达标 | 目标值 | 未达标程度 |",
             "|---|---|---|---|---:|---:|"]
    for trial in trials:
        ev = trial.get("evaluation") or {}
        lines.append("| " + " | ".join(_cell(x) for x in [trial['id'], trial['phase'], trial['status'], ev.get('feasible', False), ev.get('objective', '—'), ev.get('violation', '—')]) + " |")
    if best:
        lines += ["", "## 已确认配置", "", "```json", json.dumps(best['parameters'], ensure_ascii=False, indent=2), "```", "", "## 最终验收明细", "",
                  "```json", json.dumps(best['evaluation'], ensure_ascii=False, indent=2), "```"]
    lines += ["", "## 复现与适用范围", "", "使用原始配置、相同权重/数据集/依赖版本和 `best.json` 参数复测。`trials/` 保留每轮原始数据与进程日志；`state.json` 保留全部记录。", "",
              "本工具不自动部署获胜配置、不修改源码/权重、不推断未配置的精度门槛，也不保证任意硬件上的性能收益。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return best
