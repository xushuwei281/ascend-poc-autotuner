#!/usr/bin/env python3
"""Regenerate the four explanatory SVGs with Python 3.10+ (stdlib only).

Run from the repository root to rebuild:
    python3 scripts/render_docs_diagrams.py
Check that committed files match the generator without changing them:
    python3 scripts/render_docs_diagrams.py --check
Absolute script paths also work from any directory.

Outputs in docs/assets/ (width x height, in SVG pixels):
    optimization-loop.svg: 1200 x 1424
    parameters.svg:        1200 x 1210
    strategies.svg:        1200 x 1070
    config-map.svg:        1200 x 1180

The files are deterministic and have no embedded scripts or external assets.
"""
import argparse
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
CHECK = False
INK = "#17324d"
MUTED = "#52677a"
BLUE = "#2463b5"
TEAL = "#087f8c"
PURPLE = "#7158bb"
AMBER = "#9b6314"
RED = "#b34a54"


class SVG:
    def __init__(self, width, height, title, desc):
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(desc)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#60768c"/></marker></defs>',
            '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif} .code{font-family:"DejaVu Sans Mono",monospace}</style>',
        ]
        self.rect(0, 0, width, height, "#f4f7fb", r=0)

    def rect(self, x, y, w, h, fill="white", stroke=None, r=18):
        border = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"{border}/>')

    def text(self, x, y, text, size=23, fill=INK, weight=400, anchor="start", code=False):
        cl = ' class="code"' if code else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"{cl}>{escape(str(text))}</text>')

    def lines(self, x, y, lines, size=23, gap=34, fill=MUTED, weight=400, anchor="start"):
        for i, line in enumerate(lines):
            self.text(x, y + i * gap, line, size, fill, weight, anchor)

    def arrow(self, points, dashed=False, both=False):
        dash = ' stroke-dasharray="7 7"' if dashed else ""
        start = ' marker-start="url(#arrow)"' if both else ""
        d = "M" + " L".join(f"{x} {y}" for x, y in points)
        self.parts.append(f'<path d="{d}" fill="none" stroke="#60768c" stroke-width="3" stroke-linejoin="round" marker-end="url(#arrow)"{start}{dash}/>')

    def pill(self, x, y, w, text, bg="#e8f0fb", fg=BLUE, size=20):
        self.rect(x, y, w, 38, bg, r=19)
        self.text(x + w / 2, y + 26, text, size, fg, 600, "middle")

    def heading(self, title, subtitle):
        self.text(44, 60, title, 35, weight=700)
        self.text(44, 100, subtitle, 22, MUTED)

    def save(self, name):
        path = OUT / name
        content = "\n".join(self.parts + ["</svg>"]) + "\n"
        if CHECK:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Outdated or missing diagram: {path}")
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def optimization_loop():
    s = SVG(1200, 1424, "自然语言驱动的开放推理调优", "用户先写明想优化什么和哪些条件不能改变。系统生成评分目标与显式约束，未提的推理项开放。大模型分析实测反馈、查询资料、提出完整候选。候选通过结构与客户条件检查后，必须经MCP或A2A到MCP核查文档与源码，再由大模型审查具体改动，获准后才能启动测试。独立ABA复核通过且提升足够后晋升；初始参照不合格时，候选三次独立启动全部达标可建立合格基线。失败和证据不足也反馈到下一轮。")
    s.heading("说需求，让 Agent 用实测寻找更好的配置", "autotune：大模型分析与提案，核查文档和源码，审查通过后再实验。")
    s.rect(44, 132, 1112, 105, "#e6effb")
    s.text(70, 173, "你的文字需求 → 正式评分目标 + 明确限制", 27, BLUE, 700)
    s.text(70, 211, "只有写明的条件才锁住；没有提出的质量、延迟或精度要求，不自动补上默认门槛。", 22)
    s.arrow([(600, 238), (600, 269)])
    steps = [
        (282, "1  先分析，再查询资料", ["大模型读取当前成绩和失败原因，提出瓶颈假设。", "通过 DeepWiki 研究实际版本下的推理方法。"], "#edf0fb", PURPLE),
        (456, "2  提出完整候选配置", ["本机能力目录提供可用参数，可提议开关和嵌套 JSON。", "先检查结构、资源与客户明确条件，不要求预批准 profile。"], "#e8f5f5", TEAL),
        (630, "3  查文档与源码，大模型审查", ["通过 MCP 或 A2A→MCP 核查本轮具体改动。", "来源不全、兼容性未明或预算不足，不执行该候选。"], "#e8f5f5", TEAL),
        (804, "4  审查通过后，真正启动测量", ["按候选配置启动被测服务，固定评测器记录结果。", "启动失败、内存不足、参数不兼容也留下反馈。"], "#edf0fb", PURPLE),
        (978, "5  独立复核，再决定是否保留", ["有合格参照时，ABA 复核检查环境波动和总体提升。", "初始参照不合格时，候选三次独立启动须全部达标。"], "#eaf1fb", BLUE),
    ]
    for y, title, lines, bg, color in steps:
        s.rect(230, y, 740, 139, bg)
        s.text(258, y + 40, title, 27, color, 700)
        s.lines(258, y + 79, lines, 21, 32)
    for y in (422, 596, 770, 944):
        s.arrow([(600, y), (600, y + 27)])
    s.arrow([(460, 1118), (460, 1150), (305, 1150), (305, 1174)])
    s.arrow([(740, 1118), (740, 1150), (895, 1150), (895, 1174)])
    s.rect(70, 1184, 470, 113, "#e5f3ea")
    s.text(305, 1223, "达标且提升 / 建立合格基线", 25, "#267247", 700, "middle")
    s.text(305, 1261, "按复核规则保留，作为下一轮参照", 22, INK, anchor="middle")
    s.rect(660, 1184, 470, 113, "#fff0df")
    s.text(895, 1223, "失败 / 未提升 / 证据不足", 25, AMBER, 700, "middle")
    s.text(895, 1261, "保留原配置，带着原因继续分析", 22, INK, anchor="middle")
    s.arrow([(70, 1239), (44, 1239), (44, 351), (220, 351)], dashed=True)
    s.arrow([(1130, 1239), (1156, 1239), (1156, 351), (980, 351)], dashed=True)
    s.pill(69, 868, 133, "结果反馈", "#eef1f6", MUTED)
    s.pill(995, 868, 133, "结果反馈", "#eef1f6", MUTED)
    s.rect(44, 1328, 1112, 65, "white")
    s.text(600, 1368, "到达预算就结束 · 源码核查不能代替现场实测 · 不改评分脚本，不把模拟数字当实测", 21, MUTED, anchor="middle")
    s.save("optimization-loop.svg")


def slot_row(s, x, y, filled, total, color, width=43, gap=9):
    for i in range(total):
        s.rect(x + i * (width + gap), y, width, 44, color if i < filled else "#e6ebf2", r=9)


def parameters():
    s = SVG(1200, 1210, "开放参数调优的三个常见例子", "max-num-seqs限制每轮可处理的序列数。max-num-batched-tokens限制每轮处理的token预算。gpu-memory-utilization改变实例的设备内存预算。开放模式还可以提议精度、并行、缓存、上下文和嵌套JSON设置。只有客户明确要求限制的推理项才被锁住，可执行性还取决于实际引擎、资产与接线。")
    s.heading("三个例子，开放范围不限于此", "“参数”是工作方法中的设置；Agent 还可以发现并尝试其他推理选项。")
    data = [
        (135, "01  一轮接待多少个序列", "max-num-seqs", "像餐厅同一轮能照顾多少桌", ["调小：每轮处理的序列较少", "可能更易控制延迟，也可能吃不满算力"], ["调大：每轮可处理更多序列", "可能提高吞吐，也可能增加内存与延迟"], "slots"),
        (425, "02  一轮处理多少份文字", "max-num-batched-tokens", "像厨房这一轮最多处理多少工作量", ["调小：每轮 token 预算较少", "长输入可能分更多轮；效果取决于负载"], ["调大：每轮 token 预算较多", "可能提高输入处理效率，也可能拖慢输出"], "tokens"),
        (715, "03  给实例多少设备内存预算", "gpu-memory-utilization", "像为服务划出仓库预算，不是算力利用率", ["调小：留给其他开销的空间较多", "KV 缓存容量可能减少，更容易排队"], ["调大：本实例可用内存预算较多", "可能容纳更多 KV，但余量少、易内存不足"], "memory"),
    ]
    for y, title, param, analogy, small, large, kind in data:
        s.rect(44, y, 1112, 270)
        s.text(70, y + 40, title, 27, weight=700)
        s.text(1126, y + 40, param, 22, BLUE, anchor="end", code=True)
        s.text(70, y + 78, analogy, 21, MUTED)
        s.rect(68, y + 98, 512, 148, "#eff1fb")
        s.rect(600, y + 98, 532, 148, "#e9f5f3")
        if kind == "slots":
            slot_row(s, 92, y + 114, 3, 8, PURPLE)
            slot_row(s, 624, y + 114, 7, 8, TEAL)
        elif kind == "tokens":
            slot_row(s, 92, y + 114, 3, 10, PURPLE, 33, 9)
            slot_row(s, 624, y + 114, 9, 10, TEAL, 33, 9)
        else:
            for x, fraction, color in [(92, .60, PURPLE), (624, .85, TEAL)]:
                s.rect(x, y + 116, 430, 40, "#dce4ed", r=9)
                s.rect(x, y + 116, 430 * fraction, 40, color, r=9)
                s.text(x + 18, y + 143, "本实例预算", 19, "white", 600)
        s.lines(92, y + 192, small, 21, 32, INK)
        s.lines(624, y + 192, large, 21, 32, INK)
    s.rect(44, 1025, 1112, 153, "#fff1df")
    s.text(70, 1063, "精度、TP 并行、缓存、上下文、嵌套 JSON 也可提议", 25, AMBER, 700)
    s.lines(70, 1100, ["例如 additional-config；能否运行由实际引擎、资产和接线决定。", "只有用户明确的条件才锁定。序列名额 ≠ 外部并发用户数；图形仅示意方向。"], 21, 34)
    s.save("parameters.svg")


def strategies():
    s = SVG(1200, 1070, "推理策略是改变做事方法", "前缀缓存复用相同token前缀的KV计算；分块预填充让新请求的长输入与其他已读完输入请求的解码交错；量化减少数值表示所需空间但要验证质量和性能；张量并行分摊模型计算也带来通信成本。开放模式可提议这些策略，能否执行取决于实际引擎与可用资产。")
    s.heading("四种方法：先理解收益，再实测代价", "“策略”改变做事方法；能否使用、是否更快，取决于模型、昇腾软件版本和客户负载。")
    for x, y in [(44, 132), (612, 132), (44, 568), (612, 568)]:
        s.rect(x, y, 544, 412)
    # Prefix caching.
    s.text(70, 176, "前缀缓存：重复的开头少算一次", 25, BLUE, 700)
    s.pill(70, 195, 295, "Agent 可提议开关与相关参数", size=18)
    s.text(70, 271, "请求 A", 20, MUTED)
    s.rect(151, 244, 205, 40, "#dfeafa", r=8)
    s.text(253, 271, "共同的 token 开头", 18, BLUE, 600, "middle")
    s.rect(368, 244, 180, 40, "#e8e1f6", r=8)
    s.text(458, 271, "问题 A", 19, PURPLE, 600, "middle")
    s.arrow([(253, 287), (253, 308)])
    s.rect(151, 315, 205, 40, "#e2f1ee", r=8)
    s.text(253, 342, "保存可复用的 KV", 18, TEAL, 600, "middle")
    s.text(70, 402, "请求 B", 20, MUTED)
    s.rect(151, 375, 205, 40, "#e2f1ee", r=8)
    s.text(253, 402, "同前缀，命中时复用", 18, TEAL, 600, "middle")
    s.rect(368, 375, 180, 40, "#e8e1f6", r=8)
    s.text(458, 402, "新问题 B", 19, PURPLE, 600, "middle")
    s.arrow([(253, 357), (253, 369)])
    s.lines(70, 461, ["复用输入计算，不是直接返回旧答案。", "很少共享前缀时，收益可能很小。"], 21, 32)
    # Chunked prefill.
    s.text(638, 176, "分块预填充：长任务分段穿插", 25, TEAL, 700)
    s.pill(638, 195, 300, "Agent 可提议分块策略和预算", "#e1f2ef", TEAL, 18)
    s.text(638, 267, "P = 长输入分块，D = 其他请求输出", 20, MUTED)
    s.text(638, 310, "未分块", 20)
    s.rect(732, 285, 294, 40, "#dfd7f1", r=8)
    s.text(879, 313, "很长的 P", 20, PURPLE, 600, "middle")
    s.rect(1036, 285, 78, 40, "#cde9e4", r=8)
    s.text(1075, 313, "D", 20, TEAL, 600, "middle")
    s.text(638, 389, "分块后", 20)
    for i, label in enumerate(["P₁", "D", "P₂", "D", "P₃", "D"]):
        x = 732 + i * 65
        s.rect(x, 364, 57, 40, "#dfd7f1" if i % 2 == 0 else "#cde9e4", r=8)
        s.text(x + 28, 392, label, 20, PURPLE if i % 2 == 0 else TEAL, 600, "middle")
    s.lines(638, 449, ["D 属于其他已读完输入的请求。", "给这些正在输出的请求留调度机会。", "首字等待与输出间隔的权衡要实测。"], 21, 32)
    # Quantization.
    s.text(70, 612, "量化：换成更省空间的表示", 25, PURPLE, 700)
    s.pill(70, 631, 300, "已准备量化模型资产时可提议", "#efebf8", PURPLE, 18)
    s.rect(78, 709, 184, 88, "#e4dcf3")
    s.text(170, 745, "原数值表示", 22, PURPLE, 600, "middle")
    s.text(170, 779, "占用更多字节", 18, PURPLE, anchor="middle")
    s.arrow([(274, 753), (338, 753)])
    s.rect(350, 724, 184, 58, "#dbeee9")
    s.text(442, 762, "更少字节", 22, TEAL, 600, "middle")
    s.lines(70, 847, ["可能减少权重 / 缓存占用。", "是否提速取决于可用计算实现。", "有明确质量要求时，按该要求验收。"], 21, 34)
    # Tensor parallelism.
    s.text(638, 612, "张量并行 TP：多张卡分工", 25, BLUE, 700)
    s.pill(638, 631, 300, "在物理卡数内开放搜索 TP", size=18)
    s.rect(744, 703, 316, 47, "#e3ebfa", r=9)
    s.text(902, 735, "把模型计算切分到多张卡", 20, BLUE, 600, "middle")
    s.arrow([(800, 754), (800, 786)])
    s.arrow([(1002, 754), (1002, 786)])
    s.rect(707, 798, 179, 53, "#ddeef2", r=10)
    s.text(796, 833, "NPU 0 的部分", 20, TEAL, 600, "middle")
    s.rect(932, 798, 179, 53, "#ddeef2", r=10)
    s.text(1021, 833, "NPU 1 的部分", 20, TEAL, 600, "middle")
    s.arrow([(892, 825), (926, 825)], both=True)
    s.lines(638, 883, ["能分摊计算和模型内存，也要相互通信。", "两卡仅作示意；实际并行数可以调整。", "卡数更多不保证更快，须满足明确约束。"], 21, 32)
    s.rect(44, 1005, 1112, 43, "#e9eef5", r=12)
    s.text(600, 1034, "无需预先批准 profile；每轮先核查文档与代码，再由实测验证可用性。", 21, MUTED, anchor="middle")
    s.save("strategies.svg")


def config_map():
    s = SVG(1200, 1180, "一份文字需求和一份运行接线", "autotune读取用户文字需求与runtime配置。需求生成intent.json中的评分目标和明确约束，未提的推理项开放。runtime提供真实硬件、模型资产、API、固定评测器、环境变量与负载接口，并发现本机CLI参数。大模型诊断并提出候选，经文档与实现代码核查，再由大模型审查具体改动，允许后才运行测量和独立复核；报告保留配置条件、研究证据与测量结果。")
    s.heading("一份需求 + 一份运行接线，就能交给 Agent", "你说明要什么；工程师接好真实设备、模型和测试接口。两者一起决定实验。")
    s.rect(44, 140, 540, 225, "#efebf8")
    s.text(70, 182, "自然语言需求", 28, PURPLE, 700)
    s.text(70, 221, "--request 或 UTF-8 的 .txt 文件", 22, PURPLE)
    s.lines(70, 264, ["写清想提高的指标和不能改变的条件。", "没有提出的精度、缓存、延迟等限制，", "不会从旧模板继承为默认门槛。"], 22, 34)
    s.rect(616, 140, 540, 225, "#e7f3f1")
    s.text(642, 182, "customer-runtime.json", 26, TEAL, 700, code=True)
    s.lines(642, 228, ["真实硬件、模型资产、API、固定评测器", "可用环境变量与负载参数接口", "服务接线、实验预算和独立复核设置", "从 vLLM serve --help=all 发现参数"], 21, 34)
    s.arrow([(314, 367), (314, 407)])
    s.arrow([(886, 367), (886, 407)])
    s.rect(44, 420, 540, 160, "#ece8f5")
    s.text(70, 463, "intent.json：需求的正式解释", 26, PURPLE, 700)
    s.lines(70, 507, ["评分目标 + 原文明确约束", "请核对解释是否正确；未提的推理项开放。"], 22, 34)
    s.rect(616, 420, 540, 160, "#e2f1ee")
    s.text(642, 463, "实际可执行的候选范围", 26, TEAL, 700)
    s.lines(642, 507, ["发现的引擎选项、嵌套 JSON、实际资产", "地址、鉴权、结果路径和测试逻辑不参与调参。"], 21, 34)
    s.parts.append('<path d="M314 582 V612 H886 V582" fill="none" stroke="#60768c" stroke-width="3" stroke-linejoin="round"/>')
    s.arrow([(600, 612), (600, 642)])
    s.rect(225, 655, 750, 88, "#e8effa")
    s.text(600, 693, "大模型诊断与提案 → 文档／源码核查", 26, BLUE, 700, "middle")
    s.text(600, 726, "大模型审查通过 → 实测 → 独立复核", 21, MUTED, anchor="middle")
    s.arrow([(400, 746), (400, 779)])
    s.arrow([(800, 746), (800, 779)])
    s.rect(44, 792, 540, 158, "white", "#d6dfea")
    s.rect(616, 792, 540, 158, "white", "#d6dfea")
    s.text(70, 833, "规划大模型 API", 27, PURPLE, 700)
    s.lines(70, 876, ["分析客户目标与反馈，提出并审查候选", "它的回答不是性能成绩"], 22, 34)
    s.text(642, 833, "昇腾上的被测模型服务", 27, TEAL, 700)
    s.lines(642, 876, ["像参赛选手：实际运行并接受固定评测", "模拟测量或 dummy 权重不算真实成绩"], 22, 34)
    s.rect(44, 984, 1112, 162, "#e7edf5")
    s.text(70, 1025, "结果：report.md + 完整候选与测量证据", 27, weight=700)
    s.lines(70, 1067, ["有效晋升后导出 champion.json / champion-contract.json。", "报告同时写明模型、负载与配置条件；未执行质量测试，就不宣称质量通过。"], 22, 34)
    s.save("config-map.svg")


def main():
    global CHECK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check generated files without changing them")
    CHECK = parser.parse_args().check
    optimization_loop()
    parameters()
    strategies()
    config_map()
    print(f"{'Verified' if CHECK else 'Generated'} 4 SVG diagrams in {OUT}")


if __name__ == "__main__":
    main()
