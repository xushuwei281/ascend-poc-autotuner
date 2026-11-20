#!/usr/bin/env python3
"""Regenerate the four explanatory SVGs with Python 3.10+ (stdlib only).

Run from the repository root to rebuild:
    python3 scripts/render_docs_diagrams.py
Check that committed files match the generator without changing them:
    python3 scripts/render_docs_diagrams.py --check
Absolute script paths also work from any directory.

Outputs in docs/assets/ (width x height, in SVG pixels):
    optimization-loop.svg: 1200 x 1080
    parameters.svg:        1200 x 1160
    strategies.svg:        1200 x 1070
    config-map.svg:        1200 x 1100

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
    s = SVG(1200, 1080, "这个仓库怎样找到更合适的推理配置", "客户先固定验收规则。Agent查资料并提出允许的候选，控制器启动服务测试。通过硬门槛和独立复测且提升足够的候选晋升，否则保留原配置。结果反馈给下一轮，预算限制循环。")
    s.heading("从客户标准出发，一轮一轮验证", "把 Agent 当作实验助手：提出方案、跑测试、看证据，再决定是否保留。")
    s.rect(44, 132, 1112, 105, "#e6effb")
    s.text(70, 173, "先固定“考卷”和“及格线”", 27, BLUE, 700)
    s.text(70, 211, "同一模型与卡数 · 同一批请求 · 同一答案质量要求 · 同一延迟与错误率门槛", 22)
    s.arrow([(600, 238), (600, 269)])
    steps = [
        (282, "1  提出一个允许尝试的方案", ["Agent 结合资料与历史实验，选择策略和参数。", "只能从工程师已验证并开放的目录中选择。"], "#edf0fb", PURPLE),
        (456, "2  按固定规则实测", ["控制器启动该配置，执行性能与质量测试。", "记录快慢、失败和实际消耗；失败也留下证据。"], "#e8f5f5", TEAL),
        (630, "3  检查是否值得替换", ["先满足全部客户门槛，再进行独立复测。", "比较当前配置 → 候选 → 当前配置，检查环境波动。"], "#eaf1fb", BLUE),
    ]
    for y, title, lines, bg, color in steps:
        s.rect(230, y, 740, 139, bg)
        s.text(258, y + 40, title, 27, color, 700)
        s.lines(258, y + 79, lines, 22, 32)
    s.arrow([(600, 422), (600, 449)])
    s.arrow([(600, 596), (600, 623)])
    s.arrow([(460, 770), (460, 806), (305, 806), (305, 830)])
    s.arrow([(740, 770), (740, 806), (895, 806), (895, 830)])
    s.rect(70, 840, 470, 113, "#e5f3ea")
    s.text(305, 879, "合格，且提升足够", 25, "#267247", 700, "middle")
    s.text(305, 917, "晋升为新冠军，作为下一轮参照", 22, INK, anchor="middle")
    s.rect(660, 840, 470, 113, "#fff0df")
    s.text(895, 879, "未通过 / 证据不足", 25, AMBER, 700, "middle")
    s.text(895, 917, "保留原冠军，记录原因再选方案", 22, INK, anchor="middle")
    s.arrow([(70, 895), (44, 895), (44, 351), (220, 351)], dashed=True)
    s.arrow([(1130, 895), (1156, 895), (1156, 351), (980, 351)], dashed=True)
    s.pill(69, 522, 133, "下一轮", "#eef1f6", MUTED)
    s.pill(995, 522, 133, "下一轮", "#eef1f6", MUTED)
    s.rect(44, 984, 1112, 65, "white")
    s.text(600, 1024, "到达预算或停止条件就结束 · 每个测量配置重新启动服务 · 不自动切换生产流量", 21, MUTED, anchor="middle")
    s.save("optimization-loop.svg")


def slot_row(s, x, y, filled, total, color, width=43, gap=9):
    for i in range(total):
        s.rect(x + i * (width + gap), y, width, 44, color if i < filled else "#e6ebf2", r=9)


def parameters():
    s = SVG(1200, 1160, "当前模板真正调节的三个参数", "max_num_seqs限制每轮可处理的序列数。max_num_batched_tokens限制每轮处理的token预算。gpu_memory_utilization改变实例的设备内存预算。调大均有代价，图形只示意方向。上下文要求在客户验收中固定。")
    s.heading("三个旋钮：不是越大越好", "当前模板自动搜索这三项；“参数”是同一种工作方法中的数值设置。")
    data = [
        (135, "01  一轮接待多少个序列", "max_num_seqs", "像餐厅同一轮能照顾多少桌", ["调小：每轮处理的序列较少", "可能更易控制延迟，也可能吃不满算力"], ["调大：每轮可处理更多序列", "可能提高吞吐，也可能增加内存与延迟"], "slots"),
        (425, "02  一轮处理多少份文字", "max_num_batched_tokens", "像厨房这一轮最多处理多少工作量", ["调小：每轮 token 预算较少", "长输入可能分更多轮；效果取决于负载"], ["调大：每轮 token 预算较多", "可能提高输入处理效率，也可能拖慢输出"], "tokens"),
        (715, "03  给实例多少设备内存预算", "gpu_memory_utilization", "像为服务划出仓库预算，不是算力利用率", ["调小：留给其他开销的空间较多", "KV 缓存容量可能减少，更容易排队"], ["调大：本实例可用内存预算较多", "可能容纳更多 KV，但余量少、易内存不足"], "memory"),
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
    s.rect(44, 1025, 1112, 103, "#fff1df")
    s.text(70, 1063, "上下文长度是服务能力约定，不能为了成绩随意缩短", 25, AMBER, 700)
    s.text(70, 1100, "max_model_len 在模板中固定；序列名额 ≠ 外部并发用户数。图形仅示意，不按数值比例。", 20, MUTED)
    s.save("parameters.svg")


def strategies():
    s = SVG(1200, 1070, "推理策略是改变做事方法", "前缀缓存复用相同token前缀的KV计算；分块预填充让新请求的长输入与其他已读完输入请求的解码交错；量化减少数值表示所需空间但要验证质量和性能；张量并行分摊模型计算也带来通信成本。四种策略不是都已由仓库自动搜索。")
    s.heading("四种方法：先理解收益，再实测代价", "“策略”改变做事方法；能否使用、是否更快，取决于模型、昇腾软件版本和客户负载。")
    for x, y in [(44, 132), (612, 132), (44, 568), (612, 568)]:
        s.rect(x, y, 544, 412)
    # Prefix caching.
    s.text(70, 176, "前缀缓存：重复的开头少算一次", 25, BLUE, 700)
    s.pill(70, 195, 257, "已有 profile，默认未启用", size=18)
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
    s.pill(638, 195, 300, "模板开启；关闭对照需先验证", "#e1f2ef", TEAL, 18)
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
    s.pill(70, 631, 257, "当前无现成量化 profile", "#efebf8", PURPLE, 18)
    s.rect(78, 709, 184, 88, "#e4dcf3")
    s.text(170, 745, "原数值表示", 22, PURPLE, 600, "middle")
    s.text(170, 779, "占用更多字节", 18, PURPLE, anchor="middle")
    s.arrow([(274, 753), (338, 753)])
    s.rect(350, 724, 184, 58, "#dbeee9")
    s.text(442, 762, "更少字节", 22, TEAL, 600, "middle")
    s.lines(70, 847, ["可能减少权重 / 缓存占用。", "是否提速取决于可用计算实现。", "需兼容模型与格式，并重新验证质量。"], 21, 34)
    # Tensor parallelism.
    s.text(638, 612, "张量并行 TP：多张卡分工", 25, BLUE, 700)
    s.pill(638, 631, 300, "模板固定两卡，不自动搜索 TP", size=18)
    s.rect(744, 703, 316, 47, "#e3ebfa", r=9)
    s.text(902, 735, "把模型计算切分到多张卡", 20, BLUE, 600, "middle")
    s.arrow([(800, 754), (800, 786)])
    s.arrow([(1002, 754), (1002, 786)])
    s.rect(707, 798, 179, 53, "#ddeef2", r=10)
    s.text(796, 833, "NPU 0 的部分", 20, TEAL, 600, "middle")
    s.rect(932, 798, 179, 53, "#ddeef2", r=10)
    s.text(1021, 833, "NPU 1 的部分", 20, TEAL, 600, "middle")
    s.arrow([(892, 825), (926, 825)], both=True)
    s.lines(638, 903, ["能分摊计算和模型内存，也要相互通信。", "卡数更多不保证更快；卡数须符合约定。"], 21, 32)
    s.rect(44, 1005, 1112, 43, "#e9eef5", r=12)
    s.text(600, 1034, "目录中的方法要先由工程师验证并开放；Agent 不能只凭查到资料就启用。", 21, MUTED, anchor="middle")
    s.save("strategies.svg")


def config_map():
    s = SVG(1200, 1100, "三个配置文件和两个不同的大模型角色", "customer-agent.json引用customer-workflow.json，工作流再引用customer-contract.json。契约定义固定的硬件、模型、测试与规划API。规划大模型提出建议，昇腾上的被测模型运行真实负载。报告保存过程，冠军配置仅在通过独立验收并晋升后导出。")
    s.heading("先填三张表，再运行 Agent", "JSON 是可编辑的配置表；文件名不重要，引用关系与里面的内容才决定行为。")
    boxes = [
        (141, "customer-agent.json", "任务单", ["文字任务目标、最大步骤数、实验次数和总时限", "经验文件位置；引用下面的工作流"], PURPLE, "#efebf8"),
        (341, "customer-workflow.json", "允许尝试的方法", ["已验证策略目录、资料查询方式、复测规则", "实验预算；引用下面的客户契约"], TEAL, "#e7f3f1"),
        (541, "customer-contract.json", "固定的客户考卷", ["模型、卡数、服务与性能 / 质量测试命令", "正式目标与验收门槛、可调范围、规划 API 地址"], BLUE, "#e8effa"),
    ]
    for y, filename, title, lines, color, bg in boxes:
        s.rect(62, y, 770, 151, bg)
        s.text(89, y + 39, filename, 25, color, 700, code=True)
        s.pill(849, y + 16, 287, title, bg, color, 22)
        s.lines(89, y + 84, lines, 23, 34)
    s.arrow([(447, 294), (447, 333)])
    s.text(478, 323, 'workflow: "customer-workflow.json"', 20, MUTED, code=True)
    s.arrow([(447, 494), (447, 533)])
    s.text(478, 523, 'contract: "customer-contract.json"', 20, MUTED, code=True)
    s.text(858, 613, "相对路径均以", 21, MUTED)
    s.text(858, 646, "引用者所在目录为起点", 21, MUTED)
    s.arrow([(300, 695), (300, 744)])
    s.arrow([(700, 695), (700, 721), (900, 721), (900, 744)])
    s.rect(62, 756, 518, 160, "white", "#d6dfea")
    s.rect(612, 756, 524, 160, "white", "#d6dfea")
    s.text(89, 797, "规划大模型 API", 27, PURPLE, 700)
    s.lines(89, 840, ["像实验顾问：读指标、给候选建议", "通常是外部 API，不是被测成绩来源"], 22, 34)
    s.text(638, 797, "昇腾上的被测模型服务", 27, TEAL, 700)
    s.lines(638, 840, ["像参赛选手：运行固定客户请求", "性能与质量由固定测试脚本记录"], 22, 34)
    s.rect(62, 948, 1074, 118, "#e7edf5")
    s.text(89, 988, "运行后看输出目录", 26, weight=700)
    s.text(89, 1026, "agent_report.md：过程与结论；champion*.json：成功晋升后才有的冠军配置。", 22, MUTED)
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
