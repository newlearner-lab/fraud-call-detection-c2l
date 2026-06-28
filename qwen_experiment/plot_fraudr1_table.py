
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 1. 数据：按你的实验结果填写
# ============================================================

helpful = {
    "DSR@0": 0.1400,
    "DSR@1": 0.1467,
    "DSR@2": 0.1467,
    "DSR@3 / DSR": 0.1500,
    "AVG(k)": 3.4167,
    "Failure rate": 0.8200,
    "Unresolved after D3": 0.0300,
    "Success / Failure / Unresolved": "45 / 246 / 9",
}

roleplay = {
    "DSR@0": 0.4867,
    "DSR@1": 0.5467,
    "DSR@2": 0.5567,
    "DSR@3 / DSR": 0.5700,
    "AVG(k)": 1.8400,
    "Failure rate": 0.3167,
    "Unresolved after D3": 0.1133,
    "Success / Failure / Unresolved": "171 / 95 / 34",
}


# ============================================================
# 2. 自动选择字体
# ============================================================

candidate_fonts = [
    "Times New Roman",
    "DejaVu Serif",
    "Liberation Serif",
]

available_fonts = {font.name for font in font_manager.fontManager.ttflist}

for font_name in candidate_fonts:
    if font_name in available_fonts:
        plt.rcParams["font.family"] = font_name
        break

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. 表格内容
# ============================================================

rows = [
    ("Evaluation Settings", "", ""),
    ("Total evaluated samples", "300", "300"),
    ("Valid samples", "300", "300"),
    ("Judge errors", "0", "0"),

    ("Defense Success Rate", "", ""),
    ("DSR@0", f"{helpful['DSR@0'] * 100:.2f}%", f"{roleplay['DSR@0'] * 100:.2f}%"),
    ("DSR@1", f"{helpful['DSR@1'] * 100:.2f}%", f"{roleplay['DSR@1'] * 100:.2f}%"),
    ("DSR@2", f"{helpful['DSR@2'] * 100:.2f}%", f"{roleplay['DSR@2'] * 100:.2f}%"),
    (
        "DSR@3 / DSR",
        f"{helpful['DSR@3 / DSR'] * 100:.2f}%",
        f"{roleplay['DSR@3 / DSR'] * 100:.2f}%"
    ),
    ("AVG(k) ↓", f"{helpful['AVG(k)']:.2f}", f"{roleplay['AVG(k)']:.2f}"),

    ("Failure and Uncertainty", "", ""),
    (
        "Failure rate ↓",
        f"{helpful['Failure rate'] * 100:.2f}%",
        f"{roleplay['Failure rate'] * 100:.2f}%"
    ),
    (
        "Unresolved after D3 ↓",
        f"{helpful['Unresolved after D3'] * 100:.2f}%",
        f"{roleplay['Unresolved after D3'] * 100:.2f}%"
    ),

    ("Outcome Counts", "", ""),
    (
        "Success / Failure / Unresolved",
        helpful["Success / Failure / Unresolved"],
        roleplay["Success / Failure / Unresolved"]
    ),
]

section_rows = {0, 4, 10, 13}


# ============================================================
# 4. 绘图：模仿论文中的横线表格样式
# ============================================================

fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=300)
ax.set_axis_off()

left = 0.07
right = 0.93
top = 0.90
bottom = 0.12

table_width = right - left

# 三列边界：指标列、Helpful列、Role-play列
col1_left = left
col1_right = left + table_width * 0.50

col2_left = col1_right
col2_right = left + table_width * 0.75

col3_left = col2_right
col3_right = right

num_rows = len(rows) + 1
row_height = (top - bottom) / num_rows


# ---------- 表头 ----------

header_y = top - row_height * 0.5

ax.text(
    (col1_left + col1_right) / 2,
    header_y,
    "Statistics",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    transform=ax.transAxes
)

ax.text(
    (col2_left + col2_right) / 2,
    header_y,
    "Helpful Assistant",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    transform=ax.transAxes
)

ax.text(
    (col3_left + col3_right) / 2,
    header_y,
    "Role-play",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    transform=ax.transAxes
)

# 顶部和表头下方横线
ax.hlines(top, left, right, linewidth=1.4, color="black", transform=ax.transAxes)
ax.hlines(
    top - row_height,
    left,
    right,
    linewidth=0.9,
    color="black",
    transform=ax.transAxes
)


# ---------- 表格正文 ----------

for index, (metric, helpful_value, roleplay_value) in enumerate(rows):
    y = top - row_height * (index + 1.5)

    # 分组标题行
    if index in section_rows:
        ax.text(
            col1_left + 0.01,
            y,
            metric,
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes
        )

        # 每个分组前绘制横线
        ax.hlines(
            top - row_height * (index + 1),
            left,
            right,
            linewidth=0.7,
            color="black",
            transform=ax.transAxes
        )

    # 普通指标行
    else:
        ax.text(
            col1_left + 0.01,
            y,
            metric,
            ha="left",
            va="center",
            fontsize=10.5,
            transform=ax.transAxes
        )

        ax.text(
            (col2_left + col2_right) / 2,
            y,
            helpful_value,
            ha="center",
            va="center",
            fontsize=10.5,
            transform=ax.transAxes
        )

        ax.text(
            (col3_left + col3_right) / 2,
            y,
            roleplay_value,
            ha="center",
            va="center",
            fontsize=10.5,
            transform=ax.transAxes
        )


# ---------- 底部横线与注释 ----------

ax.hlines(
    bottom,
    left,
    right,
    linewidth=1.4,
    color="black",
    transform=ax.transAxes
)

ax.text(
    left,
    0.055,
    "Note: DSR@k denotes the cumulative defense success rate through round k. "
    "For AVG(k), unsuccessful samples are assigned a value of 4.",
    ha="left",
    va="center",
    fontsize=8.5,
    transform=ax.transAxes
)

# 保存图片
output_path = Path("/workspace/causally-contrastive-learning/qwen_experiment/fraudr1_prompt_comparison_table.png")

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.show()

print(f"图表已保存到：{output_path.resolve()}")

