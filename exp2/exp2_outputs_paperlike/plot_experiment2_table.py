from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 1. 在这里填写你的 metrics.json 路径
# ============================================================

MODEL_FILES = {
    "Chinese-RoBERTa": {
        "D0": "/workspace/causally-contrastive-learning/exp2_outputs_paperlike/D0_base/roberta_vanilla/metrics.json",
        "D1": "/workspace/causally-contrastive-learning/exp2_outputs_paperlike/D1_credibility/roberta_vanilla/metrics.json",
        "D2": "/workspace/causally-contrastive-learning/exp2_outputs_paperlike/D2_credibility_urgency/roberta_vanilla/metrics.json",
        "D3": "/workspace/causally-contrastive-learning/exp2_outputs_paperlike/D3_credibility_urgency_emotion/roberta_vanilla/metrics.json",
    },
    "C2L-Chinese-RoBERTa": {
        "D0": "/workspace/causally-contrastive-learning/exp2_outputs_paperlike/D0_base/c2l_roberta/metrics.json",
        "D1": "/workspace/causally-contrastive-learning/exp2_outputs_paperlike/D1_credibility/c2l_roberta/metrics.json",
        "D2": "/workspace/causally-contrastive-learning/exp2_outputs_paperlike/D2_credibility_urgency/c2l_roberta/metrics.json",
        "D3": "/workspace/causally-contrastive-learning/exp2_outputs_paperlike/D3_credibility_urgency_emotion/c2l_roberta/metrics.json",
    },
    "Qwen2.5-3B-Instruct + QLoRA": {
        "D0": "/workspace/causally-contrastive-learning/qwen_experiment/outputs_paperlike/D0_base/metrics.json",
        "D1": "/workspace/causally-contrastive-learning/qwen_experiment/outputs_paperlike/D1_credibility/metrics.json",
        "D2": "/workspace/causally-contrastive-learning/qwen_experiment/outputs_paperlike/D2_credibility_urgency/metrics.json",
        "D3": "/workspace/causally-contrastive-learning/qwen_experiment/outputs_paperlike/D3_credibility_urgency_emotion/metrics.json",
    },
}


# ============================================================
# 2. 数据集显示名称
# ============================================================

DATASET_ORDER = ["D0", "D1", "D2", "D3"]

DATASET_LABELS = {
    "D0": "D0 (Base)",
    "D1": "D1 (Credibility)",
    "D2": "D2 (Credibility + Urgency)",
    "D3": "D3 (Credibility + Urgency + Emotion)",
}

METRICS = ["accuracy", "precision", "recall", "f1"]


# ============================================================
# 3. 读取 metrics.json
# ============================================================

def load_metrics(json_path):
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"找不到文件: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for metric in METRICS:
        if metric not in data:
            raise KeyError(f"{json_path} 中缺少字段: {metric}")
        result[metric] = float(data[metric])

    return result


# ============================================================
# 4. 汇总所有结果
# ============================================================

all_results = {}

for model_name, dataset_files in MODEL_FILES.items():
    all_results[model_name] = {}
    for dataset_name in DATASET_ORDER:
        all_results[model_name][dataset_name] = load_metrics(
            dataset_files[dataset_name]
        )

print("=" * 80)
print("读取到的结果如下：")
for model_name, dataset_results in all_results.items():
    print(f"\n{model_name}")
    for dataset_name in DATASET_ORDER:
        m = dataset_results[dataset_name]
        print(
            f"  {dataset_name}: "
            f"accuracy={m['accuracy']:.6f}, "
            f"precision={m['precision']:.6f}, "
            f"recall={m['recall']:.6f}, "
            f"f1={m['f1']:.6f}"
        )
print("=" * 80)


# ============================================================
# 5. 字体设置（尽量接近论文风格）
# ============================================================

candidate_fonts = [
    "Times New Roman",
    "Liberation Serif",
    "DejaVu Serif",
]

available_fonts = {font.name for font in font_manager.fontManager.ttflist}

for font_name in candidate_fonts:
    if font_name in available_fonts:
        plt.rcParams["font.family"] = font_name
        break

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 6. 组织表格内容
# ============================================================

table_rows = []

for model_name, dataset_results in all_results.items():
    for i, dataset_name in enumerate(DATASET_ORDER):
        m = dataset_results[dataset_name]

        table_rows.append({
            "model": model_name if i == 0 else "",
            "dataset": DATASET_LABELS[dataset_name],
            "accuracy": f"{m['accuracy'] * 100:.2f}",
            "precision": f"{m['precision'] * 100:.2f}",
            "recall": f"{m['recall'] * 100:.2f}",
            "f1": f"{m['f1'] * 100:.2f}",
        })


# ============================================================
# 7. 绘制成一张论文风格表格图
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8.2), dpi=300)
ax.set_axis_off()

# 整个表格区域
left = 0.04
right = 0.96
top = 0.90
bottom = 0.10

# 列宽比例：Model / Dataset / Accuracy / Precision / Recall / F1
col_ratios = [0.28, 0.30, 0.105, 0.105, 0.105, 0.105]
col_edges = [left]

for ratio in col_ratios:
    col_edges.append(col_edges[-1] + (right - left) * ratio)

headers = [
    "Model",
    "Dataset",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
]

num_rows = len(table_rows) + 1
row_h = (top - bottom) / num_rows


# -----------------------------
# 标题
# -----------------------------
ax.text(
    0.5,
    0.955,
    "Experiment 2 Results on Four Test Sets",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    transform=ax.transAxes,
)

# -----------------------------
# 表头
# -----------------------------
header_y = top - row_h * 0.5

for i, header in enumerate(headers):
    x_center = (col_edges[i] + col_edges[i + 1]) / 2
    ax.text(
        x_center,
        header_y,
        header,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )

# 顶部横线、表头下横线
ax.hlines(top, left, right, linewidth=1.6, color="black", transform=ax.transAxes)
ax.hlines(
    top - row_h,
    left,
    right,
    linewidth=0.9,
    color="black",
    transform=ax.transAxes,
)


# -----------------------------
# 正文
# -----------------------------
for row_idx, row in enumerate(table_rows):
    y = top - row_h * (row_idx + 1.5)

    values = [
        row["model"],
        row["dataset"],
        row["accuracy"],
        row["precision"],
        row["recall"],
        row["f1"],
    ]

    for col_idx, value in enumerate(values):
        x_center = (col_edges[col_idx] + col_edges[col_idx + 1]) / 2

        if col_idx in [0, 1]:
            x = col_edges[col_idx] + 0.006
            ha = "left"
        else:
            x = x_center
            ha = "center"

        ax.text(
            x,
            y,
            value,
            ha=ha,
            va="center",
            fontsize=9.8,
            transform=ax.transAxes,
        )

    # 每个模型块结束后画一条横线
    if (row_idx + 1) % 4 == 0:
        ax.hlines(
            top - row_h * (row_idx + 2),
            left,
            right,
            linewidth=0.7,
            color="black",
            transform=ax.transAxes,
        )


# -----------------------------
# 底线与注释
# -----------------------------
ax.hlines(
    bottom,
    left,
    right,
    linewidth=1.6,
    color="black",
    transform=ax.transAxes,
)

ax.text(
    left,
    0.045,
    "Note: All values are percentages. "
    "D0 is the base test set, and D1-D3 are progressively enhanced test sets.",
    ha="left",
    va="center",
    fontsize=8.5,
    transform=ax.transAxes,
)


# ============================================================
# 8. 保存图片
# ============================================================

png_path = Path("/workspace/causally-contrastive-learning/exp2_outputs_paperlike/experiment2_metrics_table.png")
pdf_path = Path("/workspace/causally-contrastive-learning/exp2_outputs_paperlike/experiment2_metrics_table.pdf")

plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.15)

plt.show()

print(f"PNG 已保存：{png_path.resolve()}")
print(f"PDF 已保存：{pdf_path.resolve()}")