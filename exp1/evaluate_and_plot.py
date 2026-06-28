import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)
from transformers import BertTokenizer

from classes.modeling import BertForCounterfactualRobustness
from classes.datasets import CFIMDbDataset


def split_sentences(text):
    sp = text.split(" [SEP] ")
    if len(sp) > 1:
        return sp
    return sp[0]


def load_test_data(dataset_path):
    test_path = os.path.join(dataset_path, "test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test = json.load(f)

    anc_test_texts = [split_sentences(d["anchor_text"]) for d in test]
    pos_test_texts = [split_sentences(d["positive_text"]) for d in test]
    neg_test_texts = [split_sentences(d["negative_text"]) for d in test]
    test_labels = [d["label"] for d in test]

    y_true = np.argmax(np.array(test_labels), axis=1)

    return test, anc_test_texts, pos_test_texts, neg_test_texts, test_labels, y_true


def build_dataloader(anc_texts, pos_texts, neg_texts, labels, model_name, max_length, batch_size):
    tokenizer = BertTokenizer.from_pretrained(model_name)

    anc_encodings = tokenizer(
        anc_texts,
        truncation=True,
        padding=True,
        max_length=max_length
    )
    pos_encodings = tokenizer(
        pos_texts,
        truncation=True,
        padding=True,
        max_length=max_length
    )
    neg_encodings = tokenizer(
        neg_texts,
        truncation=True,
        padding=True,
        max_length=max_length
    )

    dataset = CFIMDbDataset(
        anc_encodings,
        pos_encodings,
        neg_encodings,
        labels
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataloader


def evaluate_one_model(checkpoint_dir, dataloader, y_true, raw_test_data):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_path = os.path.join(checkpoint_dir, "best_epoch")
    model = BertForCounterfactualRobustness.from_pretrained(
        model_path,
        num_labels=2
    )
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []

    for batch in dataloader:
        with torch.no_grad():
            anc_input_ids = batch["anchor_input_ids"].to(device)
            anc_attention_mask = batch["anchor_attention_mask"].to(device)

            if "anchor_token_type_ids" in batch:
                anc_token_type_ids = batch["anchor_token_type_ids"].to(device)
            else:
                anc_token_type_ids = None

            outputs = model(
                anc_input_ids,
                anc_attention_mask,
                anchor_token_type_ids=anc_token_type_ids
            )

            logits = outputs[0]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)[:, 1]   # fraud 概率

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Fraud"],
        digits=4,
        output_dict=True
    )

    pred_rows = []
    for i, d in enumerate(raw_test_data):
        pred_rows.append({
            "text": d["anchor_text"],
            "true_label": int(y_true[i]),
            "pred_label": int(y_pred[i]),
            "prob_normal": float(1 - y_prob[i]),
            "prob_fraud": float(y_prob[i])
        })

    pred_df = pd.DataFrame(pred_rows)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
        "report": report_dict,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "pred_df": pred_df
    }


def plot_metrics_bar(metrics_df, save_path):
    x = np.arange(len(metrics_df))
    width = 0.18

    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * width, metrics_df["Accuracy"], width, label="Accuracy")
    plt.bar(x - 0.5 * width, metrics_df["Precision"], width, label="Precision")
    plt.bar(x + 0.5 * width, metrics_df["Recall"], width, label="Recall")
    plt.bar(x + 1.5 * width, metrics_df["F1-score"], width, label="F1-score")

    plt.xticks(x, metrics_df["Model"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"]
    )
    disp.plot(values_format="d", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_single_roc(y_true, y_prob, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve_comparison(results_dict, y_true, save_path):
    plt.figure(figsize=(8, 6))

    for model_name, result in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, result["y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_classification_report(report_dict, save_path):
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(save_path, encoding="utf-8-sig")


def save_metrics_json(result_dict, save_path):
    metrics = {
        "accuracy": result_dict["accuracy"],
        "precision": result_dict["precision"],
        "recall": result_dict["recall"],
        "f1": result_dict["f1"]
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="dataset/FraudCall")
    parser.add_argument("--model-name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vanilla-path", type=str, default="checkpoints/FraudCall/roberta_vanilla")
    parser.add_argument("--c2l-path", type=str, default="checkpoints/FraudCall/c2l_roberta")
    parser.add_argument("--output-dir", type=str, default="eval_outputs")
    args = parser.parse_args()

    # 创建总输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 为两个模型和比较结果分别创建文件夹
    vanilla_out = os.path.join(args.output_dir, "roberta_vanilla")
    c2l_out = os.path.join(args.output_dir, "c2l_roberta")
    comparison_out = os.path.join(args.output_dir, "comparison")

    os.makedirs(vanilla_out, exist_ok=True)
    os.makedirs(c2l_out, exist_ok=True)
    os.makedirs(comparison_out, exist_ok=True)

    print("========== Loading test data ==========")
    raw_test_data, anc_texts, pos_texts, neg_texts, labels, y_true = load_test_data(args.dataset_path)

    dataloader = build_dataloader(
        anc_texts,
        pos_texts,
        neg_texts,
        labels,
        args.model_name,
        args.max_length,
        args.batch_size
    )

    results = {}

    # 评估基础模型
    print("\n========== Evaluating Vanilla Model ==========")
    vanilla_result = evaluate_one_model(
        checkpoint_dir=args.vanilla_path,
        dataloader=dataloader,
        y_true=y_true,
        raw_test_data=raw_test_data
    )
    results["Chinese-RoBERTa"] = vanilla_result

    # 保存基础模型结果
    vanilla_result["pred_df"].to_csv(
        os.path.join(vanilla_out, "predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    save_classification_report(
        vanilla_result["report"],
        os.path.join(vanilla_out, "classification_report.csv")
    )
    save_metrics_json(
        vanilla_result,
        os.path.join(vanilla_out, "metrics.json")
    )
    plot_confusion(
        vanilla_result["cm"],
        "Confusion Matrix - Chinese-RoBERTa",
        os.path.join(vanilla_out, "confusion_matrix.png")
    )
    plot_single_roc(
        y_true,
        vanilla_result["y_prob"],
        "ROC Curve - Chinese-RoBERTa",
        os.path.join(vanilla_out, "roc_curve.png")
    )

    # 评估 C2L 模型
    print("\n========== Evaluating C2L Model ==========")
    c2l_result = evaluate_one_model(
        checkpoint_dir=args.c2l_path,
        dataloader=dataloader,
        y_true=y_true,
        raw_test_data=raw_test_data
    )
    results["C2L-Chinese-RoBERTa"] = c2l_result

    # 保存 C2L 模型结果
    c2l_result["pred_df"].to_csv(
        os.path.join(c2l_out, "predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    save_classification_report(
        c2l_result["report"],
        os.path.join(c2l_out, "classification_report.csv")
    )
    save_metrics_json(
        c2l_result,
        os.path.join(c2l_out, "metrics.json")
    )
    plot_confusion(
        c2l_result["cm"],
        "Confusion Matrix - C2L-Chinese-RoBERTa",
        os.path.join(c2l_out, "confusion_matrix.png")
    )
    plot_single_roc(
        y_true,
        c2l_result["y_prob"],
        "ROC Curve - C2L-Chinese-RoBERTa",
        os.path.join(c2l_out, "roc_curve.png")
    )

    # 打印结果
    for model_name, result in results.items():
        print(f"\n========== {model_name} ==========")
        print("Accuracy :", result["accuracy"])
        print("Precision:", result["precision"])
        print("Recall   :", result["recall"])
        print("F1-score :", result["f1"])
        print("Confusion Matrix:\n", result["cm"])

    # 汇总指标表
    metrics_rows = []
    for model_name, result in results.items():
        metrics_rows.append({
            "Model": model_name,
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1-score": result["f1"]
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        os.path.join(comparison_out, "metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 指标柱状图
    plot_metrics_bar(
        metrics_df,
        os.path.join(comparison_out, "metrics_bar.png")
    )

    # ROC 对比图
    plot_roc_curve_comparison(
        results,
        y_true,
        os.path.join(comparison_out, "roc_curve_comparison.png")
    )

    print("\n========== 已生成文件夹 ==========")
    print(vanilla_out)
    print(c2l_out)
    print(comparison_out)


if __name__ == "__main__":
    main()
