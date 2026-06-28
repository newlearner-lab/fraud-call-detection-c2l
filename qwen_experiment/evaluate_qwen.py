import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


SYSTEM_PROMPT = (
    "你是一名中文虚假通话文本分类助手。"
    "请根据通话内容判断其是否属于诈骗。"
    "回答只能是“正常”或“诈骗”，不得输出解释。"
)


def label_to_int(label):
    """正常=0，诈骗=1。"""
    if isinstance(label, list) and len(label) >= 2:
        if label == [1, 0]:
            return 0
        if label == [0, 1]:
            return 1

    if str(label).strip() in {"0", "正常"}:
        return 0
    if str(label).strip() in {"1", "诈骗"}:
        return 1

    raise ValueError(f"无法识别标签：{label}")


def build_prompt(tokenizer, dialogue):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"通话内容：\n{dialogue}",
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def score_candidate_batch(
    model,
    tokenizer,
    prompts,
    candidate_text,
    max_length,
):
    """
    计算每条 prompt 生成指定候选标签的对数概率。
    candidate_text 为：正常 + EOS 或 诈骗 + EOS。
    """
    full_texts = [prompt + candidate_text for prompt in prompts]

    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )

    real_lengths = encoded["attention_mask"].sum(dim=1)

    if int(real_lengths.max()) > max_length:
        raise RuntimeError(
            f"发现测试样本长度达到 {int(real_lengths.max())} tokens，"
            f"超过当前 max_length={max_length}。"
        )

    device = next(model.parameters()).device

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    candidate_ids = tokenizer(
        candidate_text,
        add_special_tokens=False,
    )["input_ids"]

    candidate_len = len(candidate_ids)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits
    batch_size, seq_len, _ = logits.shape

    # 使用左侧 padding，因此候选答案一定位于每条序列的末尾。
    target_ids = input_ids[:, seq_len - candidate_len:seq_len]

    target_logits = logits[
        :,
        seq_len - candidate_len - 1:seq_len - 1,
        :
    ]

    log_probs = F.log_softmax(target_logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs.sum(dim=-1).detach().cpu()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="待评估测试集目录，例如 dataset/.../D0_base",
    )

    parser.add_argument(
        "--adapter-path",
        type=str,
        default=(
            "qwen_experiment/checkpoints/"
            "qwen2.5_3b_qlora_a100_batch8/best_adapter"
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1536,
    )

    args = parser.parse_args()

    test_path = Path(args.dataset_path) / "test.json"

    if not test_path.exists():
        raise FileNotFoundError(f"找不到测试集：{test_path}")

    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"找不到 LoRA Adapter：{args.adapter_path}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU。")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("=" * 70)
    print("Qwen2.5-3B-Instruct + LoRA 测试评估")
    print("模型：", args.model_name)
    print("Adapter：", args.adapter_path)
    print("测试集：", test_path)
    print("输出目录：", args.output_dir)
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
    )

    model.config.use_cache = True
    model.eval()

    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    labels = []
    sample_ids = []

    for sample_id, item in enumerate(data):
        prompts.append(
            build_prompt(
                tokenizer,
                str(item["anchor_text"]).strip(),
            )
        )
        labels.append(label_to_int(item["label"]))
        sample_ids.append(sample_id)

    normal_candidate = "正常" + tokenizer.eos_token
    fraud_candidate = "诈骗" + tokenizer.eos_token

    normal_scores = []
    fraud_scores = []

    for start in tqdm(
        range(0, len(prompts), args.batch_size),
        desc="评估 Qwen",
        unit="batch",
    ):
        batch_prompts = prompts[start:start + args.batch_size]

        normal_batch_score = score_candidate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            candidate_text=normal_candidate,
            max_length=args.max_length,
        )

        fraud_batch_score = score_candidate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            candidate_text=fraud_candidate,
            max_length=args.max_length,
        )

        normal_scores.extend(normal_batch_score.tolist())
        fraud_scores.extend(fraud_batch_score.tolist())

    score_tensor = torch.tensor(
        list(zip(normal_scores, fraud_scores)),
        dtype=torch.float32,
    )

    fraud_probs = torch.softmax(score_tensor, dim=1)[:, 1].numpy()
    predictions = (fraud_probs >= 0.5).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    try:
        auc = roc_auc_score(labels, fraud_probs)
    except ValueError:
        auc = None

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "model": "Qwen2.5-3B-Instruct + QLoRA",
        "dataset": str(args.dataset_path),
        "num_samples": len(labels),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": None if auc is None else float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    pd.DataFrame([
        {
            "sample_id": sample_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "fraud_score": fraud_score,
            "normal_logprob": normal_score,
            "fraud_logprob": fraud_logprob,
        }
        for sample_id, true_label, pred_label,
        fraud_score, normal_score, fraud_logprob
        in zip(
            sample_ids,
            labels,
            predictions,
            fraud_probs,
            normal_scores,
            fraud_scores,
        )
    ]).to_csv(
        Path(args.output_dir) / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([metrics]).to_csv(
        Path(args.output_dir) / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        Path(args.output_dir) / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("评估完成")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()
