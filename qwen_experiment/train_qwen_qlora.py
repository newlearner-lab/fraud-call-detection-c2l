import os
import json
import math
import inspect
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# 1. 数据集：只让模型学习 assistant 的“正常/诈骗”答案
# ============================================================

class FraudSFTDataset(Dataset):
    """
    保留完整 system + user 对话 + assistant 标签。

    不截断任何通话内容：
    - 若样本超出 max_length，会明确报错；
    - 不会悄悄丢掉对话前文或后文。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        too_long_samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                    messages = record["messages"]

                    if len(messages) < 3:
                        raise ValueError("messages 不完整")

                    # system + user，并生成 assistant 的起始标记
                    prompt_text = tokenizer.apply_chat_template(
                        messages[:-1],
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    # 正确标签：正常 或 诈骗
                    answer_text = (
                        str(messages[-1]["content"]).strip()
                        + tokenizer.eos_token
                    )

                    prompt_ids = tokenizer(
                        prompt_text,
                        add_special_tokens=False
                    )["input_ids"]

                    answer_ids = tokenizer(
                        answer_text,
                        add_special_tokens=False
                    )["input_ids"]

                    input_ids = prompt_ids + answer_ids
                    total_length = len(input_ids)

                    # 不截断，超长则记录
                    if total_length > self.max_length:
                        too_long_samples.append(
                            (line_id, total_length)
                        )
                        continue

                    labels = (
                        [-100] * len(prompt_ids)
                        + answer_ids
                    )

                    self.samples.append({
                        "input_ids": input_ids,
                        "attention_mask": [1] * total_length,
                        "labels": labels,
                    })

                except Exception as e:
                    print(
                        f"[跳过] 第 {line_id} 条："
                        f"{type(e).__name__}: {e}"
                    )

        if too_long_samples:
            max_seen = max(x[1] for x in too_long_samples)

            print("\n" + "=" * 70)
            print(f"{jsonl_path} 中有 {len(too_long_samples)} 条样本超过 "
                  f"max_length={self.max_length}。")
            print(f"最长超长样本长度：{max_seen} tokens")
            print("为保证完整对话，程序没有截断这些样本。")
            print("请把 --max-length 提高到至少：", max_seen)
            print("前 10 条超长样本：", too_long_samples[:10])
            print("=" * 70)

            raise RuntimeError(
                "检测到超长样本。请提高 --max-length 后重新训练。"
            )

        print(
            f"已加载 {jsonl_path}，"
            f"完整保留的有效样本数：{len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class FraudDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])

            batch_input_ids.append(
                item["input_ids"] + [self.pad_token_id] * pad_len
            )

            batch_attention_mask.append(
                item["attention_mask"] + [0] * pad_len
            )

            batch_labels.append(
                item["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                batch_attention_mask,
                dtype=torch.long
            ),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# ============================================================
# 2. 主训练程序
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--resume-from-checkpoint",
    type=str,
    default=None,
    help="从指定 checkpoint 恢复训练。"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct"
    )

    parser.add_argument(
        "--train-file",
        type=str,
        default="qwen_experiment/data/train_qwen.jsonl"
    )

    parser.add_argument(
        "--valid-file",
        type=str,
        default="qwen_experiment/data/valid_qwen.jsonl"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="qwen_experiment/checkpoints/qwen2.5_3b_qlora"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=768
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 CUDA GPU。Qwen2.5-3B 的 QLoRA 训练建议使用 NVIDIA GPU。"
        )

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"找不到训练文件：{args.train_file}")

    if not os.path.exists(args.valid_file):
        raise FileNotFoundError(f"找不到验证文件：{args.valid_file}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Qwen2.5-3B-Instruct + QLoRA 虚假通话分类训练")
    print("模型：", args.model_name)
    print("训练集：", args.train_file)
    print("验证集：", args.valid_file)
    print("输出目录：", args.output_dir)
    print("max_length：", args.max_length)
    print("epoch：", args.epochs)
    print("=" * 70)

    # T4 等多数消费级 GPU 使用 FP16；支持 BF16 时自动使用 BF16
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True
    )

    # Qwen 通常没有单独 pad_token，使用 eos_token 填充
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
    )

    model.config.use_cache = False

    # 为 4-bit QLoRA 训练做准备
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = FraudSFTDataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length
    )

    valid_dataset = FraudSFTDataset(
        args.valid_file,
        tokenizer,
        max_length=args.max_length
    )

    collator = FraudDataCollator(tokenizer.pad_token_id)

    # 兼容不同 transformers 版本
    training_kwargs = dict(
        output_dir=args.output_dir,
        #overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    params = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" in params:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
    )

    trainer_params = inspect.signature(Trainer.__init__).parameters

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    #trainer.train()
    if args.resume_from_checkpoint:
      print("从检查点恢复训练：", args.resume_from_checkpoint)
      trainer.train(
                      resume_from_checkpoint=args.resume_from_checkpoint
                        )
    else:
      trainer.train()

    # 保存最佳 LoRA Adapter
    best_dir = os.path.join(args.output_dir, "best_adapter")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    metrics = trainer.evaluate()

    print("\n" + "=" * 70)
    print("训练完成")
    print("最佳 Adapter：", best_dir)
    print("验证集 loss：", metrics.get("eval_loss"))
    print("困惑度 perplexity：", math.exp(metrics["eval_loss"]))
    print("=" * 70)


if __name__ == "__main__":
    main()
