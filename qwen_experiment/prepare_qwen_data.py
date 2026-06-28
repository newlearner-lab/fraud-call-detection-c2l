import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "你是一名中文虚假通话文本分类助手。"
    "请根据通话内容判断其是否属于诈骗。"
    "回答只能是“正常”或“诈骗”，不得输出解释。"
)


def label_to_text(label):
    """将 C2L one-hot 标签转换为中文分类标签。"""
    if isinstance(label, list) and len(label) >= 2:
        if label == [1, 0]:
            return "正常"
        if label == [0, 1]:
            return "诈骗"

    raise ValueError(f"无法识别标签：{label}")


def convert_file(input_path, output_path):
    """把 C2L JSON 数据转为 Qwen SFT JSONL。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    normal_count = 0
    fraud_count = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(data):
            try:
                dialogue = str(item["anchor_text"]).strip()
                label_text = label_to_text(item["label"])

                if not dialogue:
                    raise ValueError("anchor_text 为空")

                record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": f"通话内容：\n{dialogue}"
                        },
                        {
                            "role": "assistant",
                            "content": label_text
                        }
                    ]
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                total += 1
                if label_text == "正常":
                    normal_count += 1
                else:
                    fraud_count += 1

            except Exception as e:
                skipped += 1
                print(f"[跳过] 第 {idx} 条：{e}")

    print("=" * 60)
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")
    print(f"成功转换：{total}")
    print(f"正常样本：{normal_count}")
    print(f"诈骗样本：{fraud_count}")
    print(f"跳过样本：{skipped}")
    print("=" * 60)


def find_valid_file(dataset_dir):
    """自动寻找 valid.json 或 val.json。"""
    candidates = [
        dataset_dir / "valid.json",
        dataset_dir / "val.json",
        dataset_dir / "dev.json"
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "没有找到验证集，请检查 dataset/FraudCall 下是否存在 "
        "valid.json、val.json 或 dev.json。"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/FraudCall"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="qwen_experiment/data"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    train_path = dataset_dir / "train.json"
    valid_path = find_valid_file(dataset_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"找不到训练集：{train_path}")

    convert_file(
        train_path,
        output_dir / "train_qwen.jsonl"
    )

    convert_file(
        valid_path,
        output_dir / "valid_qwen.jsonl"
    )


if __name__ == "__main__":
    main()
