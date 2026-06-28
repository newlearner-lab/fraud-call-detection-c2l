import json
import random
import argparse
from pathlib import Path


LEVELS = [
    "D0_base",
    "D1_credibility",
    "D2_credibility_urgency",
    "D3_credibility_urgency_emotion",
]


def is_fraud(item):
    label = item.get("label")
    return (
        isinstance(label, list)
        and len(label) >= 2
        and label[0] == 0
        and label[1] == 1
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="dataset/FraudCall_R1_PaperLike",
    )
    parser.add_argument(
        "--output",
        default="qwen_experiment/data/fraudr1_interactive_300.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="保留多少条诈骗样本；0 表示全部。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    root = Path(args.root)

    datasets = {}
    for level in LEVELS:
        path = root / level / "test.json"
        if not path.exists():
            raise FileNotFoundError(f"找不到文件：{path}")
        datasets[level] = load_json(path)

    total = len(datasets["D0_base"])

    for level, data in datasets.items():
        if len(data) != total:
            raise RuntimeError(
                f"{level} 样本数不一致：{len(data)}，应为 {total}"
            )

    benchmark = []

    for sample_id in range(total):
        d0 = datasets["D0_base"][sample_id]
        d1 = datasets["D1_credibility"][sample_id]
        d2 = datasets["D2_credibility_urgency"][sample_id]
        d3 = datasets["D3_credibility_urgency_emotion"][sample_id]

        if not is_fraud(d0):
            continue

        rounds = [
            str(d0["anchor_text"]).strip(),
            str(d1["anchor_text"]).strip(),
            str(d2["anchor_text"]).strip(),
            str(d3["anchor_text"]).strip(),
        ]

        # 仅保留每一轮均有实际变化的诈骗样本
        if not (
            rounds[0] != rounds[1]
            and rounds[1] != rounds[2]
            and rounds[2] != rounds[3]
        ):
            continue

        benchmark.append(
            {
                "sample_id": sample_id,
                "label": "fraud",
                "rounds": rounds,
            }
        )

    random.Random(args.seed).shuffle(benchmark)

    if args.limit > 0:
        benchmark = benchmark[:args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in benchmark:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("多轮诈骗基准构建完成")
    print("保留诈骗样本数：", len(benchmark))
    print("输出文件：", output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
