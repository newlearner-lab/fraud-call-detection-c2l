import os
import json
import random
import re
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ========== 文件路径 ==========
TRAIN_CSV_PATH = "训练集结果.csv"
TEST_CSV_PATH = "测试集结果.csv"   # 如果你的测试集文件名不同，改这里

OUT_DIR = "dataset/FraudCall"

# ========== 字段名 ==========
TEXT_COL = "specific_dialogue_content"
LABEL_COL = "is_fraud"
FRAUD_TYPE_COL = "fraud_type"
CALL_TYPE_COL = "call_type"
STRATEGY_COL = "interaction_strategy"


def parse_bool(x):
    """
    将 is_fraud 转成二分类标签：
    0 = 正常通话
    1 = 诈骗/虚假通话
    """
    if pd.isna(x):
        return None

    if isinstance(x, bool):
        return 1 if x else 0

    if isinstance(x, (int, float)):
        if x == 1:
            return 1
        if x == 0:
            return 0

    s = str(x).strip().lower()

    if s in ["true", "1", "1.0", "yes", "y", "诈骗", "欺诈", "fraud"]:
        return 1

    if s in ["false", "0", "0.0", "no", "n", "正常", "normal"]:
        return 0

    return None


def clean_text(text):
    """
    简单清洗通话文本。
    保留 left/right 对话结构，因为这对通话语义有帮助。
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\ufeff", "")
    text = text.replace("音频内容：", "")
    text = text.replace("**", "")
    text = re.sub(r"\n{3,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def to_one_hot(label):
    """
    C2L 官方代码使用 one-hot label。
    label=0 正常：[1, 0]
    label=1 诈骗：[0, 1]
    """
    return [1, 0] if label == 0 else [0, 1]


def safe_sample(df_pool):
    if len(df_pool) == 0:
        return None
    return df_pool.sample(1, random_state=random.randint(0, 10**9)).iloc[0]


def load_and_clean_csv(csv_path, dataset_name):
    """
    读取并清洗单个 CSV。
    dataset_name 只是用于打印信息，比如 train 或 test。
    """
    print(f"\n========== 正在读取 {dataset_name}: {csv_path} ==========")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件：{csv_path}")

    df = pd.read_csv(csv_path)

    print(f"{dataset_name} 原始样本数：", len(df))
    print(f"{dataset_name} 原始字段：", list(df.columns))

    keep_cols = [
        TEXT_COL,
        LABEL_COL,
        FRAUD_TYPE_COL,
        CALL_TYPE_COL,
        STRATEGY_COL
    ]

    missing_cols = [col for col in keep_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"{dataset_name} 缺少字段：{missing_cols}")

    df = df[keep_cols].copy()

    df["text"] = df[TEXT_COL].apply(clean_text)
    df["label"] = df[LABEL_COL].apply(parse_bool)

    df["fraud_type"] = df[FRAUD_TYPE_COL].fillna("正常")
    df["call_type"] = df[CALL_TYPE_COL].fillna("未知")
    df["interaction_strategy"] = df[STRATEGY_COL].fillna("未知")

    # 删除无标签和空文本
    before = len(df)
    df = df.dropna(subset=["label", "text"])
    df = df[df["text"].str.len() > 0]
    after = len(df)

    print(f"{dataset_name} 删除缺失标签/空文本样本数：", before - after)

    df["label"] = df["label"].astype(int)

    # 去重
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    after_dedup = len(df)

    print(f"{dataset_name} 去重删除样本数：", before_dedup - after_dedup)
    print(f"{dataset_name} 清洗后样本数：", len(df))

    print(f"\n{dataset_name} 标签分布：")
    print(df["label"].value_counts())

    print(f"\n{dataset_name} call_type 分布：")
    print(df["call_type"].value_counts())

    print(f"\n{dataset_name} fraud_type 分布：")
    print(df["fraud_type"].value_counts())

    return df


def build_triplet_json(data):
    """
    构造 C2L 训练所需三元组：
    anchor_text：当前样本
    positive_text：同类样本
    negative_text：异类样本

    构造逻辑：
    1. 诈骗样本：positive 优先取相同 fraud_type 的诈骗样本
    2. 正常样本：positive 优先取相同 call_type 的正常样本
    3. negative：优先取相同 call_type 但 is_fraud 不同的样本
    """
    data = data.reset_index(drop=True)
    examples = []

    for idx, row in data.iterrows():
        label = int(row["label"])
        text = row["text"]
        call_type = row.get("call_type", "未知")
        fraud_type = row.get("fraud_type", "未知")

        # ---------- positive ----------
        if label == 1:
            # 诈骗样本：优先找相同诈骗类型
            pos_pool = data[
                (data["label"] == 1)
                & (data.index != idx)
                & (data["fraud_type"] == fraud_type)
            ]

            # 找不到就找任意诈骗样本
            if len(pos_pool) == 0:
                pos_pool = data[
                    (data["label"] == 1)
                    & (data.index != idx)
                ]
        else:
            # 正常样本：优先找相同通话类型
            pos_pool = data[
                (data["label"] == 0)
                & (data.index != idx)
                & (data["call_type"] == call_type)
            ]

            # 找不到就找任意正常样本
            if len(pos_pool) == 0:
                pos_pool = data[
                    (data["label"] == 0)
                    & (data.index != idx)
                ]

        pos_row = safe_sample(pos_pool)
        positive_text = pos_row["text"] if pos_row is not None else text

        # ---------- negative ----------
        # 优先选择同一 call_type 但标签相反的 hard negative
        neg_pool = data[
            (data["label"] != label)
            & (data["call_type"] == call_type)
        ]

        # 找不到就选择任意异类样本
        if len(neg_pool) == 0:
            neg_pool = data[data["label"] != label]

        neg_row = safe_sample(neg_pool)
        negative_text = neg_row["text"] if neg_row is not None else text

        examples.append({
            "anchor_text": text,
            "positive_text": positive_text,
            "negative_text": negative_text,
            "label": to_one_hot(label),
            "triplet_sample_mask": True
        })

    return examples


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ========== 1. 读取训练集和测试集 ==========
    train_df = load_and_clean_csv(TRAIN_CSV_PATH, "训练集")
    test_df = load_and_clean_csv(TEST_CSV_PATH, "测试集")

    # ========== 2. 防止训练集和测试集文本重复 ==========
    # 如果训练集和测试集有重复文本，建议从训练集中删掉这些重复文本，避免数据泄漏。
    test_texts = set(test_df["text"].tolist())
    before_remove_leak = len(train_df)

    train_df = train_df[~train_df["text"].isin(test_texts)].reset_index(drop=True)

    after_remove_leak = len(train_df)

    print("\n========== 数据泄漏检查 ==========")
    print("从训练集中删除与测试集重复的样本数：", before_remove_leak - after_remove_leak)
    print("去重后训练集样本数：", len(train_df))

    # ========== 3. 从训练集中划分 train / valid ==========
    train, valid = train_test_split(
        train_df,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=train_df["label"]
    )

    test = test_df

    print("\n========== 最终数据划分结果 ==========")
    print("train:", len(train), train["label"].value_counts().to_dict())
    print("valid:", len(valid), valid["label"].value_counts().to_dict())
    print("test :", len(test), test["label"].value_counts().to_dict())

    # ========== 4. 构造 C2L JSON ==========
    train_json = build_triplet_json(train)
    valid_json = build_triplet_json(valid)
    test_json = build_triplet_json(test)

    save_json(os.path.join(OUT_DIR, "train.json"), train_json)
    save_json(os.path.join(OUT_DIR, "valid.json"), valid_json)
    save_json(os.path.join(OUT_DIR, "test.json"), test_json)

    print("\n========== 已生成 C2L 格式数据 ==========")
    print(os.path.join(OUT_DIR, "train.json"))
    print(os.path.join(OUT_DIR, "valid.json"))
    print(os.path.join(OUT_DIR, "test.json"))

    print("\n训练样本示例：")
    print(json.dumps(train_json[0], ensure_ascii=False, indent=2)[:1000])


if __name__ == "__main__":
    main()