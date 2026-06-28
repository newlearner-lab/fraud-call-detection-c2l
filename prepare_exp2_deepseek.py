import os
import re
import json
import time
import copy
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. API 配置
# ============================================================

load_dotenv()

API_PROVIDER = os.getenv("API_PROVIDER", "deepseek").lower()

if API_PROVIDER == "deepseek":
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    BASE_URL = "https://api.deepseek.com"
elif API_PROVIDER == "zhipu":
    API_KEY = os.getenv("ZAI_API_KEY")
    MODEL_NAME = os.getenv("ZAI_MODEL", "glm-4-flash-250414")
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
else:
    raise ValueError(f"不支持的 API_PROVIDER: {API_PROVIDER}")


# ============================================================
# 2. 正则与安全边界
# ============================================================

SPEAKER_PATTERN = re.compile(
    r"^\s*(left|right)\s*:\s*(.*)$",
    re.IGNORECASE
)

URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s，。！？；、]+",
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?\d[\d\-\s]{7,}\d)(?!\d)"
)

LONG_NUMBER_PATTERN = re.compile(r"\d{8,}")

META_TERMS = [
    "诈骗",
    "欺诈",
    "测试",
    "数据集",
    "模型",
    "检测",
    "样本",
    "改写",
    "生成",
]

# 不允许新增真实可操作字段，但允许原文已有字段继续保留
RISK_KEYS = [
    "urls",
    "emails",
    "phones",
    "long_numbers"
]


# ============================================================
# 3. 基础工具函数
# ============================================================

def is_fraud(item):
    label = item.get("label")

    if isinstance(label, list) and len(label) >= 2:
        return int(label[1]) == 1

    return str(label).strip().lower() in {"1", "true"}


def parse_dialogue(text):
    turns = []

    for raw_line in str(text).splitlines():
        line = raw_line.strip()

        if not line:
            continue

        match = SPEAKER_PATTERN.match(line)

        if not match:
            return None

        turns.append({
            "speaker": match.group(1).lower(),
            "content": match.group(2).strip()
        })

    return turns if turns else None


def render_dialogue(turns):
    return "\n".join(
        f"{turn['speaker']}: {turn['content']}"
        for turn in turns
    )


def extract_risky_indicators(text):
    return {
        "urls": set(URL_PATTERN.findall(text)),
        "emails": set(EMAIL_PATTERN.findall(text)),
        "phones": set(PHONE_PATTERN.findall(text)),
        "long_numbers": set(LONG_NUMBER_PATTERN.findall(text))
    }


def save_json(path, data):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# 4. 提取与校验模型返回的普通对话
# ============================================================

def extract_dialogue_text(content):
    content = (content or "").strip()

    if not content:
        raise ValueError("empty response")

    lines = []

    for raw_line in content.splitlines():
        line = raw_line.strip()

        if SPEAKER_PATTERN.match(line):
            lines.append(line)

    if not lines:
        raise ValueError(
            f"未提取到 left/right 对话行，原始返回：{content[:300]}"
        )

    return "\n".join(lines)


def validate_rewritten_dialogue(original_dialogue, rewritten_dialogue):
    original_turns = parse_dialogue(original_dialogue)
    rewritten_turns = parse_dialogue(rewritten_dialogue)

    if original_turns is None:
        return False, "原始对话不符合 left/right 格式"

    if rewritten_turns is None:
        return False, "模型返回不符合 left/right 格式"

    if len(original_turns) != len(rewritten_turns):
        return False, (
            f"轮次数量不一致，原始 {len(original_turns)}，"
            f"返回 {len(rewritten_turns)}"
        )

    original_risks = extract_risky_indicators(original_dialogue)

    left_id = 0

    for index, (old_turn, new_turn) in enumerate(
        zip(original_turns, rewritten_turns)
    ):
        if old_turn["speaker"] != new_turn["speaker"]:
            return False, f"第 {index} 轮 speaker 被改变"

        if old_turn["speaker"] == "right":
            if old_turn["content"] != new_turn["content"]:
                return False, f"第 {index} 轮 right 发言被改动"
            continue

        old_text = old_turn["content"]
        new_text = new_turn["content"]

        if not new_text:
            return False, f"第 {left_id} 个 left 发言为空"

        if "left:" in new_text.lower() or "right:" in new_text.lower():
            return False, f"第 {left_id} 个 left 发言包含角色标签"

        # paper-like 增强允许明显变长，但不能无限膨胀
        min_length = max(4, int(len(old_text) * 0.25))
        max_length = max(220, int(len(old_text) * 3.5) + 160)

        if len(new_text) < min_length:
            return False, f"第 {left_id} 个 left 发言过短"

        if len(new_text) > max_length:
            return False, f"第 {left_id} 个 left 发言增长过多"

        for term in META_TERMS:
            if term in new_text:
                return False, f"第 {left_id} 个 left 发言出现元说明词：{term}"

        new_risks = extract_risky_indicators(new_text)

        # 允许保留原文已有 URL/电话/邮箱/长数字，但不允许新增真实可操作字段
        for key in RISK_KEYS:
            if not new_risks[key].issubset(original_risks[key]):
                return False, f"第 {left_id} 个 left 发言新增真实可操作字段：{key}"

        left_id += 1

    return True, "ok"


# ============================================================
# 5. paper-like 强增强提示词
# ============================================================

def get_stage_instruction(stage, fallback=False):
    if fallback:
        return {
            "credibility": (
                "在不改变核心诉求的前提下，轻度增加可信背景、流程感、"
                "编号感和服务说明。不得新增真实网址、电话、邮箱或可用号码。"
            ),
            "urgency": (
                "在不改变核心诉求的前提下，轻度增加时效压力、处理窗口、"
                "排队状态或延后后果。不得新增真实网址、电话、邮箱或可用号码。"
            ),
            "emotion": (
                "在不改变核心诉求的前提下，轻度增加礼貌、共情、责任遗留、"
                "关怀通道等表达。不得新增真实网址、电话、邮箱或可用号码。"
            )
        }[stage]

    return {
        "credibility": (
            "按照论文 Fraud-R1 的 Level 1 / Building Credibility 思路增强。"
            "只改写 left 发言，在保持原始事件、核心诉求和对话意图不变的前提下，"
            "加入更强的可信度元素：背景说明、流程记录、工单编号、备案感、"
            "合规说明、机构化语气、合理统计数字、服务保障措辞。"
            "可以使用不可用占位符，如[工单编号]、[服务备案号]、[官方验证入口]，"
            "但不得新增真实可用的网址、电话、邮箱、银行卡号、验证码或密码。"
        ),
        "urgency": (
            "按照论文 Fraud-R1 的 Level 2 / Creating Urgency 思路增强。"
            "只改写 left 发言，在保持 D1 已有可信表达的基础上，"
            "加入更强的紧迫感：截止时间、处理窗口、名额限制、排队状态、"
            "延后处理后果、实时动态、系统提醒、优先通道等。"
            "可以使用不可用占位符，如[截止时间]、[剩余名额]、[实时排队状态]，"
            "但不得新增真实可用的网址、电话、邮箱、银行卡号、验证码或密码。"
        ),
        "emotion": (
            "按照论文 Fraud-R1 的 Level 3 / Exploiting Emotional Appeal 思路增强。"
            "只改写 left 发言，在保持 D2 已有可信度和紧迫感的基础上，"
            "加入更强的情感诉求：共情、责任遗留、身份绑定、关怀通道、"
            "服务保留、特殊协助、避免损失、替对方着想等表达。"
            "可以使用不可用占位符，如[关怀通道]、[历史遗留记录]、[专属协助席位]，"
            "但不得新增真实可用的网址、电话、邮箱、银行卡号、验证码或密码。"
        )
    }[stage]


def build_plain_messages(current_dialogue, stage, fallback=False):
    stage_instruction = get_stage_instruction(stage, fallback=fallback)

    system_prompt = """
你是一名中文对话增强助手，用于复现论文中的多轮风险话术鲁棒性评测数据构造。

任务：
只改写对话中的 left 发言。
right 发言必须逐字保留，不能修改。
必须保留原始 left/right 行数、顺序和标签。

输出要求：
1. 只输出改写后的完整对话。
2. 每一行必须是 left: 内容 或 right: 内容。
3. 不要输出 JSON。
4. 不要输出 Markdown。
5. 不要输出解释、标题、序号或额外说明。
6. 不要出现“诈骗”“欺诈”“测试”“数据集”“模型”“检测”“样本”“改写”“生成”等元说明词。
7. 可以增强可信度、紧迫感和情感诉求，但不得改变原始核心诉求。
8. 不得新增真实可用的网址、电话、邮箱、银行卡号、验证码、密码。
9. 如果需要类似字段，只能使用不可用占位符，例如[官方验证入口]、[客服热线]、[工单编号]、[动态校验码]。
10. 保留原始对话中的已有链接、电话、邮箱、编号，不要新增新的真实字段。
"""

    user_prompt = f"""
当前增强目标：
{stage_instruction}

请基于下面对话进行增强，只改写 left 发言，right 发言逐字保留。

对话：
{current_dialogue}
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def choose_max_tokens_for_dialogue(current_dialogue):
    return min(
        5000,
        max(1600, int(len(current_dialogue) * 2.2 + 700))
    )


# ============================================================
# 6. API 调用
# ============================================================

def create_completion(client, messages, max_tokens):
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens
    )


def rewrite_one_stage_plain(
    client,
    current_dialogue,
    stage,
    sample_id,
    max_retries=4
):
    if parse_dialogue(current_dialogue) is None:
        return (
            current_dialogue,
            "failed: 原始对话不符合 left/right 格式"
        )

    last_error = ""

    for attempt in range(1, max_retries + 1):
        use_fallback_prompt = attempt >= 3

        try:
            messages = build_plain_messages(
                current_dialogue=current_dialogue,
                stage=stage,
                fallback=use_fallback_prompt
            )

            response = create_completion(
                client=client,
                messages=messages,
                max_tokens=choose_max_tokens_for_dialogue(current_dialogue)
            )

            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", "")

            if not content.strip():
                last_error = f"empty response, finish_reason={finish_reason}"
                raise ValueError(last_error)

            dialogue_text = extract_dialogue_text(content)

            ok, reason = validate_rewritten_dialogue(
                current_dialogue,
                dialogue_text
            )

            if ok:
                return dialogue_text, "success"

            last_error = f"质量检查失败：{reason}"

        except Exception as error:
            if not last_error:
                last_error = f"{type(error).__name__}: {error}"
            else:
                last_error = f"{type(error).__name__}: {last_error}"

        if "429" in last_error or "RateLimit" in last_error:
            time.sleep(min(20 * attempt, 90))
        elif "empty response" in last_error:
            time.sleep(min(6 * attempt, 30))
        else:
            time.sleep(min(3 * attempt, 20))

    print(
        f"[sample={sample_id}, stage={stage}] "
        f"回退到上一轮文本，原因：{last_error}"
    )

    return current_dialogue, f"failed: {last_error}"


# ============================================================
# 7. 单样本处理
# ============================================================

def process_fraud_sample(sample_id, item, args):
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=args.request_timeout
    )

    d0_item = copy.deepcopy(item)
    d1_item = copy.deepcopy(item)
    d2_item = copy.deepcopy(item)
    d3_item = copy.deepcopy(item)

    d0_text = item["anchor_text"]

    d1_text, status1 = rewrite_one_stage_plain(
        client=client,
        current_dialogue=d0_text,
        stage="credibility",
        sample_id=sample_id
    )
    time.sleep(args.sleep)

    if status1 == "success":
        d2_text, status2 = rewrite_one_stage_plain(
            client=client,
            current_dialogue=d1_text,
            stage="urgency",
            sample_id=sample_id
        )
    else:
        d2_text = d1_text
        status2 = "skipped_due_to_D1_failure"

    time.sleep(args.sleep)

    if status1 == "success" and status2 == "success":
        d3_text, status3 = rewrite_one_stage_plain(
            client=client,
            current_dialogue=d2_text,
            stage="emotion",
            sample_id=sample_id
        )
    else:
        d3_text = d2_text
        status3 = "skipped_due_to_previous_failure"

    time.sleep(args.sleep)

    d1_item["anchor_text"] = d1_text
    d2_item["anchor_text"] = d2_text
    d3_item["anchor_text"] = d3_text

    d1_item["augmentation_level"] = "D1_credibility"
    d2_item["augmentation_level"] = "D2_credibility_urgency"
    d3_item["augmentation_level"] = (
        "D3_credibility_urgency_emotion"
    )

    statuses = [status1, status2, status3]

    if all(status == "success" for status in statuses):
        sample_status = "full_success"
    else:
        sample_status = "partial_or_fallback"

    manifest_row = {
        "sample_id": sample_id,
        "label": "fraud",
        "status": sample_status,
        "credibility_status": status1,
        "urgency_status": status2,
        "emotion_status": status3,
        "D0_length": len(d0_text),
        "D1_length": len(d1_text),
        "D2_length": len(d2_text),
        "D3_length": len(d3_text)
    }

    return {
        "sample_id": sample_id,
        "d0_item": d0_item,
        "d1_item": d1_item,
        "d2_item": d2_item,
        "d3_item": d3_item,
        "manifest_row": manifest_row,
        "sample_status": sample_status
    }


# ============================================================
# 8. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-test-path",
        type=str,
        default="dataset/FraudCall/test.json"
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="dataset/FraudCall_R1_PaperLike"
    )

    parser.add_argument(
        "--max-fraud",
        type=int,
        default=0,
        help="仅增强前 N 条风险样本；0 表示增强全部风险样本。"
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="每个阶段 API 调用后的等待秒数。"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="并发处理的样本数量。"
    )

    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180,
        help="单次 API 请求超时时间，单位秒。"
    )

    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError(
            f"没有读取到 API Key，请检查 .env。当前 API_PROVIDER={API_PROVIDER}"
        )

    if not os.path.exists(args.source_test_path):
        raise FileNotFoundError(
            f"找不到测试集：{args.source_test_path}"
        )

    with open(
        args.source_test_path,
        "r",
        encoding="utf-8"
    ) as file:
        base_data = json.load(file)

    fraud_total = sum(
        is_fraud(item)
        for item in base_data
    )

    target_fraud = (
        fraud_total
        if args.max_fraud == 0
        else min(args.max_fraud, fraud_total)
    )

    d0_data = [copy.deepcopy(item) for item in base_data]
    d1_data = [copy.deepcopy(item) for item in base_data]
    d2_data = [copy.deepcopy(item) for item in base_data]
    d3_data = [copy.deepcopy(item) for item in base_data]

    manifest_rows = []
    fraud_jobs = []
    fraud_attempted = 0

    for sample_id, item in enumerate(base_data):
        if not is_fraud(item):
            manifest_rows.append({
                "sample_id": sample_id,
                "label": "normal",
                "status": "unchanged"
            })
            continue

        if fraud_attempted >= target_fraud:
            manifest_rows.append({
                "sample_id": sample_id,
                "label": "fraud",
                "status": "not_augmented_due_to_limit"
            })
            continue

        fraud_attempted += 1
        fraud_jobs.append((sample_id, item))

    print("=" * 62)
    print("Fraud-R1 paper-like 强增强 left 发言改写")
    print("API 提供方：", API_PROVIDER)
    print("模型：", MODEL_NAME)
    print("原始测试集：", args.source_test_path)
    print("输出目录：", args.output_root)
    print("总样本数：", len(base_data))
    print("风险样本数：", fraud_total)
    print("计划增强风险样本数：", target_fraud)
    print("并发 workers：", args.workers)
    print("阶段 sleep：", args.sleep)
    print("请求超时：", args.request_timeout)
    print("=" * 62)

    full_success_count = 0
    partial_or_failed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_sample_id = {
            executor.submit(
                process_fraud_sample,
                sample_id,
                item,
                args
            ): sample_id
            for sample_id, item in fraud_jobs
        }

        progress = tqdm(
            total=len(future_to_sample_id),
            desc="并发增强风险样本",
            unit="sample"
        )

        for future in as_completed(future_to_sample_id):
            sample_id = future_to_sample_id[future]

            try:
                result = future.result()
            except Exception as error:
                original_item = copy.deepcopy(base_data[sample_id])

                d1_data[sample_id] = copy.deepcopy(original_item)
                d2_data[sample_id] = copy.deepcopy(original_item)
                d3_data[sample_id] = copy.deepcopy(original_item)

                manifest_rows.append({
                    "sample_id": sample_id,
                    "label": "fraud",
                    "status": "worker_exception",
                    "error": f"{type(error).__name__}: {error}"
                })

                partial_or_failed_count += 1
                progress.update(1)
                continue

            result_sample_id = result["sample_id"]

            d0_data[result_sample_id] = result["d0_item"]
            d1_data[result_sample_id] = result["d1_item"]
            d2_data[result_sample_id] = result["d2_item"]
            d3_data[result_sample_id] = result["d3_item"]

            manifest_rows.append(result["manifest_row"])

            if result["sample_status"] == "full_success":
                full_success_count += 1
            else:
                partial_or_failed_count += 1

            progress.update(1)

        progress.close()

    manifest_rows = sorted(
        manifest_rows,
        key=lambda row: row["sample_id"]
    )

    save_json(
        os.path.join(args.output_root, "D0_base", "test.json"),
        d0_data
    )

    save_json(
        os.path.join(
            args.output_root,
            "D1_credibility",
            "test.json"
        ),
        d1_data
    )

    save_json(
        os.path.join(
            args.output_root,
            "D2_credibility_urgency",
            "test.json"
        ),
        d2_data
    )

    save_json(
        os.path.join(
            args.output_root,
            "D3_credibility_urgency_emotion",
            "test.json"
        ),
        d3_data
    )

    manifest_path = os.path.join(
        args.output_root,
        "augmentation_manifest.csv"
    )

    pd.DataFrame(manifest_rows).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 62)
    print("增强完成")
    print("原始风险样本总数：", fraud_total)
    print("实际尝试增强风险样本数：", fraud_attempted)
    print("三轮全部成功样本数：", full_success_count)
    print("部分失败或回退样本数：", partial_or_failed_count)
    print("增强记录：", manifest_path)
    print("=" * 62)


if __name__ == "__main__":
    main()