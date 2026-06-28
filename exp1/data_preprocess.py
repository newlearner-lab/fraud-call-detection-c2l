import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读取原始数据
print("正在加载数据集...")
df = pd.read_csv('训练集结果.csv')

# 2. 清洗数据：去除没有目标标签的数据行
df = df.dropna(subset=['is_fraud'])

# 3. 分层抽样：按照 8:2 划分训练集和验证集
# stratify=df['is_fraud'] 是关键，保证正负样本比例不失衡
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['is_fraud']
)

# 4. 保存为供模型直接读取的 CSV 格式
# 使用 utf-8-sig 防止中文出现乱码
train_df.to_csv('train_split.csv', index=False, encoding='utf-8-sig')
val_df.to_csv('val_split.csv', index=False, encoding='utf-8-sig')

print(f"✅ 划分完成！")
print(f"训练集已保存为 train_split.csv (共 {len(train_df)} 条)")
print(f"验证集已保存为 val_split.csv (共 {len(val_df)} 条)")