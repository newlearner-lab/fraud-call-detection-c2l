# Fraud Call Detection with C2L-Chinese-RoBERTa

本项目基于 AAAI 2022 论文 **C2L: Causally Contrastive Learning for Robust Text Classification** 的官方 PyTorch 实现进行修改与扩展，用于完成中文虚假通话 / 诈骗通话检测任务。

  原始 C2L 项目主要以英文 CFIMDb 数据集作为示例，本项目将其迁移到中文通话文本分类场景中，构建了基于 Chinese-RoBERTa 的虚假通话检测模型，并进一步引入 C2L 对比学习机制提升模型鲁棒性,同时引入千问开源模型进行对比。本项目分为实验一和实验二两大部分,实验一将虚假通话检测建模为中文文本二分类任务,对比 Chinese-RoBERTa 与 加了C2L方法的Chinese-RoBERTa 之间在给的对话数据集上的效果,实验二则调用deepssek的API对数据集进行改写,在改写的数据集上再次测试模型的性能,实验二加入在原始数据集上微调的千问开源模型,与实验一的两个模型一起在改写的数据集上测试得到测试结果,实验结果发现三个模型表现都接近满分。
  本研究觉得之前模型分数极高除了模型本身性能外,与二分类任务有关。为探究模型是否能够真实达到极高性能,准确识别出对话是欺诈还是正常.还加入一个额外实验即调用未微调的千问模型,在改写的数据集上抽取300条样本进行测试.该测试与前两个实验测试性能的方式不同。前两个实验是做二分类任务,模型判断对话是"虚假"还是"正常",评判指标是Accuracy、Precision、Recall 和 F1四个。补充实验的测试方法是未微调的千问模型以两种角色提示词生成开放式对话,两种角色提示词分别是Helpful Assistant 和 Role-play 两种提示方式。Helpful Assistant 表示普通助手模式，即模型以通用智能助手身份回答用户问题，主要目标是尽量提供帮助。Role-play 表示角色扮演模式，即在提示词中明确要求模型扮演具有安全意识的反诈骗助手或风险防御人员，使模型在回复时更加关注对话中的诈骗诱导、危险链接、验证码泄露、转账请求等风险因素。评价是否成功识别则调用智谱API作为评判,分为三种评判结果:suceess,failure,more_info.评判为success则成功防御,failure则说明失败,more_info则说明需要更多信息.结果发现未微调模型在开放式对话防御中效果不佳。

---


## 项目结构

```text
causally-contrastive-learning/
├── __pycache__/                              # Python 缓存文件
├── checkpoints/                              # 模型检查点
├── classes/                                  # 模型、数据处理及工具类
│
├── dataset/                                  # 数据集目录
│   ├── CFIMDb/                               # 原项目 CFIMDb 数据集
│   ├── FraudCall/                            # 诈骗通话数据集
│   ├── FraudCall_R1_PaperLike/               # FraudR1 PaperLike 实验数据
│   └── work_origin_dataset/                  # 原始工作数据集
│
├── exp1/                                     # 实验一：C2L 文本分类
│   ├── eval_outputs/                         # 评估结果及图表
│   ├── data_preprocess.py                    # 数据预处理脚本
│   ├── evaluate_and_plot.py                  # 模型评估与结果绘图
│   ├── evaluate_bert_imdb_pairwise_shellscript.py
│   │                                         # BERT/C2L 模型评估脚本
│   ├── prepare_fraud_c2l_data.py             # 构造 FraudCall C2L 数据
│   └── train_bert_imdb_pairwise_shellscript.py
│                                             # BERT/C2L 模型训练脚本
│
├── exp2/                                     # 实验二：Fraud数据增强实验
│   └── exp2_outputs_paperlike/               # 实验二数据及输出结果
│       ├── D0_base/                          # 原始基线数据
│       ├── D1_credibility/                   # 加入可信度策略的数据
│       ├── D2_credibility_urgency/           # 加入可信度和紧迫性策略
│       ├── D3_credibility_urgency_emotion/   # 加入可信度、紧迫性和情绪策略
│   ├── augmentation_sample_counts.png    # 数据增强样本数量统计图
│   ├── experiment2_metrics_table.pdf     # 实验指标表 PDF
│   ├── experiment2_metrics_table.png     # 实验指标表图片
│   ├── plot_experiment2_table.py         # 实验指标绘图脚本
│   └── prepare_exp2_deepseek.py          # 使用 DeepSeek 准备增强数据
│
└── qwen_experiment/                          # Qwen 大模型实验
    ├── __pycache__/                          # Python 缓存文件
    ├── checkpoints/                          # QLoRA 模型检查点
    ├── data/                                 # Qwen 训练与评估数据
    ├── outputs_fraudr1/                      # FraudR1 交互实验输出
    ├── outputs_paperlike/                    # PaperLike 实验输出
    ├── build_fraudr1_interactive_benchmark.py
    │                                         # 构建 FraudR1 交互式评测集
    ├── evaluate_fraudr1_interactive.py       # 执行 FraudR1 交互式评估
    ├── evaluate_qwen.py                      # 评估 Qwen 分类模型
    ├── fraudr1_prompt_comparison_table.png   # 不同提示策略的结果对比图
    ├── plot_fraudr1_table.py                 # 生成 FraudR1 实验结果图表
    ├── prepare_qwen_data.py                  # 准备 Qwen 训练数据
    └── train_qwen_qlora.py                   # 使用 QLoRA 微调 Qwen 模型
```

### 目录说明

- `exp1/`：基于 C2L 和 BERT 的诈骗通话文本二分类实验，包括数据预处理、模型训练、模型评估及结果可视化。
- `exp2/`：研究可信度、紧迫性和情绪等策略对数据增强效果的影响，并生成实验指标表。
- `qwen_experiment/`：基于 QLoRA 的诈骗通话识别实验，涵盖数据准备、参数高效微调、常规分类评估、FraudR1 交互式基准构建及结果可视化。
- `dataset/`：保存原始数据、处理后的数据和不同实验使用的数据集。
- `checkpoints/`：保存训练过程中生成的模型参数和检查点。


## 实验结果

### 实验一：数据增强样本分布

<img width="398" height="243" alt="image" src="https://github.com/user-attachments/assets/1a333b24-9d24-4530-abb3-f0ec843dd748" />


### 实验二：模型指标对比

<img width="3345" height="1984" alt="image" src="https://github.com/user-attachments/assets/019992ee-b47d-486d-b335-e1de7de198ae" />


### 实验三：FraudR1 提示策略对比
<img width="2763" height="1753" alt="image" src="https://github.com/user-attachments/assets/51d74b35-30c8-4a9f-9c68-65b6f39d5a68" />



## 以下是原项目（原论文官方GitHub）的介绍
# C2L: Causally Contrastive Learning for Robust Text Classification
Official pytorch implementation of [**C2L: Causally Contrastive Learning for Robust Text Classification**](https://ojs.aaai.org/index.php/AAAI/article/download/21296/version/19583/21045) (AAAI 2022) by Seungtaek Choi*, Myeongho Jeong*, Hojae Han, Seung-won Hwang.

## Setup
In this repository, we only treat CFIMDb dataset as an example. This also works on another datasets which are mentioned in our paper.
### Requirements
```
pip install -r requirements.txt
```

### Download Dataset
You can download dataset from the repository of [Learning the Difference that Makes a Difference with Counterfactually-Augmented Data](https://github.com/acmi-lab/counterfactually-augmented-data).
Then, please put the dataset into `dataset/CFIMDb/aclImdb/` and run the python script below.
```
cd utils
python reform_cfimdb_dataset.py
cd ..
```

However, we already pre-process the data for our training code. you can just clone this repository and run the training script below.
Also, there also exists the dataset augmented with our approach. You can train with this dataset directly.

## Train & Evaluate
### Train vanilla model
```
bash train_cfimdb_public.sh
```
### Generate counterfactually masked samples
To generate counterfactually masked samples, we provide a notebook [pairing-data-ours-public.ipynb](https://github.com/hist0613/counterfactual-robustness/blob/main/pairing-data-ours-public.ipynb). Please run all shells sequentially. After that, please run the code below to reform the output to trainable dataset.
```
cd utils
python triplets_masking_dataset.py
cd ..
```
### Train model with C2L
```
bash train_cfimdb_ours_public.sh
```
