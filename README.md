# Fraud Call Detection with C2L-Chinese-RoBERTa

本项目基于 AAAI 2022 论文 **C2L: Causally Contrastive Learning for Robust Text Classification** 的官方 PyTorch 实现进行修改与扩展，用于完成中文虚假通话 / 诈骗通话检测任务。

原始 C2L 项目主要以英文 CFIMDb 数据集作为示例，本项目将其迁移到中文通话文本分类场景中，构建了基于 Chinese-RoBERTa 的虚假通话检测模型，并进一步引入 C2L 对比学习机制提升模型鲁棒性。

---

## 1. Task Description

本项目将虚假通话检测建模为中文文本二分类任务：

```text
输入：通话文本 specific_dialogue_content
输出：是否诈骗 is_fraud

##2.
数据预处理代码:prepare_fraud_c2l_data.py
训练代码：train_bert_imdb_pairwise_shellscript.py
评估代码：evaluate_and_plot.py
评估代码生成的图表文件夹：eval_outputs



以下是原项目（改动前）的介绍
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
