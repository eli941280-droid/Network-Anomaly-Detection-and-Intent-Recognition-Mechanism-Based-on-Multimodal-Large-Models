```markdown
# 🛡️ 5G-NIDS-Multimodal: 基于多模态大模型的 5G 核心网意图识别系统

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

本项目是一个针对 5G 核心网（Open5GS + UERANSIM）复杂网络环境设计的**多模态网络入侵检测系统 (NIDS)**。通过创新性地结合 **1D-ResNet（提取流量时空特征）** 与 **DistilBERT（提取专家语义知识）**，并利用 **交叉注意力机制 (Cross-Attention)** 进行异构数据对齐，有效解决了传统 IDS 缺乏深度语义理解、过度依赖端口规则（Shortcut Learning）的痛点。

## ✨ 核心特性 (Key Features)

- 📡 **5G 隧道穿透与原生数据采集**：提供针对 5G N3/N6 接口的自动化多态流量注入引擎（涵盖 DNS 爆发、大文件下载、UDP Flood、HTTP CC 等），并攻克了 GTP-U 隧道封装导致的特征聚合塌缩问题。
- 🛡️ **Anti-Cheating 训练策略**：在预处理阶段强制剔除 Source/Destination IP 及 Port 等极易引发“数据泄露”的身份特征，迫使模型学习流量突发性与包长不对称性等真实物理行为规律。
- 🧠 **双塔多模态融合架构 (Dual-Tower Fusion)**：
  - **Traffic Tower**: 采用 1D-ResNet 提取高维数值统计特征。
  - **Semantic Tower**: 利用预训练大语言模型 (DistilBERT) 解析基于特征自动生成的专家 Prompt。
  - **Cross-Attention**: 将流量特征作为 Query，语义向量作为 Key/Value 进行动态检索对齐。
- 📊 **学术级评估体系**：集成完整的可解释性 EDA（特征小提琴图）与高级模型评估图表（ROC 曲线、PR 曲线、混淆矩阵）。

## 🏗️ 模型架构 (Architecture)

```text
       [输入: 流量数值特征]                  [输入: 文本描述 Prompt]
             (14维)                        (Sequence Length, 32)
               │                                   │
               ▼                                   ▼
    ┌─────────────────────┐             ┌─────────────────────┐
    │  流量塔 (Traffic)   │             │   语义塔 (Text)     │
    │    (1D-ResNet)      │             │    (DistilBERT)     │
    └──────────┬──────────┘             └──────────┬──────────┘
               │                                   │
        (Traffic Features)                  (Semantic Features)
          [作为 Query]                      [作为 Key / Value]
               │                                   │
               ▼                                   ▼
    ┌─────────────────────────────────────────────────────┐
    │           交叉注意力融合层 (Cross-Attention)         │
    │  "用流量的行为特征，去文本知识库中检索对应的意图语义"   │
    └──────────────────────────┬──────────────────────────┘
                               │
                       (Fused Features)
                               │
                               ▼
                    ┌─────────────────────┐
                    │   分类头 (MLP)      │
                    └──────────┬──────────┘
                               │
                               ▼
                       [输出: BENIGN / DDoS]

```

## 📂 仓库结构 (Directory Structure)

```bash
├── auto_traffic_pro.py   # 5G核心网多态流量自动化生成引擎 (运行于 UE 端)
├── data_preprocess.py    # 真实流量真值逆向标注与 Anti-Cheating 清洗工具
├── plot_features.py      # 数据分布核密度可视化脚本 (EDA)
├── dataset.py            # 多模态数据集加载与 Expert Prompt 自动生成器
├── model.py              # 双塔网络与 Cross-Attention 融合层核心代码
├── train.py              # 包含早停机制的训练逻辑与高级图表绘制 (ROC/PR)
├── main.py               # 模型训练与评估主入口
├── config.py             # 全局超参数配置文件
└── distilbert_local/     # (需自行下载) DistilBERT 预训练模型权重文件夹

```

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

```bash
git clone [https://github.com/YourUsername/5G-NIDS-Multimodal.git](https://github.com/YourUsername/5G-NIDS-Multimodal.git)
cd 5G-NIDS-Multimodal
pip install torch pandas numpy scikit-learn transformers matplotlib seaborn tqdm

```

### 2. 模型下载

由于国内网络限制，请前往 [HuggingFace 镜像站](https://www.google.com/search?q=https://hf-mirror.com/distilbert-base-uncased/tree/main) 下载 DistilBERT 的基础文件，并放入 `distilbert_local/` 目录下：

* `config.json`
* `pytorch_model.bin`
* `vocab.txt`
* `tokenizer.json`
* `tokenizer_config.json`

### 3. 数据处理与模型训练

1. **数据清洗与标注**：将 `CICFlowMeter` 提取的 CSV 文件放置于根目录，运行预处理脚本：
```bash
python data_preprocess.py

```


2. **训练模型**：
```bash
python main.py

```


3. **数据可视化 (EDA)**：
```bash
python plot_features.py

```



## 📈 实验结果 (Results)

在严格剔除 IP/Port 并在混合应用层攻击（如 HTTP CC）的复杂 5G 数据集上，模型依然表现出卓越的性能：

| 类别 (Class) | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| **BENIGN (正常)** | 0.8235 | 0.9636 | 0.8881 |
| **DDoS (攻击)** | 0.9823 | 0.9073 | 0.9433 |
| **Accuracy (全局)** | **-** | **-** | **0.9247** |

*(请在运行 `main.py` 和 `plot_features.py` 后，将生成的 `1_confusion_matrix.png`, `2_roc_curve.png`, `4_feature_violin_plots.png` 等图片上传至仓库的 `images/` 文件夹，并在 README 中展示)*

## 🤝 贡献 (Contributing)

欢迎提交 Pull Requests 或开启 Issue 讨论有关多模态网络安全的进一步优化方案。


```

安排！一份接轨国际开源社区和顶会标准的英文版 `README.md` 能够极大地提升你项目的主页“逼格”，让国外研究者和导师一眼看出你的专业素养。

你可以直接在 GitHub 仓库里新建一个 `README.md`（或者把中文版重命名为 `README_zh-CN.md`，用这个英文版作为默认主页）。

---

```markdown
# 🛡️ 5G-NIDS-Multimodal: Intent Recognition System for 5G Core Networks based on Multimodal Large Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository implements an advanced **Multimodal Network Intrusion Detection System (NIDS)** tailored for the complex environment of 5G core networks (Open5GS + UERANSIM). 

By innovatively fusing **1D-ResNet** (for extracting spatiotemporal traffic features) and **DistilBERT** (for extracting expert semantic knowledge) via a **Cross-Attention mechanism**, this project addresses the limitations of traditional IDS, such as the lack of deep semantic understanding and over-reliance on superficial port rules (Shortcut Learning).

## ✨ Key Features

- 📡 **5G Tunnel Decapsulation & Native Data Collection**: Provides an automated polymorphic traffic injection engine (covering DNS bursts, heavy downloads, UDP Floods, HTTP CC, etc.) targeting 5G N3/N6 interfaces, successfully overcoming the "feature aggregation collapse" caused by GTP-U tunneling.
- 🛡️ **Anti-Cheating Training Strategy**: Strictly removes identity features (Source/Destination IPs and Ports) during the preprocessing stage to prevent data leakage. This forces the model to learn genuine physical behavior patterns, such as traffic burstiness and forward/backward asymmetry.
- 🧠 **Dual-Tower Multimodal Fusion Architecture**:
  - **Traffic Tower**: Utilizes 1D-ResNet to process high-dimensional numerical statistical features.
  - **Semantic Tower**: Employs a pre-trained LLM (DistilBERT) to parse expert Prompts automatically generated from traffic behaviors.
  - **Cross-Attention**: Dynamically aligns heterogeneous data by using traffic features as the Query and semantic vectors as the Key/Value.
- 📊 **Academic-Grade Evaluation**: Integrates comprehensive Exploratory Data Analysis (Violin plots for feature density) and advanced model evaluation metrics (ROC curve, PR curve, Confusion Matrix).

## 🏗️ Architecture

```text
       [Input: Numerical Traffic Features]         [Input: Text Expert Prompt]
                  (14-Dim)                            (Sequence Length, 32)
                     │                                         │
                     ▼                                         ▼
          ┌─────────────────────┐                   ┌─────────────────────┐
          │    Traffic Tower    │                   │   Semantic Tower    │
          │     (1D-ResNet)     │                   │    (DistilBERT)     │
          └──────────┬──────────┘                   └──────────┬──────────┘
                     │                                         │
             (Traffic Features)                        (Semantic Features)
               [Act as Query]                        [Act as Key / Value]
                     │                                         │
                     ▼                                         ▼
          ┌───────────────────────────────────────────────────────────┐
          │             Cross-Attention Fusion Layer                  │
          │ "Querying text semantics using traffic behavior patterns" │
          └─────────────────────────────┬─────────────────────────────┘
                                        │
                                (Fused Features)
                                        │
                                        ▼
                             ┌─────────────────────┐
                             │   Classifier (MLP)  │
                             └──────────┬──────────┘
                                        │
                                        ▼
                            [Output: BENIGN / DDoS]

```

## 📂 Directory Structure

```bash
├── auto_traffic_pro.py   # Automated polymorphic traffic generator for 5G UE
├── data_preprocess.py    # Ground Truth reverse labeling & Anti-Cheating cleaner
├── plot_features.py      # Feature density visualization script (EDA)
├── dataset.py            # Multimodal dataset loader & Expert Prompt generator
├── model.py              # Dual-tower network & Cross-Attention fusion core
├── train.py              # Training logic with early stopping & metric plotting
├── main.py               # Main entry for model training and evaluation
├── config.py             # Global hyperparameters configuration
└── distilbert_local/     # (To be downloaded) DistilBERT pre-trained weights

```

## 🚀 Quick Start

### 1. Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YourUsername/5G-NIDS-Multimodal.git](https://github.com/YourUsername/5G-NIDS-Multimodal.git)
cd 5G-NIDS-Multimodal
pip install torch pandas numpy scikit-learn transformers matplotlib seaborn tqdm

```

### 2. Download Pre-trained Weights

Download the DistilBERT base files from [HuggingFace](https://huggingface.co/distilbert-base-uncased/tree/main) and place them in the `distilbert_local/` directory:

* `config.json`
* `pytorch_model.bin`
* `vocab.txt`
* `tokenizer.json`
* `tokenizer_config.json`

### 3. Data Processing & Training

1. **Data Cleaning & Labeling**: Place the raw CSV extracted by `CICFlowMeter` into the root directory and run the preprocessor:
```bash
python data_preprocess.py

```


2. **Train the Model**:
```bash
python main.py

```


3. **Data Visualization (EDA)**:
```bash
python plot_features.py

```



## 📈 Results

Evaluated on a highly complex 5G dataset with IP/Port features strictly removed and mixed application-layer attacks (e.g., HTTP CC) introduced, the model demonstrates outstanding robustness:

| Class | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| **BENIGN** | 0.8235 | 0.9636 | 0.8881 |
| **DDoS** | 0.9823 | 0.9073 | 0.9433 |
| **Accuracy (Global)** | **-** | **-** | **0.9247** |

*(Upload your generated `1_confusion_matrix.png`, `2_roc_curve.png`, and `4_feature_violin_plots.png` to an `images/` folder and showcase them here!)*

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License.

```

