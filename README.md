[English Version](#english-version) | [中文版](#中文版)

---

<a id="english-version"></a>
# 🛡️ 5G-NIDS-Multimodal: Intent Recognition System for 5G Core Networks based on Multimodal Large Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository implements an advanced **Multimodal Network Intrusion Detection System (NIDS)** tailored for the complex environment of 5G core networks (Open5GS + UERANSIM). 

By innovatively fusing **1D-ResNet** (for extracting spatiotemporal traffic features) and **DistilBERT** (for extracting expert semantic knowledge) via a **Cross-Attention mechanism**, this project addresses the limitations of traditional IDS, such as the lack of deep semantic understanding and over-reliance on superficial port rules (Shortcut Learning).

## ✨ Key Features

* **5G Tunnel Decapsulation & Native Data Collection**: Provides an automated polymorphic traffic injection engine targeting 5G N3/N6 interfaces, successfully overcoming the "feature aggregation collapse" caused by GTP-U tunneling.
* **Anti-Cheating Training Strategy**: Strictly removes identity features (Source/Destination IPs and Ports) during the preprocessing stage to prevent data leakage, forcing the model to learn genuine physical behavior patterns.
* **Traffic Tower (1D-ResNet)**: Utilizes a 1D Residual Network to process high-dimensional numerical statistical features.
* **Semantic Tower (DistilBERT)**: Employs a pre-trained LLM to parse expert Prompts automatically generated from traffic behaviors.
* **Cross-Attention Fusion**: Dynamically aligns heterogeneous data by using traffic features as the Query and semantic vectors as the Key/Value.
* **Academic-Grade Evaluation**: Integrates comprehensive Exploratory Data Analysis (Violin plots) and advanced model evaluation metrics (ROC curve, PR curve, Confusion Matrix).

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
