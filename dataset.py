import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import RobustScaler  # 对异常值更鲁棒
from transformers import AutoTokenizer


class MultiModalDataset(Dataset):
    def __init__(self, csv_file, config):
        print(f"正在加载真实数据集: {csv_file} ...")
        self.df = pd.read_csv(csv_file)

        # 清洗列名空格
        self.df.columns = self.df.columns.str.strip()

        # 清洗无效数据
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)

        # 基础列
        base_columns = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min',
            'Bwd Packet Length Max', 'Bwd Packet Length Min'
        ]
        missing = [c for c in base_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"缺少列: {missing}")

        # === 特征工程: 衍生特征 (DDoS 检测关键: 前后向不对称、吞吐量) ===
        df = self.df[base_columns].copy()
        eps = 1e-6  # 避免除零
        df['Fwd_Bwd_Ratio'] = (df['Total Fwd Packets'] + eps) / (df['Total Backward Packets'] + eps)
        df['Fwd_Byte_Ratio'] = (df['Total Length of Fwd Packets'] + eps) / (df['Total Length of Bwd Packets'] + eps)
        df['Packets_Per_Sec'] = (df['Total Fwd Packets'] + df['Total Backward Packets']) / (df['Flow Duration'] / 1e6 + eps)
        df['Avg_Fwd_Pkt_Size'] = df['Total Length of Fwd Packets'] / (df['Total Fwd Packets'] + eps)
        df['Asymmetry'] = np.abs(df['Total Fwd Packets'] - df['Total Backward Packets']) / (df['Total Fwd Packets'] + df['Total Backward Packets'] + eps)

        self.selected_columns = list(df.columns)
        self.feature_names = self.selected_columns
        raw_features = df.values
        self.raw_features = raw_features  # 保留原始值用于生成可读 prompt

        # RobustScaler 对 Flow Duration 等偏态/异常值更稳健
        self.scaler = RobustScaler()
        self.features = self.scaler.fit_transform(raw_features)

        # 处理标签
        label_col = config.LABEL_COLUMN.strip()
        if label_col not in self.df.columns:
            # 尝试自动查找 Label 列
            candidates = [c for c in self.df.columns if 'Label' in c]
            if candidates: label_col = candidates[0]

        self.encoder = LabelEncoder()
        self.labels = self.encoder.fit_transform(self.df[label_col].astype(str).values)
        print(f"标签映射: {dict(zip(self.encoder.classes_, range(len(self.encoder.classes_))))}")

        # 初始化 Tokenizer (从本地加载)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
        self.max_len = 32

    def __len__(self):
        return len(self.df)

    def generate_expert_prompt(self, raw_row):
        """使用原始特征值生成更丰富的专家 prompt，便于语言模型理解"""
        duration_us = raw_row[0]
        fwd_pkts = raw_row[1]
        bwd_pkts = raw_row[2]
        fwd_ratio = raw_row[9] if len(raw_row) > 9 else (fwd_pkts + 1) / (bwd_pkts + 1)
        asymmetry = raw_row[13] if len(raw_row) > 13 else 0.5
        # 简化为可读描述
        flow_type = "highly asymmetric" if fwd_ratio > 10 else "balanced" if fwd_ratio < 2 else "moderate asymmetry"
        prompt = (
            f"Traffic analysis: duration {duration_us/1e6:.2f}s, "
            f"forward packets {fwd_pkts:.0f}, backward {bwd_pkts:.0f}. "
            f"Flow is {flow_type} with fwd/bwd ratio {min(fwd_ratio, 999):.1f}. "
            f"Detect malicious intent."
        )
        return prompt

    def __getitem__(self, idx):
        traffic_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
        text = self.generate_expert_prompt(self.raw_features[idx])

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'traffic': traffic_tensor,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }