import torch


class Config:
    # === 修改 1: 指向刚刚生成的新文件 ===
    DATA_PATH = "real_labeled_data.csv"

    # === 修改 2: 标签列名改为纯 "Label" (去掉了之前的空格) ===
    LABEL_COLUMN = "Label"

    # 训练配置
    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    EARLY_STOP_PATIENCE = 8

    # 特征维度 (9 基础 + 5 衍生 = 14)
    TRAFFIC_INPUT_DIM = 14
    NUM_CLASSES = 2

    # 模型配置
    UNFREEZE_BERT_LAYERS = 2   # 解冻 BERT 最后 N 层以微调
    TRAFFIC_HIDDEN_DIMS = [64, 128, 256]  # 流量塔加深
    FUSION_EMBED_DIM = 256
    FUSION_HEADS = 8
    DROPOUT = 0.4

    # 文本预训练模型本地路径
    TEXT_MODEL = "./distilbert_local"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42