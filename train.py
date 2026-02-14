import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import torch.nn.functional as F


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 进度条只在训练时显示
    loop = tqdm(dataloader, leave=False, desc="Training")

    for batch in loop:
        traffic = batch['traffic'].to(device)
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(traffic, input_ids, mask)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_description(f"Train Loss: {loss.item():.4f}")

    return total_loss / len(dataloader), 100 * correct / total


def evaluate_one_epoch(model, dataloader, criterion, device):
    """验证集测试函数 (不反向传播)"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            traffic = batch['traffic'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(traffic, input_ids, mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), 100 * correct / total


def evaluate_model(model, dataloader, device, classes):
    """升级版评估函数：不仅输出报告，还生成多种高级学术图表"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # 用于存储预测概率(画 ROC 用)

    print("\n[Info] 正在生成最终评估报告与高级可视化图表...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            traffic = batch['traffic'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(traffic, input_ids, mask)
            _, predicted = torch.max(outputs.data, 1)

            # 计算预测为正类(DDoS, 索引为1)的概率
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 解析类别名称
    target_names = [str(k) for k in classes.keys()] if isinstance(classes, dict) else classes

    print("\n" + "=" * 30)
    print("       FINAL REPORT (TEST SET)")
    print("=" * 30)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    # 设置全局绘图风格
    sns.set_theme(style="whitegrid")

    # === 图表 1: 混淆矩阵 (美化版) ===
    try:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names,
                    cbar=False, annot_kws={"size": 14})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('1_confusion_matrix.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"混淆矩阵绘图失败: {e}")

    # === 图表 2: ROC 曲线 ===
    try:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='#d62728', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve for Intent Recognition', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.savefig('2_roc_curve.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"ROC曲线绘图失败: {e}")

    # === 图表 3: PR 曲线 ===
    try:
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='#2ca02c', lw=2, label='PR curve')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.tight_layout()
        plt.savefig('3_pr_curve.png', dpi=300)
        plt.close()
        print("[Info] 评估图表绘制完成！已保存：1_confusion_matrix.png, 2_roc_curve.png, 3_pr_curve.png")
    except Exception as e:
        print(f"PR曲线绘图失败: {e}")


def plot_history(train_acc, val_acc, train_loss, val_loss):
    """绘制训练vs验证曲线"""
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    # Loss 对比
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
    plt.plot(epochs, val_loss, 'g-s', label='Val Loss')
    plt.title('Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Accuracy 对比
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-o', label='Train Acc')
    plt.plot(epochs, val_acc, 'g-s', label='Val Acc')
    plt.title('Accuracy Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.close()
    print("[Info] 已保存训练曲线: training_history.png")