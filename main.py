import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import Config
from model import AdvancedMultiModalNet
from dataset import MultiModalDataset
from train import train_one_epoch, evaluate_one_epoch, evaluate_model, plot_history


def main():
    print("=" * 60)
    print("   多模态意图识别训练 (Anti-Cheating Version)")
    print("=" * 60)

    # 1. 检查数据
    if not os.path.exists(Config.DATA_PATH):
        print(f"[错误] 找不到 {Config.DATA_PATH}")
        return

    # 2. 检查模型文件夹
    if not os.path.exists(Config.TEXT_MODEL):
        print(f"[错误] 找不到本地模型文件夹: {Config.TEXT_MODEL}")
        print("请确保你已经建立了 distilbert_local 文件夹并放入了5个文件。")
        return

    print(f"[Info] 设备: {Config.DEVICE}")

    try:
        # 3. 加载全量数据
        full_dataset = MultiModalDataset(Config.DATA_PATH, Config)

        # 80% 训练, 20% 验证
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(Config.SEED)
        )

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"[Info] 数据划分完成 | 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    except Exception as e:
        print(f"[Dataset Error] {e}")
        return

    # 4. 初始化模型与优化
    model = AdvancedMultiModalNet(Config).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=getattr(Config, 'WEIGHT_DECAY', 0.01)
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_state = None
    patience = getattr(Config, 'EARLY_STOP_PATIENCE', 8)
    patience_counter = 0

    print(f"\n>>> 开始训练 (最多 {Config.EPOCHS} Epochs, Early Stop patience={patience}) <<<")

    for epoch in range(Config.EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        v_loss, v_acc = evaluate_one_epoch(model, val_loader, criterion, Config.DEVICE)
        scheduler.step(v_loss)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        print(f"Epoch [{epoch + 1}/{Config.EPOCHS}] "
              f"Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% "
              f"(Loss: {t_loss:.4f} / {v_loss:.4f}) [best: {best_val_acc:.2f}%]")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 加载最佳模型参数
    if best_state is not None:
        model.load_state_dict(best_state)

    # 保存权重
    torch.save(model.state_dict(), "final_model.pth")
    print(f"\n[Success] 模型保存完毕。")

    # 5. 生成所有图表与最终评估报告
    print(">>> 正在绘制训练曲线图...")
    plot_history(history['train_acc'], history['val_acc'],
                 history['train_loss'], history['val_loss'])

    # 最终用验证集出报告和高级学术图表
    class_names = full_dataset.encoder.classes_
    class_map = {name: i for i, name in enumerate(class_names)}
    evaluate_model(model, val_loader, Config.DEVICE, class_map)

if __name__ == "__main__":
    main()