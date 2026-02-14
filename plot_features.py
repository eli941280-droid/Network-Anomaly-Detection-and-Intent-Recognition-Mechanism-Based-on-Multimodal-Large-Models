import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_feature_distributions():
    print("=" * 50)
    print("   多模态数据特征核密度可视化 (EDA)")
    print("=" * 50)

    data_path = "real_labeled_data.csv"
    if not os.path.exists(data_path):
        print(f"[错误] 找不到 {data_path}。请确保数据清洗脚本已成功运行！")
        return

    print("[Info] 正在加载训练数据集...")
    df = pd.read_csv(data_path)

    # 挑选 4 个最具代表性的流量特征展示差异
    features_to_plot = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Length of Fwd Packets',
        'Fwd Packet Length Max'
    ]

    # 针对网络特征进行 Log 变换，消除极值影响，使分布图平滑
    plot_df = df.copy()
    for col in features_to_plot:
        # np.log1p 会计算 log(1 + x) 避免 log(0)
        plot_df[col] = np.log1p(plot_df[col])

    # 设置绘图风格，体现严谨学术感
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    print("[Info] 正在绘制小提琴图 (Violin Plot)...")
    for i, feature in enumerate(features_to_plot):
        sns.violinplot(
            x='Label', y=feature, data=plot_df,
            ax=axes[i], palette={'BENIGN': '#1f77b4', 'DDoS': '#ff7f0e'},
            inner="quartile", alpha=0.8
        )
        axes[i].set_title(f'Log Distribution of {feature}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Log(Value + 1)')

    plt.tight_layout()
    plt.savefig('4_feature_violin_plots.png', dpi=300)
    print("[Success] 数据特征可视化完成！已保存: 4_feature_violin_plots.png")


if __name__ == "__main__":
    plot_feature_distributions()