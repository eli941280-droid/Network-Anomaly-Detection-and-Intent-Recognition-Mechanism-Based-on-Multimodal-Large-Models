import pandas as pd


def label_and_clean():
    print("=" * 65)
    print("   5G 核心网真实流量标注与清洗工具 (Data Preprocessor)")
    print("=" * 65)

    # 1. 加载最新的真实数据
    file_name = "upf_gtpu5.pcap_Flow.csv"
    print(f"[*] 正在加载原始数据: {file_name}")
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"[错误] 找不到 {file_name}，请确保它在当前文件夹下！")
        return

    # 清理列名首尾可能自带的恶心空格
    df.columns = df.columns.str.strip()

    # 2. 核心逻辑：根据之前多态打流脚本的物理行为，进行精准的“逆向标注”
    def assign_label(row):
        # 1. UDP Flood (DDoS): 脚本打的是目标端口 > 10000 的高频 UDP (Proto 17)
        if row['Protocol'] == 17 and row['Dst Port'] > 10000:
            return 'DDoS'

        # 2. Port Scan (DDoS): 扫的端口通常 < 1024，绝大多数是极短的 TCP 握手失败
        if row['Protocol'] == 6 and row['Dst Port'] < 1024 and row['Dst Port'] not in [80, 443] and row[
            'Flow Duration'] < 1000000:
            return 'DDoS'

        # 3. HTTP GET Flood (DDoS): 打的是 80 端口，但由于目标(如8.8.8.8)会拒绝，导致前向包极小
        if row['Protocol'] == 6 and row['Dst Port'] == 80 and row['Total Length of Fwd Packet'] <= 100:
            return 'DDoS'

        # 剩下的（正常 Web、大文件下载、DNS解析等）全部归为正常流量
        return 'BENIGN'

    print("[*] 正在基于网络物理行为进行逆向真值标注 (Ground Truth Labeling)...")
    df['Label'] = df.apply(assign_label, axis=1)

    # 打印最终的标签分布
    counts = df['Label'].value_counts()
    print(f"[+] 标注完成！正常流量 (BENIGN): {counts.get('BENIGN', 0)} 条 | 攻击流量 (DDoS): {counts.get('DDoS', 0)} 条")

    # 3. 字段映射字典 (完美解决 CICFlowMeter 和你 dataset.py 的列名分歧)
    rename_map = {
        'Flow Duration': 'Flow Duration',
        'Total Fwd Packet': 'Total Fwd Packets',  # 修复复数后缀
        'Total Bwd packets': 'Total Backward Packets',  # 修复拼写
        'Total Length of Fwd Packet': 'Total Length of Fwd Packets',
        'Total Length of Bwd Packet': 'Total Length of Bwd Packets',
        'Fwd Packet Length Max': 'Fwd Packet Length Max',
        'Fwd Packet Length Min': 'Fwd Packet Length Min',
        'Bwd Packet Length Max': 'Bwd Packet Length Max',
        'Bwd Packet Length Min': 'Bwd Packet Length Min',
        'Label': 'Label'
    }

    # 提取需要的列并重命名 (IP 和 Port 在这一步被彻底丢弃，贯彻 Anti-Cheating 思想！)
    df_clean = df[list(rename_map.keys())].rename(columns=rename_map)

    # 4. 保存为供大模型训练的标准数据集
    output_file = "real_labeled_data.csv"
    df_clean.to_csv(output_file, index=False)
    print(f"\n[Success] 数据清洗与标注完毕！已生成标准训练集: {output_file}")
    print("=" * 65)


if __name__ == "__main__":
    label_and_clean()