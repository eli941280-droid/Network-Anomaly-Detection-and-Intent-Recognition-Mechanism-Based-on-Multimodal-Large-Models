import socket
import urllib.request
import threading
import random
import time
import os

print("=" * 60)
print("   5G 核心网全模态流量生成器 (Pro版 - 生成海量混合流)")
print("=" * 60)

# 测试目标 (建议替换为你们实验室允许扫描/访问的内外网IP)
TARGET_IP = "8.8.8.8"
URLS = [
    "http://www.baidu.com", "http://www.qq.com",
    "http://www.163.com", "http://www.bing.com"
]


def normal_web_browsing():
    """1. 模拟日常网页浏览 (中短 TCP 流)"""
    print("[+] 启动: Web 浏览模拟器 (TCP)")
    for _ in range(150):
        try:
            req = urllib.request.Request(random.choice(URLS), headers={'User-Agent': 'Mozilla/5.0'})
            urllib.request.urlopen(req, timeout=2).read(1024)  # 只读前1KB
        except:
            pass
        time.sleep(random.uniform(0.1, 0.5))


def heavy_file_download():
    """2. 模拟大文件/视频下载 (超长 TCP 流，包体积巨大)"""
    print("[+] 启动: 大文件下载模拟器 (Heavy TCP)")
    # 使用公共的测速文件链接
    test_file_url = "http://ipv4.download.thinkbroadband.com/5MB.zip"
    try:
        # 下载5MB的文件，会产生极其明显的长持续时间、大载荷特征
        urllib.request.urlopen(test_file_url, timeout=15).read()
    except:
        pass
    print("[-] 大文件下载完成")


def dns_query_burst():
    """3. 模拟高频 DNS 解析 (海量极短 UDP 流)"""
    print("[+] 启动: DNS 查询模拟器 (Short UDP)")
    domains = ["google.com", "github.com", "bilibili.com", "zhihu.com", "apple.com"]
    for _ in range(300):
        try:
            socket.gethostbyname(random.choice(domains))
        except:
            pass
        time.sleep(0.05)


def tcp_port_scan():
    """4. 模拟黑客端口扫描 (海量失败的 TCP 握手流)"""
    print("[+] 启动: Nmap 端口扫描模拟器 (TCP Port Scan)")
    # 扫描 1~1024 端口，会产生 1024 条不同的 TCP Flow (即使被拒绝也会记录)
    for port in range(1, 1025):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.05)
            s.connect((TARGET_IP, port))
            s.close()
        except:
            pass


def udp_flood_attack():
    """5. 模拟 UDP 泛洪攻击 (海量并发的大体积 UDP 流)"""
    print("[+] 启动: UDP Flood 攻击模拟器 (DDoS)")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 打 2000 个不同的随机端口，CICFlowMeter 会将其认作 2000 条不同的流
    for _ in range(2000):
        try:
            target_port = random.randint(10000, 65535)
            # 连续发送 10 个 512 Bytes 的垃圾包
            for _ in range(10):
                sock.sendto(os.urandom(512), (TARGET_IP, target_port))
        except:
            pass


def http_get_flood():
    """6. 模拟应用层 HTTP CC 攻击 (高频 TCP 交互流)"""
    print("[+] 启动: HTTP GET Flood 模拟器 (App-layer DDoS)")
    for _ in range(300):
        try:
            # 不断与 80 端口建立连接并发送无意义的 GET 请求
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            s.connect((TARGET_IP, 80))
            s.send(b"GET / HTTP/1.1\r\nHost: target\r\n\r\n")
            s.close()
        except:
            pass


if __name__ == "__main__":
    print(">>> 流量生成引擎启动！请确保已在后台开启 tcpdump 抓包！")

    # 将所有行为放入线程池并发执行，模拟真实的混乱网络环境
    threads = [
        threading.Thread(target=normal_web_browsing),
        threading.Thread(target=heavy_file_download),
        threading.Thread(target=dns_query_burst),
        threading.Thread(target=tcp_port_scan),
        threading.Thread(target=udp_flood_attack),
        threading.Thread(target=http_get_flood)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(">>> 所有模态流量注入完毕！可以按 Ctrl+C 停止 tcpdump 抓包了。")