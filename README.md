# a2
包括了三个部分：分别为测带宽的服务器和客户端与控制器

## requirements: 
- python3.7 
- gurobi
- iperf3
- tf
- aiohttp
- gurobipy
- pillow
- tcconfig

## gurobi 下载地址
https://packages.gurobi.com/8.1/gurobi8.1.1_linux64.tar.gz

## 本地运行需要注意地方:
---
### 运行带宽测量模块：　
bw_server: 中需要改动cpu_server.txt　和gpu_server.txt 这部分的ｉｐ
bw_client: 中需要改动cpu_server.txt　和gpu_server.txt　这部分的ｉｐ

bw_server启动指令：
cpu：
- `python a2/bw_server/server_bw.py -p a2/bw_server/cpu_server.txt`

gpu：
-  `python a2/bw_server/server_bw.py -p a2/bw_server/gpu_server.txt`

bw_client启动指令：
cpu: 
- `python a2/bw_client/iperf_clinet.py -p a2/bw_client/cpu_server.txt -i 10 -t 1200`
gpu: 
- `python a2/bw_client/iperf_clinet.py -p a2/bw_client/gpu_server.txt -i 10 -t 1200`
i 是指间隔几秒钟 测量一次
t 是指持续多久 单位 s
---

### 运行控制器
- `python controller_main.py -a 0.0.0.0 -d cpu -t 60 -x a2 -debug aws -n 18`
  -n 18 是开了 3 个 client process 在每个 client machine
 -debug 是指在 aws 上运行
