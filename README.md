# 3× DGX Spark 部署 DeepSeek-V4.1-Flash（TP=3 跨三机）· 可用方案

三台 DGX Spark（GB10）用 CX7 直连成**三角**拓扑，跑起 DeepSeek-V4.1-Flash 推理服务。
**结论先行**：这套三角拓扑下 NCCL 的 IB/RoCE 路径不可用（上游已受理 issue），
**可用方案是让 NCCL 走 socket/TCP over 同一批 CX7 链路**，实测单流 **18–19 tok/s**、
4 并发聚合 **约 40 tok/s**。

> 本文档描述的是**已经跑通并验证过**的配置，包含全部踩坑记录与九条 RoCE 路线的实测结论。

---

## 1. 这套方案长什么样

```
        node1 (head)              node2                    node3
        GPU: GB10                 GPU: GB10                GPU: GB10
        ┌──────────┐   W2   ┌──────────┐   W3   ┌──────────┐
        │ p1 ──────┼────────┼─ p0      │        │          │
        │ p0 ──────┼────────┼──────────┼────────┼─ p0      │
        └──────────┘   W1   └────┬─────┘        └────┬─────┘
                                 └────── p1 ──────────┘
                          W1: node1-p0 ↔ node3-p0
                          W2: node1-p1 ↔ node2-p0
                          W3: node2-p1 ↔ node3-p1
```

- **只有 3 根线**，构成一个三角（不是 rail，也没有交换机）。
- 每根线在**两张 PCI 卡上各有一个端口**（同一根线的两个 24 位子网视图），因此每台机器有
  **4 个 RDMA 设备 / 4 个 fabric 地址**，但物理上只有 2 根线。
- 这带来一个关键后果：**一个 rank 的两个邻居分属不同网卡**，而 NCCL 的 IB 传输层默认假设
  "按设备索引配对、两端同轨"。这个假设在三角里不成立 → 见 §4。

## 2. 可行的传输配置（核心）

数据面走 **socket/TCP**，但**链路上仍然是那几条 CX7 直连线**：

```bash
# 传输选择
NCCL_NET=Socket
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=enp1s0f0np0     # 数据面网卡（各机同名）
GLOO_SOCKET_IFNAME=wlP9s9          # bootstrap 走管理/WiFi 网卡，避免和数据面抢

# 其余保持配方默认
NCCL_MAX_NCHANNELS=8
NCCL_BUFFSIZE=1048576
NCCL_PROTO='^LL128'
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
```

链路侧必须做的两件事（`scripts/fabric-mtu-route.sh`，已用 netplan 持久化，见 `assets/`）：

1. **所有 fabric 口 MTU 9000**（含两张卡的 4 个口）；
2. **为每个对端加 `/32` 直连路由**，让 TCP 直接走直连线而不是绕管理网。

> ⚠️ 临时 `ip route add` 会被 NetworkManager 清掉（表现为引擎卡死在不可达路由上）。**必须写进 netplan**（示例见 `assets/netplan-99-cluster.yaml`）。

## 3. 实测性能

| 指标 | 数值 |
|---|---|
| 单流解码 | **17.9 – 19.2 tok/s** |
| 4 并发聚合 | **39.7 – 43.3 tok/s** |
| fabric 单向带宽 | 2.07 – 2.14 GB/s（MTU 9000 + 直连路由；MTU 1500 时约 1.75 GB/s） |
| 4 并发时的实际占用 | 每链路约 225 MB/s，**仅占理论上限 32%** |
| 模型服务 | `max_model_len=262144`、`max_total_num_tokens=499968` |

**为什么是 socket 而不是 RoCE**：实测瓶颈不在带宽（只用掉 32%），而在**每一步约 104 次集合通信的
延迟**（socket 约 1 ms/次，RoCE 50–100 µs）。所以"换成 RoCE"是唯一能显著提速的方向 —— 但在这套
拓扑上走不通，原因见下节。

## 4. RoCE 为什么走不通（结论 + 证据）

我们在**九条路线**上做了单变量实测，全部失败，且都收敛到同一个 NCCL 内部错误：

```
transport/net_ib/p2p.cc:703 (ncclIbCompletionEventProcess)
  NCCL WARN NET/IB: Recv comm could not retreive a request found for a successful completion
ncclInternalError: Internal check failed
```

| # | 路线 | 结果 |
|---|---|---|
| 1–2 | 直连三角的默认配置 / bootstrap 平面分离 | `ibv_modify_qp 110` |
| 3 | `NCCL_CROSS_NIC=1` | 110 |
| 4 | `NCCL_IB_MERGE_NICS=1` | QP 建起来、数据通过，但撞 `p2p.cc` 内部错误 |
| 5 | 自研补丁 NCCL（按 `NCCL_IB_HCA` 顺序注册设备）+ 按线缆图排 per-rank HCA 顺序 | **110 消失**，仍撞 `p2p.cc` |
| 6 | 上游 AICAD ring-only 补丁栈（v1+v4+stageB，逐对端设备映射） | **110 消失**、`Tree=0/Ring=1` 生效，仍撞 `p2p.cc` |
| 7 | 确保容器内**只有一套 NCCL**（覆盖镜像自带路径）+ 6 项变量矩阵 | 仍撞 `p2p.cc` |
| 8 | 每机 2 口（一根线一个口）/ 自定义 `NCCL_TOPO_FILE` | 挂住 / 仍撞 `p2p.cc` |
| 9 | 复刻上游生产组合（驱动 580.173.02 + 内核 6.17.0-1031） | 仍撞 `p2p.cc` |

完整记录见 [`docs/ROCE-INVESTIGATION.md`](docs/ROCE-INVESTIGATION.md)，已向上游提交：
**luxingcom/aicad-nccl-optimization#1**（[链接](https://github.com/luxingcom/aicad-nccl-optimization/issues/1)）。

**结论**：NCCL 的 IB 传输层无法表达"一个 rank 的两个邻居分属两张卡"的三角形形状。
出路只有两条：**加一台 RoCE 交换机**（恢复标准 rail，预期 13–17 GB/s），或**等上游修**。在此之前
socket/TCP 是最优可用解。

## 5. 快速上手

```bash
# 1) 链路：MTU 9000 + /32 直连路由（三台都执行，或用 assets/ 里的 netplan 持久化）
sudo bash scripts/fabric-mtu-route.sh

# 2) 用 socket 传输启动（在 head 上）
cp env.example .env            # 按需修改 IP/HCA/内存水位
./start.sh share               # 每次重启后必须先 share，否则 worker 挂载检查会卡住
./start.sh serve

# 3) 验证
curl -s http://127.0.0.1:8888/health          # 期望 200
curl -s http://127.0.0.1:8888/v1/models       # 期望看到 deepseek-v4.1-flash
python scripts/bench_decode.py --url http://127.0.0.1:8888
```

**运维铁律**（完整 14 条见 [`docs/PITFALLS.md`](docs/PITFALLS.md)）：

1. 重启后**先 `./start.sh share` 再 `serve`**（否则 worker 的 NFS 检查要卡十几分钟）；
2. NFSv4 `fsid=0` 导出时，客户端必须挂**伪根 `:/`**，不是子目录；
3. **绝不要 `docker ps -q | xargs docker rm -f`** —— 引擎容器也在列表里；
4. `MEM_FRACTION_STATIC` 用配方给的 0.95（低于 0.944 会出现 "weights leave no GPU memory for the KV cache"）。

## 6. 仓库结构

```
docs/DEPLOY-GUIDE.md        完整部署手册（正确流程 + 14 条踩坑）
docs/PITFALLS.md            避坑清单（按现象索引）
docs/ROCE-INVESTIGATION.md  九条 RoCE 路线的完整实测记录与证据
docs/UPSTREAM-ISSUE.md      提交给上游的 issue 全文
scripts/                    可直接复用的脚本（链路、探针、基准、内核切换、NCCL 补丁应用）
assets/                     netplan 示例、自定义 NCCL 拓扑文件
env.example                 环境变量样例（已脱敏）
```

## 7. 硬件/软件基线（实测环境）

| 项 | 值 |
|---|---|
| 节点 | 3 × DGX Spark（GB10），每机 1 颗 GPU |
| 内核 / 驱动 | `7.0.0-1019-nvidia` / `580.178.04` |
| NCCL | 官方 2.30.7（socket 路径不依赖补丁；IB 路线见 §4） |
| 容器 | 厂商 sglang 镜像 + 官方配方脚本 |

> 注：DGX 上**驱动与内核是配对的**（如 `580.173.02 ↔ 6.17.0-1031`、`580.178.04 ↔ 7.0.0-1019`），
> 换内核必须同时换驱动；相关脚本与坑见 `scripts/kernel-driver-*.sh` 与 `docs/PITFALLS.md`。

## 8. 已知限制

- RoCE 不可用（见 §4），因此单流吞吐受 socket 延迟限制，约为 RoCE 方案预期值的 1/2 ~ 1/3；
- SPS（投机解码的吞吐表）在当前构建下无法生效（三处 shape 不一致，详见 `docs/ROCE-INVESTIGATION.md` 附注）；
- 单流解码延迟有波动（0.9 s ↔ 18 s），已排除 GPU 降频（三台 2.1–2.3 GHz、节流位 0x0）与带宽瓶颈，根因未定位。

## 9. 许可与致谢

- 本仓库文档与脚本：MIT（见 `LICENSE`）。
- 不包含任何厂商源码或镜像内容；引用的第三方补丁请遵循其各自仓库的许可。
- 感谢 `luxingcom/aicad-nccl-optimization` 与 LuZ 生产栈作者公开 ring-only 补丁与构建记录，
  它们把问题从"QP 建不起来"推进到了"连接建立后死在完成队列"，为定位提供了关键对照。
