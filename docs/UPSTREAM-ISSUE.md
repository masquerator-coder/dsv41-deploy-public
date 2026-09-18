> # 🔚 结案更正（2026-09-19）：不是 NCCL 的缺陷，我们自己的环境有三处问题
>
> 以下 issue 正文描述的"第二个 communicator 失败 / `p2p.cc` 内部错误"，最终定位**不在 NCCL**，
> 而是我们现场的三处问题，全部修正后同一套三角拓扑上 RoCE 一次通过：
>
> 1. **接线错误**：有两根缆接在同一端口索引上（`p0↔p0`、`p1↔p1`），NCCL 按设备索引跨 rank 配对时
>    必然配到"不在同一根缆上"的两个口；改成**每条缆 p0↔p1**（交叉环）即解。
> 2. **内核 CMA 回归**：DGX OS `7.0.0-1019-nvidia` 上 `ibv_reg_mr_iova2` 必失败
>    （`grep CmaTotal /proc/meminfo` = `0 kB`，正常内核 `131072 kB`）；换 `6.17.0-1031-nvidia`
>    + 驱动 `580.173.02` 即解。
> 3. **容器镜像 `/etc/nccl.conf` 残留**：`NCCL_IB_USE_INLINE=1` + `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1`
>    会让 SGLang 的 PyNccl 4 字节 warmup 永久冻死；置 0 即解。
>
> 结果：fabric `all_reduce` 256 MB **13.86 GB/s**（32/32 信道 `via NET/IB`），
> 引擎单流 **28.5–35.4 tok/s**、4 并发聚合 **67.5–75.8 tok/s**。
> 感谢贵仓库公开的 ring-only 补丁与构建记录——它们在错误配置阶段提供了关键对照，
> 也**不需要**了（当前官方 2.30.7 原样即可跑通）。下方原文保留作为排查过程记录。

## 背景

我们按 `patches/README.md` 的指引，把 **`v1-ring-only.patch` + `v4-netdev-hardcode.patch` + `stageB-tuner-two-band.patch` + `stageB-hardened-two-branch.patch`** 应用到官方 NCCL 2.30.7（`v2.30.7-1`）源码上，编译产物三机 md5 一致（`cd73d299887540559838202807e07b01`），并把补丁库用 **bind mount 覆盖镜像自带路径**（保证进程里只有一套 libnccl）。

补丁**确实生效**（下方日志可证），`ibv_modify_qp 110` 也被修好了 —— 但 all_reduce 仍无法完成。我们希望确认：**3 节点三角拓扑是否在你们的支持范围内**，以及我们是否漏了某个约束。

## 环境

| 项 | 值 |
|---|---|
| 节点 | 3 × DGX Spark（GB10），每机 1 颗 GPU，TP=3 跨三机 |
| 接线 | 仅 3 根 CX7 直连线构成**三角**：`01-p0↔03-p0`、`01-p1↔02-p0`、`02-p1↔03-p1`。每根线在两张 PCI 卡上各有一个端口（我们这里表现为同一根线的"双视图"，共 **4 口/机**） |
| 宿主 | 驱动 `580.178.04` / 内核 `7.0.0-1019-nvidia` |
| 容器 | sglang 镜像（torch 自带 NCCL 2.29.7，已被覆盖） |
| NCCL | 官方 2.30.7 + 上述 4 个补丁；另用**未打补丁的 2.30.7**做对照 |

> 补充：我们也把宿主**完整切换到你们 `BUILD-IDENTITY.md` 记录的生产组合**（驱动 `580.173.02` + 内核 `6.17.0-1031-nvidia`）复测过 —— **现象完全相同**，因此可以排除内核/驱动差异。

## 最小复现

3 节点各 1 rank，`torch.distributed` + `all_reduce`（1 MB / 16 MB / 256 MB 三档均试）。通信面与你们的配置对齐：

```
NCCL_NET=IB  NCCL_IB_DISABLE=0  NCCL_IB_HCA=<每机按接线顺序的 4 口>
NCCL_IB_GID_INDEX=3  NCCL_IB_MERGE_NICS=0  NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_NET_PLUGIN=none  NCCL_CROSS_NIC=1  NCCL_ALGO=RING
NCCL_P2P_DISABLE=1  NCCL_SHM_DISABLE=1     # 与 .env.example 一致
bootstrap 走管理网卡（GLOO_SOCKET_IFNAME / NCCL_SOCKET_IFNAME）
```

## 现象

**✅ 补丁生效的证据**

```
NCCL INFO RING-ONLY v4 rank 0->2 chan 0 dev 1 (was 0)      # rank0 共 32 条映射行
NCCL INFO Using network IB
NCCL INFO Rank 0/1/2: 4 Net devices
AllReduce 算法矩阵: Tree=0  Ring=1                          # NCCL_ALGO=RING 生效
```

**✅ 相对未打补丁时的明确改善**：`ibv_modify_qp ... 110 (Connection refused)` **消失**（QP 能建起来了）。

**❌ 但 all_reduce 在连接建立后失败**（1 MB 与 256 MB 相同）：

```
transport/net_ib/p2p.cc:703 (ncclIbCompletionEventProcess)
  NCCL WARN NET/IB: Recv comm could not retreive a request found for a successful completion
ncclInternalError: Internal check failed
```

## 已排除的变量（每条都是单变量实测）

| 变量 | 结果 |
|---|---|
| `NCCL_NET_GDR_LEVEL=0`（宿主未加载 `nvidia-peermem`） | 同一错误（有时表现为 hang） |
| `NCCL_PROTO=Simple` | 同一错误 |
| `NCCL_IB_SPLIT_DATA_ON_QPS=0` | 同一错误 |
| `NCCL_IB_ADDR_FAMILY=AF_INET6` | 同一错误 |
| `NCCL_MAX_NCHANNELS=4` + `NCCL_BUFFSIZE=8388608`（你们的 `.env.tp4.example` 值） | 同一错误 |
| `NCCL_IB_MERGE_NICS=1` | QP 通、数据过（`Rank 0: 6 Net devices`），但同一 `p2p.cc` 错误 |
| bootstrap 平面换到管理网卡 | 同一错误 |
| 每机 **2 口**（一根线一个口，对应 `.env.example` 的 3 机档案） | **hang**（`init_process_group` 不返回） |
| 每机 4 口 + `(myRank,peerRank)` 逐对端映射 | 同一错误 |
| 自定义 `NCCL_TOPO_FILE`（GPU 与四卡声明为同桥兄弟，全 PIX） | 同一错误 |
| 内核+驱动切到你们的生产组合 | 同一错误 |

## 我们的两点观察（供参考，也可能是我们理解有误）

1. **4 口双视图的歧义**：我们的三角里，同一根线在两张卡上各有一个端口。接收侧看到同一 peer 从**同一根线**以两个不同 GID 到达时，`(connIndex, devIndex)` 与 QP 的对应关系可能出现歧义 —— 这与"`v4` 修好了配对（110 消失）、但完成队列仍失配（`could not retreive a request`）"的现象吻合。
2. **2 口方案必然 hang**：3 机环要求每个 rank 的两个邻居各走不同网卡，而 NCCL 的"按 `channelId` 取设备索引"模型在每机只有 2 口时，必然让其中一条连接落在不可达的线上，于是初始化挂住。

## 想请教

1. 你们的 ring-only 栈在 **3 机三角**上是否有已验证配置？`.env.example` 里的 3-Spark 档案所假设的接线，是否与我们这套（NVIDIA Sync 默认三角 + 同线双端口）一致？
2. `v4` 的 `ncclRingDevOverride(myRank, peerRank, channelId, …)` 按 `channelId % 2` 在 `{devA, devB}` 之间轮换：在 3 机场景下是否还需要额外约束（例如**同一根线的两个视图不应同时出现在同一 peer 的候选集中**）？
3. 如果方便，能否给一个"3 机下推荐的口数/接口顺序/HCA 列表"的最小示例？

如需完整材料（`NCCL_DEBUG=INFO` 全量日志、九条路对照表、编译与部署脚本、三机拓扑与 GID 表），我可以直接贴出来。

感谢你们公开补丁与文档 —— 它把我们从"QP 建不起来"推进到了"连接建立后死在完成队列"，这是明确的一步。
