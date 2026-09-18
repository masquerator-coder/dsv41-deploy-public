# RoCE 定位结论（2026-09-18，三节点 CX7 三角）

## 一句话
按"4 个口 + MERGE_NICS=0 + SUBNET_AWARE=1 + 不设 CROSS_NIC"配置后，**NCCL 确实开始走 IB 了**（之前一直静默降级到 Socket，元凶是我们 .env 里的 `NCCL_IB_MERGE_NICS=1`），
但 **QP 仍然建不起来（`ibv_modify_qp 110`）**，原因是结构性的：NCCL 按**设备索引**跨 rank 对齐配对，而三台的索引 0 都是"最小 PCI"那张卡，彼此不在同一根线上。

## 证据 1：IB 真的被选中了（对比之前）
```
NCCL_IB_MERGE_NICS set by environment to 0.
NET/IB : Using [0]rocep1s0f0:1/RoCE [1]rocep1s0f1:1/RoCE [2]roceP2p1s0f0:1/RoCE [3]roceP2p1s0f1:1/RoCE [RO]; OOB enp1s0f0np0:10.100.178.2<0>
Assigned NET plugin IB to comm          ← 之前是 Assigned NET plugin Socket
Using network IB                         ← 之前是 Using network Socket
Rank 0: 4 Net devices                    ← 之前是 1 Net devices
Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0 … 03 via NET/IB/3（4 个口轮转）
```
关键：`.env` 的 `NCCL_IB_MERGE_NICS=1` 会把"每张 PCI 卡的两个口"合成虚拟设备，导致图搜索里没有任何 IB 设备可用 → **静默退回 Socket，带宽 2.10 GB/s（= TCP，无报错）**。这就是之前"跑了但很慢"的真相。

## 证据 2：110 的确切错配
```
rank0 (head):  dev rocep1s0f1, local GID 10.100.180.2  →  remote GID 10.100.176.1
rank2 (node3): dev rocep1s0f0, local GID 10.100.178.1  →  remote GID 10.100.180.1
```
两边用的都是**自己的索引 1 / 索引 0**，而对端 GID 落在别人的线上。

## 证据 3：地址/线缆表（每台 4 口 = 2 根线 × 2 视图）
| 节点 | enp1s0f0np0 | enp1s0f1np1 | enP2p1s0f0np0 | enP2p1s0f1np1 |
|---|---|---|---|---|
| 01 (head) | 178.2 | 180.2 | 179.2 | 181.2 |
| 02 | 180.1 | 176.2 | 181.1 | 177.2 |
| 03 | 178.1 | 176.1 | 179.1 | 177.1 |

线缆（按共用 /24 判断）：**178/179 = 01-p0↔03-p0｜180/181 = 01-p1↔02-p0｜176/177 = 02-p1↔03-p1**。
每根线两个视图（同 `phys_switch_id`、同物理口），因此每条线的两端各有一个"另一个 /24"。

## 证据 4：为什么 SUBNET_AWARE 也救不了
源码 `net_ib/connect.cc` 的兜底要求"至少一端 index-0 卡已经在那条线上"（`matched == checked`）。
三角里每台的 index 0 = 最小 PCI = 每条线各占一头，不构成闭环 → 无解。
另外环形通讯要求**同一个 rank 用同一个网卡**同时服务左右两个邻居（日志：`0[0] -> 1[0] [send] via NET/IB/0` 与 `2[0] -> 0[0] [receive] via NET/IB/0`），
而"一个口 = 一根线"在三角里做不到 → 这才是 `NCCL_IB_MERGE_NICS=1` 原本想解决的问题（把同一 PCI 卡的两个口合成"能同时够到两个邻居"的设备）。
配方 README 自己也写了：4 节点的场景需要交换机。

## ⛔ 两条路都已验证走不通（2026-09-18 12:45 收尾）

| 配置 | QP | 结果 |
|---|---|---|
| 4 口 + `MERGE_NICS=0`（用户指定） | ❌ `ibv_modify_qp 110` | `local GID 180.2 (head p1) → remote GID 176.1`，索引对齐把两个不在同一根线上的口配到一起 |
| 4 口 + `MERGE_NICS=1` | ✅ **QP 建立成功、数据通** | 但 NCCL 内部记账崩：`p2p.cc:705 Recv comm could not retreive a request found for a successful completion` → `ncclInternalError: Internal check failed` |

`MERGE_NICS=1` 生成的是 NCCL 的**多平面设备**（`Rank 0: 6 Net devices` = 4 物理口 + 2 个按 PCI 卡合并的虚拟设备，信道映射为 `via NET/IB/4`、`NET/IB/5`，收发同设备 —— 形状正是三角所需）。QP 能建起来证明**合并设备在 RDMA 层是可达的**，但 NCCL 假定同一平面组的各口通向**同一对 rank**，而三角里一台的两个口通向两个不同邻居 → 收包记账崩。

→ **结论：NCCL 2.30.7 在"3 节点 × 每节点 2 口三角直连"上无法跑 RoCE**。可行形状只有两种，都被排除：
1. 每口一轨（rail 对齐）→ 三角里无解（数学已证 + 实测 110）；
2. 按卡合并为多平面 → 违反 NCCL 平面语义（实测 internal error）。
终局方案：**加 RoCE 交换机**（变成星形/标准 rail，NCCL 的标准形状），或继续走 socket/TCP over CX7（2.1 GB/s，已验证可用）。

## ★ 终局：五条路全部走不通，根因在 NCCL 自身（2026-09-18 15:48）

| # | 配置 | 结果 |
|---|---|---|
| 1 | 社区配方（4 口 + `SUBNET_AWARE=1` + `MERGE_NICS=0` + `NET_PLUGIN=none`，不设 `CROSS_NIC`） | ❌ 110（`local 180.2 → remote 176.1`） |
| 2 | ＋ bootstrap 挪到管理网 | ❌ 110 |
| 3 | ＋ `NCCL_CROSS_NIC=1` | ❌ 110 |
| 4 | `MERGE_NICS=1` | ⚠️ QP 通、数据动，但 `p2p.cc:705 could not retreive a request…` → `ncclInternalError` |
| 5 | 补丁 NCCL（按 HCA 顺序注册）+ 按线缆图排 per-rank HCA 顺序 | ⚠️ **110 消失、QP 建立成功**，但 `p2p.cc:703` 同一内部错误 |

第 5 条的闭环分配（已实测三台生效）：
```
rank0 (head) : rocep1s0f1,rocep1s0f0   → idx0 = p1(180.2)→W2→node2 ；idx1 = p0(178.2)→W1→node3
rank1 (node2): rocep1s0f1,rocep1s0f0   → idx0 = p1(176.2)→W3→node3 ；idx1 = p0(180.1)→W2→head
rank2 (node3): rocep1s0f0,rocep1s0f1   → idx0 = p0(178.1)→W1→head ；idx1 = p1(176.1)→W3→node2
```
执行要点（两个坑）：① 换 NCCL 构建必须用 **`LD_PRELOAD`**（`LD_LIBRARY_PATH` 对 torch 自带的 NCCL 无效，实测 `from /nccl` 映射为 0）；② 补丁 `.so` 必须**三台都放**，否则 worker 端静默回退。
`smoke-final/`、`smoke-preload/`、`smoke-patched2/` 等目录留有全部日志。

**结论**：三角形中一个 rank 的上下邻居挂在不同网卡上，违反 NCCL IB P2P 的"同轨/同卡"假设；两种独立配置都在 QP 建立成功后倒在同一个 `ncclInternalError` → **NCCL 对该拓扑的限制**，非配置问题。出路只有 RoCE 交换机或维持 socket。

## ★ 第六条路：AICAD 上游补丁栈（2026-09-18 16:40，已编译并实测）

来源：[`luxingcom/aicad-nccl-optimization`](https://github.com/luxingcom/aicad-nccl-optimization)（"DGX Spark TP4 vLLM NCCL 2.30.7 ring-only 优化资料包"，allreduce -81%），
配套生产栈：[`luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring`](https://github.com/luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring)。

**补丁栈**（全部文本 diff，适用 NCCL 2.30.7，与我们源码树同版本）：
| 补丁 | 作用 |
|---|---|
| `v1-ring-only.patch` | `transport.cc`：`ncclTransportP2pConnect` 跳过非环邻的跨节点 peer（3 机时是 no-op，因两个 peer 都是环邻） |
| `v4-netdev-hardcode.patch` | `transport/net.cc`：`ncclRingDevOverride()` 静态 `(myRank,peerRank)→{偶通道设备, 奇通道设备}` 表，设备号 = `NCCL_IB_HCA` 顺序 |
| `stageB-tuner-two-band.patch` / `stageB-hardened-two-branch.patch` | `enqueue.cc`：按消息大小选协议（≤40KB→LL，>40KB→Simple，仅 allreduce）+ 加固 |

**我们的做法**：源码树在 v2.30.7-1 ✓ → 回滚我的旧改动 → 应用 4 个补丁 → **把 v4 的表改写成三角版**（每台 4 口按 [线A视图1, 线B视图1, 线A视图2, 线B视图2] 排序；`rank0/1: peer→{0,2} 或 {1,3}` 等）→ `make -j20 src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"` ✓ → 产物 md5 `cd73d299887540559838202807e07b01`，三台同 md5 ✓。

**实测结果（smoke-ringonly / smoke-sizes2）**：
- ✅ 补丁生效：日志出现 `RING-ONLY v4 rank 0->2 chan 0 dev 1 (was 0)` 等映射行；`NCCL_ALGO=RING` 生效（算法矩阵五个集合操作全部 **Tree=0 / Ring=1**）
- ✅ **`ibv_modify_qp 110` 彻底消失**（这一条是五轮都没做到的）
- ❌ 但 all_reduce 仍在 `p2p.cc:703 (ncclIbCompletionEventProcess)`：`Recv comm could not retreive a request found for a successful completion` → `ncclInternalError`；**1 MB 与 256 MB 都失败**（不是尺寸问题）

**执行要点（又两个坑）**：
1. 换 NCCL 构建**必须 `LD_PRELOAD`**（`LD_LIBRARY_PATH` 对 torch 自带 NCCL 无效）；
2. 补丁版 `.so` **必须三台都放**（worker 的 `-v` 源路径在 worker 本地）；
3. smoke 默认监听 29599，残留进程会导致 `EADDRINUSE` 假失败 —— 用 `SMOKE_PORT` 换端口。

**差异分析（为什么仍失败）**：上游的 4 机环网里，一台的两个口通向两个邻居，但靠 `v4` 的**逐对端映射**即可闭环；我们 3 机三角`v4` 已经正确（映射行可见、QP 全通）却仍撞内部错误。
另外上游**自带的 3 机档案**（`.env.example`：`NNODES=3 / TP_SIZE=3`）用的是 **`IB_HCA=rocep1s0f0,rocep1s0f1`（每机 2 口）**，与该栈的 PEER_HCA 机制配合；我们这套 NVIDIA Sync 4 口双视图接线与之不同 → 照搬不完全对症。
**下一步**：按他们的**3 机档案原样**再试（2 口/机 + ring-only 库），或在对方仓库提 issue 并附本轮全部日志。

## ★ 第七条路：外部排障清单（附件）逐条对照 + 单 NCCL 覆盖测试（2026-09-18 17:30）

用户提供了一份通用 NCCL 排障 playbook，逐条对照我们的实测：

| 清单项 | 我们的实测结论 |
|---|---|
| ① 单机 `nnodes=1, tp=8` 自检 | **不适用** —— DGX Spark 每机 1 颗 GPU |
| ② `nccl-tests` 3 节点各 1 rank | **已做**（我们的 smoke 即此形态），IB 在 1 MB 与 256 MB 均失败 |
| ③ 强制 TCP 兜底 | **已做**：`NCCL_IB_DISABLE=1` + socket → **PASS**，故障锁定在 IB/RoCE 层 |
| 页锁定内存 memlock | host 与容器均 `unlimited` → 排除 |
| GPUDirect RDMA `nvidia-peermem` | **三台都未加载**（GB10 走 DMABUF 路径）→ 于是测了 `NCCL_NET_GDR_LEVEL=0`，仍失败 |
| MIG / `topo -m` / BIOS ACS | Spark 无 MIG；`nvidia-smi topo -m` 显示 GPU0 对 4 个 NIC 为 NODE/PIX/NODE/NODE |
| 版本一致性 | 三台补丁库 md5 一致（`cd73d299…`） |

**第七轮实测矩阵**（都基于 ring-only 补丁库 + 逐对端映射 + `NCCL_ALGO=RING` + `CROSS_NIC=1`）：
`NCCL_IB_SPLIT_DATA_ON_QPS=0`、`NCCL_PROTO=Simple`、`NCCL_NET_GDR_LEVEL=0`、`NCCL_IB_ADDR_FAMILY=AF_INET6`、上游生产值（`NCCL_MAX_NCHANNELS=4` + `NCCL_BUFFSIZE=8M`）—— **全部失败**。

**第七轮的重大发现（两个环境问题）**：
1. **容器里同时加载两套 NCCL**：只加 `LD_PRELOAD` 时，`/proc/self/maps` 同时出现 `/nccl/libnccl.so.2.30.7` 与镜像自带 `/opt/sglang/.../nvidia/nccl/lib/libnccl.so.2`（`torch.cuda.nccl.version()` 仍报 2.29.7）。**正确做法是把补丁库直接挂到镜像自带路径上做覆盖**（`-v <build>/libnccl.so.2.30.7:/opt/sglang/.../libnccl.so.2:ro`），此时 `LIBS` 只剩 1 条 ✓。
2. **RDMA 设备名必须逐字节精确**：真实名是 `rocep1s0f0 / rocep1s0f1 / roceP2p1s0f0 / **roceP2p1s0f1**`（不是 `roceP2p1s1`）。名字写错会被 `NCCL_IB_HCA` 静默过滤 → 只剩 3 张卡 → 补丁映射请求索引 3 时报 `NET/IB : Requested properties for vNic 3, only 3 vNics have been created`。已加逐台预检脚本。
   注：仓库 `.env` 里的 `NCCL_IB_HCA` 本身是正确的 4 口写法。

**第七轮的干净复测**：单 NCCL 覆盖 + 4 张卡全认（`Rank 0: 4 Net devices`）+ 32 条 per-peer 映射行全部生效 → **仍是 `p2p.cc:703 could not retreive a request ... successful completion`**（`NCCL_IB_SPLIT_DATA_ON_QPS=0` 同样；`NCCL_NET_GDR_LEVEL=0` 变成挂住直到超时）。

→ **结论强化**：排除"双库并存 / 网卡名错 / GDR / 协议 / 通道数 / 字族"这些外部因素后，故障稳定复现在 NCCL IB 传输层的完成队列簿记上，指向 **NCCL 对"3 节点三角（一 rank 两邻居分属两张卡）"这一形状的支持缺陷**。`socket/TCP over CX7` 为当前可用方案。

## ★ 第八条路：两个幸存假设的实测（2026-09-18 18:02）

**H1「一根线上两个口」**：前面所有尝试都给每台 4 个口，即**同一根线上有两个口**（每根线有两个 /24 视图）。对接收侧而言，同一 rank 会从同一根线以两个不同 GID 到达，peer↔QP 映射存在歧义 —— 这正好能解释 `could not retreive a request`。上游 3 机档案恰恰是**每机 2 口（一根线一个口）**。
**实测**：用**未打补丁的 2.30.7**（`libnccl.so.2.30.7.prepatch`，单 NCCL 覆盖挂载）+ `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1`（每机一根线一个口）+ `CROSS_NIC=1`：
- `NCCL_ALGO=RING` → **挂住**（200 s 超时，连 `init_process_group` 都没完成）
- 不设 `NCCL_ALGO`（让 NCCL 自选） → **同样挂住**
→ 与我的穷举结论一致：3 机环要求每个 rank 的**两个邻居各用不同网卡**，而 NCCL 的"按通道号取设备索引"模型表达不了这一点；2 口时必然有一条连接的目标线不可达 → 挂住。

**H2「自定义拓扑文件」**（外部排障清单第 ⑤ 项）：`nvidia-smi` 报 GPU 在 PCI 域 `0000000F`，四张网卡在 `0000`/`0002` 域 → NCCL 的距离模型无从判断；于是写了 `topo/fq-triangle.xml`（把 GPU 与四个 NIC 声明为同一桥下兄弟，使四口都是 PIX）配合 4 口 + 逐对端映射：
**实测** → 仍是 `Internal check failed`（无任何 topology 相关日志输出，文件实际未改变行为）。
→ 拓扑文件管的是"距离排序"，管不了"peer↔设备配对"，因此对本故障无效。

**至此八条路全部实测完毕**，结论不变：NCCL 对该三角形状存在支持缺陷；`socket/TCP over CX7` 仍是可用方案。

**执行教训（脚本层面）**：smoke 脚手架改造后必须 `bash -n` 校验且**确认变量赋值早于使用**（`set -u` 下 `SHADOW_SRC` 顺序错会让每轮秒退）；矩阵脚本要加单实例锁，否则并发跑会互相抢容器/端口、结果作废。

## ★ 第九条路：上游"已知良好组合"的内核+驱动复刻（2026-09-18 19:28，已实测并回滚）

**动机**：上游那套**能跑通**的生产栈（`BUILD-IDENTITY.md:189-191`）是 **Driver `580.173.02` + Kernel `6.17.0-1031-nvidia`**，而我们三台在 **7.0.0-1019 + 580.178.04**。外部排障清单也点名 7.0.0-1019 存在 NCCL/RoCE 回归。

**执行**（三台同步，全程可逆）：
1. 依赖约束：`linux-modules-nvidia-580-open-6.17.0-1031-nvidia` 要求 `nvidia-kernel-common-580 <= 580.173.02-1` → **换内核必须同时降级驱动**（两者是配对的：173.02↔6.17.0-1031，178.04↔7.0.0-1019）。
2. 精准集合（`apt -s` 验证过）：装 3 个包（6.17.0-1031 内核 + 其 modules + 驱动模块包）、降级 4 个驱动包到 173.02、卸载 7.0.0 的旧驱动模块；**不要**碰 `nvidia-driver-580-open` 桌面元包（会拖进 dkms/Xorg 等 16 个包）。
3. 先把 178.04 全套 + 7.0.0 驱动模块 `--download-only` 缓存 → 回滚可离线完成。
4. 切换后 `uname -r` = **6.17.0-1031-nvidia**、驱动 **580.173.02**、`nvidia-smi` 正常（GB10）✓ —— **完整复刻成功**。

**实测结果（run_kernel_test.sh）**：
- `r4p`（ring-only 库 + 4 口 + 逐对端映射 + `ALGO=RING`）→ **仍是 `p2p.cc:703 could not retreive a request`** ✗
- `p2p`（未打补丁 2.30.7 + 每机 2 口 + `ALGO=RING`）→ **挂住到超时** ✗

→ **内核/驱动假设证伪**：在上游那套已知良好组合上，故障**一模一样地复现**。差异不在内核/驱动，而在别处（上游是 4 节点环 + 自带 3 机档案的 2 口接线 + `libncclpin` 等）。
**处理**：已回滚到 7.0.0-1019 + 580.178.04（服务基线配置）✓，6.17.0-1031 内核包保留在系统里（未使用）。

**运维要点（重要，踩过坑）**：
- 本机 `grub-reboot` **不支持 `--id`**（会静默失败并让 `next_entry` 为空）→ 必须用**位置参数**；子菜单项要写路径 **`"1>2"`**（`0`=DGX OS 简单项，`1`=Advanced 子菜单，子菜单内 `0/1/2`=7.0.0 / 7.0.0-recovery / 6.17.0-1031）。
- 设置后**务必先校验** `sudo grub-editenv /boot/grub/grubenv list | grep next_entry` 再重启，否则重启即失去一次性引导。
- 降级驱动后，**当前内核的驱动模块会被卸掉** → 若这时误重启到旧内核，会得到"有系统没显卡"（ssh 可用，可恢复）。
- 脚本经 ssh 分发时文件名要一致；`setsid sudo -n bash /绝对路径` 才能真脱离会话（`~` 在引号里的变量中不展开）。

**提交上游 issue（已完成）**：2026-09-18 20:00 发布 → [`luxingcom/aicad-nccl-optimization#1`](https://github.com/luxingcom/aicad-nccl-optimization/issues/1)（标题：3 节点三角应用 v1+v4 补丁栈后 QP110 消失但 all_reduce 仍撞 `net_ib/p2p.cc:703`；含环境表、最小复现、补丁生效证据、11 项已排除变量表、两点观察、三个具体问题）。草稿留档 `UPSTREAM-ISSUE-aicad-3node-triangle.md`；发布走 GitHub OAuth 设备码（作用域 `public_repo`），令牌未落盘。账号：`masquerator-coder`。

## 下一步候选（未验证完）
1. **`NCCL_IB_MERGE_NICS=1` + 全部 4 个口 + SUBNET_AWARE=1 + 不设 CROSS_NIC**（本次 smoke 在引擎占用 GPU 时挂住，未取到结论）→ 需**停引擎后**重跑，这是三角拓扑下唯一形状正确的方案。
2. 用打过补丁的 NCCL（按 `NCCL_IB_HCA` 顺序注册设备，`~/nccl/build/lib/libnccl.so.2.30.7`）+ 每台按线缆顺序给 2 个口，构造闭环（数学上成立，但 NCCL 环形语义下仍存疑）。
3. 加 RoCE 交换机（终局）。

## 现场状态（供接手）
- 服务在线：`http://192.168.0.101:8888`，socket/TCP over CX7，`health=200`。
- `.env` 现状：`NCCL_NET=Socket`、`NCCL_IB_DISABLE=1`、`NCCL_IB_MERGE_NICS=0`、`NCCL_IB_SUBNET_AWARE_ROUTING=1`、`NCCL_NET_PLUGIN=none`、`NCCL_IB_HCA` 列全 4 口、未设 `NCCL_CROSS_NIC`（后 5 项按用户指定保留，切回 IB 时只需改前两行）。
- 备份：`.env.bak-socket-transport`（改 RoCE 前）、`.env.bak-block5-uniform`、`.env.bak-block3-compact`、`.env.working-socket-20260918`。
- 重要坑：`nccl_smoke.sh` 里有 `set -a; source .env` —— **行内环境变量会被 .env 覆盖**，做传输实验必须先改 .env，或用 `SMOKE_EXTRA_ENV`（追加在后面的 `-e` 优先生效）。
- 服务端 engram 现状：两个 worker 都是 `packed=True`，读本地 `/home/user/dsv41-3xspark/engram/engram-l{1,14}-r{1,2}of3.bin`（各 63 GiB，挂载 `/engram`），非 NFS；head 的 per-rank 分片在 `~/dsv41-engram`（63 GiB）。
