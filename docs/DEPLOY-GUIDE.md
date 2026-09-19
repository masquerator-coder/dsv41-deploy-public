> ## ⚠️ 结论更正（2026-09-19）：传输部分以 README §1–§4 为准
>
> 本文"RoCE 走不通、只能 socket、约 18–19 tok/s"的结论是在**接线错误 + 内核 CMA 回归（`7.0.0-1019-nvidia`，
> `CmaTotal: 0 kB`）+ 引擎镜像 `/etc/nccl.conf` 残留**三重问题下得出的。
> 修正后同一套三角拓扑上 RoCE 可用：单流 **28.5–35.4 tok/s**、4 并发 **67.5–75.8 tok/s**、
> fabric `all_reduce` 256 MB **13.86 GB/s**。流程与踩坑主体仍适用，但**传输配置、性能数字、
> 内核/驱动版本请按 README（`6.17.0-1031-nvidia` + `580.173.02`）**。下文 §200 起的 RoCE 章节已作废。

# DGX Spark ×3 部署 DeepSeek-V4.1-Flash 操作手册（正确流程 + 踩坑记录）

> 建立于 2026-09-18。适用于本机这套 **3 × DGX Spark（GB10 / SM121）× TP=3** 的现场。
> 关联：内部 Obsidian 专题《DGX Spark ×3 部署记录》· 传输结论以 README §1–§4 为准
> 现场文档（`AIWorkspace/dsv41-fq-3xspark/`，head 的 `~/dsv41-3xspark/` 有副本）：
> `serve-*.log`（各轮 boot 日志）· `pynccl_probe3.py`+`run_probe.sh`（1.5 分钟多 communicator 复现探针）· `kernel_switch2.sh`（内核/驱动切换）
>
> 本笔记 = **我们亲手跑过、亲手踩过的流程与坑**；社区资料整理见「避坑清单」那份，两者互补。

---

## 一、最终状态与设计取舍（先看结论）

| 项 | 值 |
|---|---|
| API | `http://192.168.0.101:8888`（无鉴权），模型名 `deepseek-v4.1-flash`，`max_model_len=262144` |
| 传输 | **socket/TCP over CX7 fabric**（`NCCL_NET=Socket`）；实测 2.07–2.14 GB/s |
| 性能（本现场实测） | 单流 **18–19 tok/s**；4 并发 **40–43 tok/s** 聚合；预填 ~2.0k tok/s |
| 参考值（社区 SGLang TP3 散文） | 单流 37.9 / 并发 4 聚合 78.6 tok/s；我们只有 ~50% → **见 §八 第 1 条（GPU 降频锁死）** |
| 上下文 | `max_total_num_tokens≈500k`（`MEM_FRACTION_STATIC=0.95`） |
| RoCE | ❌ 三角直连下 NCCL 2.30.7 建不起来（两种可能形状都实测排除，见 §六） |
| SPS 表 | 表已采出并通过自检，但 compact ragged-verify 启动即崩；当前 `static` + 表 inert |

**为什么用 socket**：CX7 三角只有 3 根线（每台 2 口），NCCL 的 rail 对齐与多平面语义都表达不了这个拓扑。权衡后：带宽低一个数量级，但**稳定、可复现**。

---

## 二、拓扑与地址（部署前必须核对）

```
管理网 WiFi wlP9s9      01=192.168.0.86（头）  02=192.168.0.84  03=192.168.0.102
容器                    dsv41-head(rank0)   dsv41-worker(rank1)   dsv41-worker(rank2)
镜像 / 部署根           dsv41-3xspark:local      ~/dsv41-3xspark
```

**CX7 四口地址（实测：每台 4 口 = 2 根物理线 × 2 个视图）**

| 节点 | enp1s0f0np0 | enp1s0f1np1 | enP2p1s0f0np0 | enP2p1s0f1np1 |
|---|---|---|---|---|
| 01 head | 10.100.178.2 | 10.100.180.2 | 10.100.179.2 | 10.100.181.2 |
| 02 | 10.100.180.1 | 10.100.176.2 | 10.100.181.1 | 10.100.177.2 |
| 03 | 10.100.176.1 | 10.100.178.1 | 10.100.177.1 | 10.100.179.1 |

**线缆（按共用 /24 判断）**

```
W1 = 01-p0 ↔ 03-p1      W2 = 01-p1 ↔ 02-p0      W3 = 02-p1 ↔ 03-p0
（178/179、180/181、176/177 分别是这三根线的两个视图）
每条缆必须 p0↔p1（交叉环）。同索引相接（p0↔p0 / p1↔p1）会让 NCCL 按设备索引配对时
配到"不在同一根缆上"的两个口 → ibv_modify_qp 110 或静默死。原因见 README §1。
（2026-09-19 更正：本节旧版写的是 01-p0↔03-p0 / 02-p1↔03-p1，那是会失败的接法。）
```

⚠️ 6 个 /24 ≠ 6 条腿，只有 3 根线。判别法：`phys_switch_id` 相同 + `ping -I <iface> <对端口IP>` 双向可达。

---

## 三、正确部署流程

### 0) 前置（一次性）
- 三台 Docker + NVIDIA 容器运行时；镜像 `dsv41-3xspark:local` 三台都在。
- 权重：head 的 `~/NewModels/DeepSeek-V4.1-Flash`，由 head 的 NFS exporter（容器 `dsv41-nfs`）导出；worker 用 docker volume `dsv41-weights` 挂载（**不落本地副本**）。
- head 免密 ssh 到两台 worker；三台 `sudo -n true` 免密。
- **Engram 本地分片**：每台放自己 rank 的行（head `~/dsv41-engram`、worker `/home/user/dsv41-3xspark/engram`，各 63 GiB / 两个文件 31.5 GiB）。用 `./start.sh pack`（幂等）。

### 1) ⚠️ 每次重启机器之后：先修 NFS，再起服务
```bash
cd ~/dsv41-3xspark
./start.sh share            # 必须！重启会丢 NFS 导出与 worker 卷挂载
```
若 worker 卷陈旧（输出 `cannot see the checkpoint over NFS`），先在 worker 上删卷再 share：
```bash
# node2 / node3
docker volume rm -f dsv41-weights
# 回到 head
./start.sh share
```
**通过标准**：两个 worker 都出现 `dsv41-weights has config.json`。

### 2) 起服务（约 13–15 分钟）
```bash
cd ~/dsv41-3xspark
(setsid nohup ./start.sh serve > serve-$(date +%m%d-%H%M).log 2>&1 &)
```
阶段：镜像检查 → NFS/GID 探测 → 推表 → 起 worker 容器 → 起 head → 权重流式加载（48 shard / 99.4 GB/节点，合计 ~7.5 min）→ 专家准备 → CUDA graph 捕获 → warm-up 18 s → 对外。

### 3) 验证（三层，缺一不可）
```bash
# ① 进程/健康
curl -s http://127.0.0.1:8888/health -w " %{http_code}\n"
docker ps --format "{{.Names}} {{.Status}}"          # head: dsv41-head + dsv41-nfs
# ② 模型
curl -s http://127.0.0.1:8888/v1/models | python3 -c "import sys,json;m=json.load(sys.stdin)['data'][0];print(m['id'],m.get('max_model_len'))"
# ③ 真生成（用 nonce 防缓存）
N=$(date +%s); curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"deepseek-v4.1-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"只回答数字：$N\"}],\"max_tokens\":24,\"temperature\":0}"
# 两台 worker 上确认 dsv41-worker healthy
```

### 4) 辅助命令
```bash
./start.sh stop      # 停机（可能超时并留下 worker 容器 → 三台 docker ps 核对）
./start.sh share     # 修 NFS（重启后必做）
./start.sh pack      # 重打 Engram 本地分片（幂等）
./start.sh doctor    # 环境自检
./start.sh logs [-f] # 引擎日志
```

---

## 四、踩坑记录（按代价排序，全部为本现场实测）

### 1. 临时路由被 NetworkManager 冲掉 → 引擎 init 永久挂起（最贵）
- **症状**：引擎 init 卡死；node2 `ss -tn` 显示对 head 的 TCP 停在 `SYN-SENT`，**源地址是 WiFi 而不是 fabric**。表现成「2 个 rank ~100% CPU、1 个 rank idle」的假死。
- **根因**：`ip route add` 加的 `/32` 直连路由在服务启动过程中被 NM 抹掉。
- **修法**：路由与 MTU 一律交给 **netplan**（三台 `/etc/netplan/99-nvidia-sync-cluster.yaml`：四口 `mtu: 9000` + node2/3 的 `/32` 直连路由）。

### 2. 重启后直接 serve → 静默卡 18 分钟
- **症状**：`./start.sh serve` 长时间无输出、不报错。
- **根因**：卡在 worker 的 NFS 检查上（重启丢了导出/挂载）。
- **修法**：见 §三-1，先 `share` 再 `serve`。

### 3. NFSv4 `fsid=0` 导出必须挂伪根，不能挂子目录
- **症状**：worker 卷挂载失败 `mount ... No such file or directory`。
- **根因**：exporter 以 `fsid=0` 导出 `/export`，客户端必须挂 `<ip>:/`；挂 `:/dsv41-native` 会被解释成 `/export/dsv41-native`。
- **修法**：`files/nfs-share.sh` 的 `NFS_DEVICE=":/"`，`.env` 的 `NFS_EXPORT_NAME` 与服务端目录名一致（现为 `export`）。备份 `files/nfs-share.sh.bak-exportname`。

### 4. `./start.sh share` 会重建 NFS 容器 → 客户端规格不一致就自伤
- **症状**：本来能用的 worker 卷，跑完 share 反而挂不上。
- **修法**：改完 share 相关脚本后，**worker 上 `docker volume rm -f dsv41-weights` 再 share**，让卷按新规格重建。

### 5. `MEM_FRACTION_STATIC` 必须 ≥0.944
- **症状**：`Loaded weights leave no GPU memory for the KV cache`。
- **根因**：早期误诊留下的 0.60/0.55；权重 99.4 GB/节点已满，0.6 分不出 KV 池。
- **修法**：`MEM_FRACTION_STATIC=0.95` + `HEAD_MEM_FRACTION_STATIC=0.95` → `max_total_num_tokens≈500k` 正常。

### 6. `./start.sh stop` 可能超时并留下 worker 容器
- **修法**：停完三台都 `docker ps` 核对。跑 NCCL smoke 前必须确认（引擎占 99 GB/rank，smoke 抢不到显存会挂）。

### 7. ⛔ 绝不能按 `docker ps -q` 批量删容器
- **教训（本人踩过）**：为清理 NCCL smoke 的临时容器，在 worker 上执行 `docker ps -q | xargs -r docker rm -f`，把**正在运行的 dsv41-worker** 一起删掉，服务中断约 20 分钟。
- **正确做法**（按名字或排除引擎容器）：
  ```bash
  docker rm -f <临时容器名>
  for c in $(docker ps -a --format '{{.ID}} {{.Names}}' | grep -v 'dsv41-' | awk '{print $1}'); do docker rm -f "$c"; done
  ```

### 8. 重启会清 `/tmp`
- 基准/探测脚本要重新 scp；固定脚本放 `~/dsv41-3xspark/scripts/`。

### 9. 传输实验：`nccl_smoke.sh` 会 `set -a; source .env`
- **症状**：行内环境变量（`NCCL_NET=IB …`）看似传了，实际被 `.env` 覆盖，测的是 .env 的值 → 容易得出错误结论（本人据此白跑一轮）。
- **修法**：先改 `.env`；或 `SMOKE_EXTRA_ENV="KEY=VAL …"`（脚本转成 docker `-e`，**追加在最后、优先生效**）。

### 10. RoCE 的两个陷阱（细节见 `ROCE-FINDINGS-2026-09-18.md`）
- `NCCL_IB_MERGE_NICS=1` 会让图搜索找不到可用 IB 设备，**静默退回 Socket**：带宽只剩 2.1 GB/s 且**不报错**，极易误判"RoCE 跑通了"。
- 列全 4 口 + `SUBNET_AWARE_ROUTING=1` + `MERGE_NICS=0` 能让 NCCL 真正选 IB（`Using network IB`、`Rank 0: 4 Net devices`），但三角下 QP 仍 110（见 §六）。

### 11. SPS 表的隐藏门槛（细节见 `SPS-TABLE-2026-09-18.md`）
- 服务端必须**同时**带 `SGLANG_DSPARK_ENABLE_SPS_RECORD=1` 与 `SGLANG_SIMULATE_ACC_LEN=1.0`（后者必须正好 1.0）；start.sh 默认都不透传。
- profiler 需 `--max-batch-size 4`（≤ 引擎已捕获的 graph 档位），否则拒采。
- 采完必须把这两个开关关掉再正常重启。
- ⚠️ 有表时 `boot.py` 会把 `SGLANG_RAGGED_VERIFY_MODE` 默认成 `compact`，而 compact 在这套构建上**启动即崩**（三处 shape 不一致）；当前用 `static`。

### 12. Engram 本地分片的路径是 `.env` 的 `WORKER_DIR`
- `start.sh` 的默认值写作 `/home/<user>/dsv41-3x**-spark**`（多一个连字符），实际目录是 `.env` 指定的 `/home/user/dsv41-3xspark`。
- 按默认值查会得出"worker 没有本地分片"的**错误结论**（本人误判过一次）。正确核对：
  ```bash
  ssh <worker> 'ls -la /home/user/dsv41-3xspark/engram/; du -sh /home/user/dsv41-3xspark/engram'
  docker logs dsv41-worker 2>&1 | grep "Engram layer"    # 期望 packed=True
  ```

### 13. 新增环境变量要显式补透传
```bash
python3 patch_startsh_envvar.py <VAR> <默认值>     # head 与 worker 两处一起补
```

### 14. 其它现场事实
- 本机**没有 `enP7s7`**（社区配方假设的 10GbE 管理口）；Store/gloo 走 `wlP9s9`。
- `DSPARK_BLOCK_SIZE=3` 是调优值（社区也确认 k=5 会过度起草），勿随意改。
- 容器名固定；`docker ps` 里同时有引擎容器与临时容器时，删东西一律按名字。

---

## 五、备份与回滚清单（head `~/dsv41-3xspark/`）

| 文件 | 用途 |
|---|---|
| `.env.working-socket-20260918` | 可用 socket 配置快照 |
| `.env.bak-socket-transport` | 切 RoCE 前快照 |
| `.env.bak-block3-compact` / `.env.bak-block5-uniform` | SPS 实验快照 |
| `start.sh.bak-socket-20260918` / `nccl_smoke.sh.bak-nocrossnic` | 脚本改前备份 |
| `files/nfs-share.sh.bak-exportname` | NFS 挂载规格改前 |
| `boot.py.bak-pre-align` | ragged-verify 开关改前 |
| `~/nccl/build/lib/libnccl.so.2.30.7{,.prepatch}` | 按 HCA 顺序注册设备的补丁版 NCCL |

**从 RoCE 退回 socket**：改 `.env` 两行（`NCCL_NET=Socket`、`NCCL_IB_DISABLE=1`）→ 三台重启 → `share` → `serve`。

---

## 六、两个"此路不通"的结论（避免重复投入）

### RoCE（三角直连）—— 五条路全部实测走不通，根因在 NCCL 自身
| # | 配置 | 结果 |
|---|---|---|
| 1 | 社区配方：4 口全列 + `MERGE_NICS=0` + `SUBNET_AWARE=1` + `NET_PLUGIN=none` + 不设 `CROSS_NIC` | ❌ `ibv_modify_qp 110`：`local GID 180.2(head p1) → remote GID 176.1`，索引对齐把不在同一根线上的口配到一起 |
| 2 | ＋ bootstrap 平面挪到管理网（清单"两个平面别配混"） | ❌ 110，错配一模一样（`OOB wlP9s9` 已生效） |
| 3 | ＋ `NCCL_CROSS_NIC=1`（唯一允许两端用不同网卡的开关） | ❌ 110 |
| 4 | `MERGE_NICS=1`（按 PCI 卡合并成"能同时够到两个邻居"的多平面设备） | ⚠️ QP 建立成功、数据通（`Rank 0: 6 Net devices`、`via NET/IB/4`），但 all_reduce 崩溃：`p2p.cc:705 Recv comm could not retreive a request found for a successful completion` → `ncclInternalError` |
| 5 | **补丁版 NCCL（按 `NCCL_IB_HCA` 顺序注册设备）+ 按线缆图排 per-rank HCA 顺序**（真正闭环的索引分配） | ⚠️ **110 消失**（三台顺序确认为 `rank0/1: [0]rocep1s0f1 [1]rocep1s0f0`、`rank2: [0]rocep1s0f0 [1]rocep1s0f1`，QP 建立成功），但同样是 `p2p.cc:703 ... could not retreive a request` → `ncclInternalError` |
| 6 | **AICAD 上游补丁栈**（`luxingcom/aicad-nccl-optimization`：v1 ring-only + v4 逐对端设备映射 + stageB 协议 tuner，见 `ROCE-FINDINGS-2026-09-18.md` 末节） | ✅ 补丁生效（`RING-ONLY v4 rank 0->2 chan 0 dev 1`、算法矩阵 Tree=0/Ring=1）✅ **`ibv_modify_qp` 彻底消失** ❌ 但仍 `p2p.cc:703 could not retreive a request` → `ncclInternalError`，**1 MB 与 256 MB 均失败** |
| 7 | **单 NCCL 覆盖 + 外部排障清单逐项**（把补丁库直接挂到镜像自带 `.../nvidia/nccl/lib/libnccl.so.2` 路径，消除"双库并存"；再逐项测 `SPLIT_DATA_ON_QPS=0` / `PROTO=Simple` / `NET_GDR_LEVEL=0` / `ADDR_FAMILY=AF_INET6` / 上游生产值） | ❌ 全部失败。干净复测下：`nccl_libs:1`（只剩一套）、`Rank 0: 4 Net devices`、32 条 per-peer 映射行全生效 → 仍同一个 `p2p.cc:703` |

| 8 | **两机 2 口（一根线一个口）** + 未打补丁 2.30.7；以及**自定义 `NCCL_TOPO_FILE`**（GPU 与四口声明为同桥兄弟 → 全 PIX） | ❌ 2 口方案**挂住**（`ALGO=RING` 与自选算法都挂）；拓扑文件方案仍是 `Internal check failed` |

> **第 8 条结论**：拓扑文件只管"距离排序"，管不了"peer↔设备配对"；2 口方案与我的穷举结论一致（3 机环要求一个 rank 的两个邻居各用不同网卡，NCCL 的"按通道号取设备索引"模型表达不了）。**八条路全部实测完毕，结论稳定：NCCL 对该三角形状存在支持缺陷**；当前方案为 socket/TCP over CX7。

| 9 | **复刻上游"已知良好组合"**：Driver `580.173.02` + Kernel `6.17.0-1031-nvidia`（上游 BUILD-IDENTITY 记录的生产组合） | ❌ **假设证伪**：4 口方案仍是同一个 `p2p.cc:703`，2 口方案挂住 —— 在**同款内核+驱动**上故障一模一样地复现。已回滚到 7.0.0-1019 + 580.178.04 |

> **第 9 条的运维知识（值得记住）**：
> - 驱动与内核是**配对**的：`173.02 ↔ 6.17.0-1031`、`178.04 ↔ 7.0.0-1019`（`linux-modules-nvidia-580-open-<内核>` 依赖 `nvidia-kernel-common-580` 的版本上限）。换内核必须同时换驱动。
> - 降级驱动会**卸掉当前内核的驱动模块** → 若此时重启到旧内核会得到"有系统没显卡"（ssh 可用，可恢复）。
> - 本机 `grub-reboot` **不支持 `--id`**：必须用位置参数；子菜单项写 **`"1>2"`**（`0`=DGX OS 简单项，`1`=Advanced 子菜单，子菜单内 `0/1/2`=7.0.0 / 7.0.0-recovery / 6.17.0-1031）；设置后**先校验** `grub-editenv list | grep next_entry` 再重启。
> - 回滚用 `~/dsv41-3xspark/kernel_revert.sh`（178.04 全套已缓存，可离线执行）。

> **第 7 条留下的两个环境教训（重要）**：
> 1. **只加 `LD_PRELOAD` 会同时加载两套 NCCL**（补丁库 + 镜像自带，`torch.cuda.nccl.version()` 仍报 2.29.7）。要确保只有一套，须把补丁库**直接挂到镜像自带路径上覆盖**：`-v <build>/libnccl.so.2.30.7:/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:ro`（`/proc/self/maps` 里 `LIBS` 只剩 1 条即为准）。
> 2. **RDMA 设备名逐字节精确**：`rocep1s0f0 / rocep1s0f1 / roceP2p1s0f0 / roceP2p1s0f1`（末者是 `P2p1s0f1`，**不是** `P2p1s1`）。写错会被 `NCCL_IB_HCA` 静默过滤 → 只剩 3 卡 → 报 `Requested properties for vNic 3, only 3 vNics have been created`。

> **注意第 5 条的正确执行方式**：`LD_LIBRARY_PATH` **覆盖不了** torch 自带的 NCCL（镜像里 torch 走 `/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2`，实测 `from /nccl` 映射数为 0），必须用 **`LD_PRELOAD=/nccl/libnccl.so.2.30.7`** 才生效；而且补丁版 `.so` 必须**三台都放**（worker 容器的 `-v` 源路径在 worker 本地）。

**根因**：三角形里一个 rank 的"上邻居"和"下邻居"分别挂在不同网卡上，而 NCCL 的 IB P2P 路径假定同轨/同卡（环语义 + 请求-完成配对）。两种独立配置（合并设备、正确索引顺序）都走到 QP 建立成功却在同一处 `ncclInternalError` 倒下 → **是 NCCL 对本拓扑的限制，不是配置问题**。
**出路**：加 RoCE 交换机（标准 rail/星形），或维持 socket。可拿去报上游（附本文档 5 条证据）。

### SPS 表（compact ragged verify）
崩链三处（同一类问题）：target-verify 图捕获（`engram.py:296`，`12 vs 4×4`）→ 加 `SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE=1` 后 target 通过、warm-up 崩（`18 vs 4×6`）→ 再加 align 开关后 draft-verify 捕获崩（`Expected size 5 but got size 4`）。
结论：此构建下不可用；`static` 模式下表 inert，但性能仍优于早期基线。

---

## 七、收尾检查清单（每次部署/重启后逐条打勾）

- [ ] 三台 `uptime` 正常、GPU 空闲（~15 W）
- [ ] `./start.sh share` 两个 worker 都有 `has config.json`
- [ ] `./start.sh serve` 日志出现 `Load weight end`（48/48 shard）
- [ ] 出现 `Warm-up done` 且**无** `Scheduler hit an exception`
- [ ] `/health=200`、`/v1/models` 返回 `max_model_len=262144`
- [ ] nonce 生成测试回显正确
- [ ] 三台 `docker ps` 对应容器 `(healthy)`
- [ ] 吞吐抽查：单流 ~18–19 tok/s、4 并发 ~40 tok/s 聚合

---

## 八、尚未核对的社区清单项（源自「避坑清单」，按预期收益排序，待办）

1. **✅ GPU 降频锁死：已核对（2026-09-18 14:50），三台全部健康，已排除** —— fp16 4096³ matmul 烧 18 秒，第 12 秒采样：

   | 节点 | 空闲 | 负载 | 节流位掩码 |
   |---|---|---|---|
   | head (rank0) | 2405 MHz / 14.9 W | **2268 MHz / 89.6 W / 96%** | `0x0` |
   | node2 (rank1) | 2411 MHz / 14.9 W | **2223 MHz / 89.3 W / 96%** | `0x0` |
   | node3 (rank2) | 2405 MHz / 13.5 W | **2112 MHz / 89.2 W / 96%** | `0x0` |

   锁死特征为 700–950 MHz / <20 W 且 `nvidia-smi` 看不出异常 → 与本现场不符。脚本：`AIWorkspace/dsv41-fq-3xspark/gpu_burn.py`。
   顺带确认：head 上建第二个 CUDA context 未影响引擎（跑完 `health=200`、容器仍 healthy）。
   ⚠️ 踩坑：用 `-v /tmp/x.py:/tmp/x.py:ro` 挂脚本时，若宿主文件不存在，docker 会**创建同名目录**（本次遇到，root 拥有，需 `sudo rm -rf` 清掉）。
   → 所以"吞吐只有参考值 ~50%、单流 0.9 s↔18 s 抖动"要另找原因，当前首要嫌疑是**第 3 条 socket 带宽差 4 倍**。
2. **`sudo swapoff -a`（三台）**：有 swap 时内存超限会换页卡死节点（需断电）；无 swap 则只是 OOM 杀 worker（可重启）。
3. **✅ socket 带宽嫌疑：已量化（2026-09-18 15:0x），不是瓶颈** —— 4 并发解码实测（57 秒窗口，`measure_fabric.sh`）：

   ```
   head  rx p0 223 MB/s   tx p1 223 MB/s         单向环：01 →(p1/W2)→ 02 →(p1/W3)→ 03 →(p0/W1)→ 01
   node2 rx p0 225 MB/s   tx p1 225 MB/s         三台 fabric 合计 rx 680 MB/s，反向每路仅 ~1.2 MB/s
   node3 rx p1 228 MB/s   tx p0 228 MB/s         WiFi 侧 <0.1 MB/s（纯控制面，已确认）
   ```

   每链路 **~225 MB/s**，折合 **~84 MB/step/rank**（按 2.7 步/秒），仅占 socket 上限 2.1 GB/s 的 **32%** → **带宽不是瓶颈**。
   真正的时间去向：bs=1 时引擎实测 **123 ms/step**（SPS profiler 数据），而社区 RoCE 的 collectives 开销只有 **13–16 ms/step** → **每步 ~104 个 collective 在 socket 上单次 ~1 ms（RoCE 是 50–100 µs）**就是那 2× 差距的主体，也是 TCP-over-CX7（无 RDMA）的结构性代价。
   **可动的杠杆**：① `NCCL_SOCKET_NTHREADS` / `NCCL_NSOCKS_PERTHREAD`（当前**未设**，默认 4×1）——值得在 smoke 里试一版；② **提高 `MAX_RUNNING_REQUESTS` 与 CUDA graph 档位**（现 4 → 8），用更大 batch 摊薄每步固定延迟，是目前最直接的提吞吐手段。
4. **视觉分支与工具调用单独验**：纯文本冒烟测不到视觉编码器；DSML 标签在 V4.1 带前导空格，自研解析器不更新会**静默失败**。
5. **page cache 回收守护**：Engram 行读取会把 MemFree 压到 3.7 GiB（GB10 allocator stall 区）；社区提醒 `flusher2.sh` 的判据恒为负永不触发，要用 `memfree_flusher`。
6. **可复现性基准**：并发 >1 的 batch 组成会变，所有"可复现"基准应在 concurrency=1 下做，并注明 `reasoning_effort`（SGLang 默认 50 vs 发布方默认 75，会让 thinking 数字不可比）。
