# 3× DGX Spark 部署 DeepSeek-V4.1-Flash（TP=3 跨三机）· RoCE 方案

三台 DGX Spark（GB10）用 CX7 直连成**三角**拓扑，跑起 DeepSeek-V4.1-Flash 推理服务。

**结论先行（2026-09-19 更新）**：这套三角拓扑上 **NCCL 的 IB/RoCE 路径可用**，实测
单流 **28.5–35.4 tok/s**、4 并发聚合 **67.5–75.8 tok/s**、预填 **≈3.0k tok/s**、fabric
`all_reduce` 256 MB **13.86 GB/s**。

> ⚠️ **本仓库早期版本（commit `3c7a016`）的结论"RoCE 走不通、只能 socket"是错的**——
> 那是在**错误接线 + 有 CMA 回归的内核 + 引擎镜像里残留的 NCCL 开关**三重问题上得出的。
> 三条都修正后一次通过。旧记录里"九条路线实测"的价值只剩"哪些旋钮是负收益"，
> 请以本文为准；`docs/ROCE-INVESTIGATION.md` 已标注为**历史记录（已作废）**。

---

## 1. 接线：这是能不能走 RoCE 的第一决定因素

```
        正确（交叉环）                          错误（会 110 / 静默死）
        W1: n1.p0 ↔ n3.p1                      W1: n1.p0 ↔ n3.p0
        W2: n1.p1 ↔ n2.p0                      W2: n1.p1 ↔ n2.p0
        W3: n2.p1 ↔ n3.p0                      W3: n2.p1 ↔ n3.p1
```

**心智模型**：每台 Spark 的 2 个物理 QSFP 口会在**两个 PCI 域各暴露一个 netdev**
（`phys_switch_id` 相同、`phys_port_name` 为 `p0/p1/p0/p1`），所以每台看起来有 4 个 RDMA 设备、
4 个 fabric 地址、全网 6 个 `/24` —— **物理上只有 3 根线**，域 2 那批是同一根缆的第二层视图。
**一个 `/24` 只属于一根缆**。

**为什么必须是"每条缆 p0↔p1"**：NCCL 按 channel→NIC 的**设备索引**跨 rank 配对
（`ncclTopoSearchCheckNet`，索引 0 = PCI 域最小的卡 = 三台都是 `p0`）。只要有两根缆接在**同一端口索引**上，
就必然存在"被配到一起的两个口不在同一根缆上"的索引 → `ibv_modify_qp 110 Connection timed out`
（重试 35 次后放弃），或表现为无 NCCL 告警的静默死。改成每条缆 **p0↔p1** 后，配合
`NCCL_IB_SUBNET_AWARE_ROUTING=1`（接收侧按对端当前 GID 的同 `/24` 逐连接选本地口），NCCL 层一次通过。

现役地址表（实测）：

| 节点 | `p0·域0` | `p1·域0` | `p0·域2` | `p1·域2` |
|---|---|---|---|---|
| n1 head | 10.100.178.2 | 10.100.180.2 | 10.100.179.2 | 10.100.181.2 |
| n2 | 10.100.180.1 | 10.100.176.2 | 10.100.181.1 | 10.100.177.2 |
| n3 | 10.100.176.1 | 10.100.178.1 | 10.100.177.1 | 10.100.179.1 |

> 从错误接线改成正确接线，本集群只动了 **n3 的两个 QSFP 插头**，并把 **n3 的四个 fabric IP 整体对调**
> （`p0` 那对 ↔ `p1` 那对），`/32` 路由的 `dev` 跟着改。

**每台必做的链路配置**（示例见 `assets/`）：

1. 四个 fabric 口 **MTU 9000**；
2. 为**每个对端**加 `/32` 直连路由，让流量走直连线而不是绕管理网；
3. 以上两项**必须写进 netplan / NetworkManager**——临时 `ip route add` 会被 NM re-apply 冲掉，
   症状是到 fabric 的 TCP 停在 `SYN-SENT` 且**源地址是 WiFi**，引擎 init 永久挂起；
4. **`/etc/nvidia/cx7-hotplug-enabled` 必须移走**（三台都查）：留着的话某台重启会把 ConnectX 口
   从**邻居**的 PCI 总线上摘掉，fabric 局部静默消失。

```bash
# 应用网络改动（不要用 netplan apply，免得连带重置管理网 WiFi）
sudo netplan generate && nmcli con reload && nmcli dev reapply <iface>
```

**接线验证（四件套）**：

```bash
# ① 四条 ConnectX function 都满速：期望 32.0 GT/s x4
for f in $(lspci -D -d 15b3: | awk '{print $1}'); do
  echo $f $(cat /sys/bus/pci/devices/$f/current_link_speed) x$(cat /sys/bus/pci/devices/$f/current_link_width); done
# ② IB 口全 ACTIVE（某台离线时，1: DOWN / phys 3: Disabled 的口就是通往那台的缆）
for d in /sys/class/infiniband/*; do echo "$(basename $d) $(cat $d/ports/1/state)"; done
# ③ 源口 ping（同缆通、异缆必须不通）
ping -I enp1s0f0np0 10.100.178.1 && echo 同缆OK
```

> ⚠️ 只靠 `ping -I` 判同缆**不可靠**：存在 `/32` 路由时，指定源地址的包仍按路由表选 `dev`，
> 会对异缆地址"通"。严格判定请用 L2 ARP 探针（AF_PACKET，不带网段语义）。

**控制面必须走管理网**：`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` 用 WiFi 管理口 `wlP9s9`。
它决定每个 rank 公布的 bootstrap 监听地址；设成 fabric 口时各节点公布自己那个口的 fabric IP，
三角里这些 IP 两两不在同一根缆上、又无路由 → 内核改走默认路由 → 永久 `SYN-SENT`、
**NCCL 一行日志都不打**、三台空转。诊断：`sudo ss -tnp | grep 10.100`。

## 2. 另两条前提（同样致命，且都不在 NCCL 里）

| # | 问题 | 判据 | 修法 |
|---|---|---|---|
| ① | **内核 CMA 回归**：`7.0.0-1019-nvidia` 上 `ibv_reg_mr_iova2` 必然 `Cannot allocate memory`（连 **1 KB** 区域都失败，而此时节点有 9 GB 空闲）⇒ 引擎建后续 communicator 时 `ncclSystemError` | `grep CmaTotal /proc/meminfo` = **`0 kB`**（正常内核 `131072 kB`） | 内核/驱动配对切到 **`6.17.0-1031-nvidia` + `580.173.02`**；`GRUB_DEFAULT="1>2"` + `update-grub` + `apt-mark hold` 防自动升回 |
| ② | **引擎镜像 `/etc/nccl.conf` 残留**：`NCCL_IB_USE_INLINE=1` + `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1` ⇒ pynccl 的 **4 字节** warmup all_reduce 永久冻死，日志停在 `sglang is using nccl==2.30.7`，GPU 96% 但功耗仅 ~16 W（自旋） | `docker exec <容器> cat /etc/nccl.conf` | 给 `start.sh` 补透传并置 0：`patch_startsh_envvar.py NCCL_IB_USE_INLINE 0`、`... NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS 0`（进程 env 优先于 conf 文件） |

> ② 之所以长期被误诊：所有"能通过"的独立 smoke / 探针跑的**都是基础镜像**（不带该文件），
> 于是形成"单测能过、引擎卡死"的假矛盾。**复现必须用引擎镜像**跑探针。
> 别被 memlock 误导：引擎容器带 `--ulimit memlock=-1:-1`；**裸 `docker run` 的容器内默认只有 8192**。

## 3. 传输配置

### 3.1 RoCE（推荐）

```bash
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1      # 每台两个逻辑口 = 域0 的两个物理口
NCCL_IB_GID_INDEX=3                     # RoCE v2 + fabric IPv4
NCCL_IB_SUBNET_AWARE_ROUTING=1          # 交叉环下靠它逐连接选对网卡（不要关）
NCCL_IB_MERGE_NICS=0
NCCL_NET_PLUGIN=none
NCCL_CROSS_NIC=0
NCCL_SOCKET_IFNAME=wlP9s9               # 控制面走管理网
GLOO_SOCKET_IFNAME=wlP9s9
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_MAX_NCHANNELS=8
NCCL_BUFFSIZE=1048576
NCCL_PROTO='^LL128'
NCCL_DEBUG_SUBSYS=INIT,NET,ENV
NCCL_HOST_DIR=/nonexistent-nccl-host-dir   # 禁挂自建 NCCL，用镜像自带
```

### 3.2 socket 回退（RoCE 出问题时的一键退路）

```bash
NCCL_NET=Socket
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=enp1s0f0np0          # 数据面仍走 CX7 直连线
GLOO_SOCKET_IFNAME=wlP9s9
```

## 4. 实测性能（同机同配方，只换传输）

| 指标 | **RoCE（当前）** | socket 回退 |
|---|---|---|
| 单流解码 | **28.5 / 29.8 / 33.8 tok/s** | 17.9 – 19.2 |
| 4 并发聚合 | **67.5 / 75.6 / 75.7 / 75.8 tok/s** | 39.7 – 43.3 |
| 预填 | **2.0–2.1k tok/s**（唯一随机 prompt，≈4k 档均值 2045） | ~2.0k\* |
| fabric `all_reduce` 256 MB | **13.86 GB/s**（1 MB 8.1 / 16 MB 13.2） | 2.07 – 2.14 GB/s |
| 模型服务 | `max_model_len=262144`、`max_total_num_tokens=499968` | 同 |

**为什么差这么多**：瓶颈不是带宽，而是**每一步约 104 次集合通信的延迟**——socket 单次约 1 ms、
RoCE 50–100 µs。切到 RoCE 后单流 **+60~85%**、4 并发 **+75%**。
（并发档仍略低于社区同配方参考 78.6 tok/s，剩余空间在 SPS 表与 batch 档位。）

\* socket 档预填为**重复 filler** 所测（可能被前缀缓存虚高）；要点：**预填是算力受限，RoCE 的增益主要在解码**，两档预填同量级。
`scripts/bench_prefill.py` 已改为**每题唯一随机 prompt + 扣解码修正**（重复 prompt 会让预填虚高 1.5–2×，这是本文早前 "≈3.0k" 的错因）。

## 5. 快速上手

```bash
# 0) 前置：内核/驱动配对（§2 ①）+ engine 侧两个开关（§2 ②）+ 接线与链路（§1）

# 1) 链路：MTU 9000 + /32 直连路由（三台都执行，或用 assets/ 里的 netplan 持久化）
sudo bash scripts/fabric-mtu-route.sh

# 2) 起服务（在 head 上；svc.sh 会把预检/share/等就绪/三层验证串起来，约 13–15 分钟）
cp env.example .env            # 按需修改 IP/HCA/内存水位
./svc.sh start                 # 预检 → share → serve → 等就绪 → 三层验证
# 其余子命令：./svc.sh status（只读体检）｜stop｜restart｜preflight｜logs -f
# 若不用脚本，手工等价步骤：
#   ./start.sh share && (setsid nohup ./start.sh serve > serve-$(date +%m%d-%H%M).log 2>&1 &)
# 开机自启（已启用）：systemd 单元 dsv41.service → scripts/svc-boot.sh
#   systemctl status dsv41 ｜ sudo systemctl stop dsv41 ｜ sudo systemctl disable dsv41

# 3) 三层验证（缺一不可）
curl -s http://127.0.0.1:8888/health -w " %{http_code}\n"                     # 200
docker ps --format "{{.Names}} {{.Status}}"                                    # 三台容器 healthy
L=$(ls -t serve-*.log|head -1)                                                 # 传输归属
grep -m2 -E "Using network|Assigned NET plugin" $L; grep -c "via NET/IB" $L; grep -c ibv_reg_mr_iova2 $L
#   期望：Using network IB / Assigned NET plugin IB / via NET/IB ≈64 / reg_mr 失败 = 0
python scripts/bench_decode.py --url http://127.0.0.1:8888
```

**运维铁律**（完整清单见 [`docs/PITFALLS.md`](docs/PITFALLS.md)）：

1. **一轮失败后必须三台重启再开下一轮**——残留容器与挂起上下文会让下一轮在更早的地方假失败；
2. 重启后**先 `./start.sh share` 再 `serve`**（否则 worker 的 NFS 检查静默卡十几分钟）；
3. NFSv4 `fsid=0` 导出时，客户端必须挂**伪根 `:/`**，不是子目录；
4. **绝不要 `docker ps -q | xargs docker rm -f`** —— 引擎容器也在列表里；
5. `MEM_FRACTION_STATIC` 用 0.95（低于 0.944 会报 "weights leave no GPU memory for the KV cache"）；
6. 新增 `.env` 变量必须补透传（`patch_startsh_envvar.py`），`bash -n` 查不出这类拼接错误。

## 6. 仓库结构

```
docs/DEPLOY-GUIDE.md        完整部署手册（流程 + 踩坑；结论以本 README §1–§4 为准）
docs/PITFALLS.md            避坑清单（按现象索引）
docs/ROCE-INVESTIGATION.md  历史排查记录（旧接线 / 坏内核 / 镜像残留，已作废，见文首更正）
docs/UPSTREAM-ISSUE.md      提交给上游的 issue 全文 + 结案更正
scripts/                    可直接复用的脚本（链路、探针、基准、内核切换）
  svc.sh                    服务启停与体检：preflight / start / stop / restart / status / logs
  svc-boot.sh               开机自启包装（等 worker 就绪 → svc.sh start；被下面的 unit 调用）
  dsv41.service             systemd 单元（放到 /etc/systemd/system/ 后 systemctl enable --now）
  run_probe.sh              + pynccl_probe3.py：多 communicator 复现探针驱动/探针（**必须用引擎镜像跑**，1.5 分钟一轮）
  kernel_switch2.sh         内核/驱动配对切换（6.17.0-1031 + 580.173.02，带回滚包缓存）
  patch_startsh_envvar.py   给 start.sh 补环境变量透传（head + worker 两处）
  ping_matrix.sh            接线验证：逐口源地址 ping 矩阵（判定哪两个口同缆）
  arp_probe.py              L2 ARP 探针：反推"对端物理口插在本机哪个口"（不依赖 IP 规划）
  verify_extras.py          视觉分支 + 工具调用（tool_calls）验证
  make_vision_test.py       生成自检用测试图（蓝方块 / 红圆 / 黑条）
  bench_decode.py / bench_prefill.py / bench_concurrency.py   单流 / 预填 / 并发基准
  fabric-mtu-route.sh       MTU 9000 + /32 直连路由（用 netplan 持久化）
  gpu_burn.py               GPU 烧机自检（排除降频锁死）
  __legacy__/               已作废排查路线的证据，保留备查（**不要照做**）：
                            apply_aicad_patches.py、patch_nccl_userorder.py、patch_smoke_*.py、
                            nccl-variant-matrix.sh、nccl_probe_matrix.py、probe-roce-gids.sh、
                            kernel-driver-switch.sh / -revert.sh（旧内核脚本）
fleet/                      线上引擎侧**文件本体**快照（md5 与 head 逐字节一致，见 fleet/README.md）
  start.sh                  head 上实际在跑的启动器（含 9 个透传 + NCCL_CROSS_NIC 默认 0）
  boot.py                   容器入口（DSPARK_ALIGN_VERIFY_TO_TIER 守护，bind-mount 覆盖镜像内旧版）
  Dockerfile                剥掉基础镜像残留的 /etc/nccl.conf（否则 pynccl warmup all_reduce 冻死）
  files/nfs-share.sh        NFS 导出/挂载（mount-spec 修法；start.sh 以相对路径 source）
  patch_dockerfile_ncclconf.py / patch_bootpy_align_optin.py   上述两处改动的生成器（可重放）
assets/                     netplan 示例（node3 已按改线后地址更新）+ 已作废的 NCCL 拓扑文件
env.example                 环境变量样例（已脱敏，RoCE 档 + socket 回退注释）
```

## 7. 硬件/软件基线（实测环境）

| 项 | 值 |
|---|---|
| 节点 | 3 × DGX Spark（GB10），每机 1 颗 GPU，121.7 GiB 统一内存 |
| 内核 / 驱动 | **`6.17.0-1031-nvidia` / `580.173.02`**（配对标；`7.0.0-1019` 有 CMA 回归，见 §2） |
| NCCL | 镜像自带 2.30.7（**不需要任何补丁**；早期"补丁 NCCL"实验是为绕开错误接线，现已不需要） |
| 容器 | 厂商 sglang 镜像 + 官方配方脚本（注意 §2 ② 的 `/etc/nccl.conf`） |

> DGX 上**驱动与内核是配对的**（`580.173.02 ↔ 6.17.0-1031`、`580.178.04 ↔ 7.0.0-1019`），
> 换内核必须同时换驱动；降级驱动会卸掉当前内核的驱动模块，所以"降驱动 + 引导旧内核"必须成对做。

## 8. 已知限制

- SPS（投机解码吞吐表）在当前构建下无法生效（`compact` ragged-verify 启动即崩，三处 shape 不一致），
  当前用 `static`，表 inert；对并发 ≥2 的场景本可有收益；
- **已验证（2026-09-19）**：视觉分支（自造 256×256 测试图，蓝方块/红圆/黑条的颜色、形状、位置全对）、工具调用（返回标准 `tool_calls`，DSML 解析器工作）、三台 `swapoff -a`（`/etc/fstab` 已注释，重启不复活）；
- 单流解码延迟仍有波动（0.9 s ↔ 18 s），已排除 GPU 降频（三台 2.1–2.3 GHz、节流位 0x0）与带宽瓶颈。

## 9. 许可与致谢

- 本仓库文档与脚本：MIT（见 `LICENSE`）。
- **例外**：`fleet/` 下派生自上游配方的文件 —— 4 个部署文件（`start.sh`、`boot.py`、`Dockerfile`、
  `files/nfs-share.sh`）与 `fleet/adapter/` 下 16 个源文件 —— 派生自
  [上游配方](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)（AGPL-3.0-or-later），
  在该目录内按 **AGPL-3.0-or-later** 分发，全文见 `LICENSE.AGPL`。
  其中 8 个解码 adapter 源自 `knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4`（AGPL-3.0-or-later）；
  `LICENSE.upstream-MIT` 保留的 0xSero MIT 声明必须随本仓库一并保留。
  第三方署名与**本地改动声明**见 `NOTICE`；详见 `fleet/README.md`。
- 不包含任何厂商源码、模型权重或镜像内容；引用的第三方补丁请遵循其各自仓库的许可。
- 感谢 `luxingcom/aicad-nccl-optimization` 与 LuZ 生产栈作者公开 ring-only 补丁与构建记录：
  它们在我们接线错误、内核有 CMA 回归的阶段提供了关键对照，也促成了最终定位
  （结论见 `docs/UPSTREAM-ISSUE.md` 的结案更正）。
