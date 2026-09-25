# DGX Spark ×3 部署 DeepSeek-V4.1-Flash 部署记录（RoCE 档）

> 建立：2026-09-19 凌晨（首份**传输走 RoCE/IB** 的可用记录）。适用：3 × DGX Spark（GB10 / SM121，2 × ConnectX-7）× TP=3。
> 本文取代并删除三份旧笔记（旧《部署手册》《断点》《避坑清单》）。旧记录的"RoCE 此路不通 / 出路只有加交换机 / 只有 18–19 tok/s"是**在错误接线 + 坏内核 + 镜像残留配置上得出的结论，已作废**。
> 实测性能见 §二 第 8 步（**2026-09-19 基线，已被 2026-09-25 解码栈升级刷新 —— 现行数字见文末《2026-09-25 解码栈升级》§3**）。回滚物料见 §二 第 9 步。

---

## 一、硬件连接方法（能不能走 RoCE 的第一决定因素）

### 1. 先建立正确的心智模型：6 个 /24 ≠ 6 条腿

每台 Spark 的 2 个物理 QSFP 口，会在**两个 PCI 域各暴露一个 netdev**（`phys_switch_id` 相同、`phys_port_name` 为 `p0/p1/p0/p1`）：

| 视图 | 网卡名 | IB 设备名 |
|---|---|---|
| 物理口 0 · 域 0 | `enp1s0f0np0` | `rocep1s0f0` |
| 物理口 1 · 域 0 | `enp1s0f1np1` | `rocep1s0f1` |
| 物理口 0 · 域 2 | `enP2p1s0f0np0` | `roceP2p1s0f0` |
| 物理口 1 · 域 2 | `enP2p1s0f1np1` | `roceP2p1s0f1` |

⇒ 看起来每台 4 口、全网 6 个 /24，**物理上只有 3 根 QSFP 线**；域 2 那批是同一根缆的第二层视图。
⇒ **一个 /24 只属于一根缆**，绝不要把两个视图接在同一根缆上或混用。

### 2. 正确接法：交叉环（每条缆 p0↔p1，每台的 p0/p1 通向两个不同邻居）

```
W1: n1.p0 ↔ n3.p1        (10.100.178.0/24 · 10.100.179.0/24)
W2: n1.p1 ↔ n2.p0        (10.100.180.0/24 · 10.100.181.0/24)
W3: n2.p1 ↔ n3.p0        (10.100.176.0/24 · 10.100.177.0/24)
```

**为什么必须是这个形状**：NCCL 按 channel→NIC 的**设备索引**跨 rank 配对（`search.cc` 的 `ncclTopoSearchCheckNet`），索引 0 = PCI 域最小的卡。三条缆若有两个落在同一端口索引上（例如 `p0↔p0`、`p1↔p1`），就必然存在"配到一起的两个口不在同一根缆上"的索引 → `ibv_modify_qp 110 Connection timed out` 或静默死锁。改成**每条缆都 p0↔p1** 后，配合 `NCCL_IB_SUBNET_AWARE_ROUTING=1`（接收侧按对端当前 GID 的同 /24 逐连接选本地口），NCCL 层一次通过。

现役地址表（改线后的实测值）：

| 节点 | `enp1s0f0np0`(p0·域0) | `enp1s0f1np1`(p1·域0) | `enP2p1s0f0np0`(p0·域2) | `enP2p1s0f1np1`(p1·域2) |
|---|---|---|---|
| n1 head | 10.100.178.2 | 10.100.180.2 | 10.100.179.2 | 10.100.181.2 |
| n2 | 10.100.180.1 | 10.100.176.2 | 10.100.181.1 | 10.100.177.2 |
| n3 | 10.100.176.1 | 10.100.178.1 | 10.100.177.1 | 10.100.179.1 |

**改线最小动作**：只对调 n3 的两个 QSFP 插头，并把 n3 的四个 fabric IP 整体对调（p0 那对 ↔ p1 那对），`/32` 路由的 dev 跟着改。

### 3. 每台必做的链路配置（三台一致）

1. **四个 fabric 口 MTU 9000**；
2. **为每个对端加 `/32` 直连路由**（n2 / n3），让 TCP 直连而不绕管理网；
3. 以上两项**必须写进 netplan / NetworkManager**，临时 `ip route add` 会被 NM re-apply 冲掉；
4. **`/etc/nvidia/cx7-hotplug-enabled` 必须移走**（三台都查）：留着的话某台重启会把 ConnectX 口从**邻居**的 PCI 总线上摘掉，fabric 局部静默消失。

```bash
# 应用方式（避免连带重置管理网 WiFi，不要用 netplan apply）
sudo netplan generate && nmcli con reload && nmcli dev reapply <iface>
```

### 4. 接线验证（四件套，缺一不可）

```bash
# ① 四条 ConnectX function 都满速
for f in $(lspci -D -d 15b3: | awk '{print $1}'); do
  echo $f $(cat /sys/bus/pci/devices/$f/current_link_speed) x$(cat /sys/bus/pci/devices/$f/current_link_width); done
#   期望：32.0 GT/s x4 ×4 条

# ② IB 口状态全 ACTIVE
for d in /sys/class/infiniband/*; do echo "$(basename $d) $(cat $d/ports/1/state)"; done
#   期望：全部 4: ACTIVE。若某台离线，`1: DOWN / phys 3: Disabled` 的口就是通往那台的缆（免 ping 反推接线表）

# ③ 源口 ping 矩阵（正/反各一次，且必须跑一条"异缆"对照）
ping -I enp1s0f0np0 10.100.178.1     # 同缆 ⇒ 通
ping -I enp1s0f0np0 10.100.176.1     # 异缆 ⇒ 必须不通
```

> ⚠️ 只用 `ping` 判定同缆**不可靠**：`/32` 路由存在时，`-I` 指定源地址的包仍按路由表选 dev，会对异缆地址"通"。要严格判定用 L2 ARP 探针（AF_PACKET，不带网段语义），见技能 `dgx-spark-cluster-fabric` 的 `scripts/arp_probe.py`。

### 5. 控制面必须走管理网

`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` 用 WiFi 管理口 `wlP9s9`（本集群没有社区配方假设的 `enP7s7` 10GbE 管理口）。
原因：它决定每个 rank 公布的 bootstrap 监听地址；设成 fabric 某口时，各节点公布的是自己那个口的 fabric IP，而三角里这些 IP 两两不在同一根缆上、又无路由 → 内核改走默认路由（源地址=管理网），表现为**永久 `SYN-SENT`、NCCL 一行日志都不打、三台空转**。诊断：`sudo ss -tnp | grep 10.100`。

---

## 二、部署步骤

### 0) 前置（一次性）
- 三台 Docker + NVIDIA 容器运行时；基础镜像 `lmsysorg/sglang:dev-dsv41` + 本地 overlay（`dsv41-3xspark:local`），三台都有。
- 权重在 head（`~/NewModels/DeepSeek-V4.1-Flash`），由 head 的 NFS exporter 容器（`dsv41-nfs`）导出；worker 用 docker volume `dsv41-weights` 挂载，**不落本地副本**。
- head 免密 ssh 到两台 worker；三台 `sudo -n true` 免密。
- **Engram 本地分片**：每台放自己 rank 的行（head `~/dsv41-engram`、worker `/home/fuqiang/dsv41-3xspark/engram`，各 ~63 GiB）。`./start.sh pack`（幂等）。

### 1) ⚠️ 内核 + 驱动配对（RoCE 能不能通的第二决定因素）

```
必须：Kernel 6.17.0-1031-nvidia  +  Driver 580.173.02
禁止：Kernel 7.0.0-1019-nvidia（CMA 回归 → RoCE 内存注册必然失败）
```

`7.0.0-1019-nvidia` 上 NCCL 会以 `ibv_reg_mr_iova2 failed with error Cannot allocate memory` 收场（连 1 KB 区域都失败，而节点有 9 GB 空闲）——**不是 RAM、不是 memlock，是内核 CMA 分配器坏了**。

```bash
uname -r                          # 期望 6.17.0-1031-nvidia
grep -E "^Cma(Total|Free)" /proc/meminfo   # 期望 CmaTotal: 131072 kB；坏内核是 CmaTotal: 0 kB
nvidia-smi --query-gpu=driver_version --format=csv,noheader   # 期望 580.173.02
```

- 切换/回滚：`~/dsv41-3xspark/kernel_switch2.sh`（会先缓存 178.04 全套回滚包，可离线回滚）。
- **固化**（否则一次普通重启就回到坏内核，且旧内核此时已没有驱动模块）：
  ```bash
  sudo sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT="1>2"/' /etc/default/grub   # 1=Advanced 子菜单，2=6.17.0-1031 项（0 基）
  sudo update-grub
  sudo apt-mark hold nvidia-kernel-common-580 nvidia-utils-580 libnvidia-compute-580 \
       nvidia-kernel-source-580-open linux-image-7.0.0-1019-nvidia linux-image-nvidia-hwe-24.04
  ```
- 驱动与内核是**配对**的：`173.02 ↔ 6.17.0-1031`、`178.04 ↔ 7.0.0-1019`。降级驱动会**卸掉当前内核的驱动模块**，所以"降驱动 + 一次性引导到旧内核"必须成对做，别只做一半。

### 2) 修共享（每次重启机器后第一步）

```bash
cd ~/dsv41-3xspark && ./start.sh share
```
**通过标准**：两个 worker 都出现 `dsv41-weights has config.json`。
若报 `cannot see the checkpoint over NFS`：先在 worker 上 `docker volume rm -f dsv41-weights`，再 share。

### 3) `.env` 传输段（RoCE 档，实测可用）

```ini
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

内存与其余关键项：
```ini
MEM_FRACTION_STATIC=0.95          # 必须 ≥0.944（权重 ~99.4 GB/节点）
HEAD_MEM_FRACTION_STATIC=0.95
MAX_RUNNING_REQUESTS=4
DSPARK_BLOCK_SIZE=3               # 调优值，勿改
SGLANG_RAGGED_VERIFY_MODE=static  # compact 在这套构建上启动即崩
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

### 4) ⚠️ 补两个环境变量透传（**必做**，否则引擎必然冻死在 pynccl）

引擎镜像 `dsv41-3xspark:local` 里带了 `/etc/nccl.conf`，内容是 `NCCL_IB_USE_INLINE=1` + `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1`。这两个开关会让 pynccl 的 4 字节 warmup all_reduce 永不完成。`start.sh` 只转发它显式列出的 NCCL 变量，所以必须补：

```bash
cd ~/dsv41-3xspark
cp -a start.sh start.sh.bak-preinline-$(date +%H%M)
python3 patch_startsh_envvar.py NCCL_IB_USE_INLINE 0
python3 patch_startsh_envvar.py NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS 0
grep -n "NCCL_IB_USE_INLINE\|NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS" start.sh   # 回读：head + worker 两处都要有
```
（env 变量**优先于** `/etc/nccl.conf`，置 0 即等效于删掉那两行。）

### 5) 起服务（约 13–15 分钟）

```bash
cd ~/dsv41-3xspark
(setsid nohup ./start.sh serve > serve-$(date +%m%d-%H%M).log 2>&1 &)
```
阶段与时间预算：镜像/NFS 检查 → 推表 → 起 worker 容器 → 起 head → 权重流式加载（48 shard / 99.4 GB 每节点，**~7.5 min**）→ 专家准备 → CUDA graph 捕获（target verify 2 s / draft verify 3 s）→ warm-up（14 s）→ 对外。

### 6) 三层验证（缺一不可）

```bash
curl -s http://127.0.0.1:8888/health -w " %{http_code}\n"                       # ① 200
curl -s http://127.0.0.1:8888/v1/models | python3 -c "import sys,json;m=json.load(sys.stdin)['data'][0];print(m['id'],m.get('max_model_len'))"
docker ps --format "{{.Names}} {{.Status}}"                                      # ② 三台对应容器 healthy
L=$(ls -t serve-*.log|head -1)                                                   # ③ 传输归属
grep -m2 -E "Using network|Assigned NET plugin" $L ; grep -c "via NET/IB" $L ; grep -c ibv_reg_mr_iova2 $L
#   期望：Using network IB / Assigned NET plugin IB / via NET/IB ≈64 / reg_mr 失败 = 0
# ④ nonce 生成测试（防缓存）
N=$(date +%s); curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
 -d "{\"model\":\"deepseek-v4.1-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"只回答数字：$N\"}],\"max_tokens\":24,\"temperature\":0}"
```

### 7) 辅助命令
```bash
./start.sh stop | share | pack | doctor | logs [-f]
```
⚠️ `stop` 可能超时并留下 worker 容器 → **三台 `docker ps` 逐台核对**。

### 8) 实测性能（2026-09-19 01:00，RoCE 档，本集群）

> ⚠️ **本节是 2026-09-19 的基线，已被 2026-09-25 的解码栈升级超越。**
> 现行数字见文末《2026-09-25 解码栈升级》§3（单流 +13.5 % / 代码 +34.5 % / C4 +10.1 %）。
> 本表口径为 `bench_decode.py`（**重复 prompt**）+ `bench_conc.py`；升级轮改用
> `bench_migration.py`（**每请求唯一 prompt**），两者不可直接相减。

| 项                        | RoCE 档（本次）                                                     | 旧 socket 档       | 配方参考             |
| ------------------------ | -------------------------------------------------------------- | ---------------- | ---------------- |
| 单流解码                     | **28.5 / 29.8 / 33.8 tok/s**                                   | 17.9 – 19.2      | 25.4             |
| 4 并发聚合                   | **67.5 / 75.6 / 75.7 / 75.8 tok/s**                            | 39.7 – 43.3      | 78.6             |
| 预填                       | **2.0–2.1k tok/s**（唯一随机 prompt，1.9k→30k token 全档；≈4k 档均值 2045） | ~2.0k*           | 3.35 – 3.78k @4K |
| fabric all_reduce 256 MB | **13.86 GB/s**（1 MB 8.1 / 16 MB 13.2）                          | 2.07 – 2.14 GB/s | —                |

判据：单流超过配方参考、4 并发达其 96% ⇒ **传输瓶颈已消除**；旧记录"只有参考值 ~50%"的结论只对 socket 档成立。

\* socket 档的"~2.0k"是用**重复 filler** 测的（可能同样被前缀缓存虚高），未用唯一 prompt 复测。要点：**预填是算力受限，RoCE 的增益主要体现在解码**（单流 +60~85%、4 并发 +75%），预填两档同量级。
测量条件：`bench_decode.py 300`（temp 0.7, thinking off）、`bench_conc.py 4×200`（temp 0）、预填用非流式 usage 口径（流式路径拿不到 `prompt_tokens`）。

### 9) 备份与回滚物料（head `~/dsv41-3xspark/`）

| 文件 | 用途 |
|---|---|
| `.env.ib-reach-20260918` / `.env.socket-fallback-20260919` | IB 档 / socket 档 .env 快照 |
| `start.sh.bak-preinline-*` | 补透传前 |
| `/etc/default/grub.bak-20260919` | 改 GRUB_DEFAULT 前 |
| `/var/cache/apt/archives`（178.04 全套 + 7.0.0 驱动模块） | 离线回滚内核/驱动 |
| `~/nccl/build/lib/libnccl.so.2.30.7{,.prepatch}` | 旧实验的补丁 NCCL（现不需要） |

**退回 socket 档**：`.env` 三行 `NCCL_NET=Socket`、`NCCL_IB_DISABLE=1`、`NCCL_SOCKET_IFNAME=enp1s0f0np0`（`GLOO` 保持 `wlP9s9`）→ 三台重启 → `share` → `serve`。

---

## 三、服务启停与体检（日常运维）

所有操作都在 **head** 上、于 `~/dsv41-3xspark` 目录内执行。

### 1. 一键脚本 `svc.sh`（推荐）

```bash
./svc.sh status          # 只读体检：/health、传输归属、内核/驱动、CmaTotal、三台容器与 GPU
./svc.sh start           # 预检 → share → serve → 等就绪 → 三层验证（约 13–15 分钟）
./svc.sh stop            # 停服 + 三台逐台核对容器/显存（约 6 分钟）
./svc.sh restart         # = stop + start
./svc.sh preflight       # 只跑启动前预检（不起服务）
./svc.sh logs -f         # 跟随引擎日志
```

脚本把本记录里所有踩过的坑固化成了检查项，不用记：

| 阶段 | 检查/动作 |
|---|---|
| 预检 | 内核 = `6.17.0-1031-nvidia`、`CmaTotal ≠ 0`、驱动 `580.173.02`、镜像存在、`start.sh` 两个 NCCL 透传已置 0、三台 `/etc/nvidia/cx7-hotplug-enabled` 不存在、无 GPU 计算进程残留、四个 IB 口全 `ACTIVE` |
| 双启动保护 | 服务已在运行时 `start` 会被拒绝（提示改用 `restart`/`stop`），不会起出两个引擎抢显存 |
| share | 断言两个 worker 都出现 `dsv41-weights has config.json`（缺一即中止） |
| serve | 后台启动 + 日志落盘；轮询 `/health` 就绪（默认上限 1500s）；**并早发现崩溃**（扫 `Scheduler hit an exception` / head 容器消失，立刻打印 `ibv_reg_mr_iova2`、`ncclSystemError` 等关键行，而不是干等 25 分钟） |
| 三层验证 | `/health=200`、`/v1/models` 返回 id+`max_model_len`、三台容器 `healthy`、`via NET/IB` 计数 >0 且 `ibv_reg_mr_iova2` 计数 = 0（IB 档）、nonce 防缓存生成 |
| stop | `start.sh stop` 后**逐台**核对容器与 GPU 计算进程；有残留时打印**按确切名字**的清理命令（并显式警告不要用 `docker ps -q \| xargs docker rm -f`） |

退出码：预检/验证未过 → 非 0，可直接用于自动化。

> **本脚本已实机演练**（2026-09-19 07:40）：`./svc.sh stop` **6m22s**、三台逐台核对无残留；
> `./svc.sh start` 预检全绿 → `share` 2/2 → `health=200` 用时 **770s**（12m50s）→ 三层验证全绿
> （`via NET/IB` 64 条、`reg_mr` 失败 0、nonce 正确）；双启动保护与"服务在跑 ≠ 残留"判定也已实测。

### 2. 手工流程（脚本不可用时的等价步骤）

```bash
# 起
./start.sh share                      # ← 重启机器后必做，两个 worker 都要 "has config.json"
(setsid nohup ./start.sh serve > serve-$(date +%m%d-%H%M).log 2>&1 &)   # 约 13–15 分钟
curl -s http://127.0.0.1:8888/health -w " %{http_code}\n"               # 200
# 关
./start.sh stop                       # 常超时，属正常
for h in '' fq-dgx-02.local fq-dgx-03.local; do                       # 逐台核对（本机留空）
  ssh${h:+ $h} 'docker ps --format "{{.Names}} {{.Status}}" | grep dsv41'
done
```

### 3. 起停判据（什么时候该起、什么时候别起）

- **该起**：预检全绿；或刚重启完机器（先 `share`）。
- **别起**：① 同一套配置上一轮在**同一处**失败过 → 先重启三台（残留状态会让这一轮在**更早**的地方假失败，本轮实测连 socket 档都会被拖死）；② 还有计算进程/容器占着 GPU → 先 `stop` 并解决残留。
- **起完必须看三个数字**：`/health=200`、三台容器 `healthy`、日志里 `via NET/IB` 有值且 `ibv_reg_mr_iova2` 计数为 **0**。三者缺一都不算成功（例如带宽掉到 ≈2.1 GB/s 就是又落回 socket 了）。

### 4. 开机自启（systemd，已实测）

head 上启用了 `dsv41.service`（unit → `~/dsv41-3xspark/svc-boot.sh`）：

- **触发**：`multi-user.target` 之后，`After=network-online.target docker.service`；
- `svc-boot.sh` 先**等本机 docker 与两台 worker 的 docker/ssh 就绪**（最多 10 分钟，重启后 worker 常慢一拍），再判断：已在跑→跳过；有残留容器但 `/health` 不通→先 stop；否则 `./svc.sh start 1800`；
- **日志**：`~/dsv41-3xspark/boot-autostart-*.log`（svc-boot）与 `systemd-dsv41.log`（unit 的 stdout/stderr）；`journalctl -u dsv41` 看 systemd 侧；
- **实测**（2026-09-19 08:07 真实重启）：unit 自动触发 → 预检全绿 → share 2/2 → `health=200` → 三层验证全绿 → `rc=0`，全程约 13 分钟；
- **常用操作**：`systemctl status dsv41`｜`sudo systemctl start dsv41`（手动跑一遍全流程，会阻塞到验证结束）｜`sudo systemctl stop dsv41`（= 停服务）｜`sudo systemctl disable dsv41`（关自启）；
- ⚠️ 判定"服务是否在跑"**只能看 `dsv41-head`/`dsv41-worker`**：`dsv41-nfs`（权重共享容器）常驻、重启后还会被 docker 自动拉起，用 `grep dsv41-` 会把预检误判成"服务已在运行"（脚本早期就这么错过一次，已修为 `^dsv41-(head|worker)$`）。

---

## 四、避坑指南（按代价排序，全部本现场实测）

### 1. ⛔ 引擎镜像里的 `/etc/nccl.conf` 是隐形杀手（本轮最贵）
- **症状**：引擎日志冻在 `sglang is using nccl==2.30.7`；py-spy（三台一起抓）栈为 `synchronize(torch/cuda/streams.py) ← __init__(pynccl.py:131) ← GroupCoordinator.__init__(parallel_state.py:457)`；GPU 96% 占用但功耗仅 ~16 W（内核自旋）；主机卡在 `libcuda`。
- **根因**：镜像自带 `/etc/nccl.conf` 的 `NCCL_IB_USE_INLINE=1` + `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1`（早期实验遗留）。
- **为什么长期误诊**：所有"能通过"的 smoke / 独立探针跑的都是**基础镜像**（无此文件），于是形成"单测能过、引擎卡死"的假矛盾。**要用引擎镜像跑探针才能复现**。
- **修法**：`patch_startsh_envvar.py` 把两者透传为 0（见 §二 第 4 步）。
- **快速定位**：`docker exec <容器> cat /etc/nccl.conf`，与基础镜像对比。

### 2. ⛔ 内核 `7.0.0-1019-nvidia` 的 CMA 回归 → RoCE 内存注册必失败
- **症状**：`misc/ibvwrap.cc … wrap_ibv_reg_mr_iova2 … Cannot allocate memory`，在 `sendProxyConnect/recvProxyConnect → ncclIbRegMrDmaBufInternal2` 处，连 1 KB 区域都失败；`ncclSystemError` → 进程退出。
- **指纹**：`grep CmaTotal /proc/meminfo` = **0 kB**（好内核 131072 kB）。
- **已排除**：memlock（宿主 `ulimit -l` unlimited、容器带 `--ulimit memlock=-1:-1`；注意**裸跑 `docker run` 的容器内默认只有 8192**，别被误导）、物理内存（9 GB 空闲）、`vm.max_map_count`（1048576）、libmlx5 版本（宿主与容器同为 1.24.50，都没 `mlx5dv_reg_dmabuf_mr`，属无害告警）。
- **修法**：内核/驱动配对切到 `6.17.0-1031-nvidia` + `580.173.02`（§二 第 1 步）。

### 3. ⛔ 一轮失败后必须三台重启，再开下一轮
失败轮会留下残留容器 + 挂起 CUDA 上下文 + NIC 侧未回收资源，**下一轮会在更早、更莫名其妙的地方假失败**（本轮实测：连 socket 档都会卡在 pynccl）。重启后逐台 `uptime -p` 确认（不要用一次 ssh 循环打三台，reboot 会切断连接导致静默漏掉）；重启会清 `/tmp`，探测脚本要放 repo 目录或重传。

### 4. ⛔ 绝不能 `docker ps -q | xargs docker rm -f`
引擎容器（`dsv41-head`/`dsv41-worker`）也在列表里，会当场删掉正在服务的 worker。只按确切名字删，或先 grep 掉 `dsv41-` 前缀。

### 5. 重启后必须先 `share` 再 `serve`
重启会丢 NFS 导出与 worker 卷挂载，直接 `serve` 会**静默卡 ~18 分钟且无任何报错**，看起来像"引擎启动慢"。

### 6. NFSv4 `fsid=0` 导出必须挂伪根 `:/`
挂子目录的报错是 `mount … No such file or directory`。`files/nfs-share.sh` 的 `NFS_DEVICE=":/"`、`.env` 的 `NFS_EXPORT_NAME` 与服务端目录名一致（现为 `export`）。改完 share 相关脚本后，worker 上先 `docker volume rm -f dsv41-weights` 再 share。

### 7. 新增环境变量必须显式补透传
`start.sh` 只转发它显式列出的变量，`.env` 里加了不生效、还白花一轮 boot：`python3 patch_startsh_envvar.py <VAR> <默认值>`（head + worker 两处），改完**回读打印生效值**。注意 `bash -n` 查不出这类拼接错误。

### 8. 临时路由会被 NetworkManager 冲掉
`ip route add` 的 `/32` 路由在服务启动过程中被 NM 抹掉 → 到 fabric 的 TCP 停在 `SYN-SENT` 且**源地址是 WiFi** → 引擎 init 永久挂起（表现为两个 rank ~100% CPU、一个 idle）。诊断组合：`ip route get <对端fabric IP>`、`ip route get <对端fabric IP> from <本机管理IP>`、`sudo ss -tlnp | grep 10.100`。

### 9. 集群运维小坑
- 刚重启的节点 `.local` 别名可能解析失败（`Could not resolve hostname`）→ 用 IP，或等 mDNS 稳定；裸 IP 不会套用 ssh config 里的 `IdentityFile`，会退化成密码认证而**挂住**，所以要么等别名恢复，要么显式 `-i`。
- `stop` 超时留残留容器：跑 NCCL smoke 前必须确认引擎真停了（引擎占 ~99 GB/rank）。
- `pkill -f <pattern>` 经 ssh 远端执行会匹配到自己的命令行：用 `pkill -f "foo[.]sh"` 规避。

### 10. `nccl_smoke.sh` 的判据是坏的（别被它骗）
- 收尾判据 `grep -q "done"` 永远失败：rank0 打的是 `SWEEPDONE`（大写），且 worker 日志里没有该串。
- `RESULT … correct=False` 也是脚本 bug：`x` 在循环里被反复 all_reduce 累乘（1→3→9→…→3^13），不可能等于 `world`。
- **只看**：`Using network IB` / `Assigned NET plugin IB` / `grep -c "via NET/IB"` / `busbw`。≈2.1 GB/s 就是落回 Socket 了，不能算 IB 通过。

### 11. 卡住时抓栈，不要猜（工具组合）
```bash
sudo ~/.local/bin/py-spy dump --pid <scheduler pid> --nonblocking   # 三台一起抓
docker logs <容器> 2>&1 | grep -E "Init START|Init COMPLETE|commId"  # communicator 进度
grep -oE "NCCL_[A-Z_]+ set by environment to [^ ]+" <日志> | sort | uniq -c   # 真正生效的变量
docker exec <容器> cat /etc/nccl.conf                                # 镜像级差异
```
py-spy 用 `pip install --user` 会被 PEP 668 拦，直接从装过的节点拷二进制即可。

### 12. 统一内存 = host RAM 就是显存
任何 pin 住 host 内存的东西（NCCL buffer、page cache、tokenizer、healthcheck）都在跟模型抢显存；TP 是同步的，一个 rank 内存耗尽三台全部停摆。判断标准是 `MemAvailable` 不是"显存够不够"。**`MEM_FRACTION_STATIC` ≥ 0.944**，别当"内存压力诊断旋钮"往下调（README 原话：lowering it does not free RAM, it only starves KV）。

### 13. 长跑与日志纪律
一律后台 + `timeout` + 落盘日志（`setsid nohup … > serve-<date>.log 2>&1 &`）；不要用超过工具前台上限的 `sleep` 等结果；脚本读密码时 `printf '\n\n\n' | setsid nohup …`（**绝不要**追加 `< /dev/null`，会吞掉管道让 `set -e` 静默退出）。

### 14. 基准测得对（否则数字会假）
- **预填必须用"每题唯一"的 prompt**：用重复 filler 造长 prompt 会被 radix cache **前缀复用**，越长的 prompt 越"快"，实测虚高 **1.5–2×**（本记录早先误记的"预填 ≈3.0k"就是这么来的，唯一 prompt 重测只有 **2.0–2.1k**）。
- **预填还要用非流式口径**：本服务流式响应里拿不到 `usage.prompt_tokens`（恒为 0），用流式测只会得到 `prompt_tokens=0`。
- **解码并发档要注明并发数**：batch 组成会变，跨档位比较无意义；单流基准在 concurrency=1 下做，并写明 `reasoning_effort`（SGLang 默认 50 vs 发布方 75）。
- 现成脚本：`bench_prefill.py`（唯一随机 prompt + 扣解码修正）、`bench_decode.py`、`bench_conc.py`。

### 15. 待办与已补验

**待办**：
- SPS 表（`compact` ragged-verify）在本构建启动即崩 → 当前 `static`、表 inert（收益主要在并发 ≥2）；上游 issue 草稿在 `dsv41-fq-3xspark/UPSTREAM-ISSUE-DRAFT.md`；
- 在新内核下对 `NCCL_IB_USE_INLINE` / `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS` 做单变量 A/B（`run_probe.sh` + 引擎镜像，1.5 分钟一轮），确认两者是否仍需强制为 0。

**已补验（2026-09-19，全部实测通过）**：
- **三台 `sudo swapoff -a`**，并把 `/swap.img` 在 `/etc/fstab` 中注释（重启不会复活）→ 内存超限时是 OOM 杀 worker（可重启），而不是换页卡死节点（需断电）；
- **视觉分支**：自造 256×256 测试图（先按像素自检：左上蓝方块 / 中央红圆 / 底部黑条 / 白底），模型回答"蓝色正方形：左上角；红色圆形：中心；黑色长方形：底部；背景：白色" —— **全对**；
- **工具调用**：带 `tools` 请求返回标准 `tool_calls`（`get_weather` / `{"city": "北京"}`），`deepseekv41` DSML 解析器工作正常，**无需**因"标签带前导空格"打补丁。

---

## 五、2026-09-19 停机调优实验（复测 → 矩阵 → 结论）

> 授权条件：少数人用、做重型任务、可停机重启、数据不保留。服务已实测复现原基线后再做实验。
> 结论先行：**可达参数空间内没有吞吐红利**；现役配置本来就接近最优。唯一剩下的结构性空间是 SPS/compact 路线（被上游断言 + 镜像层挡住，见 5.6）。

### 5.1 基线复核（同口径复现，服务未退化）

| 项 | 记录（01:00） | 09:40 复测 | 4×RTX PRO 6000 配方 |
|---|---|---|---|
| 单流解码 | 28.5/29.8/33.8 | 29.4/30.7/29.4（均值 29.8） | 25.4 |
| 4 并发聚合 | 67.5–75.8 | 71.2/73.7/73.3/73.5（均值 72.9） | 78.6（我们 93%） |
| 预填 4k 档 | 2.0–2.1k | 2.05–2.09k | 3.35–3.78k（**不同硬件，不可比**） |

追加特征（本机新测）：
- **吞吐天花板 ≈74 tok/s 且与并发数无关**：4/8/16 并发聚合 74.6/72.3/71.1，单流中位 21.4/16.9/8.6 → 加并发只摊薄单人速度。
- **预填 TTFT 拟合**：固定 **0.15 s** + 边际 **2150 tok/s**（666 token 的 1.36k 只是被固定项拉的）。
- prefix cache 生效：同一 4189 token prompt 第二次 3.08 s → 1.09 s。
- 内存：119.5/124.6 GB 已用，**可用仅 ~5 GB** → 提并发/加缓存没有余量。

### 5.2 NCCL 小消息延迟矩阵（12 变体，引擎停机时跑）

payload = `nccl_lat.py`（16/256/1024 KiB all_reduce），**引擎镜像 + 引擎同款 env**（`nccl_smoke.sh` 的 `SMOKE_EXTRA_ENV` 追加成靠后的 `-e`）；驱动器 `mtx.sh`，每轮 ~30 s。

| 变体 | 16 KiB µs | 256 KiB µs | 判定 |
|---|---|---|---|
| **base（现役）** | 30.4 | 132.2 | 基准 |
| GDR_LEVEL=SYS / PHB | 30.2 / 30.5 | 126.2 / 128.7 | 噪声级，未采用 |
| PROTO=LL,LL128,Simple（自动） | 31.2 | — | 更差 |
| PROTO=LL128（强制） | **57.2** | — | **差一倍** |
| PROTO=LL | 33.5 | — | 更差 |
| MAX_NCHANNELS=4 / 16 | 30.6 / 32.1 | — | 无收益 |
| CROSS_NIC=1 | 31.3 | — | 无收益 |
| **MERGE_NICS=1 + SUBNET_AWARE=0** | **FAIL（0/3）** | — | **卡死**（旧接线的组合，交叉环上不可用） |
| **USE_INLINE=1 + PREPOST=1** | **FAIL（0/3）** | — | **卡死**（与"pynccl 4 字节 warmup 冻死"同源，独立复现） |
| **BUFFSIZE=4 MB** | **29.1** | **108.5** | **唯一小赢 → 已采用** |

结论：
1. **现役 NCCL 配置已是最优**，且 `NCCL_PROTO='^LL128'`、`NCCL_IB_SUBNET_AWARE_ROUTING=1` 两项**是载重项而非排查残留**（强制 LL128 → 延迟翻倍；去掉 subnet-aware → 卡死）。§一 里"新接线尚未单变量 A/B"的遗留疑问到此闭合。
2. `NCCL_IB_USE_INLINE=1` + `PREPOST=1` 会**直接卡死集合通信**：这是"镜像 `/etc/nccl.conf` 残留导致引擎冻死"的第二次独立证据。
3. **集合通信只占每步 ~5%**：16 KiB 一次 all_reduce 30 µs，按每步约百个 collective 估算 ≈3 ms，而实测步时 65 ms ⇒ **fabric 不是 decode 瓶颈，别再往这里投时间**（也解释了为什么 13.86→17.81 GB/s 那点带宽红利不值得追）。

### 5.3 DSPARK block size：每步产出 vs 步时（同 prompt 同温度对照）

| 配置 | accept(tok/step) | 步时 | /generate(temp 0) | chat 档(temp 0.7) |
|---|---|---|---|---|
| **block=3（采用）** | 2.08 | **65.6 ms** | 31.7 tok/s | **29.8 / 72.9** |
| block=5 | 2.56（+23%） | 74.5 ms（+14%） | 34.4（+8.5%） | 26.8 / 66.6（**−10% / −9%**） |

⇒ **采样生成（temp>0）下 block 5 是负优化**：多出的 2 个草稿槽只换来 +0.48 accepted token，却让每步贵 14%。仅在 greedy/高可预测文本上为正。**已复位 3**（也与 SPS 表 manifest 的 `verify_num_draft_tokens: 4` 一致）。
副产品：accept 完全由内容决定（实测 1.03–3.45），同样硬件下单流从 15 到 51 tok/s ——**每步产出才是唯一大杠杆，而它已被 block=3 榨到当前内容下的上限**。

### 5.4 engram 行存缓存：0% 命中率 → 给缓存也没用

- 现象：引擎日志 `Engram layer=1 lookups=45361 hit_rate=0.0% reads=45361 cache=0.0GiB(0 slots)` —— `.env` 的 `DSV41_CACHE_GIB=0` 把行存缓存整个关掉了，**每次查表都是冷读**（`DSV41_RESIDENT_SCALES=0` 还额外放弃 pin scale 分片）。
- 实测：`DSV41_CACHE_GIB=2` → 1.0 GiB/层（394 万 slots，覆盖 3.1% 行）→ 步时 65.6 → **64.9 ms（噪声级）**，chat 档 +3%（噪声内），`/generate` batch4 −3.5%。
- 机制结论：96 个 I/O 线程把读**重叠**掉了，engram I/O **不在 critical path**。
- 决定：**回退 `DSV41_CACHE_GIB=0`**（收益不成立，而它占 2 GB 硬内存；本机统一内存只剩 ~5 GB，§四.12 的教训）。

### 5.5 两个必须记住的坑（新增）

1. **`--enable-forward-pass-metrics` 在本构建上必崩**：首个 batch 即 `Scheduler hit an exception` →
   `metrics_reporter.py:343 _build_scheduled_request_metrics: for sl in batch.seq_lens_cpu` → **`TypeError: 'NoneType' object is not iterable`**，
   引擎在 warm-up 阶段自杀（白等一轮 13 分钟）。**只用 `--enable-metrics`**（已开：`/metrics` 200，85 个指标族，
   `kv_available_tokens / full_token_usage / cache_hit_rate / TTFT/ITL 桶` 都有）。`--enable-mfu-metrics` 未验，别与它同用。
2. **宿主 `boot.py` 的补丁没进镜像**：宿主 `~/dsv41-3xspark/boot.py` 有 `DSPARK_ALIGN_VERIFY_TO_TIER` 守护（08-18 打好），
   但容器内 `/opt/dsv41/boot.py:323` **无条件**加 `--speculative-dspark-align-verify-tokens-to-graph-tier`
   （`server_args` 里恒为 `True`，`.env=0` 不起作用；static 下 no-op 无害）。⇒ **SPS 笔记"下一步 #1（compact + 去掉 align flag）"的前置条件是"更新镜像层里的 boot.py"**，
   在宿主改文件没用。

### 5.6 当时的现行配置（2026-09-19 11:40 起）与收尾实测

> ⚠️ **此为 09-19 时点快照，已不在生效**。现行配置见文末《2026-09-25 解码栈升级》§6
> （EP1 + k=5+cap + wo_a twin + draft head + autotune/engram + chunk 768）。

相对原配置只留两处（`env.head-live-20260919` 快照）：

```ini
NCCL_BUFFSIZE=4194304                                                  # 矩阵唯一小赢：16KiB −4% / 256KiB −18%
EXTRA_SGLANG_ARGS="--fp8-gemm-backend flashinfer_cutlass --watchdog-timeout 1800 --enable-metrics"
# 其余全部回原值：DSPARK_BLOCK_SIZE=3 / DSV41_CACHE_GIB=0 / SGLANG_RAGGED_VERIFY_MODE=static / DSPARK_ALIGN_VERIFY_TO_TIER=0
```

收尾三层验证全绿（`/health=200`、三容器 healthy、`via NET/IB` 64 条、`reg_mr` 失败 0、nonce 正确）；
实测：单流 **30.0**（29.4/28.2/32.5）、4 并发 **74.0**（73.7–74.3）、预填 **2.0k**、单流 accept 2.08 / 步时 ≈65 ms ——**与原基线同级**。

**诚实的总结**：这轮做了 4 次重启、12 个 NCCL 变体、3 组 DSPARK/engram 对照，**没有拿到任何显著的吞吐提升**；
价值在于把"还有没有优化空间"从猜测变成了结论：传输已到底、每步产出已被内容限死、内存没有余量，
**唯一剩下的结构性空间是 SPS/compact 那条路**（需先解上游 `engram.py:296` 断言 + 重建镜像层 boot.py）。

### 5.7 工作区快照（本次新增）

| 文件 | 用途 |
|---|---|
| `env.head-live-20260919` | head 上实际生效的 `.env` 快照。**⚠️ 仓库里的 `env.fq-dgx-cluster` 是 09-18 01:29 的 socket 档旧档**（`NCCL_NET=Socket`、`MERGE_NICS=1`、4 口 HCA），照它回滚会把服务带回错误配置 |
| `mtx.sh` | 12 变体 NCCL 矩阵驱动器（解析 rank 日志，自带容器清理） |
| `bench_replicate.py` | 与记录同口径的 chat 档基准（单流 300 / 4 并发 200×4） |
| `bench_accept_vs_batch.py` · `bench_temp.py` | DSPARK 每步产出随 batch / 温度的变化 |
| `bench_prefill_ttft.py` · `bench_cache_ceiling.py` · `bench_probe.py` | 预填 TTFT 分离、并发天花板、prefix cache |

---

### 5.8 compact/SPS 攻坚（2026-09-19 下午，6 次起服 + 上游证据）

**结论**：compact（ragged verify + SPS 表）从"第一次捕获就崩"推进到"**两个图捕获成功 + warm-up 完成**"，
最后卡在**运行时被裁剪的 verify 批次**上——这一层**上游没有修复**（issue #39173 仍 open，报告人结论相同）。

#### 失败链（逐层，含修法）

| # | 失败点 | 原文证据 | 修法 |
|---|---|---|---|
| 1 | target verify 图捕获 · engram 断言 | `engram target-verify expects one equal block per request, got 12 tokens for 4 requests of 4` | **上游 PR #39257**：`_ragged_capture_slots` 必须返回 `num_tokens // captured_req_width` 行；原实现 `min(num_tokens, max_bs)` 在 bs<max_bs 的档位上多摊行（12 token 摊成 4 行而不是 3 行） |
| 2 | 同一根因的下游消费者 | attention `dsv41_sparse.token_req_indices` → `repeat_interleave: Invalid output_size, expected 16 but got 12` | 同 #1（修一个根因，多个消费者一起好） |
| 3 | （**我自己的错误尝试**）spec_info 的 `-1` | 强制 draft 宽度=4 → draft sampler `view(bs, gamma, -1)` → `shape '[5, 3, -1]' is invalid` | **回退**：draft-verify 本来就该是 3（gamma），不能全局改宽度 |
| 4 | draft verify 图捕获 · 置信头 | `Sizes of tensors must match … Expected size 5 but got size 4`（`deepseek_v4_dspark.compute_confidence` → `dspark.py:441` 的 `torch.cat`） | **上游 PR #31016**：置信头要用**运行时 gamma**（`--speculative-dspark-block-size`）而非 checkpoint 内置 gamma |
| 5 | **运行时**首批 verify | `engram target-verify expects one equal block per request, got 8 tokens for 3 requests of 4` | **无上游修复**（见下） |

#### 第 5 层的机制（本轮最有价值的结论）

- compact 的收益来自**按 SPS 表裁剪 verify 窗口**：实测运行时批次是 **8 token / 3 请求**（等宽应为 3×4=12）
  ⇒ **SPS 预算调度确实生效了**（这是本集群第一次真正跑起这套机制）。
- 但 engram（`engram.py:296`）与 attention（`dsv41_sparse.token_req_indices` 里
  `repeat_interleave(req, draft_token_num, output_size=num_tokens)`）都假设**等宽**；
  变长布局只有部分消费者支持（`RaggedVerifyLayout.padded_to_bucket`）。补 engram 只会把同款崩溃推给 attention。
- 上游 issue **#39173**（同一模型 + Engram；报告人 4×GB10、TP4/EP4、block=5）结论一致：
  "compact ragged verify currently appears unusable for this model"；其评论给出的修复 PR 正是 #39257。
- 因此**这不是配置问题，也不是本集群接线问题**：要跑通 compact，需要上游先让 engram / attention 支持变长 verify 布局。

#### 采用的上游补丁（原文，非自造）

- **PR #39257** `fix: derive compact ragged CUDA graph slots from request width`（source hunk 已应用，含整除断言）
- **PR #31016** `Fix(DSpark): Use Runtime Gamma In DSpark Confidence Head During CUDA Graph Capture`
- 落地方式：**bind-mount 覆盖镜像内文件**（宿主文件与镜像内那份只差指定 hunk，33 GB 镜像不必重分发）；
  **head 与两台 worker 都要挂**（各 rank 的 verify 形状必须一致，否则 TP 集合通信会挂）。
  同样的手法也用于 `boot.py`（镜像层比仓库旧，导致 align flag 关不掉）。
- 验证判据：启动日志出现 `Capture target verify CUDA graph end`、`Capture draft verify CUDA graph end`、
  `DSpark draft proposal … folded into the draft cuda graph`、`Warm-up done`（本轮前三条已达标）。

#### 复现 / 回滚物料

| 文件 | 用途 |
|---|---|
| `start.sh.upstream-fixes-14xx` | 带三个补丁挂载的版本（重试 compact 直接用） |
| `*.patched.py` / `sglang-patches-20260919/` | 上游补丁后的源码副本（三台 md5 已核对一致） |
| `start.sh.bak-bootmount-1151` | 无任何挂载的原始 `start.sh`（当前线上用的就是它） |
| `serve-0919-1419.log` | 关键日志：两个捕获成功 + 运行时 engram 断言 |
| `apply_compact_fixes.py` · `patch_startsh_compact.py` | 补丁/挂载脚本（幂等） |

**已上游反馈**：2026-09-19 把本轮复现数据发到 issue #39173
（https://github.com/sgl-project/sglang/issues/39173#issuecomment-5740217118）：
TP3/EP3 + block=3 的完整失败链、align flag 证伪、#39257+#31016 之后的终点（捕获+warm-up 通过、
首个请求死在 engram 变长布局）、以及"15 分钟可复现、愿意测补丁"。**决定：维持 static 等上游修复（不改本地源码）。**

#### 运维附带发现（本轮的额外代价来源）

`svc.sh stop` 会把权重导出容器 `dsv41-nfs` **SIGKILL（Exited 137）**，随后紧跟的 `share`
重建同名容器会撞名失败（`start.sh` 里那句 `docker rm -f … || true` 把失败静默吞掉了）；
表现为 `[x] share 失败`。解法：`docker rm -f dsv41-nfs` 重试到名字空闲（几秒）再 start。
另外：失败 boot 会在 worker 上留下 **99.6 GB 的引擎进程残骸**，预检会拦下（"有 2 个 GPU 计算进程残留"）——
按记录里的教训**重启那两台**即可（本轮实测：重启后 residue 清空、下一轮预检全绿）。

---

## 附：现场文档（head `~/dsv41-3xspark/`）
**`svc.sh`（服务启停与体检，见 §三：status / start / stop / restart / preflight / logs）**· `serve-*.log`（各轮 boot 日志）· `pynccl_probe3.py` + `run_probe.sh`（1.5 分钟多 communicator 复现探针，**必须用引擎镜像**跑）· `nccl_smoke.sh`（fabric 冒烟；注意其自带判据是坏的，只看 `Using network IB` / `via NET/IB` / busbw）· `bench_decode.py` / `bench_conc.py` / `bench_prefill.py`（吞吐；预填要用非流式 usage 口径）· `kernel_switch2.sh`（内核/驱动切换，含回滚包缓存）· `patch_startsh_envvar.py`（补环境变量透传）· `.env.*`（各档快照：`ib-reach-20260918` / `socket-fallback-20260919`；**09-25 升级轮的备份链在 `state/env.before-*`**）。

**2026-09-25 升级轮新增的工具**（在仓库 `dsv41-deploy-public`，不在 head）：

| 工具 | 位置 | 用途 |
|---|---|---|
| `bench_migration.py` | 仓库 `scripts/` | A/B 基准：C1 散文 sampled/greedy + **C1 代码** + C4；**每请求唯一 prompt**（避免 radix 前缀复用虚高）。改配置前后对比必须用它，**别混用 `bench_decode.py`**（口径不同，见升级 §3 的口径说明） |
| `quality_gate.py` | worker `~/` | 质量门：6 项可判定任务 + 乱码检测。**每批重启后必跑**（catch accept 塌陷） |
| `fleet/adapter/` | 仓库 | head 上实际在跑的 16 个 adapter 源文件快照（含 AGPL 合规文件 `LICENSE.AGPL` / `NOTICE`） |
| `docs/BATCH-MIGRATION-2026-09-25.md` | 仓库 | 本轮完整记录（逐批证据、三个坑、未验证项） |

> 服务侧 adapter 仍是 `scp` 手工同步到 head `~/dsv41-3xspark/adapter/`，**重启不会自动从仓库同步**。

---

## 2026-09-21 并发 / 长上下文调参（KV 750k · max_req 8 · min-free-slots-delay 1）

dsh agent-team 会话（1 主 + 4 子代理 = 5 路长上下文并发，15k→126k token/路）实测只有 4 tok/s，
逐 step 复算坐实（22,465 tok / 4,913 s = 4.57 tok/s，拐点＝子代理上线那一刻）。
根因＝ 5×126k=630k 需求撑爆 500k KV 池 → 只跑 2 路、其余 FCFS 排队（6-token 请求 TTFT 248 s），
且池满导致前缀缓存被驱逐、整段前缀重算（冷 193k 前缀 TTFT 93.9 s vs 命中 4.2 s）。

改动（head `.env`，备份 `.env.bak-perf1-0921-1206`）：`MAX_TOTAL_TOKENS 500000→750000`、
`MAX_RUNNING_REQUESTS 4→8`、`EXTRA_SGLANG_ARGS` 追加 `--min-free-slots-delay 1`；
`CHUNKED_PREFILL_SIZE` 等仍保持 1024（README 的 head 内存约束，不动）。
改后实测：单流解码 32.4 tok/s、4 并发聚合 53.0、**8 并发聚合 80.3 tok/s（TTFT 2.6 s，queue=0）**。

详见同目录 `DGX三节点调参-并发与长上下文-2026-09-21.md`（含逐字段 diff、回滚命令、剩余瓶颈＝socket/TCP fabric）。

---

## 2026-09-25 解码栈升级：迁移上游 TP3 overnight campaign（单流 +13.5 % · 代码 +34.5 % · C4 +10.1 %）

背景：上游 `MiaAI-Lab` 在分支 `origin/tp3-overnight-decode`（tip `97c46ac`，核心提交 `5f7de1c`）
放出了一轮**专为 3× Spark / TP3 做的通宵调优**（8 个新 adapter + EP1 + k=5+cap），
本集群完全没吃到。§5.6 当时"唯一剩下的结构性空间是 SPS/compact"的结论，正好被这一轮覆盖。

### 1. 动手前的核对（关键：本集群引擎与上游不同）

| 项 | 本集群实测 | 上游测量所用 |
|---|---|---|
| 基础镜像 id | `sha256:381b27ffa19b` | `37939c26` |
| 内部 build | `da64c5cbb` | `37939c26` |
| manifest digest | `sha256:4a5d132a…`（即上游`5757d1b`点名的"会漂移的那个") | `sha256:3dbc3130…` |

⇒ 上游记录 **40 个 SGLang 源文件差异**，**其绝对数字不可直接引用**，只能借方向、自己实测。

动手前逐个核对了 adapter 依赖的 **9 个 hook 模块**与
`_autotune_cache_digest` / `flashinfer_autotune_context` / `attach_shared_modules` /
`_logits_from_x_post_hc` / `gather_and_crop_vocab` / `build_dspark_v4_confidence_head`
等符号**全部存在**，才开工。

### 2. 改动（四批，每批一次重启 + 三层验证 + 质量门）

| 批次 | 内容 | 新增 adapter |
|---|---|---|
| 0 | `EP_SIZE 3 → 1` | — |
| 1 | `AUTOTUNE_KEEP=1`、`ENGRAM_PREFETCH=1` | 2 个 |
| 2 | `DSPARK_BLOCK_SIZE 3 → 5` + `VERIFY_CAP=conf:0.1` + `BLOCK_VERIFY=1` | 4 个 |
| 3 | `WO_A_W8` + `_MID` + `_DROP` = 1、`DRAFT_HEAD_FP8=1` | 2 个 |
| 安全 | `CHUNKED_PREFILL_SIZE 1024 → 768` | — |

`start.sh` 用既有生成器 `patch_startsh_envvar.py` 补 **12 个 `DSV41_*` 透传**（head 与 worker
两处都补）；补完**回读 worker 启动行**确认 23 个 `-e` token 完整无粘连（该脚本注释警告过
`NCCL_DEBUG=$NCCL_DEBUG-e NCCL_…` 那种故障）。

### 3. 实测（同一口径，相对「批次 0+1」）

> ⚠️ **口径说明**：基准脚本在本轮中途**增加过 code 负载**。v1（仅散文 + C4）与 v2（+ code）下
> **同一配置的 C4 相差约 5 %**（批次1 配置：v1 得 72.22、v2 得 68.55）。
> **跨口径不可比**；下表全部是 v2 同口径。

| 配置 | C1 散文 greedy | C1 代码 greedy | C1 散文 sampled | C4 聚合 |
|---|---|---|---|---|
| 批次 0+1 基线（EP1 + autotune + engram，k=3） | 30.84 | 58.71 | 29.59 | 68.55 |
| ＋ 批次 2（k=5 + cap + block_verify） | 31.61 | **74.60** | **33.88** | 67.78 |
| ＋ 批次 3（wo_a twin + draft head） | 34.07 | 78.92 | 34.77 | 72.23 |
| 最终（＋ chunk 768） | 36.67 | 79.07 | 34.23 | 75.74 |
| 最终复测（零改动重启后） | 33.30 | 78.80 | 33.37 | 75.19 |
| **均值** | **34.99** | **78.94** | **33.80** | **75.47** |
| **相对基线** | **+13.5 %** | **+34.5 %** | **+14.2 %** | **+10.1 %** |

`C1 代码` 与 `C4` 两次测量几乎重合（78.80/79.07、75.19/75.74）——**最稳**；
散文逐次波动 ±5 %（单轮出现 33.26 与 36.74），**看均值不看单次**。

相对原配（口径 v1，仅作历史对照）：散文 sampled 29.31 → **33.80**（+15.3 %）、
散文 greedy 27.99 → **34.99**（+25.0 %）、C4 65.27 → **75.47**（+15.6 %）。

**预填**：chunk 1024 = 2154 tok/s、chunk 768 = **2018 tok/s（−6.3 %）**——本轮唯一有代价的改动。
（§二 第 8 步记的 2045 是 chunk 1024 下的口径；768 后≈回到该量级。）

### 4. 质量与可复现性

- **质量门 6/6**（算术 ×2 / JSON / 计数 / 列表 / 散文无乱码），**每批重启后都跑**，无退化。
- **回归哨兵 `accept len` 始终 2.5–3.05**：上游 E1b 的失败模式是塌到 **1.0** 且输出变垃圾，
  而那种情况 **tok/s 反而"变快"** ⇒ 只看速度会误判，必须同时看 accept len 与质量门。
- **autotune 跨重启复用已确认**：零改动重启后日志出现
  `[autotune_keep] reused …rank_tp0_pp0_dp0.json` ×2（此前每次都是 `tuned and saved`）。
- 同一条 greedy prompt 连跑 3 次，**sha256 完全一致**（`distinct = 1 of 3`）。
  ⇒ 这正是 §四 里那条"单流解码延迟 0.9 s ↔ 18 s 抖动"的**对症项**。

### 5. 本轮真踩到的三个坑（按代价排序）

**① ⛔ `wo_a_w8` 与本集群引擎版本不兼容 —— 会直接起不来**

上游 adapter 写于 `37939c26`，其 bridge 签名是 `_apply(o, wo_a, is_decode=False)`；
而本集群引擎（`da64c5cbb`）**多传两个 kwarg**：

```python
_apply_wo_a_bf16_matmul(o, wo_a,
    is_decode=forward_batch.forward_mode.is_decode(),
    is_target_verify=forward_batch.forward_mode.is_target_verify(),   # 本集群独有
    fuse_mxfp8_quant=(...),                                          # 本集群独有
)
```

⇒ `TypeError: _install_einsum_bridge.<locals>._apply() got an unexpected keyword argument 'is_target_verify'`，
发生在 **FlashInfer autotune 的 warm-up**（一次 target-verify 前向）⇒ **引擎永远不就绪**。
**修法**：bridge 的 `_apply` 签名加 `**kw`，两处透传分支补 `**kw`（与该文件自己在 237–250 行的写法一致）。
已入仓库 `fleet/adapter/wo_a_w8.py`。

> 附带结论：本集群 `kernels/ops/attention/dsv4/` 下**没有 `wo_a_bf16.py`**，所以 `wo_a_w8`
> 走的是 **einsum bridge 模式**（日志 `[wo_a_w8] einsum bridge armed`）——正是上游在 TP3 验证过的路径；
> 且 bridge 会正确**保留 draft 的 `wo_a` 为 bf16**（上游 E1b 的坑），日志可见
> `[wo_a_w8] draft wo_a left on bf16 (this engine's draft does not use the shared dispatch)`。

**② ⛔ `./start.sh build 2>&1 | tail -60` 会用管道退出码掩盖构建失败**

第一次 build 时 rsync 到 worker 失败（`topo/fq-triangle.xml` 是 **root 所有**的残留目录，
rsync `--delete` 无权删），但管道让整体返回 **0** ⇒ **head 镜像已更新、两个 worker 仍是旧镜像**
（正是上游记录过的"混合 build"故障模式）。
**修法**：`sudo chown -R fuqiang:fuqiang ~/dsv41-3xspark/topo`（两台 worker）后重建。
**纪律**：build / restart 一律 `cmd > log 2>&1; echo EXIT=$?`，**不接管道**。

**③ 抽取上游文件别用 PowerShell 管道（工具链教训）**

`git show <ref>:<path> | Set-Content -NoNewline` 会把**换行折叠成空格**，产出单行损坏文件
（Python 报 `SyntaxError: invalid syntax`，且 traceback 显示 line 1 含整个文件尾部）；
更糟的是它**骗过了 md5 校验**——因为校验的是同一个被损坏的本地文件（循环校验）。
**修法**：改用 `git archive --format=tar … | tar -x`，并以**行数**
（99/118/97/38/167/38/314/455，与上游 diff stat **逐个吻合**）+ `ast.parse` 双重校验。

### 6. 现行配置（2026-09-25 03:20 起）

```ini
EP_SIZE=1
DSPARK_BLOCK_SIZE=5
DSV41_VERIFY_CAP=conf:0.1
DSV41_BLOCK_VERIFY=1
DSV41_WO_A_W8=1
DSV41_WO_A_W8_MID=1
DSV41_WO_A_W8_DROP=1
DSV41_DRAFT_HEAD_FP8=1
DSV41_AUTOTUNE_KEEP=1
DSV41_ENGRAM_PREFETCH=1
CHUNKED_PREFILL_SIZE=768
# 保持关闭：DSV41_WO_A_W8_DRAFT=0 / DSV41_DRAFT_TAU=1 / DSV41_FOLDED_FENCE=0
```

`MEM_FRACTION_STATIC=0.95`、`MAX_TOTAL_TOKENS=750000`、`MAX_RUNNING_REQUESTS=8`、
`CONTEXT_LENGTH=262144`、`SGLANG_RAGGED_VERIFY_MODE=static` 均**未改动**。

备份链（head `~/dsv41-3xspark/state/`）：
`env.before-ep1-*` → `before-batch1-*` → `before-batch2-*` → `before-k3-ab-*` →
`before-batch3-*` → `before-revert-batch3-*` → `before-batch3b-*` → `before-chunk768-*`

### 7. 回滚

```bash
# 整轮回滚（回到本轮之前的原配）
cd ~/dsv41-3xspark && cp state/env.before-ep1-20260925-0045 .env && ./svc.sh restart
#   adapter 仍在镜像里，但每个都 env 门控，全关即等于原行为

# 单项回滚（改一行 + restart，约 10 分钟）
EP_SIZE=3                                              # 关 EP1
DSPARK_BLOCK_SIZE=3 + DSV41_VERIFY_CAP=0 + DSV41_BLOCK_VERIFY=0   # k=5 与 cap 必须成组
DSV41_WO_A_W8=0                                        # 连带 MID/DROP 一起关
DSV41_DRAFT_HEAD_FP8=0
DSV41_ENGRAM_PREFETCH=0
DSV41_AUTOTUNE_KEEP=0                                  # 回到"每启动丢弃并重 tune"
CHUNKED_PREFILL_SIZE=1024

# 想强制重 tune（换了战术时）——三台都要
rm -f ~/.cache/sglang/flashinfer/autotune/*/sm121/*/rank_*.{json,launch}
```

### 8. 后续与遗留

- **⛔ 长上下文压测（最高优先）**：本轮**没跑 >32k 的 prompt**，768 的实际保护效果**未验证**。
  这是目前唯一仍有**灾难性失败模式**的方向（上游在 1024 下被 ~200 k prompt 打爆 → 主机挂死 → 硬复位）。
  压测时用 `scripts/verify/memguard.py` 护航，目标定出 768 下的安全上界（64k / 128k / 200k）。
- **`wo_a_w8` 的本地修复只做了功能验证**（起得来、accept len 正常、质量 6/6），
  未做逐 token 数值对照（上游对 DROP 的说明本就是 "dequantized copy not bit-identical"）。
- 上游剩余大头：b12x dense MXFP8 GEMM **17.4 ms/step**、MoE grouped GEMM **21.1 ms/step**、
  SPS/STS 表（主要在并发 ≥2）、k=4 + cap。§5.6 的 SPS 结论仍待重估。
- 仓库侧：`adapter/` 已纳入 `dsv41-deploy-public` 的 `fleet/adapter/`（16 个源文件，
  含 AGPL-3.0 合规文件 `LICENSE.AGPL` / `NOTICE` / `LICENSE.upstream-MIT`），
  提交 `fbbf547` + `e0b507e`。`librow_store.so` 是构建产物未入库。
- 服务侧 adapter 仍是 `scp` 手工同步，**重启不会自动从仓库同步**。

完整记录另见仓库 `docs/BATCH-MIGRATION-2026-09-25.md`（含逐批证据、未验证项）
与 `docs/ADAPTER-MIGRATION-PLAN.md`（事前方案）。
