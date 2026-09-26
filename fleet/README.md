# fleet/ — 线上部署侧文件本体快照（3× DGX Spark · TP=3）

`scripts/` 放的是**工具**（探针、基准、内核切换、启停封装）；`fleet/` 放的是**引擎侧实际在跑的文件本身**。

这些文件此前只存在于 head 的 `~/dsv41-3xspark/`——一个**没有 .git 的 rsync 副本**，一旦重装节点、
或误跑 `start.sh build`（`rsync --delete`）就会丢。此处按**内容一致**存档（下表 md5 为存档文件自身的
md5，已与 head 归一化后逐字节比对通过；`fleet/` 被 `.gitattributes` 钉成 LF，见文末「行尾」一节）。

> ⚠️ **2026-09-26 更正**：`start.sh` 一行原先记的 `13512310…`/889 行是 **batch-1 迁移之前**的版本，
> 已过期 13 行内容（12 个 `DSV41_*` 透传只落在 head、未回收）。现更正为 head 现状
> `dad342df…`/903 行。**引这份表做基准前请先核对 md5。**

| 文件 | md5（= 存档文件字节） | 行数 | 与上游配方的差异 |
|---|---|---:|---|
| `start.sh` | `dad342dff680e696cffcabdf621da110` | 903 | 参数表为旧式 inline 风格。相对基线 `cfd405d` 的透传增补：早期 9 个（`NCCL_IB_USE_INLINE`、`NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS`、`DSPARK_ALIGN_VERIFY_TO_TIER`、`SGLANG_RAGGED_VERIFY_MODE=static`、`SGLANG_SIMULATE_ACC_LEN`、`SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE`、`SGLANG_DSPARK_ENABLE_SPS_RECORD`、`NCCL_NET_GDR_LEVEL`、`NCCL_DMABUF_ENABLE`）；**2026-09-25 batch-1** 再增 12 个 `DSV41_*`（清单见 `docs/BATCH-MIGRATION-2026-09-25.md` §3.2）；**2026-09-26** 再增 `DSV41_INDEXER_CHUNKED`、`DSV41_INDEXER_LOGITS_BUDGET_BYTES`（见 `docs/INDEXER-CHUNKED-TP3-RESULTS.md`）。`NCCL_CROSS_NIC` 默认 0 |
| `boot.py` | `37ed44c36f0aa0b18d01299adac55173` | 420 | `DSPARK_ALIGN_VERIFY_TO_TIER` 守护（align flag 改为 opt-in）；镜像层里那份仍是旧的无条件版本，故用 bind-mount 覆盖 |
| `Dockerfile` | `b4895452eb208aed3e3e56598b63d702` | 29 | 剥掉基础镜像自带的 `/etc/nccl.conf`（`NCCL_IB_USE_INLINE=1` + `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1`）——该残留会让 pynccl 的 4 字节 warmup all_reduce 永不完成 |
| `files/nfs-share.sh` | `a74e6250090e7be70380c9ba17dab313` | 188 | 相对上游的 mount-spec 修法（生成器见 `scripts/fix_nfsshare_mountspec.py`）。保留 `files/` 这一级路径，因为 `start.sh` 以 `source files/nfs-share.sh` 引用 |
| `patch_dockerfile_ncclconf.py` | `e3450653d874748924d294b2a1656f5e` | 42 | 上面 `Dockerfile` 那处改动的生成器（可重放、幂等） |
| `patch_bootpy_align_optin.py` | `643f58f3302885ced4536d7a15fb265a` | 39 | 上面 `boot.py` 那处改动的生成器（可重放、幂等） |

**已在 `scripts/` 里且与 head 内容一致**（无需重复存档）：`svc.sh`、`svc-boot.sh`、`dsv41.service`、
`patch_startsh_envvar.py`（生成 `start.sh` 的透传改动）。

## `fleet/adapter/` — 引擎侧 adapter 快照（2026-09-25 新增）

`adapter/` 此前**完全不在版本控制里**（只在 head 上），是漂移风险最大的一块。
2026-09-25 把 head 上实际在跑的 16 个源文件全部归档；**2026-09-26 增至 17 个**（新增
`indexer_chunked.py`，见 `docs/INDEXER-CHUNKED-TP3-RESULTS.md`）。库内存的是 **LF 归一**版本
——head 上有 3 个文件是 CRLF（见文末「行尾」），故这几个文件的 md5 与 head 原始字节**不同**：

| 来源 | 文件 | 说明 |
|---|---|---|
| 上游 `main`（**未改**） | `encoding_compat.py`、`engram_backend.py`、`loop_abort.py`、`mxfp8_b12x.py`、`prefill_empty_cache.py`、`row_store.cpp`、`tp3_pad.py` | 7 个；其中 Engram row-store 系源自 0xSero（MIT 声明见 `LICENSE.upstream-MIT`） |
| 上游 `indexer_chunked.py`（**未改**） | `indexer_chunked.py` | 1 个；sglang#39187 backport，**2026-09-26 加入**，见 `docs/INDEXER-CHUNKED-TP3-RESULTS.md` |
| `knapcio` 经上游 `tp3-overnight-decode`（**未改**） | `block_verify.py`、`draft_head_fp8.py`、`draft_tau.py`、`engram_prefetch.py`、`folded_result_fence.py`、`verify_cap.py` | 6 个；AGPL-3.0-or-later |
| **本地修改** | `sitecustomize.py`（新增 8 个 env 门控 hook，纯增补；**2026-09-26 再增 indexer 分支**）、`wo_a_w8.py`（einsum bridge 补 `**kw` 透传）、`autotune_keep.py`（**2026-09-26 把 `DSV41_INDEXER_CHUNKED` 加入 `_VOLATILE`**，避免切换开关触发重 tune） | 3 个；改动原因见 `NOTICE`、`docs/BATCH-MIGRATION-2026-09-25.md`、`docs/INDEXER-CHUNKED-TP3-RESULTS.md` |

> `librow_store.so` 是 `row_store.cpp` 的构建产物，**不入库**（`Dockerfile` 里现编译）。

## 来源（provenance）

- 快照时间：2026-09-19T20:03+08:00；节点 `fq-dgx-01`；内核 `6.17.0-1031-nvidia`。
- 上游配方：`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks`（AGPL-3.0-or-later），取用时的合并基线
  `cfd405d`（PR#19）；其上游祖先为 `0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000`（MIT）。
- 本快照同时存在于开发克隆的分支 `fq-dgx-3xspark-fleet` @ `7583178`（合并提交 `eccb350`），
  可与上游历史对照 diff。
- head 上有同内容的 `PROVENANCE.md`，记录提交与 md5，便于在集群侧反查"这份文件是哪一版"。

## 两条纪律

1. **不要**把本仓库或上游克隆整体 rsync 到 head：`start.sh build` 用的是 `rsync --delete`，会删掉 head-only 的
   实验脚本与 `PROVENANCE.md`，并把上表三处补丁冲回上游原样。同步单文件请用 `scp` + md5 回读比对。
2. **改什么、怎么生效**——按改动位置分三类，别把"重启服务"当成万能：

   | 改了什么 | 生效方式 |
   |---|---|
   | **`adapter/` 下任何文件**（含 `sitecustomize.py`） | **必须 `./start.sh build` 重建镜像**，再 `./svc.sh stop && ./svc.sh start` |
   | `Dockerfile` / `boot.py` / `files/` | 同上：重建镜像 + 重启 |
   | `start.sh` | 下次 `serve` 生效，**不需要**重启正在跑的服务 |
   | `.env` | 重启生效（`./svc.sh stop && ./svc.sh start`，约 9–13 分钟） |

   > ⚠️ **`adapter/` 是 `COPY` 进镜像的，不是 bind-mount。** 依据：`Dockerfile:5`
   > `COPY adapter /opt/dsv41/adapter`；容器 mounts 实测只有 `models`/`state`/`.cache`/`engram`，
   > **没有 adapter**。
   >
   > 后果：**只改 head 的 `adapter/` 并重启服务完全无效**——容器里跑的是镜像里那份旧代码。
   > 更隐蔽的是 `start.sh` 传的 env 会**照常生效**，于是表现为"配置生效了、代码没生效"的
   > **静默失败**，没有任何报错。2026-09-26 实测踩到过：改 `adapter/sitecustomize.py` 后
   > 重启服务，容器内仍是旧文件（`grep -c deepseek_v4_backend` = 0），而 `DSV41_*` 变量
   > 已经进去了。
   >
   > 排查用时最先该跑的两条：
   > ```bash
   > docker run --rm --entrypoint sh dsv41-3xspark:local -c 'ls -1 /opt/dsv41/adapter/*.py | wc -l'
   > docker run --rm --entrypoint sh dsv41-3xspark:local -c 'md5sum /opt/dsv41/adapter/<你改的文件>'
   > ```
   > 与 head 的 `adapter/` 对不上 → 忘了 `build`。

## 行尾（为什么有的文件 md5 与 head 不同）

`.gitattributes` 把 `fleet/`（`fleet/*` 与 `fleet/**`）钉成 **LF**——这些文件是 `scp` 到 Linux
直接跑的，CRLF 会让 bash 报 `$'\r': command not found`、systemd 单元与 Dockerfile 也会出错。

但 **head 上并不全是 LF**。2026-09-26 实测：

| head 文件 | head 原始行尾 | 库内 | md5 等于 head 原始字节？ |
|---|---|---|---|
| `start.sh` | LF（0 CR / 903 LF） | LF | ✅ 相同 |
| `adapter/sitecustomize.py` 及多数 adapter | LF | LF | ✅ 相同 |
| `adapter/autotune_keep.py` | **CRLF**（104 CR） | LF | ❌ 不同 |
| `adapter/verify_cap.py` | **CRLF**（314 CR） | LF | ❌ 不同 |
| `adapter/wo_a_w8.py` | **CRLF**（466 CR） | LF | ❌ 不同 |

⇒ **核对"库内这份是不是 head 那一版"时，不要直接比 md5**；先归一行尾再比内容，或比
`CR` 计数（差值应恰好等于 `CR` 数）。这三个 CRLF 文件是历史上在 Windows 侧改过留下的，
功能无影响（Python 两种都吃），但会让"逐字节一致"的表述失真——2026-09-26 已更正为
「内容一致」。

## 许可

`fleet/` 下派生自上游配方的文件 —— 4 个部署文件（`start.sh`、`boot.py`、`Dockerfile`、
`files/nfs-share.sh`）与 **`adapter/` 下全部 17 个源文件** —— 按 **AGPL-3.0-or-later** 分发，
全文见仓库根的 `LICENSE.AGPL`；第三方署名与本地改动声明见仓库根的 `NOTICE`。

**对应源码即本目录所存文件本身**（含本地对 `adapter/sitecustomize.py` 与
`adapter/wo_a_w8.py` 的修改），不是"见上游某提交"——本目录就是那一版的源码。

`fleet/adapter/` 中另有 8 个文件源自 `knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4`
（AGPL-3.0-or-later），经上游转述；`LICENSE.upstream-MIT` 保留的 0xSero MIT 声明
必须随本仓库一并保留。

本目录下的运维脚本（`svc.sh`、`svc-boot.sh`、`dsv41.service`）与 4 个补丁生成器为本仓库原创，
与本仓库其它内容同为 MIT。其中 `scripts/patch_autotune_volatile.py`（2026-09-26 新增）
用于重放 `adapter/autotune_keep.py` 的 `_VOLATILE` 改动，幂等且带回滚；它在处理 CRLF 文件时
会保留原行尾风格。