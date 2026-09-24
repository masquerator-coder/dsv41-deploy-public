# fleet/ — 线上部署侧文件本体快照（3× DGX Spark · TP=3）

`scripts/` 放的是**工具**（探针、基准、内核切换、启停封装）；`fleet/` 放的是**引擎侧实际在跑的文件本身**。

这些文件此前只存在于 head 的 `~/dsv41-3xspark/`——一个**没有 .git 的 rsync 副本**，一旦重装节点、
或误跑 `start.sh build`（`rsync --delete`）就会丢。此处按**逐字节一致**存档（下表 md5 与 head 实测比对通过）。

| 文件 | md5（= head 实测字节） | 行数 | 与上游配方的差异 |
|---|---|---:|---|
| `start.sh` | `13512310ca325dc400b24e7247f052fc` | 889 | 参数表为旧式 inline 风格；多出 9 个透传（`NCCL_IB_USE_INLINE`、`NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS`、`DSPARK_ALIGN_VERIFY_TO_TIER`、`SGLANG_RAGGED_VERIFY_MODE=static`、`SGLANG_SIMULATE_ACC_LEN`、`SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE`、`SGLANG_DSPARK_ENABLE_SPS_RECORD`、`NCCL_NET_GDR_LEVEL`、`NCCL_DMABUF_ENABLE`）；`NCCL_CROSS_NIC` 默认 0 |
| `boot.py` | `37ed44c36f0aa0b18d01299adac55173` | 420 | `DSPARK_ALIGN_VERIFY_TO_TIER` 守护（align flag 改为 opt-in）；镜像层里那份仍是旧的无条件版本，故用 bind-mount 覆盖 |
| `Dockerfile` | `b4895452eb208aed3e3e56598b63d702` | 29 | 剥掉基础镜像自带的 `/etc/nccl.conf`（`NCCL_IB_USE_INLINE=1` + `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1`）——该残留会让 pynccl 的 4 字节 warmup all_reduce 永不完成 |
| `files/nfs-share.sh` | `a74e6250090e7be70380c9ba17dab313` | 188 | 相对上游的 mount-spec 修法（生成器见 `scripts/fix_nfsshare_mountspec.py`）。保留 `files/` 这一级路径，因为 `start.sh` 以 `source files/nfs-share.sh` 引用 |
| `patch_dockerfile_ncclconf.py` | `e3450653d874748924d294b2a1656f5e` | 42 | 上面 `Dockerfile` 那处改动的生成器（可重放、幂等） |
| `patch_bootpy_align_optin.py` | `643f58f3302885ced4536d7a15fb265a` | 39 | 上面 `boot.py` 那处改动的生成器（可重放、幂等） |

**已在 `scripts/` 里且与 head 内容一致**（无需重复存档）：`svc.sh`、`svc-boot.sh`、`dsv41.service`、
`patch_startsh_envvar.py`（生成 `start.sh` 的透传改动）。

## `fleet/adapter/` — 引擎侧 adapter 快照（2026-09-25 新增）

`adapter/` 此前**完全不在版本控制里**（只在 head 上），是漂移风险最大的一块。
2026-09-25 把 head 上实际在跑的 16 个源文件全部归档（LF 归一，与 head 逐字节内容一致）：

| 来源 | 文件 | 说明 |
|---|---|---|
| 上游 `main`（**未改**） | `encoding_compat.py`、`engram_backend.py`、`loop_abort.py`、`mxfp8_b12x.py`、`prefill_empty_cache.py`、`row_store.cpp`、`tp3_pad.py` | 7 个；其中 Engram row-store 系源自 0xSero（MIT 声明见 `LICENSE.upstream-MIT`） |
| `knapcio` 经上游 `tp3-overnight-decode`（**未改**） | `autotune_keep.py`、`block_verify.py`、`draft_head_fp8.py`、`draft_tau.py`、`engram_prefetch.py`、`folded_result_fence.py`、`verify_cap.py` | 7 个；AGPL-3.0-or-later |
| **本地修改** | `sitecustomize.py`（新增 8 个 env 门控 hook，纯增补）、`wo_a_w8.py`（einsum bridge 补 `**kw` 透传） | 2 个；改动原因见 `NOTICE` 与 `docs/BATCH-MIGRATION-2026-09-25.md` |

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
2. 改 `Dockerfile` / `boot.py` / `.env` 后必须**重建镜像或重起服务**才生效（`./svc.sh stop && ./svc.sh start`，约 13 分钟）；
   只改 `start.sh` 则下次 `serve` 生效，**不需要**重启正在跑的服务。

## 许可

`fleet/` 下派生自上游配方的文件 —— 4 个部署文件（`start.sh`、`boot.py`、`Dockerfile`、
`files/nfs-share.sh`）与 **`adapter/` 下全部 16 个源文件** —— 按 **AGPL-3.0-or-later** 分发，
全文见仓库根的 `LICENSE.AGPL`；第三方署名与本地改动声明见仓库根的 `NOTICE`。

**对应源码即本目录所存文件本身**（含本地对 `adapter/sitecustomize.py` 与
`adapter/wo_a_w8.py` 的修改），不是"见上游某提交"——本目录就是那一版的源码。

`fleet/adapter/` 中另有 8 个文件源自 `knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4`
（AGPL-3.0-or-later），经上游转述；`LICENSE.upstream-MIT` 保留的 0xSero MIT 声明
必须随本仓库一并保留。

本目录下的运维脚本（`svc.sh`、`svc-boot.sh`、`dsv41.service`）与 3 个补丁生成器为本仓库原创，
与本仓库其它内容同为 MIT。