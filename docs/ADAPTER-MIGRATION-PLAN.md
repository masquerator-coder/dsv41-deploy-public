# 迁移方案：把上游 TP3 解码优化栈移植到本 fleet（预览，未执行）

> 状态：**方案预览，未改动集群与代码**。所有"实测"均引自 2026-09-25 对本 fleet 的只读核对
> （ssh `fq-dgx-01/02/03.local`）或上游仓库的真实测量记录，二者分开标注。
>
> 目标仓库：`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks` 分支 `origin/tp3-overnight-decode`（tip `97c46ac`）。
> 核心提交：`5f7de1c`（解码优化，17 个文件）、`5757d1b`（镜像钉 digest）、`97c46ac`（chunk 1024→768）。

---

## 1. 本 fleet 现状（2026-09-25 只读实测）

| 项 | 实测值 | 证据 |
|---|---|---|
| 基础镜像 id（三台） | `sha256:381b27ffa19b`（2026-09-11） | `docker image inspect` ×3，**三台一致** |
| 基础镜像 manifest digest | `sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860` | `docker image inspect --format '{{json .RepoDigests}}'` |
| 引擎 build | `da64c5cbb`（= 上游记为"较新、会漂移"的那个） | 见 §4.1：该 digest 正是上游 `5757d1b` 点名的 `4a5d132a` |
| overlay `dsv41-3xspark:local` | head `aed8521a96e0` / n2 `1ed8faae8961` / n3 `f4a2e885d76d` | 三台各自 build，id 不同属正常 |
| `BASE_IMAGE` 引用 | **移动 tag** `lmsysorg/sglang:dev-dsv41` | `fleet/start.sh:88`、`fleet/Dockerfile:3` |
| 现役 `.env` | `EP_SIZE=3`、`DSPARK_BLOCK_SIZE=3`、`MEM_FRACTION_STATIC=0.95`、`CHUNKED_PREFILL_SIZE=1024`、`MAX_TOTAL_TOKENS=750000` | head `~/dsv41-3xspark/.env` |
| 服务 | `dsv41-head` / 两个 `dsv41-worker` 均 `Up 3 days (healthy)` | `docker ps` |
| packed Engram | head `/engram/engram-l{1,14}-r0of3.bin` 各 ~33.8 GB | `ls -la /engram` → 前置条件已满足 |
| `adapter/` 在 head | 8 个文件，**两个仓库都未归档**（`fleet/` 只存了 4 个部署文件） | `ls ~/dsv41-3xspark/adapter` |

**关键差异**：head 的 `adapter/sitecustomize.py`（73 行）**比本仓库 `fq-dgx-3xspark-fleet` 分支的版本少一个分支**
（`DSV41_SERIAL_WEIGHT_LOAD`，env 门控、默认关）。即 head 的 adapter 是**第三份变体**——
迁移必须以 **head 上的文件为基线**，不能直接用分支版本覆盖。

---

## 2. 上游变更的收益（引自上游测量，**非本 fleet 实测**）

上游 `docs/overnight-results.md:14-25`，5 次重复 / 512 token / 从 spark2 发起：

| 工作负载 | 原配 (b0) | 最终 (F2) | 变化 |
|---|---|---|---|
| C1 散文 greedy | 29.28 | 34.15 | +16.6% |
| C1 代码 greedy | 44.65 | 62.24 | **+39.4%** |
| C1 sampled chat | 28.87 | 37.20 | +28.9% |
| C4 聚合 | 63.78 | 75.79 | +18.8% |
| bs=1 step | 73.2 ms | 59.0 ms | −14 ms |

⚠️ **这些数字测于 `37939c26`，本 fleet 是 `da64c5cbb`**（`docs/overnight-results.md:68,135-139`）。
方向可信，**幅度必须以本 fleet 自测为准**。

### 逐项增益（上游归因）

| 变更 | 上游实测 | 是否需要 adapter |
|---|---|---|
| `EP_SIZE 3→1` | C1 +7.8~12%、C4 +9.7%、NCCL 12.9→5.6 ms/step | ❌ 不需要 |
| `wo_a` FP8 twin + DROP + MID | bs=1 step 72→68-69 ms；NET **−640 MB/rank** | ✅ |
| `k=5` + `VERIFY_CAP=conf:0.1` + `BLOCK_VERIFY=1` | code +9.6%，prose/C4 中性 | ✅ |
| `ENGRAM_PREFETCH=1` | step −2.3~−2.9 ms，C1 +3.7~4.2% | ✅ |
| `DRAFT_HEAD_FP8=1` | step −0.1~−1.3 ms，+210 MiB | ✅ |
| `AUTOTUNE_KEEP=1` | 跨重启复用 tactic；greedy 输出可复现 | ✅ |
| 镜像钉 digest | 防"某台被换引擎" | ❌ |
| `CHUNKED_PREFILL_SIZE 1024→768` | 防 ~200k prompt 预填 OOM（上游曾硬复位） | ❌ |

---

## 3. 兼容性核对（本 fleet 实测）

### 3.1 ✅ 全部 hook 点存在

`sitecustomize.py` 需 hook 的 **9 个模块全部存在**（`ls` 逐个确认）：

```
OK  srt/model_executor/runner/flashinfer_autotune.py      ← autotune_keep
OK  srt/models/deepseek_v4.py                              ← wo_a_w8, verify_cap
OK  srt/models/deepseek_v4_dspark.py                       ← wo_a_w8, verify_cap, draft_head_fp8
OK  srt/speculative/dspark_components/dspark_draft_sampler.py  ← draft_tau
OK  kernels/ops/speculative/dspark/dspark_accept.py        ← block_verify
OK  srt/speculative/dspark_components/dspark_verify.py     ← folded_fence, verify_cap
OK  kernels/ops/moe/moe_fused_gate.py                      ← verify_cap
OK  srt/speculative/dspark_components/dspark_draft.py      ← verify_cap
OK  srt/speculative/dspark_components/dspark_planner.py    ← verify_cap
OK  srt/layers/engram.py                                   ← engram_prefetch
```

具体函数级核对：

| adapter | 需要 | 本 fleet 位置 |
|---|---|---|
| `autotune_keep` | `_autotune_cache_digest` | `flashinfer_autotune.py:209` ✅ |
| | `flashinfer_autotune_context` | `:250` ✅ |
| | `flashinfer_autotune_cache_path` | `:136` ✅ |
| `draft_head_fp8` | `attach_shared_modules` | `deepseek_v4_dspark.py:885` ✅ |
| | `_logits_from_x_post_hc` | `:985` ✅ |
| | `gather_and_crop_vocab` | `dspark.py:34` ✅ |
| `verify_cap` | `build_dspark_v4_confidence_head` | `:525` ✅ |
| | `read_ragged_verify_mode`（import 进该模块） | `:65` ✅ |
| | `RaggedVerifyMode.CAP_ACCEPT` | `ragged_verify.py:15` ✅ |
| `wo_a_w8` | `_apply_wo_a_bf16_matmul` | `deepseek_v4.py:461` ✅ |

### 3.2 ⚠️ `wo_a_w8` 在本 fleet 会走 **bridge（einsum）模式**，不是 kernel 模式

`wo_a_w8.py:418-433` 的选择逻辑：

```python
try:
    kern = importlib.import_module("sglang.kernels.ops.attention.dsv4.wo_a_bf16")
except ModuleNotFoundError:
    kern = None
if kern is not None and hasattr(kern, "wo_a_bf16_small_batch"):
    _patch_kernels(...)          # kernel 模式
else:
    _install_einsum_bridge(...)  # ← 本 fleet 走这里
```

本 fleet `kernels/ops/attention/dsv4/` 下只有 `fp8_wo_a.py`、`wo_a_bf16_gemv.py`、
`wo_a_bf16_small_batch.py`，**没有 `wo_a_bf16.py`** → `ModuleNotFoundError` → **bridge 模式**。

> 注：`deepseek_v4.py:34` 确实 import 了 `wo_a_bf16_small_batch`，但 adapter 找的是**另一个模块名**
> `wo_a_bf16`。两者不是同一个东西，勿混淆。

**这其实是好消息**：上游在 TP3 上验证过的正是 bridge 模式（其 `37939c26` 同样没有 `wo_a_bf16`），
且 bridge 模式的两个已知坑上游都已修好：
- `E1b`：bridge 模式下 DROP 会把 draft 的 `wo_a` 换成零标签 → 每个 draft 变垃圾（acceptance 1.0）。
  上游已在 adapter 里修（draft 永不 DROP，`overnight-results.md:88,98-101`）。
- `MID`：9–192 行的 verify 需要 MID，否则 C4 −8%（`E1c`）。

### 3.3 ✅ `verify_cap` 与 `SGLANG_RAGGED_VERIFY_MODE=static` 兼容

本 fleet 强制 `static`（`fleet/start.sh:299`，因为 compact 启动即崩）。而
`build_dspark_v4_confidence_head` 在 static 下**直接返回 None**（`deepseek_v4_dspark.py:528`）：

```python
if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
    return None
```

`verify_cap.install_dspark` 正是为此设计——构建期临时把 `read_ragged_verify_mode`
换成 `CAP_ACCEPT` 再还原（`verify_cap.py:305-312`）。所以 `conf:0.1` 在 static 下可用。

另外 `config.json` 无 `enable_confidence_head` 字段 → 引擎按**启用**处理并打 warning
（`deepseek_v4_dspark.py:530-534`），符合 adapter 预期。

---

## 4. 最小改动方案（分 4 批，逐个可独立验证与回滚）

### 批次 0：只改 `.env`（**零 adapter，可立即做**）

| 改动 | 值 | 依据 |
|---|---|---|
| `BASE_IMAGE` 钉 digest | `lmsysorg/sglang:dev-dsv41@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860` | 冻结现状 = 本 fleet 自己的测量基线；对应 `da64c5cbb` |
| `EP_SIZE` | `3` → `1` | 上游最大的单项收益，且**不需要 adapter** |

**收益**：EP1 据上游 TP3 实测 C1 +7.8~12%、C4 +9.7%、NCCL all-reduce 12.9→5.6 ms/step。
**风险**：低。EP1 是引擎级并行设置，与 adapter 无关。
**回滚**：改回 `EP_SIZE=3` + `./svc.sh restart`。
**未验证**：本 fleet 是 `da64c5cbb`，EP1 在此 build 上未测过。

同步改动（同一批，仅文档）：
- `fleet/start.sh:88` 与 `fleet/Dockerfile:3` 默认值改为 digest 形式；
- `env.example` 补 `EP_SIZE` 等缺失键（现文件第 47-49 行还留着**已作废**的"RoCE 不可用"注释）。

### 批次 1：低风险 adapter（可复现性 + Engram 预取）

新增文件到 head `~/dsv41-3xspark/adapter/`：

| 文件 | 上游来源 | 作用 |
|---|---|---|
| `autotune_keep.py` | `tp3-overnight-decode` | 跨重启保留 FlashInfer autotune cache |
| `engram_prefetch.py` | 同上 | Engram 行查询走 side stream |

`sitecustomize.py` 增 2 个 hook（engram 的 prefetch 挂在 `engram_backend` 之后；
`flashinfer_autotune` 新增一个 elif 分支）。

`.env` 增：`DSV41_AUTOTUNE_KEEP=1`、`DSV41_ENGRAM_PREFETCH=1`
（可选 `DSV41_ENGRAM_PREFETCH_CHECK=1` 做一次性校验，会打印 mismatch 计数）。

**收益**：step −2.3~−2.9 ms；greedy 输出跨重启可复现（直接治 README §8 的"0.9s↔18s 抖动"）。
**风险**：低。两者都有 fail-fast 符号检查，缺失即 `RuntimeError` 而非静默出错。
**回滚**：`.env` 两键置 0 + restart。

### 批次 2：k=5 + 置信度上限（解码质量档）

新增 2 个 adapter：`block_verify.py`、`verify_cap.py`（+ `draft_tau.py`、`folded_result_fence.py`，
后者默认关，为完整性一并放入）。

`.env` 增：
```
DSPARK_BLOCK_SIZE=5
DSV41_VERIFY_CAP=conf:0.1
DSV41_BLOCK_VERIFY=1
DSV41_DRAFT_TAU=1
DSV41_FOLDED_FENCE=0
```

**收益**：code +9.6%，prose/sampled/C4 中性（上游 E4）。
**风险**：中。**`k=5` 必须与 cap 同开**——上游 E3 证明 k=5 不带 cap 会 prose −8~−16%、C4 −10%。
**回滚**：`DSPARK_BLOCK_SIZE=3` + `DSV41_VERIFY_CAP=0` + restart。

### 批次 3：`wo_a` FP8 twin + draft head（内存 + 速度）

新增 `wo_a_w8.py`、`draft_head_fp8.py`。

`.env` 增：
```
DSV41_WO_A_W8=1
DSV41_WO_A_W8_MID=1
DSV41_WO_A_W8_DROP=1
DSV41_WO_A_W8_DRAFT=0
DSV41_DRAFT_HEAD_FP8=1
```

**收益**：`wo_a` bf16 einsum 从 ~11.5 ms/step 降下；NET **−640 MB/rank**；
draft head +210 MiB 换 ~1 ms/step。
**风险**：中高。三条约束缺一不可：
1. **W8 不能单独开**——上游 E1 证明 +688 MiB 会撞 `minimum viable = 0.9503` 启动失败；
   **必须与 DROP 同开**才能保住 0.95 水位；
2. **DROP 不能丢 MID**——否则 C4 −8%（E1c）；
3. `DSV41_WO_A_W8_DRAFT` 保持 0（上游 E11 测得中性）。

**回滚**：`DSV41_WO_A_W8=0`（连带关掉 MID/DROP）+ restart。

---

## 5. 执行步骤（每批相同）

```bash
# ① 在 head 上更新文件（本仓库纪律：scp 单文件 + md5 回读比对，不要 rsync 整棵树）
scp adapter/<new>.py  fq-dgx-01.local:~/dsv41-3xspark/adapter/
ssh fq-dgx-01.local 'md5sum ~/dsv41-3xspark/adapter/<new>.py'   # 与本地比对

# ② 补 start.sh 透传（复用本仓库既有生成器，幂等）
ssh fq-dgx-01.local 'cd ~/dsv41-3xspark && python3 patch_startsh_envvar.py DSV41_AUTOTUNE_KEEP 0'
#   批 0/1/2/3 需要透传的键见下表；每键跑一次

# ③ 改 .env（先备份）
ssh fq-dgx-01.local 'cd ~/dsv41-3xspark && cp .env state/env.before-<label> && vi .env'

# ④ 重建 + 分发镜像（start.sh:536 本地 build；:547 rsync 到 worker 并远端 build）
ssh fq-dgx-01.local 'cd ~/dsv41-3xspark && ./start.sh build'

# ⑤ 重启 + 三层验证
ssh fq-dgx-01.local 'cd ~/dsv41-3xspark && ./svc.sh restart'
```

需要补透传的键（`patch_startsh_envvar.py` 每键一次）：

| 批次 | 键 |
|---|---|
| 1 | `DSV41_AUTOTUNE_KEEP`、`DSV41_ENGRAM_PREFETCH`、`DSV41_ENGRAM_PREFETCH_CHECK` |
| 2 | `DSPARK_BLOCK_SIZE`（已有）、`DSV41_VERIFY_CAP`、`DSV41_BLOCK_VERIFY`、`DSV41_DRAFT_TAU`、`DSV41_FOLDED_FENCE` |
| 3 | `DSV41_WO_A_W8`、`DSV41_WO_A_W8_MID`、`DSV41_WO_A_W8_DROP`、`DSV41_WO_A_W8_DRAFT`、`DSV41_DRAFT_HEAD_FP8` |

> ⚠️ `patch_startsh_envvar.py` 的锚点是 `NCCL_CUMEM_ENABLE`（`start.sh:294` 与 `:396` 两处均存在，已核对）。
> 该脚本注释明确警告：**`bash -n` 查不出拼接错误**，改完必须回读那一行确认。

**验证**（沿用 `./svc.sh` 的三层验证 + 上游的 bench）：
- `curl -s localhost:8888/health` → 200；
- 三台容器 healthy；
- 日志 `via NET/IB` 计数、`ibv_reg_mr_iova2` 失败 = 0；
- `python scripts/bench_decode.py --url http://127.0.0.1:8888`，与 README §4 基线（28.5–35.4 单流 / 67.5–75.8 四并发）对比；
- adapter 生效证据：容器日志里对应的 `[...] armed` 行。

---

## 6. 风险与未验证项

1. **build 同源风险**：`./start.sh build` 会 `rsync -aH --delete` 把 head 的
   `~/dsv41-3xspark/` 覆盖到 worker（`start.sh:547-551`）。所以**改动必须先落在 head**，
   且**不要**用本仓库或上游克隆去 rsync head（`fleet/README.md` 纪律 1）。
2. **引擎 build 不同**：本 fleet `da64c5cbb` ≠ 上游测量用的 `37939c26`。所有增益需自测。
   上游明确列了 40 个源文件差异，含 `deepseek_v4.py` 与 DSpark worker
   （`overnight-progress.md:19-27`）。
3. **`wo_a_w8` bridge 模式在 TP3 上仅上游验证过**；本 fleet 虽更接近该模式，仍未实测。
4. **`CHUNKED_PREFILL_SIZE`**：上游在新解码栈下 1024 会 OOM（~200k prompt，硬复位），改 768。
   本 fleet 现为 1024。批次 3 会增加内存占用（draft head +210 MiB），同时 DROP 释放 1280 MB。
   **建议**：批次 0–2 保持 1024；批次 3 后做一次长 prompt 压测，必要时降 768。
5. **未运行任何验证**：本文所有"实测"仅为只读核对（`docker inspect` / `ls` / `grep`），
   **未启动过任何 adapter，未跑过任何 benchmark**。
6. **许可**：新增 adapter 派生自 `knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4`（AGPL-3.0），
   经上游 `NOTICE` 标注。若归档进本仓库，`adapter/` 需按 AGPL-3.0-or-later 分发
   （与本仓库现有 `fleet/` 的许可处理方式一致，见 `fleet/README.md` §许可）。
7. **`env.example` 已过时**：第 47-49 行仍是已作废的"RoCE 实测不可用"结论，且缺 `EP_SIZE`、
   `MAX_TOTAL_TOKENS`、`DSPARK_BLOCK_SIZE` 等实际生效的键——与实际 `.env` 大幅脱节，建议同批修正。

---

## 7. 建议顺序

**批次 0 → 1 → 2 → 3**，每批一次重启（约 13 分钟）+ 一轮 bench。
批次 0 单独就能拿到最大单项收益（EP1）且零 adapter 风险，建议**先只做批次 0 并观察一夜**。
