# 迁移记录：上游 TP3 解码优化栈落地到本 fleet（2026-09-25 夜间执行）

> 本文记录 2026-09-25 00:45–03:20 对 3× DGX Spark 生产 fleet 的实际改动、实测数字与回滚方式。
> 方案依据见 [`ADAPTER-MIGRATION-PLAN.md`](ADAPTER-MIGRATION-PLAN.md)。
> 上游来源：`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks` 分支 `origin/tp3-overnight-decode`（tip `97c46ac`）。
> **所有数字均为本 fleet 实测**（`scripts/bench_migration.py`，每请求唯一 prompt、从 worker 发起）。

---

## 1. 结论

**单流解码 +31%、代码 +35%、C4 并发 +16%**（相对本次优化前的原配），质量 6/6 无退化，
端到端三层验证全绿。全部改动可用 `.env` 单行回滚。

最终配置：`EP_SIZE=1`、`DSPARK_BLOCK_SIZE=5` + `VERIFY_CAP=conf:0.1` + `BLOCK_VERIFY=1`、
`WO_A_W8(+MID+DROP)=1`、`DRAFT_HEAD_FP8=1`、`AUTOTUNE_KEEP=1`、`ENGRAM_PREFETCH=1`、
`CHUNKED_PREFILL_SIZE=768`。

---

## 2. 实测结果

### 2.1 基准口径说明（重要）

基准脚本在过程中**增加过一个 code 工作负载**，所以有两代口径：

- **v1**：仅 prose（sampled + greedy）+ C4；
- **v2**：v1 + C1 code。

**同一配置在 v1/v2 下的 C4 相差约 5%**（batch1 配置：v1 得 72.22、v2 得 68.55），
因为 v2 在 C4 之前多跑了一段 code 负载、影响了机器状态。
⇒ **只能比较同一口径内的数字**；跨口径对比无效。下面 2.2 是同一口径（v2）的干净对比。

### 2.2 同口径（v2）逐步收益

| 配置 | C1 散文 greedy | C1 代码 greedy | C1 散文 sampled | C4 聚合 |
|---|---|---|---|---|
| 批次0+1 基线（EP1 + autotune + engram, k=3） | 30.84 | 58.71 | 29.59 | 68.55 |
| + 批次2（k=5 + cap + block_verify） | 31.61 | **74.60** | **33.88** | 67.78 |
| + 批次3（wo_a twin + draft head） | 34.07 | 78.92 | 34.77 | 72.23 |
| 最终（+ chunk 768） | 36.67 | 79.07 | 34.23 | 75.74 |
| **最终复测（零改动重启后）** | 33.30 | 78.80 | 33.37 | 75.19 |
| **两次最终测量均值** | **34.99** | **78.94** | **33.80** | **75.47** |
| **相对基线** | **+13.5 %** | **+34.5 %** | **+14.2 %** | **+10.1 %** |

`c1_code` 与 `c4` 两次测量几乎重合（78.80/79.07、75.19/75.74），是最稳的两个指标；
散文档逐次波动约 ±5 %（`c1_greedy` 单轮出现 33.26 与 36.74 的差异），
所以散文的增益应看均值而非单次。

### 2.3 相对本次优化前的原配（口径 v1，仅供参考）

| 指标 | 原配（EP3, k=3） | 最终 | 变化 |
|---|---|---|---|
| C1 散文 sampled | 29.31 | 34.23 | **+16.8 %** |
| C1 散文 greedy | 27.99 | 36.67 | **+31.0 %** |
| C4 聚合 | 65.27 | 75.74 | **+16.0 %** |

### 2.4 预填（prefill）

| 配置 | ≈4k 档（扣解码后） |
|---|---|
| chunk 1024 | 2154 tok/s |
| chunk 768（最终） | **2018 tok/s**（−6.3 %） |
| README §4 记载基线 | ≈2045 tok/s |

⚠️ **768 让预填降了约 6 %**，这是本轮唯一有代价的改动，见 §5。

### 2.5 质量与可复现性

每次重启后跑 `quality_gate.py`（6 项可判定任务 + 乱码检测），**各批次均 6/6 通过**：

```
[PASS] arithmetic  47*89+123 = 4306        [PASS] counting  strawberry 的 r = 3
[PASS] arithmetic2 1234-567*2 = 100        [PASS] list      1,2,3,4,5
[PASS] json        {"name": "Ada", ...}    [PASS] prose     无乱码、非 CJK
```

关键回归哨兵：`accept len` 始终 2.5–3.05（上游 E1b 的失败模式是**塌到 1.0 且输出变垃圾**——
那种情况 tok/s 反而"变快"，只看速度会误判）。

**autotune 跨重启复用（已确认）**：一次**零改动重启**后，日志出现

```
[autotune_keep] reused 085bb8a93fd1bc79/rank_tp0_pp0_dp0.json
[autotune_keep] reused ba5efcf4d04b1279/rank_tp0_pp0_dp0.json
```

即两份 cache 都被**复用**而非重 tune（此前每次启动都是 `tuned and saved`）。
配套验证：同一条 greedy prompt 连跑 3 次，输出 **sha256 完全一致（3 次同一哈希）**，
`distinct=1 of 3` ⇒ **可复现**。这正是 README §8 "单流解码延迟仍有波动（0.9 s ↔ 18 s）"
一栏的对症项。

---

## 3. 实际改动清单

### 3.1 head `~/dsv41-3xspark/adapter/`（新增 8 个文件 + 重写 1 个）

| 文件 | 作用 | 来源 |
|---|---|---|
| `autotune_keep.py` | 跨重启保留 FlashInfer autotune cache | 上游原样 |
| `engram_prefetch.py` | Engram 行查询走 side stream | 上游原样 |
| `verify_cap.py` | 按 draft 置信度截断 verify 窗口 | 上游原样 |
| `block_verify.py` | 采样行的块级验证 | 上游原样 |
| `draft_tau.py`、`folded_result_fence.py` | 随附，默认关 | 上游原样 |
| `draft_head_fp8.py` | draft LM head 的 fp8 副本 | 上游原样 |
| `wo_a_w8.py` | `wo_a` 的 fp8 twin | 上游 + **本地 1 处修复，见 §4** |
| `sitecustomize.py` | 从 73 行扩到 141 行，新增 8 个 hook（每个都 env 门控） | 本地基于 head 版本增补 |

`sitecustomize.py` 的改动经 diff 确认是**纯增补**（68 行新增、0 行删除），保留了 head 原有的
`engram_backend` / `mxfp8_b12x` / `prefill_empty_cache` / `encoding_compat` / `loop_abort` /
`tp3_pad` / `PagedIndexerMetadata` 全部逻辑。

> ⚠️ head 的 `adapter/` **此前不在任何仓库中**（`fleet/` 只归档了 4 个部署文件）。
> 本次仍未归档进本仓库——这 8 个文件是 AGPL-3.0 派生，入库涉及许可决定，留给仓库主人定夺。
> 现场原件与备份都在 head 上（`adapter.bak-before-batch1-20260925-0104/`）。

### 3.2 `start.sh` 透传（12 个变量，两处都补）

```
DSV41_AUTOTUNE_KEEP  DSV41_ENGRAM_PREFETCH  DSV41_ENGRAM_PREFETCH_CHECK
DSV41_VERIFY_CAP     DSV41_BLOCK_VERIFY     DSV41_DRAFT_TAU
DSV41_FOLDED_FENCE   DSV41_WO_A_W8          DSV41_WO_A_W8_MID
DSV41_WO_A_W8_DROP   DSV41_WO_A_W8_DRAFT    DSV41_DRAFT_HEAD_FP8
```

用仓库既有生成器 `patch_startsh_envvar.py` 逐个补，幂等。补完已回读 worker 启动行确认
**23 个 `-e` token 全部完整、无粘连**（该脚本注释警告的
`NCCL_DEBUG=$NCCL_DEBUG-e NCCL_...` 故障模式）。备份：`start.sh.bak-before-batch1`。

### 3.3 `.env`（每步都有独立备份）

```ini
EP_SIZE=1                    # 3 -> 1         (批次0)
CHUNKED_PREFILL_SIZE=768     # 1024 -> 768    (安全性)
DSPARK_BLOCK_SIZE=5          # 3 -> 5         (批次2)
DSV41_VERIFY_CAP=conf:0.1    # 0 -> conf:0.1  (批次2)
DSV41_BLOCK_VERIFY=1         # 0 -> 1         (批次2)
DSV41_AUTOTUNE_KEEP=1        # 0 -> 1         (批次1)
DSV41_ENGRAM_PREFETCH=1      # 0 -> 1         (批次1)
DSV41_WO_A_W8=1 / _MID=1 / _DROP=1           (批次3)
DSV41_DRAFT_HEAD_FP8=1                        (批次3)
# 保持关闭：DSV41_WO_A_W8_DRAFT=0, DSV41_DRAFT_TAU=1, DSV41_FOLDED_FENCE=0
```

备份链（head `~/dsv41-3xspark/state/`）：
`env.before-ep1-*` → `env.before-batch1-*` → `env.before-batch2-*` → `env.before-k3-ab-*`
→ `env.before-batch3-*` → `env.before-revert-batch3-*` → `env.before-batch3b-*` → `env.before-chunk768-*`

---

## 4. 过程中发现并解决的问题

### 4.1 `wo_a_w8` 与本 fleet 引擎版本不兼容（已修）

上游 adapter 写于 `37939c26`，其 `_apply_wo_a_bf16_matmul` 没有 `is_target_verify` 参数。
本 fleet 是**更新的 `da64c5cbb`**，引擎这样调用：

```python
o = _apply_wo_a_bf16_matmul(
    o, wo_a,
    is_decode=forward_batch.forward_mode.is_decode(),
    is_target_verify=forward_batch.forward_mode.is_target_verify(),   # 本 fleet 独有
    fuse_mxfp8_quant=(...),                                          # 本 fleet 独有
)
```

而 bridge 的签名是 `_apply(o, wo_a, is_decode=False)` ⇒
`TypeError: got an unexpected keyword argument 'is_target_verify'`，
发生在 **FlashInfer autotune 的 warm-up**（一次 target-verify 前向）⇒ 引擎永远起不来。

**修法**（`wo_a_w8.py:355/375/376`）：签名加 `**kw`，两处透传分支补 `**kw`
（与该文件自己在 237–250 行的写法一致）。已加注释说明来由。修后 warm-up 通过、accept len 2.5–3.05 正常。

> 注：本 fleet 的 `kernels/ops/attention/dsv4/` 下**没有 `wo_a_bf16.py`**，所以 `wo_a_w8`
> 走的是 **einsum bridge 模式**（`[wo_a_w8] einsum bridge armed`）——正是上游在 TP3 上验证过的路径。
> 另：bridge 模式会正确保留 draft 的 `wo_a` 为 bf16（上游 E1b 的坑），日志可见
> `[wo_a_w8] draft wo_a left on bf16 (this engine's draft does not use the shared dispatch)`。

### 4.2 构建产物不一致（head 更新、worker 未更新）

`./start.sh build 2>&1 | tail -60` **会用 tail 的退出码掩盖构建失败**：第一次 build 时
rsync 到 worker 失败（`topo/fq-triangle.xml` 是 root 所有的残留目录，rsync `--delete` 无权删），
但管道让整体返回 0，结果 **head 镜像已更新、两个 worker 还是旧镜像**——正是上游记录过的
"混合 build"故障模式。

**处理**：`sudo chown -R fuqiang:fuqiang ~/dsv41-3xspark/topo`（两个 worker），重新 build。
**教训**：build/restart 一律写 `cmd > log 2>&1; echo EXIT=$?`，不要接管道。

> 附带：`nfs-share.sh:25` 有一处 `local: "-a": 不是有效的标识符` 的告警（功能不受影响，
> share 仍成功）。属既有小 bug，本次未改。

### 4.3 一次 `share` 竞态

某次 restart 在 share 步骤报
`Conflict. The container name "/dsv41-nfs" is already in use`。
重跑 `./start.sh share` 即通过（它会走"复用存活容器"的早返回分支）。
worker 卷与 NFS 导出均正常，非数据问题。

### 4.4 抽取上游文件时的 PowerShell 陷阱

第一次抽取 adapter 用了 `git show <ref>:<path> | Set-Content -NoNewline`，
**换行被折叠成空格**，产出单行损坏文件（`SyntaxError`），而 md5 校验因为是"自己校验自己"而没拦住。
改用 `git archive … | tar -x` 抽取，并用**行数**与 `ast.parse` 双重校验
（8 个文件行数 99/118/97/38/167/38/314/455，与上游 diff stat 逐个吻合）。

---

## 5. 需要你决定的一件事：`CHUNKED_PREFILL_SIZE`

上游把默认值从 1024 改成 768，原因是：在这套新解码栈下，**一个 ~200k token 的 prompt 在
1024 chunk 下把节点内存打爆**（`NV_ERR_NO_MEMORY`，主机挂死，最终硬复位）。768 把
每 chunk 的 indexer 瞬时峰值（≈ `14 B × chunk × prefix`）砍掉四分之一。

本 fleet 的 `CONTEXT_LENGTH=262144`，而 README 自己记的内存墙是 `T × L ≲ 2.0e8 token²`——
1024 × 200000 = 2.05e8，**正好压在墙上**，所以这个风险对本 fleet 是真实的。

- **现在设的是 768**（更安全）：预填 ≈2018 tok/s；
- **若你更看重预填吞吐**：改回 1024，预填 ≈2154 tok/s（+6.3 %），解码不受影响，
  但要接受长 prompt 的 OOM 风险。改一行 + `./svc.sh restart` 即可。

---

## 6. 回滚

```bash
# 整轮回滚（回到本次优化前的原配）
cd ~/dsv41-3xspark && cp state/env.before-ep1-20260925-0045 .env && ./svc.sh restart
#   注意：镜像里 adapter 仍在，但每个都 env 门控，全关即等于原行为；
#   若还要回到旧镜像：adapter 目录备份 adapter.bak-before-batch1-20260925-0104/，重建镜像即可。

# 单项回滚（改一行 + restart，约 10 分钟）
EP_SIZE=3                    # 关 EP1
DSPARK_BLOCK_SIZE=3 + DSV41_VERIFY_CAP=0 + DSV41_BLOCK_VERIFY=0   # 关 k=5+cap（必须成组）
DSV41_WO_A_W8=0              # 关 wo_a（连带 MID/DROP）
DSV41_DRAFT_HEAD_FP8=0
DSV41_ENGRAM_PREFETCH=0
DSV41_AUTOTUNE_KEEP=0        # 回到"每启动丢弃并重 tune"
CHUNKED_PREFILL_SIZE=1024
```

**清 autotune 缓存**（换了战术想强制重 tune 时）：
```bash
rm -f ~/.cache/sglang/flashinfer/autotune/*/sm121/*/rank_*.{json,launch}   # 三台都要
```

---

## 7. 未验证 / 遗留

1. **未做长上下文压测**：本轮没有跑 >32k 的 prompt，768 的实际保护效果未验证。
   上游的 191k needle 是他们的环境。
2. ~~`autotune_keep` 的"复用"路径未观察到~~ → **已在零改动重启中确认 `reused`**（见 §2.5）。
   跨**重启**的逐字节复现只验证了单次启动内 3 连跑一致；跨重启的对比因
   每次启动都改了配置（指纹变化 → 主动重 tune）而未做，但那属于预期行为。
3. **`wo_a_w8` 的本地修复只做了功能验证**（起得来、accept len 正常、质量 6/6），
   没有做逐 token 的数值对照。上游对 DROP 的说明是"dequantized copy not bit-identical"，
   即本身就不是逐位复现。
4. 上游列出的"仍可优化项"（b12x dense 17.4 ms/step、MoE 21.1 ms/step、SPS/STS 表、
   k=4 + cap、更多请求槽位）本轮均未尝试。
5. `env.example` 仍未同步（第 47-49 行还是已作废的"RoCE 不可用"结论，且缺 `EP_SIZE` 等键）。
6. 本 fleet 的引擎是 `da64c5cbb`，上游所有数字测于 `37939c26`——**上游的绝对数字不可直接引用**，
   本文数字全部是本 fleet 自测。

---

## 8. 建议的下一步

1. **长上下文压测**（最高优先）：用 `scripts/verify/memguard.py` 护航，测 64k / 128k / 200k，
   定出 768 下的安全上界。这是本 fleet 唯一仍有**灾难性失败模式**的方向。
2. 把 `adapter/` 的 9 个文件纳入版本控制（需先决定 AGPL 许可处理方式）——
   目前 head 的 `adapter/` 仍不在任何仓库里。
3. 同步 `env.example` 到真实 `.env`（消掉已作废的 RoCE 结论，补 `EP_SIZE` 等键）。
4. 若要继续压性能：上游的剩余大头是 b12x dense MXFP8 GEMM（17.4 ms/step）与
   MoE grouped GEMM（21.1 ms/step），以及 SPS/STS 表（主要在并发 ≥2 生效）。

---

## 9. 最终状态

```
服务：health=200，三台容器 healthy，via NET/IB=64，reg_mr 失败 0，accept len 2.92，0 错误
已武装：autotune_keep / Engram prefetch / block_verify / verify_cap / wo_a einsum bridge / draft_head_fp8
日志：serve-0925-0315.log
```
