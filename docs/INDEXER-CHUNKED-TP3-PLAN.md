# indexer_chunked 移植可行性评估（TP3 / 3× DGX Spark）

> 状态：**已于 2026-09-26 完成真机实测并落地。结果见
> [`INDEXER-CHUNKED-TP3-RESULTS.md`](INDEXER-CHUNKED-TP3-RESULTS.md)（本文的实测续篇）。**
> 探测结论 `V1`；质量零退化（BROKE=0, p=1.000）；预填在 `chunk=1024` 下 2197 tok/s；
> 254,811 token 长上下文单次成功。**实测中发现一个本文未预见的关键机制**：
> adapter 是 `COPY` 进镜像的，改 adapter 必须 `./start.sh build` 重建镜像，只重启无效
> ——详见结果文档 §1。
>
> 撰写日期：2026-09-26。所有结论均标注证据出处；未验证项集中在结果文档 §7。

## 0. 结论（已实测，不再是推断）

**探测结果：`V1`** —— head `fq-dgx-01` 上的实机探测（2026-09-26），证据：

| 项 | 实测值 | 出处 |
|---|---|---|
| 引擎 build | `0.0.0.dev1+gda64c5cbb` | 探测输出 `dist:sglang` |
| 后端文件 | `/sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py` | 探测输出 `module_file` |
| `self.candidate_masks` | **True** | 源码第 41、68 行 |
| `candidate_metadata` | **False** | 全文无 |
| `_publish_or_consume_candidates` | **True** | 源码第 68 行 |
| `_DENSE_INDEXER_LOGITS_BUDGET_BYTES` | **False** | 即**尚未**原生带 #39187 |
| v1 守卫 | **通过**（`v1_missing` 为空） | |
| v3 守卫 | 不通过（缺 `mask_topk_scores`/`CandidateMasks`/`published_masks`） | |

⇒ **走 v1（`adapter/indexer_chunked.py`）路线。v3 不可用。**

v1 所需符号在容器内**全部存在**（实测）：
`_dense_fp4_mqa_logits`、`topk_transform_ragged_v2`、`_mask_topk_scores`、
`select_candidate_blocks`、`ceil_align`、`_as_int_list`、`DeepseekV4AttnBackend`、
`cls._low_ratio_index_topk_dense`、`quantize_fp4_indexer_tensor` —— 均 True/OK。

**而且引擎尚未原生携带 #39187**，所以这个 backport 不是重复劳动，确实有收益空间。

现状（`head:~/dsv41-3xspark/.env` 实测）：`CHUNKED_PREFILL_SIZE=768`、`EP_SIZE=1`；
`adapter/` 下**没有** indexer 相关文件。即当前正承受那 −6.3% 的预填损失。

---

## 0.1 一句话结论（原始判断，已被上面实测确认）

上游 TP4 线的 14 个新 adapter 中，**只有 `indexer_chunked`（v1/v3）对 TP3 有真实价值**：
它从源头削掉 dense prefill indexer 的 fp32 峰值显存，直接对应本仓库 README §8
「长上下文未压测 / `CHUNKED_PREFILL_SIZE=768` 的保护效果未验证」这一未结项，
并有望把那 −6.3% 的预填损失换回来。

---

## 1. 为什么是它：问题与收益

### 1.1 现状（本仓库已知）

| 事实 | 出处 |
|---|---|
| `CHUNKED_PREFILL_SIZE` 从 1024 降到 768，预填 −6.3%（2154 → 2018 tok/s） | `README.md:161` |
| 降它的原因是"换取长 prompt（~200k）的 OOM 余量" | `README.md:165-166` |
| **长上下文从未压测**，`CHUNKED_PREFILL_SIZE=768` 的保护效果未验证 | `README.md:266-267` |
| 上游在该解码栈 + chunk 1024 下被 ~200k prompt 打爆（`NV_ERR_NO_MEMORY`，主机挂死需硬复位） | `README.md:167` |
| 已知 ≤32k 安全 | `README.md:267` |

即：**当前是用降低 chunk 换安全，付出了 6.3% 的性能，而且这份"安全"没有被验证过。**

### 1.2 该 adapter 做什么

它把 stock 的 `_low_ratio_index_topk_dense` 从"整个 prefill chunk 对全压缩上下文打一个
fp32 `[T, lc]` 大张量"改成"按行分块、预算封顶"，掩码直接写进预分配张量：

- 峰值 transient ≈ `14 B × chunk × prefix` per rank（`indexer_chunked.py:5-6`）
- 上游实测（4× GB300）：**300K cold prompt transient 49.7 → 10.7 GB**，
  1M prompt 降到 9.6 GB，**prefill 时间不变到更好**，
  且 `page_indices / raw_indices / masks` 与原路径 **bitwise 相等**
  （依据：DeepGEMM `fp8_fp4_mqa_logits` 对行数不变）——`indexer_chunked.py:12-14`

最后一条是关键：**它不是近似，是等价重写**。这正是它可以用来"把 chunk 调回 1024"的前提。

---

## 2. 兼容性门：必须先探测的事实

### 2.1 两个变体互斥

| | v1 (`indexer_chunked.py`) | v3 (`indexer_chunked_v3.py`) |
|---|---|---|
| 目标 | 镜像 `e087e662` | 分支 `f80c91a4b`+ |
| 掩码存放 | `self.candidate_masks`（list） | `forward_metadata.candidate_metadata`（`CandidateMasks`） |
| helper | `_mask_topk_scores` | `mask_topk_scores`、`published_masks` |
| 附加上游依赖 | 无 | 可选 SG18 prefill TP split（`SPARK_PREFILL_TP_SPLIT=1`，依赖 `spark_prefill_dense.py`） |
| 出处 | `indexer_chunked.py:16-19, 217-238` | `indexer_chunked_v3.py:7-13, 352-360` |

两者各自的 `install()` 都有守卫，判据**互斥**：v1 要求源码含 `self.candidate_masks`；
v3 要求含 `candidate_metadata` **且不含** `self.candidate_masks`。装错的那个会
`raise RuntimeError(...); refusing to boot`（`indexer_chunked.py:235-238`、
`indexer_chunked_v3.py:356-357`）。

**这是好设计**：失败模式是**启动时大声拒绝**，不是静默算错。

### 2.2 本集群的引擎版本是第三个值（已实测确认）

`docs/ADAPTER-MIGRATION-PLAN.md:17` 记录本 fleet 引擎 build 是 **`da64c5cbb`**，
且该文件 `:152` 明确写着"**未验证**：本 fleet 是 `da64c5cbb`"。
上游两个 adapter 的目标分别是 `e087e662` 与 `f80c91a4b` —— **都不是 `da64c5cbb`**。

⇒ 这是为什么不能推断、只能探测。**探测已于 2026-09-26 完成，结果为 `V1`**（见 §0）。

### 2.3 探测工具与实测结果

`scripts/probe_indexer_variant.sh`（**只读**）：

```bash
# 在 head 上、与 start.sh/.env 同目录，服务运行中执行
bash probe_indexer_variant.sh --dry-run     # 先看要跑什么
bash probe_indexer_variant.sh               # 真跑
```

它复刻了两个 adapter 的守卫判据（逐字对应），输出 6 种结论之一：

| verdict | 含义 | 后续 |
|---|---|---|
| `V1` | 命中 v1 守卫 | 走 v1 路线 ← **本集群实测结果** |
| `V3` | 命中 v3 守卫 | 走 v3 路线（注意 SG18 依赖） |
| `NATIVE` | 引擎已原生带 #39187 | **无需移植**，仅需验证并考虑调回 1024 |
| `NEITHER` | 两者都不满足 | **停手上报**，函数已被上游改写 |
| `AMBIGUOUS` | 两者都过 | 人工读源码（当前判据下实际不可达） |
| `REFUSE` | 类/方法不存在 | 停手 |

退出码：`0`=V1/V3/NATIVE｜`2`=容器不可用｜`3`=import 失败｜`4`=NEITHER/REFUSE｜`5`=AMBIGUOUS｜`6`=未判定。

**已实机执行（2026-09-26，head `fq-dgx-01`，服务运行中）**：

```
结论：V1  → 可移植 indexer_chunked.py（v1）        EXITCODE=0
```

关键实测值（原始 JSON 输出）：

```
dist:sglang              = 0.0.0.dev1+gda64c5cbb
module_file              = /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py
uses_self_candidate_masks= true        (源码第 41、68 行)
uses_candidate_metadata  = false
uses_publish_or_consume  = true        (源码第 68 行)
already_has_39187_budget = false       ← 引擎尚未原生带 #39187，backport 不是重复劳动
v1_guard_passes          = true        (v1_missing 为空)
v3_guard_passes          = false       (缺 mask_topk_scores / CandidateMasks / published_masks)
```

v1 所需符号在容器内**全部存在**（实测逐项 True/OK）：
`_dense_fp4_mqa_logits`、`topk_transform_ragged_v2`、`_mask_topk_scores`、
`select_candidate_blocks`、`ceil_align`、`_as_int_list`、`DeepseekV4AttnBackend`、
`cls._low_ratio_index_topk_dense`、`quantize_fp4_indexer_tensor`。

探测对服务**无影响**：前后 `/health` 均 **200**，`dsv41-head` 状态 `Up 3 hours (healthy)` 未变。

> 📌 执行中发现并修掉脚本自身两个缺陷（已修复并复验）：
> 1. `rec()` 无返回值却被写成 `not rec(...)` —— 会在真机上 `TypeError` 崩溃；
> 2. **verdict 提取在本机 grep/sed 上失效**：head（GNU grep 3.11）上任何含双引号的
>    `grep -o` / `grep -oE` 模式实测 **0 命中**，`sed -n "s/...\"...\"/\1/p"` 同样 0 命中
>    （同一输入下 `awk -F'"'` 与 `grep`+`cut` 均正确）。已改用 `awk`，并锚定
>    `^[[:space:]]*"verdict":` 避免命中 `verdict_reason`。
>    **这是"脚本跑通、退出码 0，但结论判错"的典型陷阱，建议记入 PITFALLS。**

现状（head `.env` 实测）：`CHUNKED_PREFILL_SIZE=768`、`EP_SIZE=1`；
`adapter/` 下**没有** indexer 相关文件 —— 即当前正承受那 −6.3% 的预填损失。

---

## 3. 移植改动清单（按 V1 路线）

### 3.1 需改动的文件

| 文件 | 改动 | 风险 |
|---|---|---|
| head `~/dsv41-3xspark/adapter/indexer_chunked.py` | 新增该文件（v1，249 行，自包含：仅 stdlib + torch） | 低（新文件，head 上不存在，无覆盖） |
| head `adapter/sitecustomize.py` | 加 1 个 `elif` + finder 白名单 1 行（**纯增补 13 行**，草稿见 §3.2） | 中——须放在 `else` 之前，否则静默误伤（§3.2.2） |
| `start.sh` | 用既有生成器补 2 个变量透传（两处，**3 增 1 改**，草稿见 §3.3） | 低，幂等；须实测 token 数防粘连 |
| `.env` | `DSV41_INDEXER_CHUNKED=1`（+可选 budget） | 低 |
| `fleet/start.sh`（归档） | **顺带修正**：现归档已过期 13 行（§3.3.3） | 低，但建议同批做 |
| `fleet/adapter/*`（归档） | 同步新文件快照 + md5 | 低 |
| `NOTICE`、`LICENSE.AGPL` 声明 | 新文件为 AGPL-3.0-or-later 派生 | 低（合规必做） |

> 容器内 adapter 实际路径为 `/opt/dsv41/adapter`（`sys.path` 实测），对应 head 的
> `~/dsv41-3xspark/adapter/`。改 head 侧文件后由既有挂载机制带进容器。

### 3.2 具体注入改动（**草稿 diff 已产出并验证**）

上游 `sitecustomize.py:60-73` 把 indexer 分支整条挂在 `_tp4_launcher()` 下：

```python
elif module.__name__ == 'sglang.srt.layers.attention.deepseek_v4_backend' and _tp4_launcher():
```

本集群的 `fleet/adapter/sitecustomize.py` 走的是**无条件**列表式 finder
（`sitecustomize.py:112-133`），其中**尚不含** `deepseek_v4_backend`，
但**已含** `dsv4.metadata`。

⇒ 移植需在 finder 列表加入 `sglang.srt.layers.attention.deepseek_v4_backend`，
并在 `EngramLoader` 里加一个**独立于 TP4 门控**的 indexer 分支，沿用本仓库自己的
env 门控写法（`DSV41_INDEXER_CHUNKED` 命中才 import）。

> ⚠️ 不要照抄上游的 `_tp4_launcher()` 结构——本集群没有 `DSV41_LAUNCHER` 变量，
> 照抄会导致分支永不触发（静默不生效），这比报错更难排查。

#### 3.2.1 草稿 diff（对 `fleet/adapter/sitecustomize.py`，**纯增补 13 行 / 删 0 行**）

```diff
@@ -93,6 +93,18 @@ class EngramLoader(importlib.abc.Loader):
             if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                 from verify_cap import install_planner as install_verify_cap_planner
                 install_verify_cap_planner(module)
+        elif module.__name__ == 'sglang.srt.layers.attention.deepseek_v4_backend':
+            # indexer 2026-09-26: bound the dense prefill indexer transient (sglang#39187
+            # backport, adapter/indexer_chunked.py). The stock method scores a whole prefill
+            # chunk against the full compressed context in ONE fp32 [T, lc] tensor; the
+            # ~200k-prompt OOM margin is currently bought by lowering CHUNKED_PREFILL_SIZE
+            # to 768. This scores in row chunks instead. Gated on DSV41_INDEXER_CHUNKED.
+            # The adapter re-checks its own guard and raises (refusing to boot) if the
+            # backend has drifted to the candidate_metadata variant -- so a mismatch here
+            # is a loud boot failure, not silent corruption.
+            if os.environ.get('DSV41_INDEXER_CHUNKED', '0').strip() not in ('0', 'off', 'false', ''):
+                from indexer_chunked import install as install_indexer_chunked
+                install_indexer_chunked(module)
         else:
             # V4.1 ratio-1/2 indexers always call the FP4 DeepGEMM kernel.
             # SM120 needs its split-128 planner even when the legacy FP8
@@ -125,6 +137,7 @@ class EngramFinder(importlib.abc.MetaPathFinder):
                             'sglang.kernels.ops.moe.moe_fused_gate',
                             'sglang.srt.speculative.dspark_components.dspark_draft',
                             'sglang.srt.speculative.dspark_components.dspark_planner',
+                            'sglang.srt.layers.attention.deepseek_v4_backend',
                             'sglang.srt.layers.attention.dsv4.metadata'):
             return None
         spec = importlib.machinery.PathFinder.find_spec(fullname, path)
```

#### 3.2.2 为什么新 `elif` 必须放在 `else` 之前

现有的 `else` 分支（`fleet/adapter/sitecustomize.py:96-108`）**无条件**取
`module.PagedIndexerMetadata` 并改写其 `__post_init__`。

实测（容器内）：`deepseek_v4_backend` **也有** `PagedIndexerMetadata` 属性 ——
所以若新分支不存在，`deepseek_v4_backend` 一旦被 finder 放行、就会**落进这个 else**，
被错误地当成 indexer metadata 模块打补丁。**这不是崩溃，是静默改错对象**，更难发现。

⇒ 两处改动**必须同时上**：只加 finder 白名单而不加 `elif`，会产生上述静默误伤。

#### 3.2.3 草稿的验证记录（均已在真机跑过）

| 检查 | 方法 | 结果 |
|---|---|---|
| 语法 | `ast.parse` | OK（141 → 154 行） |
| 纯增补 | `git diff --stat` | **13 insertions(+), 0 deletions** |
| 分支命中 | 容器内比对 `module.__name__` | `deepseek_v4_backend` 命中新分支 = **True** |
| 不误伤 | 同上 | `dsv4.metadata` 仍落 `else` = **True** |
| 门控行为 | 遍历 5 个取值 | `0`/`off`/空 → 不触发；`1`/`on` → 触发 ✅ |
| 守卫可用 | 复刻 adapter `install()` 检查 | `missing=[]`、v1 守卫 = True ⇒ 不会 raise |

> 说明：草稿 diff 保存在会话临时目录，**未写入本仓库、未上传 head**。
> 容器内 adapter 路径为 `/opt/dsv41/adapter`（`sys.path` 实测，对应 head 的
> `~/dsv41-3xspark/adapter/`），新文件放该目录即可被 `from indexer_chunked import ...` 找到。

### 3.3 `start.sh` 透传（**草稿已产出并实证**）

`start.sh` 逐条列举它转发的每个变量，**不在清单里的一律静默丢弃**——`.env` 改了看起来
像没生效，还要花 13 分钟启动才能证伪（该警告写在 `scripts/patch_startsh_envvar.py:4-6`）。

用仓库**既有幂等生成器**补，**不要手改**（它会同时处理 head 与 worker 两处）：

```bash
cd ~/dsv41-3xspark
cp start.sh start.sh.bak-before-indexer-$(date +%Y%m%d-%H%M)   # 先备份
python3 patch_startsh_envvar.py DSV41_INDEXER_CHUNKED 0
python3 patch_startsh_envvar.py DSV41_INDEXER_LOGITS_BUDGET_BYTES ""
```

生成器幂等：变量已存在时打印 `already contains ...` 并退出 0。

#### 3.3.1 实测生成的 diff（针对 head 当前 `start.sh`，901 行 / md5 `bf07deab22543685b2d2efccdf1e39c4`）

```diff
@@ -292,6 +292,8 @@ docker_common_args() {
     -e "NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-0}"
     -e "NCCL_IB_SUBNET_AWARE_ROUTING=${NCCL_IB_SUBNET_AWARE_ROUTING:-1}"
     -e "NCCL_CUMEM_ENABLE=0"
+    -e "DSV41_INDEXER_LOGITS_BUDGET_BYTES=${DSV41_INDEXER_LOGITS_BUDGET_BYTES:-}"
+    -e "DSV41_INDEXER_CHUNKED=${DSV41_INDEXER_CHUNKED:-0}"
     -e "DSV41_DRAFT_HEAD_FP8=${DSV41_DRAFT_HEAD_FP8:-0}"
@@ -405,7 +407,7 @@ worker_env_lines() {
         -e NCCL_IB_SUBNET_AWARE_ROUTING=${NCCL_IB_SUBNET_AWARE_ROUTING:-1} \
-        -e NCCL_CUMEM_ENABLE=0 -e DSV41_DRAFT_HEAD_FP8=${DSV41_DRAFT_HEAD_FP8:-0} ...
+        -e NCCL_CUMEM_ENABLE=0 -e DSV41_INDEXER_LOGITS_BUDGET_BYTES=${DSV41_INDEXER_LOGITS_BUDGET_BYTES:-} -e DSV41_INDEXER_CHUNKED=${DSV41_INDEXER_CHUNKED:-0} -e DSV41_DRAFT_HEAD_FP8=${DSV41_DRAFT_HEAD_FP8:-0} ...
```

共 **3 insertions / 1 deletion**（worker 那行是整行替换，内容上只增不减）。

#### 3.3.2 验证记录（均在本地草稿上实跑）

| 检查 | 方法 | 结果 |
|---|---|---|
| 生成器锚点仍命中 | 在 head 的 `start.sh` 上实跑生成器 | `anchor1`/`anchor2` 各 1 次，**通过** |
| 幂等性 | 目标变量查重 | 二者原本均不存在，可安全执行 |
| `bash -n` | 语法检查 | exit 0 |
| **worker 行未粘连** | 统计 `-e ` token 数 | **23 → 25**，正好 +2 ✅ |
| **透传真的生效** | `eval` 真实展开该行参数 | `DSV41_INDEXER_CHUNKED=1`、`DSV41_INDEXER_LOGITS_BUDGET_BYTES=2147483648` **都在参数里**（共 50 个参数）✅ |
| 回落行为 | 不设环境变量 | `DSV41_INDEXER_CHUNKED=0`、`DSV41_INDEXER_LOGITS_BUDGET_BYTES=`（空）✅ |

> ⚠️ 生成器自身注释警告：`bash -n` **查不出**粘连故障（`NCCL_DEBUG=$NCCL_DEBUG-e NCCL_...`
> 这种语法合法但语义错误）。所以必须像上表那样**实测 token 数并真实展开**，不能只靠 `bash -n`。
> 本次已做（23→25）。

#### 3.3.3 ⚠️ 归档漂移：`fleet/start.sh` 不是 head 的现状

起草时发现 **`fleet/start.sh` 已过期**，这不是本次改动引入的，但会误导后续所有人：

| | head 实际 | `fleet/start.sh` 归档 |
|---|---|---|
| 行数 | **901** | 889 |
| md5 | `bf07deab22543685b2d2efccdf1e39c4` | `13512310ca325dc400b24e7247f052fc` |
| 12 个 batch-1 透传<br>（`DSV41_AUTOTUNE_KEEP`、`DSV41_VERIFY_CAP`、`DSV41_WO_A_W8*`、`DSV41_DRAFT_HEAD_FP8`、`DSV41_ENGRAM_PREFETCH*` 等） | **有**（`start.sh:297-306`、`:408`） | **完全没有**（grep 命中 0） |

原因：`fleet/README.md:10` 声明的 md5/行数对应的是**batch-1 迁移之前**的版本
（`docs/BATCH-MIGRATION-2026-09-25.md:118-129` 记录的 12 个透传改动只落在 head，未回收进 `fleet/`）。

⇒ **两条行动项（独立于本移植）**：
1. 用 head 现版重新归档 `fleet/start.sh` 并更新 md5/行数；
2. 在 `fleet/README.md:10` 的说明里补一句"该行 md5 为 batch-1 之前版本"或直接更正。

> 这也解释了为什么本次透传草稿必须以 **head 的 901 行版**为基准：拿 889 行的旧归档去
> 生成 diff 会得出错误的锚点上下文。

#### 3.3.4 `.env` 改动（同批）

```ini
DSV41_INDEXER_CHUNKED=1
# 可选：默认 2 GiB（indexer_chunked.py:31 DEFAULT_BUDGET_BYTES = 1 << 31）
# DSV41_INDEXER_LOGITS_BUDGET_BYTES=2147483648
```

`DSV41_INDEXER_CHUNKED=0` 即为关闭开关，**无需回滚代码**。

### 3.4 回滚

- 每个文件改前备份（沿用本仓库既有习惯：`*.bak-before-*`）。
- `DSV41_INDEXER_CHUNKED=0` 即可**在不回滚代码的情况下**关掉。（`indexer_chunked.py:214`）
- 最坏情况：`./svc.sh stop` → 还原 `adapter/` 与 `start.sh` → `./svc.sh start`（约 13–15 分钟）。

---

## 4. 执行顺序（每步独立可验证、可回滚）

1. **探测**（本文 §2.3）。只读，无需停服。
   - `NEITHER`/`AMBIGUOUS` → **停止**，转为向上游报告。
2. **基线重测**（移植前，同口径）。必须先有基线，否则无法判断收益与退化。
   - `python3 scripts/bench_migration.py`（本仓库既有，唯一 prompt 口径）
   - 记录 C1 散文/代码、C4 并发、预填 tok/s
3. **单文件落地 + 门控开启**，只跑一次启动验证：
   - `./svc.sh start`，确认日志出现 adapter 的 `ARMED` 行（v3 的 `indexer_chunked_v3.py:377`）
   - 若出现 `refusing to boot` → 探测结论与实机不符，**回滚**并回到步骤 1
4. **质量门对照**（上游脚本可复用）：`scripts/qeval.py` 是 55 题配对评分 + McNemar 检验
   （`qeval.py:9, 98, 140-184`），比现有 `verify_extras.py` 更系统。
5. **性能对照**：重跑步骤 2 的同一命令，**同口径比较**。
6. **长上下文压测**（本次移植的真正目的）：~200k prompt。
   - 通过 → 可尝试把 `CHUNKED_PREFILL_SIZE` 调回 1024 并重测预填（拿回那 6.3%）。
   - 不通过 → 保留 768，移植收益仅剩"显存余量"，需重新评估是否值得留。

---

## 5. 成本与风险

### 成本估算

| 阶段 | 估时 | 备注 |
|---|---|---|
| ~~探测~~ | ~~~2 分钟~~ | **已完成（2026-09-26），V1** |
| 基线重测 | ~1 小时 | 含一次 `svc.sh start`（13–15 min） |
| 落地 + 启动验证 | ~1 小时 | |
| 质量门（55 题） | ~1–2 小时 | 取决于并发 |
| 性能 + 长上下文压测 | ~2–3 小时 | 长 prompt 本身慢 |
| **剩余合计** | **约 1 个工作日** | 不含上游回报往返 |

### 主要风险

1. **`da64c5cbb` 是"会漂移"的构建**（`ADAPTER-MIGRATION-PLAN.md:17` 转述上游 `5757d1b` 的说法）。
   上游 adapter 写于别的 commit，本集群已因 build 差异踩过一次坑——
   `wo_a_w8.py` 的 `is_target_verify` kwarg 不匹配导致引擎起不来
   （`fleet/adapter/wo_a_w8.py:356-367`）。**indexer 是同类风险，且有前例。**
   探测已确认**接口层面匹配**（符号齐全、守卫通过），但**行为层面是否等价仍未验**。
2. **长上下文压测本身有硬复位风险**：上游在这条线上被 ~200k prompt 打爆过主机
   （`README.md:167`）。压测应在**可接受硬复位的窗口**内做。
3. **预填是算力受限**（`README.md:141`）。削显存不必然提速；收益主要来自"能调回 1024"。
   若最终不能调回，移植的净收益接近零。
4. **许可**：新文件是 AGPL-3.0-or-later 派生，需同步 `NOTICE` 与 `LICENSE.AGPL` 声明
   （沿用 `fleet/README.md:51-56` 的既有做法）。

---

## 6. 工具验证状态（诚实声明）

`scripts/probe_indexer_variant.sh`：

**本地离线验证（已完成）**
- `bash -n` 语法检查：通过
- 内嵌 Python 探针体 `ast.parse`：通过
- **用仿真 sglang 包覆盖全部分支**：`V1`、`V3`、`NATIVE`、`NEITHER`、`REFUSE`、
  以及导入失败（rc=3）均已实测命中。
  另测一例"两种特征同时出现"，实际归入 `V1`（因 v1 守卫用的是 `or _publish_or_consume_candidates`，
  比 v3 守卫更宽）——**`AMBIGUOUS` 分支在当前判据下实际不可达**，保留它只是防御性代码。

**实机验证（2026-09-26，已完成）**
- 在 head `fq-dgx-01` 上对运行中的 `dsv41-head` 容器执行，返回 `V1`，退出码 0。
- 服务无影响：探测前后 `/health` 均 200，容器 `Up 3 hours (healthy)` 未变。
- 独立交叉验证：直接 `docker exec` 拉取 `_low_ratio_index_topk_dense` 全文（88 行）与符号表，
  与探针 JSON 输出**一致**（`self.candidate_masks` 见第 41/68 行，
  `_publish_or_consume_candidates` 见第 68 行，无 `candidate_metadata`，无 #39187 预算常量）。

**执行中修掉的三个真实缺陷**（这是本次最有价值的部分——前两个只会在真机暴露）
1. `rec()` 无返回值却写 `not rec(...)` → 真机上 `TypeError` 崩溃；
2. **`grep -o` / `sed s///p` 在本机对含双引号模式 0 命中** → 结论判错（见 §2.3 注）；
3. 中文写入 JSON 在非 UTF-8 locale 下解码失败 → 改 `ensure_ascii=True`，中文走 stderr。

---

## 7. 未验证项（不要当成已知）

1. ~~`da64c5cbb` 的 indexer 属于哪个变体~~ —— **已测得 `V1`**（见 §0）。
2. `indexer_chunked` 在 **TP=3 / EP1** 下的**行为正确性** —— 上游实测是 4× GB300，**不同拓扑**。
   探测只证明了**接口匹配**（符号齐全、守卫通过），**不证明数值等价**。
3. `CHUNKED_PREFILL_SIZE` 调回 1024 的 +6.3% —— 是**从旧数据反推的假设**，非实测。
4. 长上下文（~200k）在当前栈下是否真的会 OOM —— 从未压测过。
5. 本仓库 `fleet/adapter/sitecustomize.py` 与上游 HEAD 的合并冲突实际规模 —— 只知
   归一化后差 158 行（21 增 / 158 删），未逐段核对。
6. v1 adapter 与本集群 build 的**运行时兼容性**（如 `wo_a_w8` 那类 kwarg 不匹配）——
   只有真正装上并启动才能暴露。

---

## 8. 不建议移植的另外两个（结论与理由）

| 候选 | 结论 | 理由 |
|---|---|---|
| `fast_load.py` | **不建议** | 优化的是**权重加载**阶段（上游实测 230–290 s）。本仓库的性能口径里没有任何加载耗时项，`svc.sh start` 的 13–15 分钟是固定预算。它自己有通用的 `tp_slicing()`（`fast_load.py:180-191`），在 EP1 下会生效——但加速的是没在优化的环节。代价是 57 KB 复杂加载器 + 对 `weight_utils`/`deepseek_v4`/`dspark` 三处 monkey-patch（`fast_load.py:784/1179/1302`）。 |
| `hc_fused.py` | **低优先级** | 有 Triton 版本硬门（`hc_fused.py:187`：`Triton {version} != {_TRITON}; lowering unaudited`）。本集群引擎与上游 canary 线（`sources.manifest:14` 的 `f80c91a4b9`）不同 build，Triton 版本大概率不匹配，一装就抛。收益也只在 prefill。 |

---

## 9. 附：其它 TP4-only adapter 的处置

`moe_b12x_next.py`、`prefill_sp.py`、`l2_prefetch.py`、`shared_pad_k.py`、
`replicated_split.py`、`roce_gather.py`、`router_live.py`、`draft_main_proj.py`、
`draft_head_fp8_tp4.py`、`spark_prefill_dense.py`
—— 其中 **`indexer_chunked` 已在 §0/§1 单独评估并选定 v1**；`fast_load` / `hc_fused` 见 §8。
其余全部绑定 TP4 的 ring/RoCE/EP 拓扑或 `DSV41_LAUNCHER=tp4` 门控，**对 TP3 无可用路径**。

> 例外待查：`moe_b12x_next.py` 的注释提到它 "also runs EP_SIZE=1 at TP4 (N=576 per rank)"
> （`31cd1b0`），与本集群 EP1 有交集，但依赖 `b12x_next` 包与 `runtime/` 的源码拉取。
> **未做依赖分析，列为后续可选课题**。