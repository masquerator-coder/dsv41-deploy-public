# 避坑清单：DGX Spark ×3 部署 DeepSeek-V4.1-Flash

> 采集日期：2026-09 · 适用硬件：3 × NVIDIA DGX Spark（GB10 / SM121，单机 121.7 GiB 统一内存，2 × ConnectX-7）
> 关注点：**稳定性（不崩、不乱码、可复现）+ 速度（吞吐 / 延迟）**
> 关联笔记：[[DGX-Spark-三节点-DSV4.1部署断点]] · 现场交接 `AIWorkspace/dsv41-fq-3xspark/SESSION-HANDOFF-2026-09-18.md`
> 体例：**【实测】** = 有公开测量记录；**【推断】** = 由资料推导。文末列全部来源。

---

## ★ 0. 与本地现场的逐条对照（先看这节）

| 社区记录的坑                                                                                   | 你的现场状态                                                                                                                 |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **CX7「三角」实际只有 3 根线；NCCL 按索引对齐 channel→NIC，在 3 节点 2 口拓扑上无解**                              | ✅ **你已独立实测闭环**（`search.cc:609`，index-0 = 最小 PCI = 三台都是 p0，穷举 0 解）。社区同名结论见 LLMKube「Three-Spark ring」                    |
| 环配方要求 `NCCL_IB_HCA` **列全部四个 ConnectX 口** + `NCCL_IB_SUBNET_AWARE_ROUTING=1`（NCCL ≥ 2.30） | ⚠️ **建议核对**：你试过 MERGE / CROSS / SUBNET 组合，但断点笔记未记录 `NCCL_IB_HCA` 是否含全部 4 个 netdev。你的 NCCL 是 2.30.7，**版本满足要求**          |
| 社区明确：**不要**设 `NCCL_IB_ADDR_RANGE` / `NCCL_CROSS_NIC`                                     | ⚠️ 你试过 `CROSS_NIC` —— 该组合在社区配方里是被点名避免的                                                                                 |
| 环实测带宽：RoCE **23.2 GB/s** ／ socket 兜底 **7.9 GB/s**                                        | 你：socket **2.07–2.14 GB/s**、RoCE 最佳 smoke **13.81 GB/s**。数量级一致，socket 路径确认是瓶颈                                          |
| `MEM_FRACTION_STATIC` 必须 ≥ 0.944                                                         | ✅ 你已是 0.95，且已踩过 0.60/0.55 → `Loaded weights leave no GPU memory for the KV cache`                                      |
| `DSPARK_BLOCK_SIZE=3` 是调优值；k=5 在聊天/散文上**过度起草**                                           | ⚠️ **有冲突**：你在做 `set_env_block5_uniform.py` / 「换 `DSPARK_BLOCK_SIZE=5` 重采」。你自己的铁律 5 与社区 README 都指向 k=3 更优，**注意别回归**     |
| SPS `compact` 路径与 Engram 块大小接口不一致                                                        | 你的 `engram.py:296 assert num_tokens == bs*block`（12 vs 16）与社区「能启动的模式不用表，用表的模式不能启动」的判断一致；收益主要在**并发 ≥2**                 |
| **GPU 降频锁死：`nvidia-smi` 完全看不出，必须拔电源 30–60 s**                                            | ❌ **你还没做过这个检查**。单流 0.9 s ↔ 18 s 的抖动、以及只有参考值 70–75%，**应先排除它**。你的铁律 4「看功耗判断真假占用」正是这个检查的基础                                |
| NCCL buffer 默认吃 4.7 GiB pinned host 内存                                                   | ✅ 你已设 `NCCL_MAX_NCHANNELS=8`                                                                                           |
| Engram 必须 node-local NVMe                                                                | ✅ 你已做（`DSV41_CACHE_GIB=0`；`hit_rate=0.0%` 是设计如此）                                                                       |
| 重启竞态：新 worker 加入旧 head 的 rendezvous                                                      | ✅ 你已有「`./start.sh stop` 超时留 worker 容器 → 三台 `docker ps` 检查」铁律                                                           |
| **每台 `swapoff -a`**（有 swap → 节点卡死需断电；无 swap → OOM 杀 worker 可重启）                          | ❌ 断点笔记未记录，**建议补**                                                                                                      |
| 视觉分支要单独验（纯文本冒烟测不到）                                                                       | ❌ 你的 bench / SPS 都是文本，视觉分支可能尚未验                                                                                        |
| `expandable_segments` 必须 `False`（设 True → 任何 >64 token prefill 返回 NaN）                   | ⚠️ 需核对你三台的 `PYTORCH_CUDA_ALLOC_CONF`                                                                                   |
| 长 prefill 内存墙：`T × L ≲ 2.0e8 token²`（indexer 瞬时峰值 ≈ 14 B × chunk × prefix）               | 你 `MAX_TOTAL_TOKENS=500000`；0.9 s ↔ 18 s 抖动你已怀疑与 `DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192` + Engram 全 NVMe 读有关，社区资料指向同一处 |
| 页缓存把 MemFree 压进 GB10 allocator stall 区（< 4 GiB）                                          | 你的 Spark 同款；注意社区实测 `flusher2.sh` 的判据 `Cached - Mapped - Shmem` **恒为负、永不触发**，要用 `memfree_flusher`                       |
| decode 不要数 SSE chunk 算（会低报约 3.5×）                                                        | 建议核对 `bench_decode.py` 的统计口径                                                                                           |

### 社区给的**环**配方（逐字，供你核对差异）

```ini
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_IB_SUBNET_AWARE_ROUTING=1     # 关键：NCCL 按 peer 选同 subnet 的本地口
NCCL_IB_MERGE_NICS=0
NCCL_NET_PLUGIN=none
# ibHCA 必须列全部四个口：
#   rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
# 不要设 NCCL_IB_ADDR_RANGE / NCCL_CROSS_NIC
```

> 社区机制说明与你 `connect.cc:524 ncclIbFindDevBySubnet` 的结论是**同一个机制**：接收侧按对端当前 GID 的同 /24 选卡，因此「一条连接能落到正确线缆」的前提确实是至少一端 index-0 已在线上（你的 `matched == checked`）。

---

## 1. 最重要的心智模型：host RAM 就是 GPU 显存

这一条决定了后面一半的坑。

- Spark 是统一内存架构，**任何 pin 住 host 内存的东西都在跟模型抢显存**：NCCL buffer、row cache、tokenizer 进程、mmap 映射、page cache、docker healthcheck。
- TP 是**同步**的：head 内存耗尽 → 三个 rank 全部停摆，不是只慢一点。
- 推论：判断标准不是「显存够不够」，而是 **`MemAvailable` 够不够**；page cache 会让同一配置在两次启动间差 ±0.5 GB。
- 这里没有 CPU offload 退路。**绝对不要设 `OFFLOAD_MODE=ram`** —— Engram 表会把模型挤出去。

---

## 2. 装得下吗：显存预算

| 项 | 数值 |
|---|---|
| 官方检查点磁盘占用 | 510.3 GB |
| EXL3 检查点磁盘占用 | 460.0 GB（只换 40 × 384 路由专家，其余逐字节相同） |
| Engram 记忆表（第 1、14 层，MXFP8） | 188.8 GiB —— **必须搬到 NVMe** |
| 搬走 Engram 后 GPU 常驻 | ≈ 305 GiB（MXFP4 专家 + FP8 密集权重） |
| TP3 后每台常驻 | ≈ 101 GiB / 121.7 GiB → **刚好放得下** |

【实测·SGLang TP3】resident ≈ 101 GiB/rank，服务时头节点剩 ~6 GB。
【实测·vLLM TP3 + EXL3】每台 84.2 GiB（含 DSpark drafter + 视觉编码器）；text-only 无 DSpark 时 79.9 GiB。

**坑**：`--gpu-memory-utilization` 调小**救不了加载期**。加载期峰值发生在 KV 分配之前，那类「内存不够」崩溃与 gmu 无关（你现场 99.4 GB 权重那次的教训是同一件事的另一面）。要治就治 Engram、page cache 和 mmap。

**路线分水岭**：官方发布版在 vLLM 下默认把 203 GB Engram 放 host 内存，而 Spark 的 host 内存就是 GPU 显存 → 3 台装不下。要么走 SGLang 的 Engram-on-disk 适配（你的路线），要么换 EXL3 量化检查点。

---

## 3. 稳定性坑（崩 / 乱码 / 不可复现）

### 3.1 Engram 表：三个连环坑

1. **必须 on-disk**（常驻就装不下）。
2. **必须每 rank 本地 NVMe，不能走 NFS**。
   【实测】worker 从 head 的 NFS 读 Engram 行，每步 5.9–7.8 ms；head 本地读仅 2.8 ms。每个 TP step 都等最慢 rank → 直接掉速。每台把属于自己 rank 的行 pack 到本地盘（TP3 约 47–63 GiB/node）。
3. **rank offset 必须正确**。
   【实测·审计发现】磁盘 reader 曾忽略每个 rank 的行偏移，rank 1–3 **静默读 rank 0 的行** —— 不报错、不崩溃，只是结果错。**最难发现的一类坑。**
   TP3 行范围（可用于校验）：
   - rank 1：`1:128000880:256002934 14:128004290:256009984`
   - rank 2：`1:256002934:384006168 14:256009984:384016682`

### 3.2 TP3 的整除问题（3 是素数，专门坑人）

以下全部不能被 3 整除，**每一个都会让你启动失败或静默变慢**：

| 项 | 值 | 处理 |
|---|---|---|
| attention heads | 64 | pad 到 72（vLLM）或 96（SGLang） |
| output groups | 8 | pad 到 9（vLLM）或 12（SGLang） |
| vocab | 129,280 | pad 到 129,408 |
| DSpark drafter 专家数 | 128 | 启动时 assert 失败；需放宽检查或 pad 到 129 |

**padding 带来的反量化陷阱（隐蔽）**
全零权重的 block-scale 内存里是 denormal，MXFP8 重新编码会**拒绝**这些块 → 86 层**静默**回退到 Triton（慢 3×）。
【实测】修法：把全零块的 scale 显式设为 1.0（0 × 1 = 0，数学不变）。
SGLang 侧对应现象：rank 2 的 attention 分片**整块都是 padding**；`wo_b` 取零列、`wo_a`/`wq_b` 重复最后一个真实 group。

**不要用 `--hf-overrides` 改 heads**
V4.1 这个 build 的 `text_config` 是 dict 类型，dict 形式会**整体替换子配置** → 报 `text_config ... does not have num_attention_heads`。必须直接改 `config.json`。
注意：重跑 prestage 时脚本发现 config.json 大小与 HF 不同会重新拉取 → **改完 config 后要再改一次**。

### 3.3 内核 / JIT：启动期与运行期的崩溃源

| 现象 | 原因 | 修法 |
|---|---|---|
| 全部节点同时 watchdog 复位 / 卡死 | 运行时 JIT 编译 FlashInfer MXFP8 GEMM，22 个并行 job 打爆 host 内存 | 镜像**预编译内核** + `MAX_JOBS=2`（保守用 1） |
| 预编译了还回退慢路径 | 运行时 nvcc 参数与预编译不一致 → 是另一个 cache entry | 环境变量与镜像预热时**逐字一致** |
| `No common block size for 64` | vLLM 选了清单里最小的 block size，V4 indexer 后端拒绝 | `--block-size 128` |
| DeepGEMM `block_kv == 32 or block_kv == 64` | ratio-1 indexer cache 每块 128 states | SM12x indexer page = 64 states |
| 长上下文直接崩 | `persistent_topk` 过订阅 48 个 SM，其 fallback 需 **128 KB** shared memory，而 GB10 只有 **99 KB** | 改用 `top_k_per_row_decode`（顺带快 1.6–3.6×） |
| 投机 batch 挂死 | adaptive verification 会 pad 投机 batch，padded batch 会 hang SM120 sparse MLA | **adaptive verification 保持关闭** |
| decode warmup 起不来 | FlashInfer 0.6.18 在 SM120 没有 `page_block_size=32` 的 sparse-MLA decode kernel | 升级 FlashInfer + SM12x page 补丁 |

### 3.4 重启动竞态

新 worker 会加入**尚未退出的旧 head** 的 rendezvous 端口 → 半死不活。
**铁律：重启前先停掉所有节点，先停 head。** 启动顺序反过来：先 worker，再 head。

### 3.5 版本回归：不要盲追最新版

- 【实测】vLLM **0.26.0** 在多机 + DSpark 下，长 prefill 会在 flashmla `sparse_prefill_fwd` 崩：TMA 描述符 `gmem_address 0`（空指针）→ `Assertion res == CUDA_SUCCESS failed`。同配置在 **0.25.x 完全正常**；崩在哪个 TP rank 上是随机的。回退即解。
- 【实测】vLLM 的 `dsv41-feat` 分支被 force-push 过。**必须 build 在 pinned commit 上**，否则引擎能起来但只输出一个重复的垃圾 token（DSpark 什么都不接受）——极具迷惑性的失败模式。

### 3.6 swap 与 OOM：后果完全不同，必须提前处理

- **每台执行 `sudo swapoff -a`**。
- 有 swap：内存超限 → 换页把 kubelet 换出去 → **节点卡死，需断电重启**。
- 无 swap：内存超限 → OOM 杀掉 worker → **operator 重启即可**。
- 加内存守护：free 低于 4–5 GiB 主动停掉三个容器（别等它自己崩）。

### 3.7 乱码 / 可复现性

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` —— 绝对不要设 True。**
【实测】在这套 stack 上，expandable segments 下**任何 > 64 token 的 prefill 都返回 NaN logits**（65 ~ 4000 token 的 prompt 全是垃圾，并发 ≥3 也一样）。
代价：原生分配器在长 prompt 上碎片化 → 头节点 262,144 的 context **实际只有 ~32k 可用**。
缓解：每个 prefill chunk 后归还 allocator cache（把长 prompt 成本从「累加」变成「单 chunk 瞬时」）。

**Fused MoE finalize 必须关。**
【实测】FlashInfer 的 fused finalize 用 atomic bf16 加法把 6 个 expert 输出求和；autotuner 只在 32-token bucket 选它 → greedy 输出**逐次运行不同**（首 token logprob 差到 ~1 nat），且每次加法都取整到 bf16。设 `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0` 后所有测试长度逐字节一致，代价约 0.3 ms/step。

**并发 >1 时 batch 组成会变** → 即使关了 finalize 结果仍可能逐次不同。要真正可复现必须 `--enable-deterministic-inference`。
→ **推论：所有「可复现」的基准测试都必须在 concurrency=1 下做**，否则测的是噪声。

### 3.8 三个「看起来像模型坏了」其实不是

1. **思考模式默认开启且强度 50**。`max_tokens` 给小了 → 预算全花在思考链 → 返回**空 content + `finish_reason=length`**。显式关思考或放宽预算。
2. **`reasoning_effort` 映射不一致（性能数字不可比的元凶）**。
   SGLang 内置 V4.1 encoder 把 `low/high/xhigh/max` → 25/50/75/100，**默认 high = 50**；而发布方 `encoding/encoding.py` 是 50/75/100，**默认 75**。
   不修则每个思考请求预算都是错的，且**思考预算会改变 DSpark 接受率** → 所有 thinking 模式的性能数字都不可比。
   **报任何 thinking 数字时必须写明 `reasoning_effort`。** vLLM 会**拒绝** `minimal` 和 `medium`。
3. **工具调用格式变了**。V4.1 的 DSML 标签名**带前导空格**：`<｜DSML｜ calls>`，与 V4 的 `<｜DSML｜tool_calls>` 不兼容。自研解析器不按模型卡 `encoding/encoding.py` 更新 → 工具调用**静默失败**。

### 3.9 watchdog 抓不住输出循环

`--watchdog-timeout` 在 token 持续产出时**永不触发**。
【实测】有人跑到 714k tokens / 约 2.6 小时（语义循环，非 GPU 挂死）。必须加 decode 侧检测：n-gram 循环、连续同 token 长 run、重复行。

### 3.10 视觉得单独验

视觉编码器是独立分支，**纯文本冒烟完全测不到**。
【实测】文本 / 工具 / 视觉在 TP3 下均 7/7 通过 —— 但必须专门发一条带图请求。单图最小像素 295,936；单请求最多 4 图。

---

## 4. 速度坑（按收益排序）

### 4.1 GPU 降频锁死 —— 最隐蔽，最容易被误判成「模型慢」

【实测】某台 Spark 的 EC（嵌入式控制器）会把 GPU DVFS **锁在 631–949 MHz**，而 `nvidia-smi` **完全看不出异常**：P0 正常、persistence 正常、无 thermal / power cap、无 clock event、内核日志干净。

- **普通重启 / 软关机清不掉** —— EC 在适配器插着时保持待机供电。
- **必须拔掉电源适配器 30–60 秒**再插回开机。
- 顺带检查适配器是否原装、是否插紧（EC 见到供电不足也会限频）。

**自检（跑任何 benchmark 之前必做）**：fp16 4096×4096 matmul 烧 15 秒，第 12 秒采样 `nvidia-smi`。

| | 健康 | 锁死 |
|---|---|---|
| 频率 | 2.2–2.4 GHz | 700–950 MHz |
| 功耗 | ≥ 80 W | < 20 W |
| fp16 算力 | 75–90 TFLOPS | — |

【实测收益】修好后 count 41.5 → 60.8 tok/s、code 32.9 → 57.1 tok/s。
TP 下**一台慢 = 全部慢**，因为每个 collective 都等最慢的 rank。

### 4.2 NCCL buffer 吃光内存（最划算的一个改动）

【实测·默认配置】512 个连接 buffer × 9.19 MiB × 3 种协议（Simple / LL128 / LL）≈ **4.7 GiB pinned host 内存/节点**，表现为**不可回收的 Shmem**。而 decode 的 all-reduce 只有 **61 KB**，走 LL 协议就够。

```ini
NCCL_BUFFSIZE=1MiB
NCCL_LL128_BUFFSIZE=256KiB
NCCL_PROTO=^LL128
NCCL_MAX_NCHANNELS=8
```

【实测效果】降到 **139 MB**；头节点 free 从 0.2–0.8 GB → **~6 GB**，顺带消掉启动时的「KV 抽奖」和回收停顿。

### 4.3 CUDA graph 必须开

【实测】eager 约 **200 ms/step**，host-bound（GPU 基本空转）—— 这是吞吐天花板。
【实测·SGLang】开 graph 后吞吐 +19% ~ +50%（C1 1.50×、C6 1.20×）。

- **前提**：必须先把 Engram 查表从 forward 挪到 `prepare_inputs`（并并行读所有行），否则有 host 侧 lookup 就**抓不了图**。
- **注意**：capture size 要按投机规则给（k 与 k+1 的倍数），否则 DSpark batch 会被 pad → 可能触发 §3.3 的挂死。
- **注意**：开 graph 后**第一个请求要检查 logprobs 是否 NaN**。

### 4.4 DSpark 投机解码：参数要按内容选

| 参数 | 建议 | 理由 |
|---|---|---|
| SGLang `DSPARK_BLOCK_SIZE` | **3**（4-token 验证窗） | k=5 在聊天 / 散文上**过度起草** |
| vLLM `num_speculative_tokens` | 5 | 该路线实测配置 |
| `enable_adaptive_verification` | **false** | 会 pad batch → hang SM120 sparse MLA |

【实测收益】混合集 1.5×、代码 2.4×、算术 2×、散文 1.2–1.3×。
【实测代价】KV pool 掉到约 1/3：TP3 下 678,950（开）vs 1,995,725（不开）。
【实测接受长度】平均 3.57 token/step，范围 1.88–5.92；计数 / 代码 / 表格接近上限 6，散文 / 叙事只有 ~2 —— **这解释了为什么速度按内容从 10 到 92 tok/s 分布**。

### 4.5 MXFP8 密集 GEMM 走错了内核

【实测】模型用 32×32 ue8m0 block，SGLang 会把非 128×128 的 block 丢给 **Triton**（慢）。
- 必须 `--fp8-gemm-backend flashinfer_cutlass`。
- SM120/121 上 FlashInfer 默认选的 CUTLASS 内核（128×32 tile）在 M=6 时并不快；真正合适的是 FlashInfer 的 **`b12x` warp-level MMA 内核**（16|32 行 tile），但 **SGLang 的 enum 没暴露它** → 要在 adapter 里手动路由。

【实测效果】dense FP8 projection 从 52 ms（Triton）/ 50 ms（CUTLASS SM120）→ **17 ms**。

### 4.6 长 prefill 的内存墙

【实测】峰值由 V4 的 **low-ratio indexer** 决定，不是 KV。每 chunk 瞬时峰值 ≈ `c × T × L`，实测 `c ≈ 14 B`（T = chunk，L = 当时前缀长度）。

实测规约：**`T × L ≲ 2.0e8 token²`**
- 2048 × 100k = 2.05e8 ✅ ／ 2048 × 133k = 2.7e8 ❌（guard 触发）
- 1024 × 208k = 2.13e8 ✅ ／ 1024 × 256k = 2.6e8 ❌

换算：1024 chunk 在 1M context 下 ~15 GB 瞬时；4096 需 ~60 GB → **chunk 只能给 1024**。
进阶：装自适应 chunk sizer（按 `L` 反比给 `T`），100k 以下可跑满 2048，长 prompt 自动降到 512/256。

### 4.7 Engram 页缓存把 MemFree 压进危险区

【实测】Engram 行读取会把 page cache 从 6 → 10 GiB，把 MemFree 压到 **3.7 GiB** —— GB10 GPU allocator 会 **stall** 的区间。
修法：MemFree 低于阈值就 drop cache 的 flusher。
**注意**：`flusher2.sh` 的判据 `Cached - Mapped - Shmem` 在这些机器上**恒为负**（Mapped 已计入 shmem 映射）→ **永不触发**；要用 `memfree_flusher`。

### 4.8 性能测量本身的坑

- **不要数 SSE chunk 算 decode**：DSpark 一个 chunk 打包好几个 token，会**低报约 3.5×**。用响应 `usage.completion_tokens`。
- **不要在 head 上跑压测 / 分析**：head 兼 HTTP server、tokenizer、detokenizer、NFS export，是其他 rank 等待的对象。
- **head 上不能建第二个 CUDA context**（`cudaMemGetInfo` 本身都会失败）。
- 其他容器会引入 jitter（有台机器相同 GEMM 慢 5–8%）。服务时停掉无关容器。
- autotune cache 在每节点 `~/.cache/sglang/flashinfer/autotune/`（root 拥有）。**改了任何 kernel flag，要在所有节点清掉。**

---

## 5. 网络 / Fabric

- **3 台 Spark 每台 2 个 ConnectX-7 口 → 可组成全互联三角**（直连，无需交换机）。**4 台就不行**，必须交换机或 ring。
- 走环（P0→P1 链）时有两个致命细节：
  1. `fabric.address` 是 **bootstrap 面**。环上任何 /30 都非全网可达 → **必须用管理网 IP**，`socketInterface` 也填管理网卡。带宽无所谓，bootstrap 只有几条小消息。
  2. `ibHCA` 是 **数据面**，要列出**全部 4 个口**，并设 **`NCCL_IB_SUBNET_AWARE_ROUTING=1`**（需 NCCL ≥ 2.30）。
     **不设的后果**：NCCL 按 channel index 在两端配对网卡，而该拓扑**任何 index 分配都无法满足** → channel setup 阶段 `ncclSystemError` / `ibv_modify_qp 110`。
  3. 配套：`NCCL_IB_MERGE_NICS=0`、`NCCL_NET_PLUGIN=none`；**不要**设 `NCCL_IB_ADDR_RANGE` 或 `NCCL_CROSS_NIC`。
- **环默认腿地址是 192.168.0.0/24 ~ 192.168.5.0/24**。若管理网是 192.168.1.0/24 会**撞网段**，bootstrap 起不来 → 先把腿挪到别的网段。
- **两个平面别配混**：Gloo bootstrap 走 LAN 网卡（`GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME`），NCCL 走 RoCE（`NCCL_NET=IB`、`NCCL_IB_HCA`）。GID index 要对。
- 【实测带宽】三角/环 all-reduce（256 MB，3 ranks）RoCE **23.2 GB/s** bus bandwidth；退化成 socket 走同样链路只有 **7.9 GB/s**。
- 【实测】TP3 下每步 **104 个 collective**，约 13–16 ms（LL 协议 over RoCE，每个中位数 50–100 µs）。这是 TP3 的延迟地板之一。

---

## 6. 启动与运维

- **冷启动 12–13 分钟**（8 分钟读 476 GiB + drafter + KV pool + graph capture）→ 把 NFS 挂载、Engram 行、GPU 烧机检查做进 `doctor`，别等 13 分钟才发现挂载没起来。
- **权重加载**：本地盘 129 s vs NFS 309/470 s → **每台放本地盘明显更快**。
- **大文件 staging 后要 drop page cache**：否则 vLLM worker 会因 init 时 CUDA-free 低于 `gpuMemoryUtilization × total` **拒绝启动**（460 GB stage 后大部分统一内存是 page cache）。
- **每个成员都需要全部 54 个 checkpoint 文件本地**，含两个 101 GB 的 Engram 表。
- **用预热好 JIT 内核的镜像**：否则在服务节点编译一个 CUTLASS GEMM，**每个 nvcc 进程约 6 GB host 内存**，叠加在 79.5 GiB 已 pin 的权重上。
- 用 `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,NET` 跑一次启动，看 NCCL 实际选了哪些网口和协议；**验证通过后就关掉**。
- **不要在启动脚本运行中编辑脚本**（bash 增量读取）。
- 运维判据：**看功耗判断真假占用**（~15 W 空闲；>100 W 真负载）—— 这条与 §4.1 的降频检查天然配套。

---

## 7. 健康基线：用来判断「是不是真的出了问题」

### 单步耗时（vLLM TP3 EXL3，fast state）
- 计数 prompt：63 ms/step，5.85 token/step → ~92 tok/s
- 代码 prompt：67–71 ms/step，4.97 token/step → ~77 tok/s
- 其中 88 个 all-reduce 约 5 ms；Engram staging 2–3 ms；其余是 MoE 与 attention 内核

### 吞吐（SGLang TP3，散文，256 token 输出）
| 并发 | 聚合 tok/s | 单流 tok/s | TTFT |
|---|---|---|---|
| 1 | 37.9 | 37.9 | 248 ms |
| 2 | 58.9 | 30.5 | 424 ms |
| 3 | 71.2 | 24.5 | 311 ms |
| 4 | 78.6 | 20.9 | 383 ms |

### 吞吐（vLLM TP3 EXL3，8 类 prompt 均值）
| C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|
| 46.0 | 73.4 | 100.1 | 118.4 | 134.4 | 152.9 |

### Prefill（SGLang TP3）
| 4K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|
| 3,350 | 3,782 | 3,768 | 3,531 | 3,251 tok/s |

---

## 8. 一页纸速查

**上电前**
- [ ] 三台都做 fp16 烧机检查（≥80 W / 2.2–2.4 GHz / 75–90 TFLOPS）；异常则**拔电源 30–60 s**
- [ ] `sudo swapoff -a`（三台）
- [ ] 确认适配器原装且插紧

**启动前**
- [ ] 三台本地 NVMe 都有完整 54 个 checkpoint 文件 + 自己 rank 的 Engram 行
- [ ] TP3 `config.json` 改好（heads 72/9 或 96/12 + vocab pad + `virtual_heads_from`），且 **prestage 后重跑过**
- [ ] 全零块的 block scale 已设为 1.0
- [ ] 先停掉所有节点（head 优先），再按 worker → head 顺序起
- [ ] drop page cache
- [ ] NCCL env：`NCCL_BUFFSIZE=1MiB` / `NCCL_LL128_BUFFSIZE=256KiB` / `NCCL_PROTO=^LL128` / `NCCL_MAX_NCHANNELS=8`
- [ ] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`
- [ ] `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`、`enable_adaptive_verification=false`
- [ ] 起内存守护（free < 5 GiB 停三台）

**服务中**
- [ ] 第一个请求检查 logprobs 是否 NaN
- [ ] 发一条**带图**请求验视觉
- [ ] 发一条工具调用请求验 DSML 解析
- [ ] 起 MemFree flusher（用 `memfree_flusher`，不要用 `flusher2`）
- [ ] 基准测试在 concurrency=1 下做，并记录 `reasoning_effort`
- [ ] decode 用 `usage.completion_tokens` 算，不数 SSE chunk
- [ ] 压测从 worker 发，不在 head 上跑

---

## 9. 来源

- [DeepSeek V4.1 Flash 发布公告（官方）](https://api-docs.deepseek.com/zh-cn/news/news260910/)
- [MiaAI-Lab / DeepSeek-v4.1-Flash-DGX-Sparks（SGLang，3–4 台 Spark）](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)
- [MiaAI-Lab / chunked-prefill-memory.md（长 prefill 内存分析）](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/main/docs/chunked-prefill-memory.md)
- [tonyd2wild / DeepSeek-V4.1-Flash-vLLM-DGX-Spark（vLLM TP4 + EXL3 TP3）](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
- [tonyd2wild / docs/EXL3-TP3.md（TP3 逐次启动复盘）](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/EXL3-TP3.md)
- [tonyd2wild / docs/gpu-clock-latch.md（GPU 降频锁死）](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/gpu-clock-latch.md)
- [defilantech / LLMKube PR #1816（三台 Spark 环形互联 + V4.1-Flash EXL3 TP3）](https://github.com/defilantech/LLMKube/pull/1816)
- [vLLM issue #49883（0.26.0 flashmla TMA 回归）](https://github.com/vllm-project/vllm/issues/49883)
- [SGLang issue #18799（V4.1 PD 分离崩溃）](https://github.com/sgl-project/sglang/issues/18799)
- [七牛云 / DeepSeek V4.1 Flash 部署完整指南（权重拆解、四条路线）](https://news.qiniu.com/archives/1789024203476)
- [NVIDIA / Deploy on DGX Spark（NIM 文档）](https://docs.nvidia.com/nim/large-language-models/1.15.0/deploy-on-dgx-spark.html)
