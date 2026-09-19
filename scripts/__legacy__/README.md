# __legacy__ — 已作废排查路线的证据（保留备查，**不要照做**）

这些脚本属于 2026-09-18 的 RoCE 排查阶段。当时我们把失败归因于 NCCL 本身，于是尝试了：
NCCL 补丁栈（AICAD ring-only / netdev 硬编码）、按 `NCCL_IB_HCA` 顺序注册设备、变量矩阵穷举、
smoke 脚手架改造（shadow / per-rank HCA / hooks）等。

**结论更正（2026-09-19）**：根因是我们自己环境的三处问题 ——

1. **接线**：有两根缆接在同一端口索引上（`p0↔p0`、`p1↔p1`），而 NCCL 按**设备索引**跨 rank 配对，
   必然把"不在同一根缆上"的两个口配到一起 → `ibv_modify_qp 110` 或静默死。
   改成**每条缆 `p0↔p1`（交叉环）** + `NCCL_IB_SUBNET_AWARE_ROUTING=1` 即解；
2. **内核 CMA 回归**：DGX OS `7.0.0-1019-nvidia` 上 `grep CmaTotal /proc/meminfo` = `0 kB`，
   导致 `ibv_reg_mr_iova2` 报 `ENOMEM`（连 1 KB 区域都失败）。换 `6.17.0-1031-nvidia` + 驱动 `580.173.02` 即解；
3. **引擎镜像 `/etc/nccl.conf` 残留**：`NCCL_IB_USE_INLINE=1` + `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1`
   会冻死 SGLang 的 PyNccl 4 字节 warmup。置 0 即解。

三处修正后，**官方 NCCL 2.30.7 原样即可跑通 RoCE，不需要任何补丁**（`all_reduce` 256 MB 13.86 GB/s）。
请以仓库根目录 `README.md` 的 §1–§4 为准；本目录仅供追溯"当时为什么走了弯路"。
