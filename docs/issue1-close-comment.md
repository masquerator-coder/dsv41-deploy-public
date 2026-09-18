结案更正：本 issue 的失败根因**不在 NCCL**，而在我们自己的环境 ——

1. **接线**：原接线里有两根缆接在同一端口索引上（`p0↔p0` / `p1↔p1` 各一根），而 NCCL 按设备索引跨 rank 配对，
   必然把"不在同一根缆上"的两个口配到一起 → `ibv_modify_qp 110` / `p2p.cc` 内部错误。
   改成**每条缆 p0↔p1（交叉环）** + `NCCL_IB_SUBNET_AWARE_ROUTING=1` 后一次通过。
2. **内核 CMA 回归**：`7.0.0-1019-nvidia` 上 `CmaTotal: 0 kB`，导致 `ibv_reg_mr_iova2` ENOMEM（1 KB 都失败）；
   换 `6.17.0-1031-nvidia` + 驱动 `580.173.02` 解决。
3. **容器镜像残留 `/etc/nccl.conf`**：`NCCL_IB_USE_INLINE=1` + `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1`
   会冻死 SGLang PyNccl 的 4 字节 warmup；置 0 解决。

修正后：`all_reduce` 256 MB **13.86 GB/s**（32/32 信道 `via NET/IB`），SGLang TP=3 单流 **28.5–35.4 tok/s**、
4 并发聚合 **67.5–75.8 tok/s**。官方 NCCL 2.30.7 **无需任何补丁**。

感谢贵仓库公开的补丁与构建记录 —— 它们在错误配置阶段提供了关键对照。为此前把根因指向 NCCL 致歉。
详细记录见上面更正的 issue 正文。
