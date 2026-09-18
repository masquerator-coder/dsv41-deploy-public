#!/usr/bin/env python3
"""Size-sweep NCCL probe for the fabric matrix (installed as /smoke.py in the container).

Differences vs the stock nccl_smoke.py:
  * prints which libnccl actually got loaded (from /proc/self/maps) — LD_PRELOAD
    vs the torch-bundled copy was previously ambiguous;
  * sweeps 1/16/256 MB so a variant that only breaks on big buffers is visible;
  * machine-readable RESULT lines so the driver can aggregate.
"""
import os
import time

import torch
import torch.distributed as dist


def nccl_libs():
    paths = set()
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                if "nccl" in line:
                    path = line.split()[-1]
                    if path.startswith("/"):
                        paths.add(path)
    except Exception:  # noqa: BLE001
        pass
    return sorted(paths)


rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
dist.init_process_group(backend="nccl", rank=rank, world_size=world, device_id=dev)

if rank == 0:
    print(f"MYRANK {rank} NCCL {torch.cuda.nccl.version()}", flush=True)
    for p in nccl_libs():
        print(f"LIBS {p}", flush=True)

for mb in [1, 16, 256]:
    n = mb * 1024 * 1024 // 4
    x = torch.ones(n, device=dev)
    try:
        for _ in range(3):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        iters = 10 if mb <= 16 else 3
        t0 = time.time()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        ok = bool(torch.allclose(x, torch.full_like(x, float(world))))
        bw = n * 4 * 2 * (world - 1) / world / dt / 1e9
        if rank == 0:
            print(f"RESULT {mb}MB {dt*1000:8.2f}ms {bw:7.2f}GB/s correct={ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"RESULT {mb}MB FAILED {repr(e)[:130]}", flush=True)
        break

if rank == 0:
    print("SWEEPDONE", flush=True)
dist.barrier()
dist.destroy_process_group()
