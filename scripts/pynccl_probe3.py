#!/usr/bin/env python3
"""Multi-communicator reproduction probe (engine-like ordering).

Stage 1: torch NCCL PG (world)
Stage 2: gloo subgroup -> PyNcclCommunicator (exactly what parallel_state.py:457 does)
Stage 3: N sequential torch NCCL groups, each with a small all_reduce — this is what the
         engine does later (TP/EP/draft groups) and where the ibv_reg_mr_iova2 ENOMEM
         shows up in the real boot.

Every stage prints a machine-readable line so the driver can tell how far it got.
"""
import os
import time
import traceback

import torch
import torch.distributed as dist
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
NCOMM = int(os.environ.get("PROBE_NCOMM", "6"))
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
LIB = os.environ.get("NCCL_LIB_PATH") or None


def log(m):
    print(f"[rank{rank}] {m}", flush=True)


def stage(name, fn):
    t0 = time.time()
    try:
        fn()
        log(f"STAGE-OK   {name}  {time.time()-t0:.2f}s")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"STAGE-FAIL {name}  {time.time()-t0:.2f}s  {repr(e)[:200]}")
        traceback.print_exc()
        return False


log(f"start lib={LIB} ncomm={NCOMM}")

ok = stage("torch_pg_nccl", lambda: dist.init_process_group(
    "nccl", rank=rank, world_size=world, device_id=dev))
if not ok:
    log("PROBE-ABORT")
    raise SystemExit(1)

gloo_pg = None
ok = stage("gloo_subgroup", lambda: globals().__setitem__(
    "gloo_pg", dist.new_group(ranks=list(range(world)), backend="gloo")))

comm = None
if ok:
    def _pynccl():
        global comm
        comm = PyNcclCommunicator(group=gloo_pg, device=0, library_path=LIB)
        log(f"  pynccl available={getattr(comm,'available',None)} disabled={getattr(comm,'disabled',None)}")
    ok = stage("pynccl_comm", _pynccl)

if ok:
    # use it for real (engine does this under change_state(enable=True) in graphs)
    def _use():
        with comm.change_state(enable=True):
            x = torch.ones(4096, device=dev) * (rank + 1)
            comm.all_reduce(x)
            torch.cuda.synchronize()
            got = float(x[0].item())
            exp = float(world * (world + 1) // 2)
            log(f"  pynccl all_reduce got={got} expect={exp} -> {'OK' if abs(got-exp) < 1e-3 else 'MISMATCH'}")
    stage("pynccl_all_reduce", _use)

# sequential extra communicators (engine-like registration pressure)
for i in range(NCOMM):
    def _grp(i=i):
        g = dist.new_group(ranks=list(range(world)), backend="nccl")
        t = torch.ones(262144, device=dev)
        for _ in range(3):
            dist.all_reduce(t, group=g)
        torch.cuda.synchronize()
    if not stage(f"extra_comm_{i+1}", _grp):
        log("PROBE-STOPPED-EARLY")
        break

log("PROBE-DONE")
