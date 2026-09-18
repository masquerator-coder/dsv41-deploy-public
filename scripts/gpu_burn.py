#!/usr/bin/env python3
"""fp16 4096x4096 matmul burn-in: exposes a GPU clock latch that nvidia-smi alone hides.

Community reference (DGX Spark / GB10):
  healthy : SM clock 2.2-2.4 GHz, power >= 80 W under load, ~75-90 TFLOPS fp16
  latched : SM clock 700-950 MHz, power < 20 W  -- and nvidia-smi looks perfectly normal
            (P0, no thermal/power cap, no clock event). Needs a physical power cycle
            (unplug the PSU 30-60 s); a soft reboot does NOT clear it."""
import subprocess
import time

import torch

Q = "clocks.sm,clocks.max.sm,power.draw,power.limit,utilization.gpu,temperature.gpu,clocks_throttle_reasons.active"
dev = "cuda:0"
print("device:", torch.cuda.get_device_name(0), flush=True)
a = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
b = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
print("baseline (idle):", subprocess.run(["nvidia-smi", "--query-gpu=" + Q, "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip(), flush=True)

for _ in range(5):
    c = a @ b
torch.cuda.synchronize()

t0 = time.time()
n = 0
sampled = False
while True:
    for _ in range(10):
        c = a @ b
        n += 10
    torch.cuda.synchronize()
    el = time.time() - t0
    if el >= 12 and not sampled:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + Q, "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        print("under load @12s:", out, flush=True)
        sampled = True
    if el >= 18:
        break

el = time.time() - t0
tflops = 2 * 4096 ** 3 * n / el / 1e12
print(f"matmuls={n}  elapsed={el:.1f}s  ->  {tflops:.1f} TFLOPS fp16", flush=True)
print("VERDICT_healthy = clock 2.2-2.4GHz & power>=80W & ~75-90 TFLOPS", flush=True)
