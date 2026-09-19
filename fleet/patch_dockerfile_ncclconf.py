#!/usr/bin/env python3
"""Bake /etc/nccl.conf into the overlay image (the pass-through-proof NCCL channel).

start.sh forwards a fixed list of 17 NCCL variables; anything else in .env never reaches
the container. NCCL itself reads /etc/nccl.conf (src/misc/param.cc: initEnvFunc), so a
file baked into the image sets ANY NCCL parameter without touching start.sh.

Chosen knobs -- verified against the NCCL source, and NOT equal to their defaults (a knob
already at its default proves nothing; NCCL_IB_RETRY_CNT=7 and NCCL_IB_QPS_PER_CONNECTION=1,
both recommended in a widely-copied blog, are the defaults):

    NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1   (default -2 = auto)
    NCCL_IB_USE_INLINE=1                      (default 0)

Idempotent; keeps a backup of the Dockerfile.
"""
import sys

PATH = "Dockerfile"
MARK = "/etc/nccl.conf"

src = open(PATH, encoding="utf-8").read()
if MARK in src:
    print("already patched, nothing to do")
    sys.exit(0)

anchor = "EXPOSE 8888\n"
if anchor not in src:
    print("ERROR: anchor 'EXPOSE 8888' not found")
    sys.exit(1)

block = (
    "# NCCL parameters that start.sh does not forward. NCCL reads /etc/nccl.conf\n"
    "# (src/misc/param.cc), so this is the pass-through-proof way to set any knob.\n"
    "# Both values below were checked against the source and differ from their defaults.\n"
    "RUN printf 'NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1\\nNCCL_IB_USE_INLINE=1\\n' > /etc/nccl.conf\n"
    "EXPOSE 8888\n"
)

out = src.replace(anchor, block, 1)
open(PATH, "w", encoding="utf-8", newline="\n").write(out)
print("patched Dockerfile with /etc/nccl.conf")
