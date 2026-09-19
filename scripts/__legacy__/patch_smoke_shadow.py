#!/usr/bin/env python3
"""Add a SMOKE_SHADOW_BUNDLED hook to ~/dsv41-3xspark/nccl_smoke.sh (idempotent).

Why: with only LD_PRELOAD, the container ends up with TWO libnccl copies mapped
(/nccl/libnccl.so.2.30.7 and the torch-bundled .../nvidia/nccl/lib/libnccl.so.2),
so the communicator can be half-patched — the device override fires while the
completion bookkeeping runs in the other copy. Mounting the ring-only build over
the bundled path guarantees exactly one NCCL is live.
"""
import pathlib
import shutil
import sys

p = pathlib.Path.home() / "dsv41-3xspark" / "nccl_smoke.sh"
src = p.read_text()
if "SMOKE_SHADOW_BUNDLED" in src:
    print("already patched")
    sys.exit(0)

bak = p.with_name("nccl_smoke.sh.bak-pre-shadow")
if not bak.exists():
    shutil.copy2(p, bak)
    print("backup:", bak)

anchor = "  # fq: per-rank HCA"
k = src.index(anchor)  # raises if the anchor moved -> fail loudly rather than guess
block = (
    "  # fq: shadow the torch-bundled NCCL so exactly one libnccl is live.\n"
    "  # LD_PRELOAD alone leaves both copies mapped (torch dlopens its own by path).\n"
    "  if [ \"${SMOKE_SHADOW_BUNDLED:-0}\" = \"1\" ] &&"
    " [ -f \"$HOME/nccl/build/lib/libnccl.so.2.30.7\" ]; then\n"
    "    ARGS_BASE+=(-v \"$HOME/nccl/build/lib/libnccl.so.2.30.7:"
    "/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:ro\")\n"
    "    echo \"[+] shadowing torch-bundled NCCL with the ring-only build\"\n"
    "  fi\n"
)
p.write_text(src[:k] + block + "\n" + src[k:])
print("patched:", p)
