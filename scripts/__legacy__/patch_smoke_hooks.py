#!/usr/bin/env python3
"""Add two more hooks to ~/dsv41-3xspark/nccl_smoke.sh (idempotent):

  SMOKE_SHADOW_SRC   which host .so to mount over the torch-bundled lib path
                     (default: the AICAD ring-only build)
  SMOKE_EXTRA_MOUNT  extra bind mount(s), comma separated "src:dst"

Needed to test (a) the upstream 3-node profile, which uses the plain 2.30.7 build
(no per-peer map compiled in, HCA list = 2 ports) and (b) a custom NCCL_TOPO_FILE.
"""
import pathlib
import shutil
import sys

p = pathlib.Path.home() / "dsv41-3xspark" / "nccl_smoke.sh"
src = p.read_text()
if "SMOKE_EXTRA_MOUNT" in src:
    print("already patched")
    sys.exit(0)

bak = p.with_name("nccl_smoke.sh.bak-pre-hooks")
if not bak.exists():
    shutil.copy2(p, bak)
    print("backup:", bak)

anchor = "  # fq: per-rank HCA"
k = src.index(anchor)
block = (
    "  # fq: choose which libnccl to shadow the bundled one with (default: ring-only build).\n"
    "  SHADOW_SRC=\"${SMOKE_SHADOW_SRC:-$HOME/nccl/build/lib/libnccl.so.2.30.7}\"\n"
    "  # fq: arbitrary extra bind mounts, comma separated \"src:dst\".\n"
    "  if [ -n \"${SMOKE_EXTRA_MOUNT:-}\" ]; then\n"
    "    IFS=',' read -r -a _mnts <<< \"$SMOKE_EXTRA_MOUNT\"\n"
    "    for _m in \"${_mnts[@]}\"; do ARGS_BASE+=(-v \"${_m}:ro\"); done\n"
    "  fi\n"
)
p.write_text(src[:k] + block + "\n" + src[k:])
# point the existing shadow hook at SHADOW_SRC
src = p.read_text()
src = src.replace(
    "  if [ \"${SMOKE_SHADOW_BUNDLED:-0}\" = \"1\" ] &&"
    " [ -f \"$HOME/nccl/build/lib/libnccl.so.2.30.7\" ]; then\n"
    "    ARGS_BASE+=(-v \"$HOME/nccl/build/lib/libnccl.so.2.30.7:"
    "/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:ro\")\n",
    "  if [ \"${SMOKE_SHADOW_BUNDLED:-0}\" = \"1\" ] && [ -f \"$SHADOW_SRC\" ]; then\n"
    "    ARGS_BASE+=(-v \"$SHADOW_SRC:"
    "/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:ro\")\n",
)
p.write_text(src)
print("patched:", p)
