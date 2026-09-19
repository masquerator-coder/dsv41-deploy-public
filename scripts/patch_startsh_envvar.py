#!/usr/bin/env python3
"""Forward an extra NCCL variable from .env into the engine containers (generic).

start.sh names every NCCL variable it forwards explicitly, so anything not on that
list is silently dropped -- a .env edit looks like a no-op and costs a 13-minute boot
to disprove. Usage:

    python3 patch_startsh_envvar.py NCCL_NET_GDR_LEVEL LOC

Inserts the variable at both sites (the head's docker_common_args and the worker ssh
launch), with the .env value winning and the given default as fallback. Idempotent.
"""
import sys

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(2)

var, default = sys.argv[1], sys.argv[2]
PATH = "start.sh"

src = open(PATH, encoding="utf-8").read()
if var in src:
    print(f"already contains {var}, nothing to do")
    sys.exit(0)

anchor1 = '    -e "NCCL_CUMEM_ENABLE=0"\n'
# unquoted form appears only in the worker ssh launch line; anchor on the line's own
# start so repeated patches accumulate instead of invalidating each other
anchor2 = "-e NCCL_CUMEM_ENABLE=0 "
if anchor1 not in src or anchor2 not in src:
    print("ERROR: anchors not found -- start.sh changed upstream")
    sys.exit(1)

line_dq = f'    -e "{var}=${{{var}:-{default}}}"\n'
line_bs = f"-e {var}=${{{var}:-{default}}} "
# NOTE: append ONLY the new pair. An earlier version of this script also re-emitted
# "-e NCCL_DEBUG=..." here and silently mangled the worker line into
#   "-e NCCL_DEBUG=$NCCL_DEBUG-e NCCL_DMABUF_ENABLE=..."
# which is syntactically valid (so `bash -n` passes) but sets a garbage variable and
# drops the intended one. Always re-read the patched line, do not trust bash -n alone.
out = src.replace(anchor1, anchor1 + line_dq, 1)
out = out.replace(anchor2, anchor2 + line_bs, 1)

open(PATH, "w", encoding="utf-8", newline="\n").write(out)
print(f"patched {var} (default {default}): {out.count(var)} occurrence(s)")
