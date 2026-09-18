#!/bin/bash
# Roll the node back to the previous known-good pair:
#     Driver 580.178.04 + Kernel 7.0.0-1019-nvidia
# The packages were cached before the switch, so this normally runs offline.
# The one-shot GRUB entry is already consumed, so the next boot returns to the
# default entry (7.0.0-1019) by itself.
set -u
export DEBIAN_FRONTEND=noninteractive
LOG=/tmp/krevert.log
exec > "$LOG" 2>&1

OLD=580.178.04-0ubuntu0.24.04.1
echo "=== $(hostname) 回滚开始 $(date '+%F %T') ==="
echo "当前: 内核 $(uname -r)  驱动 $(modinfo nvidia 2>/dev/null | grep ^version | awk '{print $2}')"

apt-get install -y \
  "nvidia-kernel-common-580=$OLD" "libnvidia-compute-580=$OLD" "nvidia-utils-580=$OLD" \
  "nvidia-kernel-source-580-open=$OLD" \
  "linux-modules-nvidia-580-open-7.0.0-1019-nvidia=7.0.0-1019.19~24.04.2+1"
rc=$?
echo "  apt exit=$rc"

echo
echo "--- 核对 ---"
dpkg -l 2>/dev/null | grep -E "nvidia-kernel-common-580|libnvidia-compute-580|linux-modules-nvidia-580-open-(7\.0|6\.17.0-1031)" | awk '{print "  " $2, $3}'
echo -n "  7.0.0 驱动模块: "; ls /lib/modules/7.0.0-1019-nvidia/kernel/nvidia-580-open/nvidia.ko 2>/dev/null || echo "缺失"
echo "=== $(hostname) 回滚完成 $(date '+%F %T') rc=$rc ==="
