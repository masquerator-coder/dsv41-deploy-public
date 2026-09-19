#!/bin/bash
# Switch one DGX Spark node to the upstream-known-good pair:
#     Driver 580.173.02 + Kernel 6.17.0-1031-nvidia
# Surgical set (verified with apt -s): 3 installs, 4 downgrades, 1 removal.
# Reversible: the 178.04 packages AND the 7.0.0 driver-module package are cached
# first, so a rollback needs no network.
set -u
export DEBIAN_FRONTEND=noninteractive
LOG=/tmp/kswitch.log
exec > "$LOG" 2>&1

NEW=580.173.02-0ubuntu0.24.04.1
OLD=580.178.04-0ubuntu0.24.04.1
KREL=6.17.0-1031-nvidia

PKGS_NEW=(
  "nvidia-kernel-common-580=$NEW" "libnvidia-compute-580=$NEW" "nvidia-utils-580=$NEW"
  "nvidia-kernel-source-580-open=$NEW"
  "linux-image-$KREL" "linux-modules-$KREL" "linux-modules-nvidia-580-open-$KREL"
)
PKGS_OLD=(
  "nvidia-kernel-common-580=$OLD" "libnvidia-compute-580=$OLD" "nvidia-utils-580=$OLD"
  "nvidia-kernel-source-580-open=$OLD"
  "linux-modules-nvidia-580-open-7.0.0-1019-nvidia=7.0.0-1019.19~24.04.2+1"
)

echo "=== $(hostname) 开始 $(date '+%F %T') ==="
echo "切换前: 内核 $(uname -r)  驱动 $(modinfo nvidia 2>/dev/null | grep ^version | awk '{print $2}')"

echo
echo "--- 1) 缓存回滚包（178.04 全套 + 7.0.0 驱动模块）---"
apt-get install -y --reinstall --download-only "${PKGS_OLD[@]}" >/dev/null 2>&1
if [ $? -eq 0 ]; then echo "  回滚包已缓存到 /var/cache/apt/archives"; else echo "  [警告] 回滚包缓存不完整（回滚需联网）"; fi

echo
echo "--- 2) 执行切换 ---"
apt-get install -y --allow-downgrades "${PKGS_NEW[@]}"
rc=$?
echo "  apt exit=$rc"

echo
echo "--- 3) 核对 ---"
dpkg -l 2>/dev/null | grep -E "nvidia-kernel-common-580|libnvidia-compute-580|nvidia-utils-580|linux-(image|modules)-$KREL|linux-modules-nvidia-580-open-$KREL" | awk '{print "  " $2, $3}'
echo -n "  6.17 内核下的驱动模块: "
modinfo -k "$KREL" nvidia 2>/dev/null | grep ^version | awk '{print $2}'
echo -n "  ko 文件: "; ls /lib/modules/$KREL/kernel/nvidia-580-open/nvidia.ko 2>/dev/null || echo "缺失"

echo
echo "--- 4) GRUB 条目 id ---"
grep -oE "gnulinux-[0-9][^']*" /boot/grub/grub.cfg 2>/dev/null | sort -u | sed 's/^/  /'

echo
echo "=== $(hostname) 完成 $(date '+%F %T') rc=$rc ==="
