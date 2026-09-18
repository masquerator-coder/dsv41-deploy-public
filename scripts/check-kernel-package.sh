#!/bin/bash
# Pre-flight for the 6.17.0-1031 kernel experiment: download the package (no system
# change) and read the NVIDIA driver version embedded in it. If it does not match the
# userspace driver (580.178.04) the nodes would boot without a usable GPU stack.
set -u
cd /tmp || exit 1
rm -f linux-modules-6.17.0-1031-nvidia*.deb
apt-get download linux-modules-6.17.0-1031-nvidia 2>&1 | tail -1

deb=$(ls linux-modules-6.17.0-1031-nvidia*.deb 2>/dev/null | head -1)
if [ -z "$deb" ]; then
  echo "下载失败"
  exit 1
fi
echo "包: $deb  大小 $(du -h "$deb" | cut -f1)"

rm -rf /tmp/km && mkdir -p /tmp/km
dpkg-deb -x "$deb" /tmp/km

echo
echo "=== 包内 NVIDIA 模块 ==="
find /tmp/km -name 'nvidia*.ko*' 2>/dev/null | sed 's|/tmp/km||' | head -10

echo
echo "=== 驱动版本（与用户态 580.178.04 对比）==="
ko=$(find /tmp/km -name 'nvidia.ko*' 2>/dev/null | head -1)
if [ -n "$ko" ]; then
  modinfo "$ko" 2>/dev/null | grep -E '^version|^filename' || echo "  modinfo 读不出（可能是 .zst 未解压）"
else
  echo "  包内没有 nvidia.ko —— 该内核不带驱动模块"
fi

echo
echo "=== 内核模块树顶层 ==="
ls /tmp/km/lib/modules/6.17.0-1031-nvidia/kernel/ 2>/dev/null | head -12
