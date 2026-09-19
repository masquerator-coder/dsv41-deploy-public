#!/usr/bin/env bash
# 源口 ping 矩阵：判定本机每个 fabric 口能到达哪些对端 fabric IP
# 用法：bash ping_matrix.sh <self:1|2|3>
set -u
SELF=${SELF:-1}
declare -A IF
case $SELF in
 1) IF[p0d0]=enp1s0f0np0; IF[p1d0]=enp1s0f1np1; IF[p0d2]=enP2p1s0f0np0; IF[p1d2]=enP2p1s0f1np1
    PEERS=(10.100.176.1 10.100.177.1 10.100.178.1 10.100.179.1 10.100.180.1 10.100.181.1);;
 2) IF[p0d0]=enp1s0f0np0; IF[p1d0]=enp1s0f1np1; IF[p0d2]=enP2p1s0f0np0; IF[p1d2]=enP2p1s0f1np1
    PEERS=(10.100.176.1 10.100.177.1 10.100.178.2 10.100.179.2 10.100.180.2 10.100.181.2);;
 3) IF[p0d0]=enp1s0f0np0; IF[p1d0]=enp1s0f1np1; IF[p0d2]=enP2p1s0f0np0; IF[p1d2]=enP2p1s0f1np1
    PEERS=(10.100.176.2 10.100.177.2 10.100.178.2 10.100.179.2 10.100.180.2 10.100.181.2);;
esac
echo "### self=$SELF $(hostname)"
for k in p0d0 p1d0 p0d2 p1d2; do
  i=${IF[$k]}
  printf '%-6s %-16s' "$k" "$i"
  for p in "${PEERS[@]}"; do
    if ping -c1 -W1 -I "$i" "$p" >/dev/null 2>&1; then printf ' %s=OK' "$p"; else printf ' %s=--' "$p"; fi
  done
  echo
done
echo "### addr"
ip -4 -o addr show | awk '$2 ~ /^en/ {print $2, $4}'
echo "### ib state"
for d in /sys/class/infiniband/*; do echo "$(basename $d) $(cat $d/ports/1/state) phys=$(cat $d/ports/1/phys_state)"; done
echo "### cx7 hotplug file"
ls -1 /etc/nvidia/ | grep -i cx7 || echo "no cx7 file"
