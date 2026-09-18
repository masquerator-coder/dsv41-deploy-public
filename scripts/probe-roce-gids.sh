#!/bin/bash
# Dump, for every CX7 RoCE port: which GID index carries which IPv4, and which netdev
# that GID is bound to. Needed to see which port pairs actually share a /24 with their
# cable peer (the QP 110 happens when NCCL pairs two ports that are not on the same wire).
for d in rocep1s0f0 rocep1s0f1 roceP2p1s0f0 roceP2p1s0f1; do
  b="/sys/class/infiniband/$d/ports/1"
  [ -d "$b" ] || { echo "$d: NOT PRESENT"; continue; }
  echo "--- $d ---"
  for i in $(ls "$b/gids" 2>/dev/null); do
    g=$(cat "$b/gids/$i" 2>/dev/null)
    t=$(cat "$b/gid_attrs/types/$i" 2>/dev/null)
    n=$(cat "$b/gid_attrs/ndevs/$i" 2>/dev/null)
    case "$g" in
      ::ffff:*) echo "   gid$i  $t  ndev=$n  $g" ;;
    esac
  done
done
echo "--- IPv4 on the CX7 netdevs ---"
ip -o -4 addr show | grep -E 'enp1s0f|enP2p1s0f' | awk '{print "   " $2, $4}'
