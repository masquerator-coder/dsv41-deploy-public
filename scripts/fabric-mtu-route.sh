#!/bin/bash
# cluster fabric tuning: jumbo frames + direct-link /32 routes for NCCL's socket
# transport.
#
# Why the socket transport: the three DGX Sparks are cabled as a triangle of 3
# point-to-point wires with 2 ports each. NCCL's rail model needs rail-consistent
# cabling (same-index port of every rank on one wire), which cannot exist on an
# odd cycle, and the NCCL_IB_SUBNET_AWARE_ROUTING fallback cannot cover all three
# wires either (verified experimentally: every registration/order combination
# fails -> ibv_modify_qp 110 / hung connect). So NCCL runs over TCP, which the
# kernel routes; these /32 routes keep every TCP flow on the *direct* 200GbE
# link (no transit hop through the head).
#
# Wiring (verified): W1 = 01-p0 <-> 03-p0 | W2 = 01-p1 <-> 02-p0 | W3 = 02-p1 <-> 03-p1
#   node 01: 178.2 / 180.2      node 02: 180.1 / 176.2      node 03: 178.1 / 176.1
set -u
HOST=$(hostname)

apply_mtu() {
  for d in enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1; do
    [ -e "/sys/class/net/$d" ] || continue
    ip link set dev "$d" mtu 9000 || true
  done
}

wait_ifaces() {
  for _ in $(seq 1 30); do
    [ -e /sys/class/net/enp1s0f0np0 ] && [ -e /sys/class/net/enp1s0f1np1 ] && return 0
    sleep 2
  done
  return 1
}

wait_ifaces || true
apply_mtu

case "$HOST" in
  node2)
    # -> node 03's p0 (178.1) over W3 (direct), -> head's p0 (178.2) over W2 (direct)
    ip route replace 10.100.178.1/32 via 10.100.176.1 dev enp1s0f1np1 || true
    ip route replace 10.100.178.2/32 via 10.100.180.2 dev enp1s0f0np0 || true
    ;;
  node3)
    # -> node 02's p0 (180.1) over W3 (direct)
    ip route replace 10.100.180.1/32 via 10.100.176.2 dev enp1s0f1np1 || true
    ;;
esac

logger -t fq-fabric "mtu+routes applied on $HOST"
exit 0
