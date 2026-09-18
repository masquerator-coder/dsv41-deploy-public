#!/bin/bash
# Measure per-step fabric traffic while a sustained 4-stream decode load runs.
# Writes snapshots to traces/ for offline math (rx and tx, plus a unix timestamp).
cd "$(dirname "$0")" || exit 1
mkdir -p traces
IFACES="enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1 wlP9s9"

snap() {  # $1 = output file
  : >"$1"
  for h in node1 node2 node3; do
    printf "%s " "${h##*-}" >>"$1"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$h.local" \
      'for i in '"$IFACES"'; do printf "%s %s %s " $i \
        $(cat /sys/class/net/$i/statistics/rx_bytes) \
        $(cat /sys/class/net/$i/statistics/tx_bytes); done' >>"$1"
    echo >>"$1"
  done
  date +%s >>"$1"
}

echo "[*] snapshot A"
snap traces/snapA.txt
tail -1 traces/snapA.txt

echo "[*] starting a 4-round 4-stream load on node2"
ssh -o BatchMode=yes -o ConnectTimeout=8 node2.local \
  '(setsid nohup python3 /tmp/bench_conc.py http://192.168.0.101:8888 5 200 > /tmp/conc2.out 2>&1 &); echo started'

sleep 55
echo "[*] snapshot B"
snap traces/snapB.txt
echo "[*] load output so far:"
ssh -o BatchMode=yes -o ConnectTimeout=8 node2.local 'cat /tmp/conc2.out'
