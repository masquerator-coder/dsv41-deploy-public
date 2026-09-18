#!/bin/bash
# Round 2 matrix: test the two hypotheses that survived round 1.
#
#  H1  "two ports on the same wire": every earlier attempt gave each rank FOUR ports,
#      i.e. two GIDs per wire. The receive side then sees the same peer arriving from
#      the same wire over two different local GIDs -> the QP<->peer bookkeeping is
#      ambiguous ("Recv comm could not retreive a request ..."). The upstream 3-node
#      profile uses TWO ports per node = exactly one port per wire.
#  H2  topology file (the external playbook's item 5): GPU lives in PCI domain
#      0000000F while the NICs are in 0000/0002, so NCCL's distance model may rank
#      devices wrongly. Declare GPU+NICs as siblings so all four NICs are PIX.
cd "$(dirname "$0")" || exit 1

LOCK=/tmp/fq-matrix2.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another run_matrix2.sh is alive - refusing to start"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

COMMON="NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_CROSS_NIC=1 NCCL_IB_MERGE_NICS=0 NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_NET_PLUGIN=none NCCL_SOCKET_IFNAME=wlP9s9 GLOO_SOCKET_IFNAME=wlP9s9"

TWO_PORT="NCCL_IB_HCA=rocep1s0f0,rocep1s0f1"
H0="rocep1s0f1,rocep1s0f0,roceP2p1s0f1,roceP2p1s0f0"
H1="rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1"

PREPATCH="$HOME/nccl/build/lib/libnccl.so.2.30.7.prepatch"
RINGONLY="$HOME/nccl/build/lib/libnccl.so.2.30.7"

cleanup_smoke() {
  for c in $(docker ps -aq --format '{{.ID}} {{.Names}}' | grep -vi dsv41 | awk '{print $1}'); do
    docker rm -f "$c" >/dev/null 2>&1
  done
}

run() {  # name, extra_env, port, use_perrank_hca(0/1), shadow_src
  name="$1"; extra="$2"; port="$3"; perrank="$4"; shadow="$5"
  rm -rf "mx2-$name" "mx2-$name.out"
  local hca_env=()
  if [ "$perrank" = "1" ]; then
    hca_env=(SMOKE_HCA_R0="$H0" SMOKE_HCA_R1="$H1" SMOKE_HCA_R2="$H1")
  fi
  env SMOKE_IMAGE=dsv41-3xspark:local SMOKE_ENTRYPOINT=python3 SMOKE_NO_GID=1 SMOKE_PORT="$port" \
  NCCL_IB_GID_INDEX=3 SMOKE_NCCL_DEBUG=WARN SMOKE_OUT="$PWD/mx2-$name" \
  SMOKE_NCCL_HOST_DIR="$HOME/nccl/build/lib" \
  SMOKE_SHADOW_BUNDLED=1 SMOKE_SHADOW_SRC="$shadow" \
  "${hca_env[@]}" \
  SMOKE_EXTRA_ENV="$COMMON $extra" \
  timeout 200 bash ./nccl_smoke.sh "mx2-$name" > "mx2-$name.out" 2>&1
  {
    echo "### $name"
    echo "  extra: $extra"
    echo "  shadow: $(basename "$shadow")  per_rank_hca=$perrank"
    grep -hE '^LIBS |^MYRANK |^RESULT |^SWEEPDONE' "mx2-$name/rank0.log" 2>/dev/null | sed 's/^/  /'
    echo "  netdev: $(grep -h 'Net devices' "mx2-$name/rank0.log" 2>/dev/null | head -1 | grep -oE 'Rank 0: [0-9]+ Net devices')"
    local err
    err=$(grep -m1 -hoE 'ibv_modify_qp[^,]*|could not retreive[^,]*|Internal check failed|only [0-9]+ vNics have been created|Could not (parse|open) topology' "mx2-$name/rank0.log" 2>/dev/null | cut -c1-110)
    echo "  err:    ${err:-none}"
    echo "  verdict: $(tail -1 "mx2-$name.out" | cut -c1-70)"
  } >> matrix2-results.txt
  cleanup_smoke
  sleep 3
}

rm -f matrix2-results.txt
echo "== prepatch lib: $(ls -la "$PREPATCH" 2>/dev/null | awk '{print $5}') B / ring-only: $(ls -la "$RINGONLY" | awk '{print $5}') B" | tee -a matrix2-results.txt

# H1: upstream 3-node profile -- plain 2.30.7, ONE port per wire, ring algo forced
run u3_2p        "NCCL_ALGO=RING $TWO_PORT"                       29900 0 "$PREPATCH"
# H1 control: same but let NCCL pick the algorithms itself
run u3_2p_noalgo "$TWO_PORT"                                      29901 0 "$PREPATCH"
# H2: four ports + per-peer map + custom topology file (all NICs PIX with the GPU)
export SMOKE_EXTRA_MOUNT="$HOME/dsv41-3xspark/topo/fq-triangle.xml:/topo.xml"
run topo4p       "NCCL_ALGO=RING NCCL_TOPO_FILE=/topo.xml"        29902 1 "$RINGONLY"
unset SMOKE_EXTRA_MOUNT
echo "MATRIX2 DONE"
cat matrix2-results.txt
