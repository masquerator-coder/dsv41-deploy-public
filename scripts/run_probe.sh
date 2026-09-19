#!/usr/bin/env bash
# 3-rank probe driver, ENGINE IMAGE by default (dsv41-3xspark:local), engine-like NCCL env.
# 用法（head 上）： ./run_probe.sh [extra -e K=V ...]
set -u
IMG=${IMG:-dsv41-3xspark:local}
PORT=${PORT:-29611}
MASTER=${MASTER:-192.168.0.86}
OUT=${OUT:-$HOME/dsv41-3xspark/probe-$(date +%H%M%S)}
PROBE=${PROBE:-$HOME/pynccl_probe3.py}
mkdir -p "$OUT"
for h in fq-dgx-02.local fq-dgx-03.local; do
  scp -q -o StrictHostKeyChecking=accept-new "$PROBE" "$h:./pynccl_probe3.py" || exit 1
done
COMMON=(--rm --entrypoint python3 --network host --ipc host --privileged --cap-add IPC_LOCK --gpus all
  --device /dev/infiniband:/dev/infiniband --ulimit memlock=-1:-1 --ulimit stack=67108864
  -e NCCL_CUMEM_ENABLE=0 -e MASTER_ADDR="$MASTER" -e MASTER_PORT="$PORT" -e WORLD_SIZE=3
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 -e NCCL_IB_GID_INDEX=3
  -e NCCL_SOCKET_IFNAME=wlP9s9 -e GLOO_SOCKET_IFNAME=wlP9s9 -e NCCL_IB_SUBNET_AWARE_ROUTING=1
  -e NCCL_NET_PLUGIN=none -e NCCL_IB_MERGE_NICS=0 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1
  -e NCCL_CROSS_NIC=0 -e NCCL_NET_GDR_LEVEL=LOC -e NCCL_DMABUF_ENABLE=1
  -e NCCL_PROTO='^LL128' -e NCCL_MAX_NCHANNELS=8 -e NCCL_BUFFSIZE=1048576 -e NCCL_LL128_BUFFSIZE=262144
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET,ENV
  -e NCCL_LIB_PATH=/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
  -v "$HOME/pynccl_probe3.py:/probe.py:ro")
echo "[+] image=$IMG out=$OUT extra: $*"
i=1
for h in fq-dgx-02.local fq-dgx-03.local; do
  ssh -o StrictHostKeyChecking=accept-new "$h" \
    "timeout 240 docker run ${COMMON[*]} -e RANK=$i $* $IMG /probe.py" >"$OUT/rank$i.log" 2>&1 &
  i=$((i + 1))
done
sleep 4
timeout 240 docker run "${COMMON[@]}" -e RANK=0 "$@" "$IMG" /probe.py >"$OUT/rank0.log" 2>&1
wait
echo "===== 结果 $OUT ====="
for f in "$OUT"/rank*.log; do
  echo "--- $f"
  grep -E "^\[rank|STAGE-|PROBE-|Error|Traceback" "$f" | tail -8
done
grep -q "PROBE-DONE" "$OUT/rank0.log" && echo "[+] PROBE: PASS" || echo "[x] PROBE: FAIL/卡住"
grep -c "ibv_reg_mr_iova2" "$OUT"/rank*.log | tr '\n' ' '; echo "(reg_mr 失败计数)"
