#!/usr/bin/env python3
"""Give nccl_smoke.sh per-rank NCCL_IB_HCA lists (SMOKE_HCA_R0/R1/R2).

Why: with the patched host NCCL (libnccl in $NCCL_HOST_DIR) the *order* of the
NCCL_IB_HCA entries becomes the net-device index order. On this triangle the only
assignment that closes both NCCL rings is:

  rank0 (head) : [W2 port -> node2, W1 port -> node3] = rocep1s0f1,rocep1s0f0
  rank1 (node2): [W3 port -> node3, W2 port -> head]  = rocep1s0f1,rocep1s0f0
  rank2 (node3): [W1 port -> head,  W3 port -> node2] = rocep1s0f0,rocep1s0f1

so channel c (which uses net device c mod nNets) always gets a ring that closes.
The extra -e is appended AFTER ${ARGS_BASE[*]}, and docker honours the last -e."""
p = 'nccl_smoke.sh'
s = open(p, encoding='utf-8').read()

old_w = '''i=1
for h in "${WORKER_HOSTS[@]}"; do
  ssh -o StrictHostKeyChecking=accept-new "$h" \\
    "docker run ${ARGS_BASE[*]} -e RANK=$i $IMG ${RUNCMD[*]}" >"$OUT/rank$i.log" 2>&1 &
  i=$((i + 1))
done
'''
new_w = '''i=1
for h in "${WORKER_HOSTS[@]}"; do
  # fq: per-rank HCA *order* (SMOKE_HCA_R1/R2). Appended after ARGS_BASE so the later -e wins.
  hca=""; eval "hca=\\${SMOKE_HCA_R$i:-}"
  extra=""; [ -n "$hca" ] && extra="-e NCCL_IB_HCA=$hca"
  ssh -o StrictHostKeyChecking=accept-new "$h" \\
    "docker run ${ARGS_BASE[*]} $extra -e RANK=$i $IMG ${RUNCMD[*]}" >"$OUT/rank$i.log" 2>&1 &
  i=$((i + 1))
done
'''
old_h = 'docker run "${ARGS_BASE[@]}" -e RANK=0 "$IMG" "${RUNCMD[@]}" >"$OUT/rank0.log" 2>&1'
new_h = ('HCA0_ARGS=(); [ -n "${SMOKE_HCA_R0:-}" ] && HCA0_ARGS=(-e "NCCL_IB_HCA=${SMOKE_HCA_R0}")\n'
         'docker run "${ARGS_BASE[@]}" "${HCA0_ARGS[@]}" -e RANK=0 "$IMG" "${RUNCMD[@]}" >"$OUT/rank0.log" 2>&1')

n1 = s.count(old_w); n2 = s.count(old_h)
if n1 != 1 or n2 != 1:
    raise SystemExit(f'anchor mismatch: workers={n1} head={n2}')
s = s.replace(old_w, new_w, 1).replace(old_h, new_h, 1)
open(p, 'w', encoding='utf-8', newline='').write(s)
print('patched per-rank HCA support into nccl_smoke.sh')
