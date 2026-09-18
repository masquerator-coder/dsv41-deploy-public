#!/usr/bin/env python3
"""Apply the AICAD ring-only NCCL patch stack to a clean v2.30.7-1 tree, with the
v4 per-peer device map rewritten for OUR 3-node triangle.

Upstream (luxingcom/aicad-nccl-optimization, patches/README.md) maps a 4-node ring:
    map[myRank][peerRank] = {evenChannelDev, oddChannelDev},  dev = NCCL_IB_HCA order 0..3
and each {a,b} pair is the two PCI *views* of the same physical wire.

Our triangle (W1 = 01-p0<->03-p0, W2 = 01-p1<->02-p0, W3 = 02-p1<->03-p1) maps each
rank's HCA list as [wireX-viewA, wireY-viewA, wireX-viewB, wireY-viewB]:

    rank0 head : [0]=rocep1s0f1(180.2,W2) [1]=rocep1s0f0(178.2,W1)
                 [2]=roceP2p1s0f1(181.2,W2) [3]=roceP2p1s0f0(179.2,W1)
    rank1 node2: [0]=rocep1s0f0(180.1,W2) [1]=rocep1s0f1(176.2,W3)
                 [2]=roceP2p1s0f0(181.1,W2) [3]=roceP2p1s0f1(177.2,W3)
    rank2 node3: [0]=rocep1s0f0(178.1,W1) [1]=rocep1s0f1(176.1,W3)
                 [2]=roceP2p1s0f0(179.1,W1) [3]=roceP2p1s1(177.1,W3)

=> {0,2} is the wire of the first pair, {1,3} the second, per rank."""
import re
import subprocess
import sys

ROOT = '/home/user/nccl'
PATCHES = ['v1-ring-only.patch', 'v4-netdev-hardcode.patch',
           'stageB-tuner-two-band.patch', 'stageB-hardened-two-branch.patch']


def run(cmd, check=True):
    p = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    print(f"$ {cmd}\n{p.stdout.strip()}{p.stderr.strip()}")
    if check and p.returncode != 0:
        sys.exit(f"FAILED: {cmd}")
    return p


print("=== 1. revert my earlier device-order experiment ===")
run('git checkout -- src/transport/net_ib/common.h src/transport/net_ib/init.cc')
run('git status --short')

print("=== 2. apply the AICAD patch stack (order per patches/README.md) ===")
for p in PATCHES:
    run(f'git apply --verbose /tmp/aicad-patches/{p}')

print("=== 3. rewrite the v4 per-peer map for the 3-node triangle ===")
src = open(f'{ROOT}/src/transport/net.cc', encoding='utf-8').read()
old_map = re.search(r'static const int map\[4\]\[4\]\[2\] = \{.*?\n  \};', src, re.S)
if not old_map:
    sys.exit('v4 map not found -- patch not applied?')
new_map = '''static const int map[4][4][2] = {
    /* fq: 3-node triangle (was the upstream 4-node ring map).
     * HCA order per rank = [wireA-view1, wireB-view1, wireA-view2, wireB-view2];
     * the two entries of each pair are the two PCI views of ONE physical wire. */
    {{0,0},{0,2},{1,3},{0,0}},   /* rank0 head  : p1=W2->rank1 {0,2} ; p2=W1->rank2 {1,3} */
    {{0,2},{0,0},{1,3},{0,0}},   /* rank1 node2 : W2->rank0 {0,2} ; W3->rank2 {1,3} */
    {{0,2},{1,3},{0,0},{0,0}},   /* rank2 node3 : W1->rank0 {0,2} ; W3->rank1 {1,3} */
    {{0,0},{0,0},{0,0},{0,0}},
  };'''
src = src[:old_map.start()] + new_map + src[old_map.end():]
open(f'{ROOT}/src/transport/net.cc', 'w', encoding='utf-8', newline='\n').write(src)

print("=== 4. markers present? ===")
for f, needle in [('src/transport.cc', 'RING-ONLY PATCH'),
                  ('src/transport/net.cc', 'ncclRingDevOverride'),
                  ('src/enqueue.cc', 'ncclPersizeTunerOverride')]:
    t = open(f'{ROOT}/{f}', encoding='utf-8').read()
    print(f"  {f}: {needle} -> {'OK' if needle in t else 'MISSING'}")
print(open(f'{ROOT}/src/transport/net.cc', encoding='utf-8').read()[
      open(f'{ROOT}/src/transport/net.cc', encoding='utf-8').read().find('ncclRingDevOverride') - 300:
      open(f'{ROOT}/src/transport/net.cc', encoding='utf-8').read().find('ncclRingDevOverride') + 700])
