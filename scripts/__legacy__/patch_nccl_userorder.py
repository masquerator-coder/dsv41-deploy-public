#!/usr/bin/env python3
"""fq patch: make NCCL register IB devices in the order of the NCCL_IB_HCA entries.

Why: on a multi-subnet direct-connect fabric (3-node, 2-port triangle) the NIC that
is reachable per peer differs per node, while NCCL aligns its channel->NIC index
across ranks. The PCI-path order is identical on every node, so it cannot express
that; the user's NCCL_IB_HCA order can, e.g. "rocep1s0f0,rocep1s0f1" on two ranks
and "rocep1s0f1,rocep1s0f0" on the third."""
import sys

h = 'src/transport/net_ib/common.h'
c = 'src/transport/net_ib/init.cc'

s = open(h).read()
anchor = "  int16_t planeIdx;\n"
assert s.count(anchor) == 1, ("common.h anchor count", s.count(anchor))
s = s.replace(anchor, anchor +
              "  int16_t userIdx; // fq: index of this device in NCCL_IB_HCA (registration order)\n")
open(h, 'w').write(s)

t = open(c).read()
a = "            ncclIbDevs[ncclNIbDevs].planeId = (userIfId >= 0) ? userIfs[userIfId].plane : -1;\n"
assert t.count(a) == 1, ("init.cc planeId anchor", t.count(a))
t = t.replace(a, a +
              "            ncclIbDevs[ncclNIbDevs].userIdx = (userIfId >= 0) ? (int16_t)userIfId : (int16_t)0x7fff;\n")

comp = "static int ncclIbCompareDevs(const void* dev1, const void* dev2) {"
assert t.count(comp) == 1, ("comparator anchor", t.count(comp))
newcomp = """// fq patch: order devices by their position in NCCL_IB_HCA. On a multi-subnet
// direct-connect fabric (e.g. a 3-node, 2-port triangle) the NIC that is reachable
// per peer differs per node, and NCCL aligns its channel->NIC index across ranks.
// The PCI-path order is identical on every node, so it cannot express that; the
// user's NCCL_IB_HCA order can (e.g. "rocep1s0f0,rocep1s0f1" on two ranks and
// "rocep1s0f1,rocep1s0f0" on the third).
static int ncclIbCompareUserOrder(const void* dev1, const void* dev2) {
  int16_t i1 = ((struct ncclIbDev*)dev1)->userIdx;
  int16_t i2 = ((struct ncclIbDev*)dev2)->userIdx;
  if (i1 != i2) return (i1 < i2) ? -1 : 1;
  return ncclIbCompareDevs(dev1, dev2);
}

""" + comp
t = t.replace(comp, newcomp, 1)

sortline = "    if (ncclParamIbDevicePciOrder()) qsort(ncclIbDevs, ncclNIbDevs, sizeof(struct ncclIbDev), ncclIbCompareDevs);"
assert t.count(sortline) == 1, ("sort anchor", t.count(sortline))
t = t.replace(sortline, sortline +
              "\n    // fq patch: then re-order by the NCCL_IB_HCA order\n"
              "    qsort(ncclIbDevs, ncclNIbDevs, sizeof(struct ncclIbDev), ncclIbCompareUserOrder);")
open(c, 'w').write(t)
print("patched OK")
