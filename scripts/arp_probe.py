#!/usr/bin/env python3
# L2 ARP 探针：从本机每个 fabric 口发出 ARP 请求，记录谁回了（sender MAC/IP）。
# 不依赖 IP 规划，直接回答“哪个远端物理口插在本机哪个口上”。
# 用法（需 root：AF_PACKET 原始套接字）： sudo -n python3 - < arp_probe.py
import socket, struct, subprocess

ALL_IPS = ["10.100.176.1", "10.100.176.2", "10.100.177.1", "10.100.177.2",
           "10.100.178.1", "10.100.178.2", "10.100.179.1", "10.100.179.2",
           "10.100.180.1", "10.100.180.2", "10.100.181.1", "10.100.181.2"]


def own_addrs():
    out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 4 and p[1].startswith(("enp", "enP")) and p[2] == "inet":
            d.setdefault(p[1], []).append(p[3].split("/")[0])
    return d


def read_mac(i):
    with open("/sys/class/net/%s/address" % i) as f:
        return f.read().strip()


def main():
    own = own_addrs()
    local_ips = {ip for v in own.values() for ip in v}
    print("HOST %s" % socket.gethostname())
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0806))
    s.settimeout(1.2)
    for iface in sorted(own):
        s.bind((iface, 0))
        src_ip = own[iface][0]
        src_mac_s = read_mac(iface)
        src_mac = bytes.fromhex(src_mac_s.replace(":", ""))
        print("IFACE %s mac=%s ip=%s" % (iface, src_mac_s, src_ip))
        for tgt in ALL_IPS:
            if tgt in local_ips:
                continue
            frame = (b"\xff" * 6 + src_mac
                     + struct.pack("!H", 0x0806)                       # ethertype
                     + struct.pack("!HHBBH", 1, 0x0800, 6, 4, 1)       # htype/ptype/hlen/plen/op=request
                     + src_mac + socket.inet_aton(src_ip)
                     + b"\x00" * 6 + socket.inet_aton(tgt))
            try:
                s.send(frame)
            except OSError as e:
                print("  send-error %s %s" % (tgt, e))
        seen = []
        end = 2.0
        while True:
            try:
                data, _ = s.recvfrom(2048)
            except socket.timeout:
                break
            if len(data) < 42 or data[12:14] != b"\x08\x06" or data[20:22] != b"\x00\x02":
                continue
            smac = ":".join("%02x" % b for b in data[22:28])
            sip = socket.inet_ntoa(data[28:32])
            if sip in local_ips:
                continue
            if (sip, smac) not in seen:
                seen.append((sip, smac))
        for sip, smac in sorted(seen):
            print("  <- %s %s" % (sip, smac))
        print("SUMMARY %s %s -> %s" % (iface, src_mac_s,
                                       ",".join("%s@%s" % (m, i) for i, m in sorted(seen))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
