#!/usr/bin/env python3
"""生成一张内容明确的 PNG（纯标准库，无 PIL 依赖），用于验证服务视觉分支。

图：白底 256×256，左上角蓝色实心方块，中央红色实心圆，底部一条黑色横条。
若模型能说出"红圆 / 蓝方块 / 黑条"，即视觉分支工作。
"""
import struct
import sys
import zlib

W = H = 256
buf = bytearray()
for y in range(H):
    row = bytearray([0])                      # filter type 0
    for x in range(W):
        r = g = b = 255                       # 白底
        if x < 80 and y < 80:                 # 左上蓝方块
            r, g, b = 0, 80, 255
        cx, cy, rad = 128, 128, 60            # 中央红圆
        if (x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2:
            r, g, b = 220, 30, 30
        if 200 <= y <= 230:                   # 底部黑条
            r, g, b = 0, 0, 0
        row += bytes((r, g, b))
    buf += row


def chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(bytes(buf), 9))
       + chunk(b"IEND", b""))

out = sys.argv[1] if len(sys.argv) > 1 else "vision-test.png"
open(out, "wb").write(png)
print(f"已生成 {out}（{W}x{H}, {len(png)} bytes）：左上蓝方块 + 中央红圆 + 底部黑条")
