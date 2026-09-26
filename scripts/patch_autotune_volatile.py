#!/usr/bin/env python3
"""把 DSV41_INDEXER_CHUNKED 加入 adapter/autotune_keep.py 的 _VOLATILE。

为什么：该开关切换时会改变 launch fingerprint，强制 FlashInfer 重新 autotune，
使任何针对它的 A/B 都被重 tune 噪声污染（2026-09-26 实测：ON/OFF 各触发一次
`tuned and saved`）。dense prefill indexer 是 DeepGEMM fp4 kernel + top-k，
不是 FlashInfer 调优的算子，与已在 _VOLATILE 里的 DSV41_DRAFT_HEAD_FP8 /
DSV41_ENGRAM_PREFETCH 同类。

设计要点：
  * 只插入注释与一个元组元素，不改写任何既有行；
  * **保留原文件的行尾风格**（head 上该文件是 CRLF，与其它 adapter 不同）；
  * 锚点精确匹配 + 幂等 + ast 自校验；
  * 不碰其它文件。

用法（在部署目录下）：
    python3 apply_autotune_volatile_patch.py --check
    python3 apply_autotune_volatile_patch.py --apply
    python3 apply_autotune_volatile_patch.py --revert
"""
import ast
import glob
import hashlib
import os
import shutil
import sys
import time

TARGET = "adapter/autotune_keep.py"
SENTINEL = '"DSV41_INDEXER_CHUNKED"'


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def fail(msg, code=1):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(code)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode not in ("--check", "--apply", "--revert"):
        fail("用法: apply_autotune_volatile_patch.py [--check|--apply|--revert]", 2)
    if not os.path.isfile(TARGET):
        fail(f"{TARGET} 不存在（请在 ~/dsv41-3xspark 下运行）")

    if mode == "--revert":
        baks = sorted(glob.glob(TARGET + ".bak-before-volatile-*"))
        if not baks:
            fail("找不到 .bak-before-volatile-* 备份")
        shutil.copy2(baks[-1], TARGET)
        print(f"已从 {baks[-1]} 还原")
        print(f"  还原后 md5={md5(TARGET)}")
        return

    raw = open(TARGET, "rb").read()
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n")
    eol = "\r\n" if crlf and crlf == lf else "\n"
    text = raw.decode("utf-8")

    print(f"目标 {TARGET}")
    print(f"  bytes={len(raw)} CRLF={crlf} LF={lf} -> 行尾={'CRLF' if eol == chr(13) + chr(10) else 'LF'}")
    print(f"  md5={md5(TARGET)}")

    if SENTINEL in text:
        print(f"already patched: 已含 {SENTINEL}，无需改动")
        return

    # 锚点：_VOLATILE 元组的最后一行（SGLANG_RUN_ID / DSV41_WO_A_W8_DRAFT）
    anchor = ('             "SGLANG_RUN_ID", "DSV41_WO_A_W8_DRAFT")' + eol)
    n = text.count(anchor)
    print(f"  锚点命中 {n} 次（期望 1）")
    if n != 1:
        fail("锚点不匹配，拒绝改动（autotune_keep.py 可能已漂移）")

    insertion = (
        '             "SGLANG_RUN_ID", "DSV41_WO_A_W8_DRAFT",' + eol +
        '             # indexer 2026-09-26: the dense prefill indexer is a DeepGEMM fp4 kernel' + eol +
        '             # plus a top-k, not a FlashInfer-tuned op, so toggling it must reuse the' + eol +
        '             # tactics; a flip that re-tunes makes any A/B over this switch meaningless' + eol +
        '             # (measured on TP3 2026-09-26: each flip logged "tuned and saved")' + eol +
        '             "DSV41_INDEXER_CHUNKED")' + eol
    )
    out = text.replace(anchor, insertion, 1)

    added = out.count("\n") - text.count("\n")
    print(f"  预计新增 {added} 行")

    try:
        ast.parse(out)
    except SyntaxError as exc:
        fail(f"补丁后语法错误：{exc}")

    import difflib
    dels = [l for l in difflib.unified_diff(text.splitlines(), out.splitlines(), n=0)
            if l.startswith("-") and not l.startswith("---")]
    # 本补丁天然含 1 处行内修改：元组末尾原本是 `...W8_DRAFT")`，需补一个逗号才能续接新元素。
    # 除这一行外不允许任何删除/改写。
    anchor_line = anchor.rstrip("\r\n")
    if dels:
        if len(dels) != 1 or dels[0] != "-" + anchor_line:
            print("  非预期的删除行：", file=sys.stderr)
            for d in dels[:5]:
                print("    " + d, file=sys.stderr)
            fail("只允许锚点行的尾逗号修改，其余不得改动")
        if anchor_line[:-1] + "," not in out:
            fail("锚点行内容未按预期保留（应仅追加尾逗号）")
        print("  含 1 处预期内修改：锚点行末尾补逗号（内容保留）")
    print(f"  删除行数 {len(dels)}（其中 0 处为内容删除）")

    if mode == "--check":
        print("\n--check 通过：锚点正常，未写入任何内容")
        return

    bak = f"{TARGET}.bak-before-volatile-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(TARGET, bak)
    open(TARGET, "wb").write(out.encode("utf-8"))
    ast.parse(open(TARGET, encoding="utf-8").read())
    print(f"\n已写入 {TARGET}")
    print(f"  备份 {bak}")
    print(f"  md5 {md5(bak)} -> {md5(TARGET)}")
    # 复读校验：行尾风格必须与原文件一致
    raw2 = open(TARGET, "rb").read()
    crlf2 = raw2.count(b"\r\n")
    print(f"  复读 CRLF={crlf2}（原 {crlf}，应增加 {added - len(dels)} 左右）")
    assert raw2.count(b"\n") - crlf2 == 0 or crlf2 == raw2.count(b"\n"), "行尾风格被破坏"
    print(f"  DSV41_INDEXER_CHUNKED 出现 {open(TARGET, encoding='utf-8').read().count(SENTINEL)} 次")


if __name__ == "__main__":
    main()
