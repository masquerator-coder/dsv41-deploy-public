#!/usr/bin/env python3
"""长请求稳健性测试：重复 + 并发，验证 CHUNKED_PREFILL_SIZE=1024 + indexer ON 不是"侥幸过一次"。

两个阶段：
  A) 顺序重复：N 次 ~200k token 的唯一 prompt，观察是否每次都成功、耗时是否稳定；
  B) 并发压力：W 个 ~80k token 的请求同时发出，观察是否 OOM / 报错。

每步后探一次 /health，捕捉"某次之后服务变坏"的情况。

用法：python3 longctx_robust.py <base_url> <label> [seq_n] [seq_tok] [conc_w] [conc_tok]
"""
import json
import random
import string
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://192.168.0.86:8888").rstrip("/")
LABEL = sys.argv[2] if len(sys.argv) > 2 else "robust"
SEQ_N = int(sys.argv[3]) if len(sys.argv) > 3 else 5
SEQ_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 200000
CONC_W = int(sys.argv[5]) if len(sys.argv) > 5 else 3
CONC_TOK = int(sys.argv[6]) if len(sys.argv) > 6 else 80000

URL = BASE + "/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
ALPHA = string.ascii_lowercase + string.digits
TOK_PER_WORD = 4.41          # 实测标定（1 个 6 字符随机词 ≈ 4.41 token）


def nonce(n=10):
    return "".join(random.choice(ALPHA) for _ in range(n))


def build(n_tokens):
    n_words = int(n_tokens / TOK_PER_WORD)
    body = " ".join("".join(random.choice(ALPHA) for _ in range(6)) for _ in range(n_words))
    return ("下面是一段随机字符串。请只回答最后一个字符是什么（只输出该字符）：\n" + body)


def post(prompt, max_tokens=8):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r["usage"], time.time() - t0


def health():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=8) as r:
            return r.status
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}"


print(f"# label={LABEL} base={BASE}")
print(f"# A) 顺序 {SEQ_N} 次 × ~{SEQ_TOK} tok   B) 并发 {CONC_W} × ~{CONC_TOK} tok")
print(f"# 起始 /health = {health()}")
rows = {"seq": [], "conc": None}

print(f"\n=== A) 顺序重复（目标 ~{SEQ_TOK} tok/次）===")
for i in range(1, SEQ_N + 1):
    p = build(SEQ_TOK)
    t0 = time.time()
    try:
        u, dt = post(p)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:250]
        print(f"  #{i}: HTTP {e.code} {body}", flush=True)
        rows["seq"].append({"i": i, "ok": False, "http": e.code, "body": body})
        break
    except Exception as e:  # noqa: BLE001
        print(f"  #{i}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
        rows["seq"].append({"i": i, "ok": False, "error": f"{type(e).__name__}: {e}"})
        break
    pt = u["prompt_tokens"]
    rate = pt / dt if dt else 0
    h = health()
    print(f"  #{i}: pt={pt} wall={dt:.1f}s prefill={rate:.0f} tok/s  health={h}", flush=True)
    rows["seq"].append({"i": i, "ok": True, "prompt_tokens": pt, "wall": round(dt, 2),
                        "prefill_tok_s": round(rate, 1), "health_after": h})

print(f"\n=== B) 并发 {CONC_W} × ~{CONC_TOK} tok ===")
res = [None] * CONC_W


def one(k):
    p = build(CONC_TOK)
    try:
        u, dt = post(p)
        res[k] = {"ok": True, "prompt_tokens": u["prompt_tokens"], "wall": round(dt, 2)}
    except Exception as e:  # noqa: BLE001
        res[k] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


t0 = time.time()
ths = [threading.Thread(target=one, args=(k,)) for k in range(CONC_W)]
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.time() - t0
rows["conc"] = {"wall": round(wall, 2), "results": res, "health_after": health()}
for k, r in enumerate(res):
    print(f"  worker{k}: {r}", flush=True)
print(f"  并发总 wall={wall:.1f}s  结束 /health={rows['conc']['health_after']}")

ok_seq = [r for r in rows["seq"] if r.get("ok")]
ok_conc = [r for r in res if r and r.get("ok")]
print(f"\n=== 结论 ===")
print(f"  顺序：{len(ok_seq)}/{SEQ_N} 成功")
if ok_seq:
    walls = [r["wall"] for r in ok_seq]
    print(f"    wall 范围 {min(walls):.1f}–{max(walls):.1f}s（离散度 {(max(walls)-min(walls))/min(walls)*100:.1f}%）")
print(f"  并发：{len(ok_conc)}/{CONC_W} 成功")

rows["label"] = LABEL
rows["seq_n"], rows["seq_tok"] = SEQ_N, SEQ_TOK
rows["conc_w"], rows["conc_tok"] = CONC_W, CONC_TOK
rows["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(rows, open(f"{LABEL}.json", "w"), ensure_ascii=False, indent=2)
print(f"# wrote {LABEL}.json")
