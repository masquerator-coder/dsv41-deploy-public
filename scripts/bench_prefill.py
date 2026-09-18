#!/usr/bin/env python3
"""预填（prefill）吞吐测量 —— 非流式 usage 口径 + 每题唯一 prompt。

两个必须避开的坑（都踩过）：
  1) 流式路径的 `usage` 里拿不到 `prompt_tokens`（实测恒为 0）→ 必须用非流式；
  2) 用**重复 filler** 造 prompt 会被 radix cache 前缀复用，越长的 prompt 越"快"
     （实测虚高 1.5–2×）→ 这里每题都生成**唯一随机串**，缓存不可能命中。

口径说明（报数时请一并说明）：
  * 非流式 wall = 预填 + 解码 max_tokens 个 token；
  * 先测一次纯解码速率，再从 wall 里扣掉解码时间 → "扣解码后"列；
  * "原始"列 = prompt_tokens / wall，便于与别人的口径对齐。

用法：python3 bench_prefill.py [base_url] [max_tokens]
"""
import json
import random
import string
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888").rstrip("/")
URL = BASE + "/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 16
ALPHA = string.ascii_lowercase + string.digits


def rand_word(n=None):
    n = n or random.randint(3, 7)
    return "".join(random.choice(ALPHA) for _ in range(n))


def unique_prompt(approx_tokens):
    """每题唯一（随机串），并保证答案极短：只回答数字。"""
    body = " ".join(rand_word() for _ in range(approx_tokens))
    return ("下面是随机字符串，请统计其中包含多少个数字字符，只回答数字：\n" + body)


def post(prompt, max_tokens):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, data=payload, headers={"Content-Type": "application/json"}), timeout=900))
    return r["usage"], time.time() - t0


def decode_rate():
    """纯解码速率（短 prompt、长输出），用于扣减解码时间。"""
    u, dt = post("请从 1 数到 200。", 200)
    return u["completion_tokens"] / dt


print(f"# 目标 {BASE}  模型 {MODEL}  非流式 temp=0 thinking=off  每题唯一随机 prompt（无缓存复用）")
dr = decode_rate()
print(f"# 纯解码基准：{dr:.1f} tok/s（用于扣减 {NTOK} tok 的解码时间）\n")
print(f"{'prompt_tok':>10} {'wall_s':>8} {'原始 pt/wall':>15} {'扣解码后 prefill':>18}")
rows = []
for approx in (500, 1000, 2000, 4000, 8000):
    prompt = unique_prompt(approx)
    u, dt = post(prompt, NTOK)
    pt, ct = u["prompt_tokens"], u["completion_tokens"]
    adj = dt - ct / dr
    raw = pt / dt if dt > 0 else 0
    corr = pt / adj if adj > 0.01 else float("inf")
    rows.append((pt, raw, corr))
    print(f"{pt:>10} {dt:>8.2f} {raw:>13.0f} tok/s {corr:>14.0f} tok/s")

big = [r for r in rows if r[0] >= 3000]
if big:
    print(f"\n# ≈4k token 档（扣解码后）均值：{sum(r[2] for r in big)/len(big):.0f} tok/s")
