#!/usr/bin/env python3
"""Sustained 4-stream decode load, for measuring fabric traffic per step.

Usage: python3 bench_conc.py <base_url> [rounds] [tokens]
Prints one line per round: aggregate tok/s."""
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.0.101:8888"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 200
URL = BASE.rstrip("/") + "/v1/chat/completions"
PROMPT = "请写一篇 800 字的短文，主题：分布式推理系统的通信优化。"


def one(results, i):
    body = json.dumps({"model": "deepseek-v4.1-flash",
                       "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": NTOK, "temperature": 0.0}).encode()
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            URL, data=body, headers={"Content-Type": "application/json"}), timeout=900))
        ct = r["usage"]["completion_tokens"]
        results[i] = (ct, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        results[i] = (0, time.time() - t0)
        print("   request failed:", repr(e)[:100], flush=True)


for rnd in range(ROUNDS):
    res = [None] * 4
    t0 = time.time()
    ths = [threading.Thread(target=one, args=(res, i)) for i in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    tot = sum(x[0] for x in res)
    print(f"round {rnd+1}/{ROUNDS}: {tot} tok in {wall:.1f}s -> aggregate {tot/wall:.2f} tok/s", flush=True)
