#!/usr/bin/env python3
"""批次迁移用的 A/B 基准：C1 单流 + C4 并发聚合，每请求唯一 prompt。

与 bench_decode.py / bench_concurrency.py 的口径对齐（非流式 usage.completion_tokens、
被 whole-request wall 除），这样可以直接和 README §4 的基线对比。

两个刻意的设计：
  * **每请求唯一 prompt**（追加随机 nonce）——重复 prompt 会被 radix 前缀复用，
    让 wall 虚低、结果不可比（见 bench_prefill.py 的教训）。
  * **从 worker 跑**，不要在 head 上跑（scripts/verify/README 的约定）。

用法：
    python3 bench_migration.py <base_url> <label> [outdir] [reps] [ntok]

输出：一行行结果 + <outdir>/<label>.json（含每个样本，便于事后复核）。
"""
import json
import os
import random
import statistics
import string
import sys
import threading
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://192.168.0.86:8888").rstrip("/")
LABEL = sys.argv[2] if len(sys.argv) > 2 else "unlabeled"
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "."
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 300

URL = BASE + "/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"

TOPICS = [
    "a lighthouse keeper's last winter", "a city that only exists at night",
    "two rivals sharing a train compartment", "a violin found in a flooded cellar",
    "the first market day after a long war", "a cartographer who maps dreams",
    "a bakery run by retired sailors", "an orchard planted on a rooftop",
    "a letter delivered forty years late", "a clockmaker's apprentice in a silent town",
    "a storm seen from a mountain hut", "a chess club in a fishing village",
]


def nonce(n=8):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def request(idx, temp, max_tokens):
    """返回 (completion_tokens, wall_seconds, finish_reason)。唯一 prompt。"""
    topic = TOPICS[idx % len(TOPICS)]
    prompt = (f"Write a vivid short story (about 500 words) about {topic}. "
              f"Prose only, no headings. (ref {nonce()})")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
        **({"top_p": 0.95} if temp > 0 else {}),
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        obj = json.load(r)
    dt = time.time() - t0
    return obj["usage"]["completion_tokens"], dt, obj["choices"][0].get("finish_reason")


CODE_PROMPT = (
    "Write a complete Python implementation of an LRU cache class with a doubly linked list "
    "and a dict, plus a small unittest suite covering eviction order, get/put updates and "
    "capacity 1. Output code only, no commentary."
)


def request_code(idx, temp, max_tokens):
    """Code workload: the k=5 + confidence-cap gain upstream measured was on code, not prose."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": CODE_PROMPT + f" (variant {idx} {nonce()})"}],
        "max_tokens": max_tokens,
        "temperature": temp,
        **({"top_p": 0.95} if temp > 0 else {}),
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        obj = json.load(r)
    dt = time.time() - t0
    return obj["usage"]["completion_tokens"], dt, obj["choices"][0].get("finish_reason")


def c1_code(temp, reps, max_tokens, tag):
    rates, samples = [], []
    for i in range(reps):
        n, dt, fr = request_code(i, temp, max_tokens)
        rate = n / dt if dt > 0 else 0.0
        rates.append(rate)
        samples.append({"tokens": n, "wall": round(dt, 3), "tok_s": round(rate, 2), "finish": fr})
        print(f"  {tag} run{i+1}: {n} tok in {dt:.2f}s -> {rate:.2f} tok/s (finish={fr})", flush=True)
    med = statistics.median(rates) if rates else 0.0
    print(f"  {tag} MEDIAN: {med:.2f} tok/s   (all: {[round(x,2) for x in rates]})", flush=True)
    return med, rates, samples


def c1(temp, reps, max_tokens, tag):
    rates, samples = [], []
    for i in range(reps):
        n, dt, fr = request(i, temp, max_tokens)
        rate = n / dt if dt > 0 else 0.0
        rates.append(rate)
        samples.append({"tokens": n, "wall": round(dt, 3), "tok_s": round(rate, 2), "finish": fr})
        print(f"  {tag} run{i+1}: {n} tok in {dt:.2f}s -> {rate:.2f} tok/s (finish={fr})", flush=True)
    med = statistics.median(rates) if rates else 0.0
    print(f"  {tag} MEDIAN: {med:.2f} tok/s   (all: {[round(x,2) for x in rates]})", flush=True)
    return med, rates, samples


def c4(temp, reps, max_tokens, tag, width=4):
    aggs, samples = [], []
    for rnd in range(reps):
        res = [None] * width

        def one(i):
            try:
                res[i] = request(rnd * width + i, temp, max_tokens)
            except Exception as exc:  # noqa: BLE001
                res[i] = (0, 0.0, "error:" + repr(exc)[:60])

        t0 = time.time()
        ths = [threading.Thread(target=one, args=(i,)) for i in range(width)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        tot = sum(x[0] for x in res)
        agg = tot / wall if wall > 0 else 0.0
        aggs.append(agg)
        samples.append({"tokens": tot, "wall": round(wall, 3), "aggregate": round(agg, 2),
                        "per_request": [{"tokens": x[0], "wall": round(x[1], 3)} for x in res]})
        print(f"  {tag} round{rnd+1}: {tot} tok in {wall:.2f}s -> aggregate {agg:.2f} tok/s", flush=True)
    med = statistics.median(aggs) if aggs else 0.0
    print(f"  {tag} MEDIAN: {med:.2f} tok/s   (all: {[round(x,2) for x in aggs]})", flush=True)
    return med, aggs, samples


def main():
    print(f"# label={LABEL} target={BASE} model={MODEL} reps={REPS} ntok={NTOK}", flush=True)
    print(f"# 每请求唯一 prompt（无 radix 前缀复用）", flush=True)
    out = {"label": LABEL, "base": BASE, "reps": REPS, "ntok": NTOK,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print("\n[C1 散文 sampled temp=0.7]", flush=True)
    out["c1_sampled_median"], out["c1_sampled_all"], out["c1_sampled_samples"] = \
        c1(0.7, REPS, NTOK, "c1_sampled")

    print("\n[C1 散文 greedy temp=0.0]", flush=True)
    out["c1_greedy_median"], out["c1_greedy_all"], out["c1_greedy_samples"] = \
        c1(0.0, REPS, NTOK, "c1_greedy")

    print("\n[C1 代码 greedy temp=0.0]", flush=True)
    out["c1_code_median"], out["c1_code_all"], out["c1_code_samples"] = \
        c1_code(0.0, REPS, NTOK, "c1_code")

    print("\n[C4 并发聚合 greedy temp=0.0]", flush=True)
    out["c4_greedy_median"], out["c4_greedy_all"], out["c4_greedy_samples"] = \
        c4(0.0, REPS, 200, "c4_greedy")

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("\n=== SUMMARY " + json.dumps({
        "label": LABEL,
        "c1_sampled": round(out["c1_sampled_median"], 2),
        "c1_greedy": round(out["c1_greedy_median"], 2),
        "c1_code": round(out["c1_code_median"], 2),
        "c4_greedy": round(out["c4_greedy_median"], 2),
    }), flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, LABEL + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"# wrote {path}", flush=True)


if __name__ == "__main__":
    main()
