import json, subprocess, time, sys

URL = "http://192.168.0.101:8888/v1/chat/completions"

def ttft(n_paras, max_tokens=16):
    # build a prompt of roughly n_paras*~40 tokens
    filler = "在分布式推理系统中，张量并行需要每一步都做集合通信。" * n_paras
    prompt = "请阅读下面的材料，然后用一句话概括它讲的是什么：\n" + filler
    body = json.dumps({
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
        "chat_template_kwargs": {"thinking": False},
    })
    t0 = time.time()
    p = subprocess.Popen(["curl", "-s", "-N", "-m", "900", URL, "-H", "Content-Type: application/json",
                          "-d", body], stdout=subprocess.PIPE, text=True)
    first = None
    ptok = 0
    for line in p.stdout:
        if line.startswith("data: ") and first is None and '"content"' in line:
            first = time.time() - t0
        if line.startswith("data: ") and '"usage"' in line:
            try:
                u = json.loads(line[6:])["usage"]; ptok = u.get("prompt_tokens", 0)
            except Exception:
                pass
    p.wait()
    if first:
        print("prompt_tokens=%d TTFT=%.2fs  prefill=%.0f tok/s" % (ptok, first, ptok / first if first else 0))
    else:
        print("no TTFT observed (prompt_tokens=%d)" % ptok)

if __name__ == "__main__":
    for n in (10, 25, 50):
        ttft(n)
