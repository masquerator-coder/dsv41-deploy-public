import json, subprocess, time, sys

URL = "http://192.168.0.101:8888/v1/chat/completions"
PROMPT = "请写一篇 800 字的短文，主题：分布式推理系统的通信优化。"

def run(mt, tag):
    body = json.dumps({
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": mt,
        "temperature": 0.7,
        "chat_template_kwargs": {"thinking": False},
    })
    t0 = time.time()
    out = subprocess.run(["curl", "-s", "-m", "600", URL, "-H", "Content-Type: application/json",
                          "-d", body], capture_output=True, text=True).stdout
    dt = time.time() - t0
    try:
        u = json.loads(out)["usage"]
    except Exception as e:
        print(tag, "parse error:", e, out[:200]); return
    n = u["completion_tokens"]
    print("%s: tokens=%d prompt=%d wall=%.2fs  decode=%.1f tok/s" % (tag, n, u["prompt_tokens"], dt, n / dt))

if __name__ == "__main__":
    mt = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for i in (1, 2, 3):
        run(mt, "run%d" % i)
