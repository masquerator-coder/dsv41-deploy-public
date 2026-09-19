#!/usr/bin/env python3
"""验证服务的两条"旁路"能力：视觉分支（带图请求）与工具调用（DSML 解析）。

用法：python3 verify_extras.py [base_url]
"""
import base64
import json
import sys
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888").rstrip("/")
URL = BASE + "/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
IMG = "vision-test.png"


def post(payload, timeout=180):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


print("=" * 62)
print("① 视觉分支：带图请求")
b64 = base64.b64encode(open(IMG, "rb").read()).decode()
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "请描述这张图：有哪些形状、什么颜色、在什么位置？只回答要点。"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}],
    "max_tokens": 160, "temperature": 0,
    "chat_template_kwargs": {"thinking": False},
}
try:
    r = post(payload)
    txt = r["choices"][0]["message"]["content"] or ""
    print("  回答：", txt.strip().replace("\n", " ")[:300])
    hit = [k for k in ("红", "蓝", "黑") if k in txt]
    print(f"  关键词命中 {hit}（期望 红/蓝/黑 全中）→ "
          + ("✅ 视觉分支工作" if len(hit) == 3 else "⚠️ 需人工判断"))
except Exception as e:  # noqa: BLE001
    print("  ✗ 视觉请求失败：", repr(e)[:300])

print("=" * 62)
print("② 工具调用：DSML 标签解析（注意 V4.1 的标签带前导空格）")
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    },
}]
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "北京现在天气怎么样？用工具查一下。"}],
    "tools": tools, "max_tokens": 256, "temperature": 0,
    "chat_template_kwargs": {"thinking": False},
}
try:
    r = post(payload)
    msg = r["choices"][0]["message"]
    tc = msg.get("tool_calls")
    if tc:
        print("  tool_calls：", json.dumps(tc, ensure_ascii=False)[:300])
        ok = tc[0]["function"]["name"] == "get_weather" and "北京" in tc[0]["function"]["arguments"]
        print("  → " + ("✅ 工具调用解析正确" if ok else "⚠️ 解析结果需人工判断"))
    else:
        print("  未返回 tool_calls；content：", (msg.get("content") or "")[:200])
        print("  → ⚠️ 可能是模型没用工具，或 DSML 解析器未生效（看服务端日志有无解析告警）")
except Exception as e:  # noqa: BLE001
    print("  ✗ 工具调用请求失败：", repr(e)[:300])
print("=" * 62)
