"""豆包 API 诊断 —— 把每个返回都打出来（不 pass）"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DOUBAO_API_KEY")
CHAT_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
EMBED_URL = "https://ark.cn-beijing.volces.com/api/v3/embeddings"

print("=" * 60)
print("  豆包 API 诊断（全量输出版）")
print("=" * 60)
print(f"Key: {API_KEY[:6]}...{API_KEY[-4:]}" if API_KEY and len(API_KEY) > 10 else "Key: 未配置")
print(f"Key长度: {len(API_KEY) if API_KEY else 0}")
print()

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# 每个模型都打，不 pass
models = [
    "doubao-seed-1.6",
    "doubao-seed-1.8",
    "doubao-pro-32k",
    "doubao-lite-32k",
    "doubao-seed-1.6-09243",
    "doubao-pro-online-32k",
    "doubao-lite-online-32k",
    "doubao-2.0-pro-32k",
]

print("  逐个测试模型:")
any_ok = False
for model in models:
    payload = {"model": model, "messages": [{"role": "user", "content": "回复OK"}], "max_tokens": 10}
    try:
        r = requests.post(CHAT_URL, json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "OK")
            print(f"    ✓ {model:30s} → {text[:40]}")
            any_ok = True
        else:
            err = r.json().get("error", {})
            code = err.get("code", "")
            msg = err.get("message", "")[:70]
            print(f"    ✗ {model:30s} → {r.status_code} [{code}] {msg}")
    except Exception as e:
        print(f"    ✗ {model:30s} → {e}")

if any_ok:
    print("\n  🎉 有模型成功！上面的✓对应的模型配置到 .env 即可使用")
else:
    print("\n  ❌ 全部失败。可能原因:")
    print("    1. Key 不对 → 重新复制")
    print("    2. 需要个人实名认证 → 去控制台完成")
    print("    3. 需要开通模型权限 → 点'使用'或去模型广场开通")
    print("    4. 可能需要先创建推理接入点（'模型推理接入点'标签）")
