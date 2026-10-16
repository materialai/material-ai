"""豆包企业API封装: chat completion + embedding"""

import json
import time
import logging
import threading
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """简单令牌桶RPS限流"""

    def __init__(self, rps: int):
        self.rps = rps
        self.interval = 1.0 / rps
        self._last_time = 0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last_time = time.time()


# 全局限流器
_limiter = RateLimiter(config.DOUBAO_RPS)


class DoubaoAPI:
    """豆包企业API客户端"""

    def __init__(self):
        self.api_key = config.DOUBAO_API_KEY
        self.chat_url = config.DOUBAO_CHAT_ENDPOINT
        self.emb_url = config.DOUBAO_EMBED_ENDPOINT
        self.model_id = config.DOUBAO_MODEL_ID
        self.timeout = config.DOUBAO_TIMEOUT
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat_completion(self, system_prompt: str, user_content: str,
                        temperature: float = 0.1, max_tokens: int = 2048) -> str:
        """调用豆包chat completion，返回纯文本响应"""
        _limiter.acquire()
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            self.chat_url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"API响应格式异常: {data}")

        return data["choices"][0]["message"]["content"]

    def get_embedding(self, text: str) -> list:
        """获取文本的embedding向量（维度1024）"""
        _limiter.acquire()
        payload = {
            "model": "doubao-embedding-text-240715",
            "input": text,
        }
        resp = requests.post(
            self.emb_url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data or not data["data"]:
            raise RuntimeError(f"Embedding API响应格式异常: {data}")

        return data["data"][0]["embedding"]

    def safe_json_chat(self, system_prompt: str, user_content: str) -> dict:
        """调用chat并尝试解析返回为JSON，支持容错"""
        raw = self.chat_completion(system_prompt, user_content)
        # 尝试去除markdown代码块包裹
        if raw.startswith("```"):
            lines = raw.split("\n")
            # 去除首行```xxx和末行```
            raw = "\n".join(lines[1:-1])
        try:
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"JSON解析失败，原始返回:\n{raw[:500]}")
            raise


# 便捷函数（模块级）
_doubao_client: Optional[DoubaoAPI] = None


def get_doubao() -> DoubaoAPI:
    """单例获取豆包API客户端"""
    global _doubao_client
    if _doubao_client is None:
        _doubao_client = DoubaoAPI()
    return _doubao_client


if __name__ == "__main__":
    # 快速连通性测试
    print("--- 测试豆包API连通性 ---")
    api = DoubaoAPI()
    try:
        result = api.chat_completion("回复OK", "测试")
        print(f"chat: {result}")
    except Exception as e:
        print(f"chat失败: {e}")

    try:
        emb = api.get_embedding("测试文本")
        print(f"embedding维度: {len(emb)}")
    except Exception as e:
        print(f"embedding失败: {e}")
