"""模型管理层：Chat走豆包API + RAG走本地embedding（无需额外权限）"""
import json
import logging
import os
import threading
import time

import requests

import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """令牌桶 RPS 限流"""
    def __init__(self, rps):
        self.interval = 1.0 / rps
        self._last = 0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last = time.time()


class ModelManager:
    """
    Chat：豆包 API（真实 AI 推理）
    Embedding：本地 hash 向量（无需 API 权限）
    """

    def __init__(self):
        self.api_key = os.getenv("DOUBAO_API_KEY", "")
        self.chat_url = os.getenv("DOUBAO_CHAT_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        self.model_id = os.getenv("DOUBAO_MODEL_ID", "doubao-seed-2-1-lite-260915")
        self.timeout = int(os.getenv("DOUBAO_TIMEOUT", "300"))
        # Fallback: 如果环境变量没 key，尝试从 SQLite 读取（登录时已保存）
        if not self.api_key:
            try:
                import db_sqlite as _db
                _db.DB_PATH = str(_db.Path(_db.__file__).parent / "pcm_knowledge.db")
                saved_key = _db.load_setting("DOUBAO_API_KEY")
                if saved_key:
                    self.api_key = saved_key
                    os.environ["DOUBAO_API_KEY"] = saved_key
                saved_mid = _db.load_setting("DOUBAO_MODEL_ID")
                if saved_mid:
                    self.model_id = saved_mid
                    os.environ["DOUBAO_MODEL_ID"] = saved_mid
            except Exception:
                pass
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._limiter = RateLimiter(int(os.getenv("DOUBAO_RPS_LIMIT", "5")))

    def close(self):
        """无状态连接无需关闭，但提供此接口供调用方安全调用"""
        pass

    def chat_completion(self, system_prompt: str, user_content: str,
                        temperature: float = 0.1, max_tokens: int = 2048) -> str:
        self._limiter.acquire()
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": min(max_tokens, 4096),
        }
        logger.debug(f"豆包请求: model={self.model_id}, user_len={len(user_content)}")
        resp = requests.post(
            self.chat_url, json=payload, headers=self.headers, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def safe_json_chat(self, system_prompt: str, user_content: str) -> dict:
        """调用 chat 并尝试解析 JSON（容错）"""
        raw = self.chat_completion(system_prompt, user_content)
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])
        try:
            return json.loads(raw)
        except Exception:
            logger.warning(f"JSON 解析失败:\n{raw[:500]}")
            raise

    def get_embedding(self, text: str) -> list:
        """本地 hash embedding（不开火山的 embedding 权限也能用）"""
        from local_embedding import embed_text
        return embed_text(text)


# 单例
_instance = None


def get_model() -> ModelManager:
    global _instance
    if _instance is None:
        _instance = ModelManager()
    return _instance


if __name__ == "__main__":
    # 连通性测试
    m = ModelManager()
    print("=== 测试 Chat ===")
    try:
        answer = m.chat_completion("回复OK", "只用一个词回复")
        print(f"  Chat OK: {answer}")
    except Exception as e:
        print(f"  Chat FAIL: {e}")

    print("\n=== 测试 Embedding ===")
    try:
        vec = m.get_embedding("竹基碳真空浸渍PCM")
        print(f"  Embedding OK: 维度={len(vec)}, 前3={vec[:3]}")
    except Exception as e:
        print(f"  Embedding FAIL: {e}")
