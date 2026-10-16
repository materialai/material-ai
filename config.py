"""系统配置：加载.env所有参数"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 加载.env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv(PROJECT_ROOT / ".env.example")

# ---------- 豆包企业API ----------
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_CHAT_ENDPOINT = os.getenv("DOUBAO_CHAT_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
DOUBAO_EMBED_ENDPOINT = os.getenv("DOUBAO_EMBED_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/embeddings")
DOUBAO_MODEL_ID = os.getenv("DOUBAO_MODEL_ID", "")

# ---------- PostgreSQL ----------
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "")
PG_DB = os.getenv("PG_DB", "material_db")

# ---------- 业务参数 ----------
DEVIATION_THRESHOLD = float(os.getenv("DEVIANCE_THRESHOLD_PERCENT", "15"))
MAX_HYPOTHESIS = int(os.getenv("MAX_HYPOTHESIS_COUNT", "5"))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS_PER_ROUND", "3"))
MAX_ROUNDS = int(os.getenv("MAX_QUESTION_ROUNDS", "3"))

# ---------- 置信度参数 ----------
CONF_HIT_DELTA = float(os.getenv("CONFIDENCE_HIT_DELTA", "0.05"))
CONF_MISS_DELTA = float(os.getenv("CONFIDENCE_MISS_DELTA", "0.1"))
CONF_MIN = float(os.getenv("CONFIDENCE_MIN", "0.0"))
CONF_MAX = float(os.getenv("CONFIDENCE_MAX", "1.0"))
CONF_INIT = float(os.getenv("CONFIDENCE_INIT", "0.5"))

# ---------- RAG ----------
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# ---------- 改善回授 ----------
FEEDBACK_DAYS = [int(d) for d in os.getenv("IMPROVE_FEEDBACK_DAYS", "7,14").split(",")]

# ---------- 豆包API限流 ----------
DOUBAO_RPS = int(os.getenv("DOUBAO_RPS_LIMIT", "5"))
DOUBAO_TIMEOUT = int(os.getenv("DOUBAO_TIMEOUT", "60"))

# ---------- 语音功能 ----------
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() == "true"
TTS_VOICE = os.getenv("TTS_VOICE", "BV406_V2_streaming")
TTS_ENCODING = os.getenv("TTS_ENCODING", "mp3")

# ---------- 提示词路径 ----------
PROMPTS_DIR = PROJECT_ROOT / "prompts"
PROMPT_FILES = {
    "extract": PROMPTS_DIR / "prompt_extract.txt",
    "hypothesis": PROMPTS_DIR / "prompt_hypothesis.txt",
    "question": PROMPTS_DIR / "prompt_question.txt",
    "solution": PROMPTS_DIR / "prompt_solution.txt",
}


def load_prompt(name: str) -> str:
    """加载指定提示词文件内容"""
    path = PROMPT_FILES.get(name)
    if not path or not path.exists():
        raise FileNotFoundError(f"提示词文件不存在: {path}")
    return path.read_text(encoding="utf-8")


def validate_config():
    """启动时校验关键配置"""
    errors = []
    if not DOUBAO_API_KEY:
        errors.append("DOUBAO_API_KEY未配置")
    if not DOUBAO_MODEL_ID:
        errors.append("DOUBAO_MODEL_ID未配置")
    if not PG_PASS:
        errors.append("PG_PASS未配置")
    if errors:
        raise RuntimeError("配置缺失：\n" + "\n".join(f"  - {e}" for e in errors))
    return True
