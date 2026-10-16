"""语音识别(ASR) + 语音合成(TTS) -- 火山引擎语音技术API
同一账号体系，复用 DOUBAO_API_KEY

ASR: POST https://openspeech.bytedance.com/api/v1/asr
TTS: POST https://openspeech.bytedance.com/api/v1/tts
"""

import base64
import json
import logging
import time
import tempfile
import os
from pathlib import Path
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)

VOLCANO_SPEECH_BASE = "https://openspeech.bytedance.com/api/v1"

# 常用音色: BV406_V2_streaming(女声), BV407_V2_streaming(男声), BV034_streaming(可爱女声)
DEFAULT_VOICE = "BV406_V2_streaming"


def recognize_speech(audio_bytes: bytes, fmt: str = "opus", rate: int = 16000) -> str:
    """语音识别: 音频字节 -> 文字

    参数:
        audio_bytes: 原始音频字节
        fmt:  编码格式 opus/pcm/wav/mp3
        rate: 采样率 (16000 or 8000)

    返回:
        识别出的文字
    """
    if not audio_bytes:
        raise ValueError("音频数据为空")

    audio_b64 = base64.b64encode(audio_bytes).decode()
    payload = {
        "user": {"uid": "material-ai-mobile"},
        "audio": {
            "format": fmt,
            "rate": rate,
            "bits": 16,
            "channel": 1,
            "base64": audio_b64,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
        },
    }

    try:
        resp = requests.post(
            f"{VOLCANO_SPEECH_BASE}/asr",
            json=payload,
            headers={
                "Authorization": f"Bearer {config.DOUBAO_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"ASR error: code={data.get('code')} msg={data.get('message')}")

        text = data.get("result", [{}])[0].get("text", "")
        return text.strip()

    except requests.RequestException as e:
        logger.error(f"ASR网络错误: {e}")
        raise RuntimeError(f"语音识别网络错误: {e}")


def text_to_speech(text: str, voice: str = DEFAULT_VOICE, encoding: str = "mp3") -> bytes:
    """语音合成: 文字 -> 音频字节

    参数:
        text: 要合成的文字
        voice: 音色ID
        encoding: 输出格式 mp3/opus/pcm/wav

    返回:
        音频字节数据
    """
    if not text:
        raise ValueError("文字为空")

    # TTS限制最大长度
    while len(text) > 200:
        text = text[:199]

    payload = {
        "reqid": f"tts-{int(time.time())}",
        "text": text,
        "voice_type": voice,
        "encoding": encoding,
        "volume": 50,
        "speed": 50,
    }

    try:
        resp = requests.post(
            f"{VOLCANO_SPEECH_BASE}/tts",
            json=payload,
            headers={
                "Authorization": f"Bearer {config.DOUBAO_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()

        # TTS返回音频字节
        audio_data = resp.content
        if len(audio_data) < 100:
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type:
                # JSON错误
                err = resp.json()
                raise RuntimeError(f"TTS error: {err}")
            raise RuntimeError(f"TTS返回数据过小: {len(audio_data)} bytes")

        return audio_data

    except requests.RequestException as e:
        logger.error(f"TTS网络错误: {e}")
        raise RuntimeError(f"语音合成网络错误: {e}")


def save_audio_tmp(audio_bytes: bytes, suffix: str = ".mp3") -> Optional[str]:
    """暂存音频字节到临时文件，返回路径"""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    except Exception as e:
        logger.error(f"保存临时音频失败: {e}")
        return None


def play_audio_in_flet(audio_bytes: bytes, suffix: str = ".mp3") -> Optional[str]:
    """保存临时文件并供Flet Audio组件播放"""
    return save_audio_tmp(audio_bytes, suffix)


if __name__ == "__main__":
    # 快速TTS测试
    api_key = config.DOUBAO_API_KEY
    if api_key:
        print("--- 测试TTS连通性 ---")
        try:
            audio = text_to_speech("你好，我是材料AI诊断助手")
            print(f"TTS音频大小: {len(audio)} bytes")
        except Exception as e:
            print(f"TTS失败: {e}")
    else:
        print("DOUBAO_API_KEY未配置")
