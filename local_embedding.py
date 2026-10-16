"""本地替代Embedding（无需豆包API调用，用numpy实现轻量级文本向量）
用Hash Trick + TF-like策略生成确定性向量——
不追求精度，只让召回逻辑能跑通。上线时换成豆包向量化即可。"""

import hashlib
import re
import numpy as np

VEC_DIM = 1024  # 与豆包向量维度一致，可直接替换


def chinese_tokenize(text: str) -> list:
    """极简中文切分：2~4 gram + 单字停用词过滤"""
    stops = {"的", "了", "和", "与", "或", "在", "是", "有", "着", "被", "把",
             "从", "到", "对", "为", "以", "及", "其", "这", "那", "不", "也",
             "就", "都", "而", "但", "很", "会", "要", "能", "可", "应", "该"}
    text = re.sub(r"[\s，。、；：！？\"'（）\(\)]+", " ", text)
    chars = [c for c in text if c not in stops]
    # 单字 + bigram + trigram
    tokens = chars[:]
    for n in (2, 3, 4):
        for i in range(len(chars) - n + 1):
            tokens.append("".join(chars[i:i + n]))
    return tokens


def embed_text(text: str) -> list:
    """生成1024维确定性文本向量"""
    tokens = chinese_tokenize(text)
    vec = np.zeros(VEC_DIM, dtype=np.float64)

    for tok in tokens:
        # 用hash确定非零维度位置
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for seed_offset in (0, 1, 2):  # 每个token激活3个维度
            dim = (h + seed_offset * 2654435761) % VEC_DIM
            # hash符号决定±
            val = 1.0 if (h >> seed_offset) % 2 == 0 else -1.0
            # 词频加权
            val *= (1.0 + tok.count(tok[0]) * 0.1) if tok else 1.0
            vec[dim] += val

    # L2归一化
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """余弦相似度"""
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


if __name__ == "__main__":
    # 本地验证
    texts = [
        "竹基碳真空浸渍放大生产料层厚度",
        "料层过厚造成传热传质阻力中心区域浸渍不足",
        "抗氧剂防止PEG氧化降解",
    ]
    vecs = [embed_text(t) for t in texts]
    print("T1 vs T2 相似度:", cosine_similarity(vecs[0], vecs[1]))  # 应较高
    print("T1 vs T3 相似度:", cosine_similarity(vecs[0], vecs[2]))  # 应较低
