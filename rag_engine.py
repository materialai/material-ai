"""RAG检索中间层：构建知识片段上下文"""

import logging

import config
import database as db
from doubao_api import get_doubao

logger = logging.getLogger(__name__)


class RAGContext:
    """单次RAG检索结果"""

    def __init__(self, retrieved_items: list):
        self.items = retrieved_items

    def to_prompt_text(self) -> str:
        """拼接为提示词可用的上下文文本"""
        if not self.items:
            return "（知识库暂无匹配记录，依赖大模型通用推理）"
        lines = []
        for i, item in enumerate(self.items, 1):
            lines.append(
                f"[{i}] 来源类型: {item['source_type']} | "
                f"来源ID: {item['source_id']} | "
                f"相似度距离: {item['distance']:.4f} | "
                f"内容: {item['content']}"
            )
        return "\n".join(lines)

    def get_rules(self) -> list:
        """提取经验规则类检索结果"""
        return [item for item in self.items if item["source_type"] == "rule"]

    def get_cases(self) -> list:
        """提取案例类检索结果"""
        return [item for item in self.items if item["source_type"] == "case"]

    def get_solutions(self) -> list:
        """提取方案类检索结果"""
        return [item for item in self.items if item["source_type"] == "solution"]


def retrieve(query_text: str, top_k: int = None) -> RAGContext:
    """
    执行RAG检索：
    1. 把query文本转为embedding向量
    2. 在PGVector中做余弦相似度检索
    3. 返回匹配知识片段的RAGContext
    """
    doubao = get_doubao()

    # 获取query的embedding向量
    try:
        query_emb = doubao.get_embedding(query_text)
    except Exception as e:
        logger.error(f"获取embedding失败: {e}")
        return RAGContext([])

    # 向量检索
    items = db.search_vectors(query_emb, top_k=top_k)

    logger.info(f"RAG检索: 查询'{query_text[:30]}...' 召回{len(items)}条 知识片段")
    return RAGContext(items)


def build_full_context(query_text: str, extra_context: str = "") -> tuple:
    """
    构建完整的提示词上下文
    返回 (rag_context对象, 拼接后的提示词文本)
    """
    rag = retrieve(query_text)
    rag_text = rag.to_prompt_text()

    full_text = f"## 用户问题\n{query_text}\n\n## 知识库检索结果\n{rag_text}"

    if extra_context:
        full_text += f"\n\n## 补充上下文\n{extra_context}"

    return rag, full_text


def index_text_to_vector(source_type: str, source_id: str, content: str):
    """
    将文本片段索引到向量库
    用于：经验规则录入、案例归档、方案入库时同步向量化
    """
    doubao = get_doubao()
    try:
        emb = doubao.get_embedding(content)
        vid = db.insert_vector(source_type, source_id, content, emb)
        logger.info(f"向量入库: {source_type}/{source_id} -> vector_id={vid}")
        return vid
    except Exception as e:
        logger.error(f"向量入库失败: {source_type}/{source_id} {e}")
        return -1


if __name__ == "__main__":
    # 快速测试
    ctx = retrieve("高分子聚合反应收率下降 原料结块 颜色偏深")
    print(ctx.to_prompt_text())
