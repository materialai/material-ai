"""全品类大宗材料知识库 → SQLite 知识索引
将工艺知识解析为文本片段，存入 knowledge_text 向量表
排查时RAG召回相关工艺片段作为AI推理上下文"""

import re
import logging
from pathlib import Path

import db_sqlite as db
from local_embedding import embed_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MD_FILE = Path(r"C:\Users\T480\workspace\投研能力\全品类大宗材料知识库完整版.md")

# 每段最大字符（太长的段落召回噪音多）
CHUNK_MAX = 500
# 每段最小字符
CHUNK_MIN = 30


def parse_markdown(text: str) -> list:
    """将markdown文本按章节+表格+段落切分为知识片段"""
    chunks = []

    # 提取文档级描述（前100字作为摘要）
    doc_desc = "全品类大宗材料知识库：涂料、胶粘剂、塑料、橡胶、建材、复合材料、电子封装、美缝剂等工业制造领域材料选型与配方知识"

    # 按一级章节分割（## 标题）
    sections = re.split(r"\n##\s+", text)

    for section in sections:
        if not section.strip():
            continue

        # 章节标题
        lines = section.strip().split("\n")
        section_title = lines[0].strip().rstrip(":：")

        # 将章节内容切分为可索引片段
        current_chunk = []
        current_len = 0

        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"):
                # 截断点：提交当前chunk
                if current_chunk and current_len >= CHUNK_MIN:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append({
                        "title": section_title,
                        "text": chunk_text[:CHUNK_MAX],
                        "category": classify_category(section_title),
                    })
                current_chunk = []
                current_len = 0
                continue

            # 表格行也作为独立知识行
            if line.startswith("|"):
                # 跳过表头分隔线
                if re.match(r"^[\s|:-]+$", line):
                    continue
                # 表格行作为独立知识片段加入
                row_text = table_row_to_text(line, section_title)
                if row_text and len(row_text) > CHUNK_MIN:
                    chunks.append({
                        "title": section_title,
                        "text": row_text[:CHUNK_MAX],
                        "category": classify_category(section_title),
                    })
            else:
                # 普通段落行
                # 超限时切片
                if current_len + len(line) > CHUNK_MAX:
                    if current_chunk and current_len >= CHUNK_MIN:
                        chunks.append({
                            "title": section_title,
                            "text": "\n".join(current_chunk),
                            "category": classify_category(section_title),
                        })
                    current_chunk = [line]
                    current_len = len(line)
                else:
                    current_chunk.append(line)
                    current_len += len(line)

        # 章节末尾的残余
        if current_chunk and current_len >= CHUNK_MIN:
            chunks.append({
                "title": section_title,
                "text": "\n".join(current_chunk)[:CHUNK_MAX],
                "category": classify_category(section_title),
            })

    return chunks


def table_row_to_text(row: str, section: str) -> str:
    """把Markdown表格行转为可读文本"""
    cells = [c.strip() for c in row.split("|") if c.strip()]
    if len(cells) < 2:
        return ""
    return f"{section}: " + " / ".join(cells)


def classify_category(title: str) -> str:
    """归类知识到材料大类别"""
    title_lower = title.lower()

    categories = {
        "高分子树脂": ["树脂", "乳液", "环氧", "聚氨酯", "pu", "pvc", "pe", "pp", "聚酯", "氟碳", "有机硅", "pe/pp", "丙烯酸"],
        "固化剂": ["固化剂", "交联剂", "过氧化物", "促进剂", "胺类", "异氰酸酯"],
        "填料颜料": ["填料", "颜料", "炭黑", "钛白粉"],
        "溶剂": ["溶剂", "稀释剂"],
        "增塑剂": ["增塑剂", "加工助剂"],
        "阻燃剂": ["阻燃"],
        "抗氧剂": ["抗氧剂", "抗老化", "光稳定剂"],
        "偶联剂": ["硅烷偶联剂"],
        "相变材料": ["相变", "pcm"],
        "流变助剂": ["流变", "增稠", "触变", "消泡", "分散"],
        "供应商": ["供应商"],
        "选型方法论": ["选型", "配方原则"],
        "跨品类": ["跨品类", "知识图谱"],
        "美缝剂": ["美缝"],
    }

    for category, keywords in categories.items():
        for kw in keywords:
            if kw in title_lower:
                return category
    return "通用知识"


def import_all():
    if not MD_FILE.exists():
        logger.error(f"文件不存在: {MD_FILE}")
        return

    db.init_db()
    text = MD_FILE.read_text(encoding="utf-8")
    chunks = parse_markdown(text)

    logger.info(f"解析知识片段总数: {len(chunks)}")

    # 删除旧知识库记录（可选：仅清空当前文件对应的知识类别）
    # 由于SQLite没有按source_type删除的便捷方式，这里直接追加knowledge类型

    # 入库 + 向量化
    count = 0
    category_counts = {}

    for chunk in chunks:
        text_content = f"{chunk['title']}\n{chunk['text']}"
        kid = f"BULK-{count+1:05d}"

        db.index_text("knowledge", kid, text_content)
        count += 1

        cat = chunk["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

        if count % 100 == 0:
            logger.info(f"  已入库 {count}/{len(chunks)}")

    # 统计
    stats = db.get_stats()
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ 知识库导入完成: {count} 条知识片段")
    logger.info(f"   分类统计:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        logger.info(f"     {cat}: {cnt}")
    logger.info(f"   数据库总量: 规则{stats['rules']} / 案例{stats['cases']} / 方案{stats['solutions']} / 向量{stats['vectors']}")
    logger.info(f"{'='*50}")

    print(f"\n✅ 全品类大宗材料知识库导入完成: {count} 条知识片段")
    print(f"   分类统计:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"     {cat}: {cnt} 条")
    print(f"\n   数据库总量: 向量{stats['vectors']} 条")


if __name__ == "__main__":
    import_all()
