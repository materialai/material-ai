"""PostgreSQL数据库连接与操作封装"""

import json
import logging
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from pgvector.psycopg2 import register_vector

import config

logger = logging.getLogger(__name__)


def get_connection():
    """获取连接（自动注册pgvector类型）"""
    conn = psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        user=config.PG_USER,
        password=config.PG_PASS,
        database=config.PG_DB,
        cursor_factory=RealDictCursor,
    )
    register_vector(conn)
    return conn


@contextmanager
def db_cursor(commit: bool = False):
    """上下文manager：自动处理连接开闭和事务"""
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ========================== 业务表操作 ==========================

def insert_experience_rule(rule_id: str, trigger_condition: str, exp_content: str,
                           source_info: str = "", rule_name: str = "") -> dict:
    """入库一条一点课经验规则"""
    with db_cursor(commit=True) as cur:
        cur.execute("""
            INSERT INTO experience_rule (rule_id, rule_name, trigger_condition, exp_content, source_info, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (rule_id) DO UPDATE SET
                trigger_condition = EXCLUDED.trigger_condition,
                exp_content = EXCLUDED.exp_content,
                source_info = EXCLUDED.source_info,
                update_time = NOW()
            RETURNING *
        """, (rule_id, rule_name, trigger_condition, exp_content, source_info, config.CONF_INIT))
        row = cur.fetchone()
        logger.info(f"经验规则入库: {rule_id} confidence={config.CONF_INIT}")
        return dict(row) if row else {}


def get_rule_by_id(rule_id: str) -> Optional[dict]:
    """按ID获取规则"""
    with db_cursor() as cur:
        cur.execute("SELECT * FROM experience_rule WHERE rule_id=%s", (rule_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def search_rules_by_keyword(keyword: str) -> list:
    """模糊搜索规则内容"""
    with db_cursor() as cur:
        cur.execute("""
            SELECT * FROM experience_rule
            WHERE exp_content LIKE %s OR trigger_condition LIKE %s
            ORDER BY confidence DESC
        """, (f"%{keyword}%", f"%{keyword}%"))
        return [dict(r) for r in cur.fetchall()]


def insert_case(case_id: str, problem_desc: str, problem_original: str = "",
                hypothesis_list: list = None, **kwargs) -> dict:
    """创建排查案例"""
    with db_cursor(commit=True) as cur:
        cur.execute("""
            INSERT INTO case_library (case_id, problem_desc, problem_original,
                                      hypothesis_list, status)
            VALUES (%s, %s, %s, %s, 'open')
            ON CONFLICT (case_id) DO UPDATE SET
                update_time = NOW()
            RETURNING *
        """, (case_id, problem_desc, problem_original,
              Json(hypothesis_list or [])))
        row = cur.fetchone()
        return dict(row) if row else {}


def update_case(case_id: str, **fields) -> dict:
    """通用更新案例字段"""
    allowed = {"hypothesis_list", "check_process", "root_cause",
               "root_cause_rule_ids", "improve_solution_id",
               "improve_result", "check_round", "status"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_case_by_id(case_id)

    set_clause = ", ".join(f"{k} = %s" for k in updates)
    values = list(updates.values()) + [case_id]

    with db_cursor(commit=True) as cur:
        cur.execute(f"""
            UPDATE case_library SET {set_clause}, update_time=NOW()
            WHERE case_id=%s RETURNING *
        """, values)
        return dict(cur.fetchone()) if cur.fetchone else {}


def get_case_by_id(case_id: str) -> Optional[dict]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM case_library WHERE case_id=%s", (case_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def insert_formula(formula_id: str, material_list: str, process_params: str,
                   theory_expected: str = "", env_params: str = "",
                   source_tag: str = "manual", formula_name: str = "") -> dict:
    with db_cursor(commit=True) as cur:
        cur.execute("""
            INSERT INTO formula_archive (formula_id, formula_name, material_list,
                                         process_params, env_params, theory_expected, source_tag)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (formula_id) DO UPDATE SET update_time=NOW()
            RETURNING *
        """, (formula_id, formula_name, material_list, process_params,
              env_params, theory_expected, source_tag))
        return dict(cur.fetchone()) if cur.fetchone() else {}


def insert_test_record(test_id: str, formula_id: str, core_index: str,
                       deviation_desc: str = "", deviation_magnitude: float = 0.0,
                       real_condition: str = "", attach_link: str = "") -> dict:
    with db_cursor(commit=True) as cur:
        cur.execute("""
            INSERT INTO test_record (test_id, formula_id, core_index, deviation_desc,
                                     deviation_magnitude, real_condition, attach_link)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
        """, (test_id, formula_id, core_index, deviation_desc,
              deviation_magnitude, real_condition, attach_link))
        return dict(cur.fetchone()) if cur.fetchone() else {}


# ========================== 向量表操作 ==========================

def insert_vector(source_type: str, source_id: str, content: str, embedding: list) -> int:
    """写入向量记录，返回自增ID"""
    import hashlib
    content_hash = hashlib.md5(f"{source_type}:{source_id}:{content}".encode()).hexdigest()

    with db_cursor(commit=True) as cur:
        # 去重检查
        cur.execute("""
            SELECT id FROM knowledge_vector
            WHERE content_hash=%s AND source_id=%s
        """, (content_hash, source_id))
        if cur.fetchone():
            return cur.fetchone()["id"]

        cur.execute("""
            INSERT INTO knowledge_vector (source_type, source_id, content, content_hash, embedding)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (source_type, source_id, content, content_hash, embedding))
        return cur.fetchone()["id"]


def search_vectors(query_embedding: list, top_k: int = None) -> list:
    """向量相似度检索"""
    top_k = top_k or config.RAG_TOP_K
    with db_cursor() as cur:
        cur.execute("""
            SELECT source_type, source_id, content, embedding <-> %s AS distance
            FROM knowledge_vector
            ORDER BY embedding <-> %s
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        return [dict(r) for r in cur.fetchall()]


def vector_count() -> int:
    """返回向量表总记录数"""
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) as cnt FROM knowledge_vector")
        row = cur.fetchone()
        return row["cnt"] if row else 0
