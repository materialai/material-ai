"""SQLite知识库存储
读取PCM知识库时，用pandas in-memory查询替代PG/向量"""

import sqlite3
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from local_embedding import embed_text, cosine_similarity, VEC_DIM

logger = logging.getLogger(__name__)
DB_PATH = Path(__file__).parent / "pcm_knowledge.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experience_rule (
            rule_id TEXT PRIMARY KEY,
            trigger_condition TEXT NOT NULL,
            exp_content TEXT NOT NULL,
            source_info TEXT,
            confidence REAL DEFAULT 0.5
        );
        CREATE TABLE IF NOT EXISTS case_library (
            case_id TEXT PRIMARY KEY,
            problem_desc TEXT NOT NULL,
            root_cause TEXT,
            check_round INTEGER DEFAULT 0,
            status TEXT DEFAULT 'open'
        );
        CREATE TABLE IF NOT EXISTS improve_solution (
            solution_id TEXT PRIMARY KEY,
            root_cause_type TEXT NOT NULL,
            plan_short TEXT
        );
        CREATE TABLE IF NOT EXISTS knowledge_text (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_kt_type ON knowledge_text(source_type);
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    return conn


def insert_rule(rule_id, trigger, content, source="", confidence=0.5):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO experience_rule
        (rule_id, trigger_condition, exp_content, source_info, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (rule_id, trigger, content, source, confidence))
    conn.commit()
    return conn.execute("SELECT * FROM experience_rule WHERE rule_id=?", (rule_id,)).fetchone()


def insert_case(case_id, problem, root_cause="", check_round=0):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO case_library
        (case_id, problem_desc, root_cause, check_round)
        VALUES (?, ?, ?, ?)
    """, (case_id, problem, root_cause, check_round))
    conn.commit()


def insert_solution(sid, root_type, plan):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO improve_solution
        (solution_id, root_cause_type, plan_short)
        VALUES (?, ?, ?)
    """, (sid, root_type, plan))
    conn.commit()


def index_text(source_type, source_id, content):
    """文本入库+embedding存储"""
    vec = embed_text(content)
    blob = np.array(vec, dtype=np.float32).tobytes()
    conn = get_conn()
    conn.execute("""
        INSERT INTO knowledge_text (source_type, source_id, content, embedding)
        VALUES (?, ?, ?, ?)
    """, (source_type, source_id, content, blob))
    conn.commit()
    return vec


def retrieve(query_text, top_k=4):
    """向量召回TopK"""
    q_vec = embed_text(query_text)
    conn = get_conn()
    rows = conn.execute("SELECT source_type, source_id, content, embedding FROM knowledge_text").fetchall()

    results = []
    for r in rows:
        vec = np.frombuffer(r["embedding"], dtype=np.float32)
        sim = cosine_similarity(q_vec, vec.tolist())
        results.append({
            "source_type": r["source_type"],
            "source_id": r["source_id"],
            "content": r["content"],
            "similarity": sim,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def search_rules(keyword):
    """关键词搜索经验规则"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM experience_rule
        WHERE trigger_condition LIKE ? OR exp_content LIKE ?
    """, (f"%{keyword}%", f"%{keyword}%")).fetchall()
    return [dict(r) for r in rows]


def get_all_rules():
    conn = get_conn()
    return [dict(r) for r in conn.execute("SELECT * FROM experience_rule").fetchall()]


def get_all_cases():
    conn = get_conn()
    return [dict(r) for r in conn.execute("SELECT * FROM case_library").fetchall()]


def get_case_by_id(case_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM case_library WHERE case_id=?", (case_id,)).fetchone()
    return dict(row) if row else None


def update_case(case_id, **kw):
    conn = get_conn()
    sets = ", ".join(f"{k}=?" for k in kw)
    vals = list(kw.values()) + [case_id]
    conn.execute(f"UPDATE case_library SET {sets} WHERE case_id=?", vals)
    conn.commit()


def get_stats():
    conn = get_conn()
    return {
        "rules": conn.execute("SELECT COUNT(*) FROM experience_rule").fetchone()[0],
        "cases": conn.execute("SELECT COUNT(*) FROM case_library").fetchone()[0],
        "solutions": conn.execute("SELECT COUNT(*) FROM improve_solution").fetchone()[0],
        "vectors": conn.execute("SELECT COUNT(*) FROM knowledge_text").fetchone()[0],
    }


def save_setting(key, value):
    """保存一项配置到SQLite"""
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
        (key, value)
    )
    conn.commit()


def load_setting(key, default=""):
    """读取一项配置"""
    conn = get_conn()
    row = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def load_all_settings():
    """读取全部配置"""
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    return {r["key"]: r["value"] for r in rows}
