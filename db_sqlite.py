"""SQLite知识库存储
读取PCM知识库时，用pandas in-memory查询替代PG/向量"""

import sqlite3
import json
import logging
import numpy as np
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

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
        -- ═══════════════════════════════════════════════════════════
        -- 新增: 决策引擎所需的表
        -- ═══════════════════════════════════════════════════════════
        CREATE TABLE IF NOT EXISTS formula_archive (
            formula_id TEXT PRIMARY KEY,
            formula_name TEXT NOT NULL,
            product_name TEXT,
            material_list TEXT,
            process_params TEXT,
            theory_expected TEXT DEFAULT '',
            source_tag TEXT DEFAULT 'manual',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS process_window (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            formula_id TEXT,
            param_name TEXT NOT NULL,
            unit TEXT DEFAULT '',
            lower_limit REAL,
            upper_limit REAL,
            nominal_value REAL,
            FOREIGN KEY (formula_id) REFERENCES formula_archive(formula_id)
        );
        CREATE TABLE IF NOT EXISTS diagnosis_session (
            session_id TEXT PRIMARY KEY,
            product_name TEXT,
            formula_name TEXT,
            test_phase TEXT,
            failure_mode TEXT,
            batch_scope TEXT,
            is_self_limiting INTEGER DEFAULT 0,
            decision TEXT,
            confidence REAL DEFAULT 0,
            full_report TEXT,
            profile_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS decision_rule (
            rule_id TEXT PRIMARY KEY,
            rule_name TEXT NOT NULL,
            condition_json TEXT NOT NULL,
            formula_score INTEGER DEFAULT 0,
            process_score INTEGER DEFAULT 0,
            description TEXT DEFAULT '',
            priority INTEGER DEFAULT 0
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


# ═══════════════════════════════════════════════════════════
# 新增: 决策引擎所需的函数
# ═══════════════════════════════════════════════════════════

# ── 配方档案 CRUD ──

def upsert_formula(formula_id, formula_name, product_name="",
                   material_list="", process_params="", theory_expected="",
                   source_tag="manual"):
    """保存或更新配方"""
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO formula_archive
        (formula_id, formula_name, product_name, material_list, process_params,
         theory_expected, source_tag, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (formula_id, formula_name, product_name, material_list, process_params,
          theory_expected, source_tag))
    conn.commit()
    return formula_id


def get_formula(formula_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM formula_archive WHERE formula_id=?", (formula_id,)).fetchone()
    return dict(row) if row else None


def list_formulas(product_name=None):
    conn = get_conn()
    if product_name:
        rows = conn.execute(
            "SELECT * FROM formula_archive WHERE product_name LIKE ? ORDER BY updated_at DESC",
            (f"%{product_name}%",)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM formula_archive ORDER BY updated_at DESC LIMIT 100").fetchall()
    return [dict(r) for r in rows]


# ── 工艺窗口 CRUD ──

def add_process_window(formula_id, param_name, lower, upper, nominal, unit=""):
    conn = get_conn()
    conn.execute("""
        INSERT INTO process_window (formula_id, param_name, unit, lower_limit, upper_limit, nominal_value)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (formula_id, param_name, unit, lower, upper, nominal))
    conn.commit()


def get_process_windows(formula_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM process_window WHERE formula_id=?",
        (formula_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ── 诊断会话记录 ──

def save_diagnosis_session(session_id, profile_dict, decision, confidence,
                           full_report=""):
    """诊断完成后归档"""
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO diagnosis_session
        (session_id, product_name, formula_name, test_phase, failure_mode,
         batch_scope, is_self_limiting, decision, confidence, full_report, profile_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        profile_dict.get("product_name", ""),
        profile_dict.get("formula_name", ""),
        profile_dict.get("test_phase", ""),
        profile_dict.get("failure_mode", ""),
        profile_dict.get("batch_scope", ""),
        1 if profile_dict.get("is_self_limiting") else 0,
        decision,
        confidence,
        full_report,
        json.dumps(profile_dict, ensure_ascii=False),
    ))
    conn.commit()


def list_diagnosis_sessions(limit=50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM diagnosis_session ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_diagnosis_session(session_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM diagnosis_session WHERE session_id=?", (session_id,)).fetchone()
    return dict(row) if row else None


# ── 决策规则 CRUD ──

def upsert_decision_rule(rule_id, rule_name, condition_json, formula_score,
                         process_score, description="", priority=0):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO decision_rule
        (rule_id, rule_name, condition_json, formula_score, process_score,
         description, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (rule_id, rule_name, condition_json, formula_score, process_score,
          description, priority))
    conn.commit()


def list_decision_rules():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM decision_rule ORDER BY priority DESC").fetchall()
    return [dict(r) for r in rows]


# ── 初始化默认知识：配方失效模式知识 ──

def seed_default_knowledge():
    """首次启动时注入核心知识（让冷启动也能输出有价值的诊断）"""
    conn = get_conn()
    
    # 配方相关的经验规则
    seeds = [
        # (rule_id, trigger, content, source, confidence)
        ("FRULE-001", "结晶 OR 凝胶",
         "结晶/凝胶的根因通常是：(1)固化剂用量偏高或分散不均 (2)反应温度过低导致预聚 (3)填料/助剂与基体相容性不足。工艺侧需检查温度控制和加料顺序。",
         "材料工程经验", 0.75),
        ("FRULE-002", "粘度升高 OR 变稠 OR 流动性差",
         "粘度异常的核心路径：(1)树脂/预聚物批次间分子量差异 (2)溶剂挥发或吸湿 (3)分散剂/增稠剂失效或用量偏移 (4)温度低于工艺窗口。先查温度，再查批次。",
         "材料工程经验", 0.7),
        ("FRULE-003", "固化不良 OR 不固化 OR 发粘",
         "固化失败三要素：(1)固化剂/催化剂配比错误或吸潮失效 (2)固化温度/时间不足 (3)阻聚剂过量或环境氧阻聚。配方层面重点检查固化体系，工艺层面检查温控。",
         "材料工程经验", 0.8),
        ("FRULE-004", "气泡 OR 针孔 OR 起泡",
         "气泡产生路径：(1)搅拌裹入气体未脱出 (2)配方中溶剂/低沸物挥发 (3)固化反应产气（如异氰酸酯遇水）(4)真空度或脱泡时间不足。通常先查真空工艺和搅拌参数。",
         "材料工程经验", 0.72),
        ("FRULE-005", "颜色异常 OR 色差 OR 发黄",
         "颜色问题优先级：(1)颜料/色浆分散不良 (2)基料黄变（热/光氧化）(3)固化过度导致热变色 (4)原料批次间色相差异。配方查颜料和稳定剂，工艺查温度曲线。",
         "材料工程经验", 0.68),
        ("FRULE-006", "强度不足 OR 剥离强度低 OR 附着力差",
         "力学性能异常的排查路径：(1)固化不完全（配比/温度/时间）(2)界面污染或底材处理不足 (3)相容性不良导致内应力 (4)填料分散不均。先检测固化度(DSC或邵尔硬度)，再查界面。",
         "材料工程经验", 0.75),
        # 决策规则
        ("DRULE-001", "新配方试产失效",
         "新配方/试产阶段出现偏差：优先调整配方（+30分），因为配方本身尚未经过量产验证",
         "决策矩阵", 0.9),
        ("DRULE-002", "连续多缸出现异常",
         "连续/多缸系统性问题：高度指向配方缺陷（+40分），工艺漂移通常表现为渐进而非连续",
         "决策矩阵", 0.85),
        ("DRULE-003", "单缸偶发且后续正常",
         "单缸偶发自限性异常：优先排查工艺波动（+35分），如温控过冲、搅拌瞬时异常、加料波动",
         "决策矩阵", 0.8),
        ("DRULE-004", "渐变衰减",
         "性能渐进衰减：优先查工艺参数漂移或设备磨损（+30分），配方问题通常导致突变",
         "决策矩阵", 0.78),
        ("DRULE-005", "换原料批次后出现",
         "更换原料批次后出现异常：配方原料批次适配性问题（+20分）或原料本身差异（需检测确认）",
         "决策矩阵", 0.72),
    ]
    
    existing = conn.execute("SELECT COUNT(*) FROM experience_rule").fetchone()[0]
    if existing > 0:
        return  # 已有规则，跳过
    
    for sid, trigger, content, source, conf in seeds:
        insert_rule(sid, trigger, content, source, conf)
    
    # 注入知识文本（用于向量检索）
    existing_kt = conn.execute("SELECT COUNT(*) FROM knowledge_text").fetchone()[0]
    if existing_kt == 0:
        kt_entries = [
            ("formula_knowledge", "F-K001",
             "美缝剂A胶结晶：美缝剂A组份（环氧树脂+填料）出现结晶/凝胶，常见原因包括：(1)固化剂（B组份）混入A组份 (2)温度低于10℃导致环氧树脂结晶析出 (3)填料吸湿结块。解决方案：升温至25℃搅拌观察是否溶解、检查原料存储条件、确认加料顺序无交叉污染。"),
            ("formula_knowledge", "F-K002",
             "胶粘剂不干/固化不良：双组份胶粘剂固化不完全，表面发粘。排查点：(1)固化剂配比是否正确（A:B比例）(2)固化温度是否达标 (3)固化剂/催化剂是否过期或吸潮失效 (4)混合是否均匀 (5)被粘材料是否有脱模剂或污染"),
            ("formula_knowledge", "F-K003",
             "环氧产品气泡/针孔：环氧灌封胶或结构胶固化后内部出现气泡。根因：(1)搅拌速度过快裹入气体未充分脱泡 (2)真空度不足或脱泡时间不够 (3)固化反应过快，气体来不及逸出 (4)配方中低沸点溶剂挥发"),
            ("process_knowledge", "P-K001",
             "温控异常对化工生产的影响：温度偏差会显著影响反应速率、粘度、分散稳定性、结晶行为。温度过低→反应不完全、结晶、粘度高；温度过高→暴聚、黄变、起胶。工艺窗口的上下限定义了安全操作区间。"),
            ("process_knowledge", "P-K002",
             "搅拌/分散工艺关键点：搅拌转速、时间、桨叶型式直接影响：(1)填料分散均匀性 (2)温度分布 (3)气泡裹入量 (4)多相体系相容性。搅拌不足→分散不均；搅拌过度→裹气、温升"),
            ("process_knowledge", "P-K003",
             "真空脱泡工艺：真空度(-0.095MPa以上)和维持时间(一般15~30min)是关键参数。常见问题：(1)真空度不够→气泡残留 (2)时间不足→脱泡不完全 (3)温度过高→物料沸腾 (4)浸胶产品需分段抽真空"),
        ]
        for stype, sid, content in kt_entries:
            index_text(stype, sid, content)
    
    conn.close()
