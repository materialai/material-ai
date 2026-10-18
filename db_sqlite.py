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
        -- ═══════════════════════════════════════════════════════════
        -- 扩展: 附录 I~S 知识表
        -- ═══════════════════════════════════════════════════════════
        -- I. 测试指标-产线工序映射
        CREATE TABLE IF NOT EXISTS test_process_mapping (
            mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_indicator TEXT NOT NULL,
            priority_process TEXT,
            secondary_process TEXT,
            exclusion_note TEXT
        );
        -- J. 材料相容性矩阵
        CREATE TABLE IF NOT EXISTS material_compatibility (
            compat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_a TEXT NOT NULL,
            material_b TEXT NOT NULL,
            static_ok TEXT,
            dynamic_high_tell TEXT,
            long_term_aging TEXT,
            conclusion TEXT,
            redline TEXT
        );
        -- K. 经验规则触发边界
        CREATE TABLE IF NOT EXISTS rule_trigger_boundary (
            boundary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_text TEXT NOT NULL,
            preconditions TEXT,
            not_applicable TEXT,
            conflict_resolution TEXT,
            confidence_level TEXT,
            source TEXT
        );
        -- M. 行业标准约束
        CREATE TABLE IF NOT EXISTS industry_standard (
            standard_id INTEGER PRIMARY KEY AUTOINCREMENT,
            performance_indicator TEXT NOT NULL,
            product_scope TEXT,
            standard_code TEXT,
            test_condition TEXT,
            threshold_value TEXT,
            notes TEXT,
            business_consequence TEXT
        );
        -- N. DOE交互效应
        CREATE TABLE IF NOT EXISTS doe_interaction (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_a TEXT NOT NULL,
            factor_b TEXT NOT NULL,
            interaction_level TEXT,
            description TEXT
        );
        -- O. 原料隐性风险
        CREATE TABLE IF NOT EXISTS material_hidden_risk (
            risk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_category TEXT NOT NULL,
            routine_checks TEXT,
            hidden_risks TEXT,
            process_consequence TEXT
        );
        -- P. 设备老化漂移
        CREATE TABLE IF NOT EXISTS equipment_drift (
            drift_id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_part TEXT NOT NULL,
            drift_symptom TEXT,
            real_consequence TEXT,
            detection_method TEXT
        );
        -- Q. 成本-性能权衡方案
        CREATE TABLE IF NOT EXISTS cost_performance_plan (
            plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario TEXT NOT NULL,
            candidate_plan TEXT,
            performance_gain TEXT,
            cost_change TEXT,
            verification_cycle TEXT,
            risk_level TEXT,
            priority_rank INTEGER DEFAULT 0
        );
        -- R. 客户工况约束
        CREATE TABLE IF NOT EXISTS customer_constraint (
            constraint_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_type TEXT NOT NULL,
            inherent_limits TEXT,
            high_freq_failures TEXT,
            exclusive_constraints TEXT
        );
        -- S. 工程师思维链案例
        CREATE TABLE IF NOT EXISTS thinking_chain_case (
            case_id TEXT PRIMARY KEY,
            phenomenon TEXT NOT NULL,
            first_guess TEXT,
            elimination_round TEXT,
            multi_hypothesis TEXT,
            risk_tradeoff TEXT,
            verification_action TEXT,
            final_root_cause TEXT,
            review_highlight TEXT
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
    """首次启动 → 注入L0-L7全量知识（经验规则 + 知识文本 + 决策规则 + 失败教训）"""
    conn = get_conn()

    # ═══════════════════════════════════════════════════════════
    # 一、经验规则（experience_rule）：触发式诊断知识
    # ═══════════════════════════════════════════════════════════
    existing = conn.execute("SELECT COUNT(*) FROM experience_rule").fetchone()[0]
    if existing == 0:
        seeds = [
            # ── L0 配方层规则 ──
            ("FRULE-L0-001", "结晶 OR 凝胶 OR 起胶",
             "【L0配方层】结晶/凝胶根因：(1)固化剂分散不均或偏高 (2)反应温度过低导致预聚 (3)填料/助剂相容性不足 (4)配比偏移Δr>5% → 先计算r_actual定位。"
             "计算链：EEW=100/环氧值，AHEW=M_amine/n_H，r_actual=(Q_B·ρ_B/AHEW_B)/(Q_A·ρ_A/EEW_A)。"
             "|Δr|≤5%可接受；5~10%强度/Tg漂移；>10%不固化/发软/漏芯高危。",
             "L0配方层", 0.85),
            ("FRULE-L0-002", "粘度 OR 稠 OR 稀 OR 流变",
             "【L0配方层】粘度异常路径：(1)树脂批次分子量差异 (2)溶剂挥发或吸湿 (3)分散剂/增稠剂失效 (4)温度低于窗口 (5)固含量偏移。"
             "公式：Einstein η=η_0(1+2.5φ)（低φ）；Krieger-Dougherty η=η_0(1-φ/φ_m)^(-[η]·φ_m)（高φ碳粉）。"
             "Arrhenius：μ(T)=μ_0·exp[E_a/R·(1/T-1/T_0)]（每降5°C粘度×2~3倍）。",
             "L0配方层", 0.8),
            ("FRULE-L0-003", "固化 OR 不固化 OR 发粘 OR 不干 OR 发脆",
             "【L0配方层】固化异常全路径："
             "(a)固化慢：t_gel=A·exp(E_a/RT)，先升温2~3°C，再补促进剂，最后换固化剂批次。"
             "(b)不固化：计算r_actual确认配比；检查固化剂活性（胺值下降=吸潮预反应）。"
             "(c)发脆：交联度过高 p_c=2/f_avg → 增韧剂+5~8phr或微降固化剂。"
             "(d)发软：E=3ν_eRT交联密度不足 → 提交联密度或降PCM含量。"
             "Kamal动力学：dα/dt=(k_1+k_2·α^m)(1-α)^n → DSC放热峰可标定。",
             "L0配方层", 0.88),
            ("FRULE-L0-004", "气泡 OR 针孔 OR 起泡 OR 漏芯",
             "【L0配方层+工艺】气泡/漏芯路径："
             "(1)真空脱泡：必须<-0.06MPa/15~20min；高碳粉体系需预烘干80°C/4h+润湿20min+真空三步缺一不可。"
             "(2)搅拌裹气：降低N或延长脱泡时间；桨尖线速度v_tip=πND，量产限4~6m/s。"
             "(3)碳粉含水→接枝水解→气泡+漏芯 → 查RH是否>75%。"
             "(4)反应产气：异氰酸酯遇水→CO2 → 原料干燥密封。"
             "碳粉含水→气泡速查：搅拌初期大量小气泡→延长搅拌；排出料含气→检查真空密封；表面冒泡>5min→真空度不足+含气超标。",
             "L0+L2+L3耦合", 0.82),
            ("FRULE-L0-005", "颜色 OR 色差 OR 变色 OR 发黑",
             "【L0配方层】颜色异常路径：(1)颜料/色浆分散不良 → 补分散剂或延长分散 (2)基料黄变（热/光氧化）→ 降T或加抗氧剂 (3)固化过度→热变色 (4)原料批次色相差异 → 查颜料批次。",
             "L0配方层", 0.7),
            ("FRULE-L0-006", "强度 OR 剥离 OR 拉伸 OR 硬度 OR 冲击",
             "【L0配方层+L4界面】力学性能路径："
             "(1)固化度不足：DSC测Tg或邵尔硬度确认固化是否完全。"
             "(2)界面失效：区分界面剥离/内聚破坏；界面剥离→偶联剂不足或底材污染。"
             "(3)接枝失效：G_actual=G_theo·η_graft·f(T)·f(γ̇)·f(t) → 高温高剪切下接枝断裂。"
             "DLVO分散稳定性：V_total=V_vdW+V_edl+V_steric>15k_BT才稳定；V_total<15k_BT易团聚→强度下降。",
             "L0+L4耦合", 0.85),
            ("FRULE-L0-007", "漏芯 OR 相变 OR 潜热 OR 包覆",
             "【L0配方层】PCM漏芯/潜热异常："
             "(1)漏芯率L_leak=m_leak/m_0×100% <5%合格 → 换壁材或降剪切。"
             "(2)潜热ΔH=ΔH_PCM·w_PCM·η_capsule·(1-L_leak)→ 查芯材含量和包覆率。"
             "(3)相变点偏移（Gibbs-Thson）：T_m=T_m_0-2σV_m/(r·ΔH_f) → 纳米胶囊<5°C可接受。"
             "(4)DSC相变峰偏移>±3°C → 真空系统异常或配方变更。",
             "L0配方层", 0.8),
            # ── L1 用户非标工况规则 ──
            ("FRULE-L1-001", "施工 OR 客户 OR 售后 OR 冬施",
             "【L1用户非标工况】化工90%玄学批次出自这里："
             "(1)设备私改（混合管节数不足/管路阻力不匹配）→ 重新标定工艺窗口。"
             "(2)加热非标（明火直烤/单点过热/桶壁过热）→ 改水浴或烘房加热。"
             "(3)低温施工（T_a<15°C）→ 冬季补偿：每降5°C料温+3~4°C；粘度比R=μ_树脂/μ_固化剂即为修正方向 → 按当量比调泵而非体积比。"
             "(4)操作不稳（施压不均/中途停机）→ 排空再启动+首件验证。"
             "(5)旧料混用>5% → 重做首件验证。",
             "L1用户非标", 0.88),
            # ── L2 环境耦合规则 ──
            ("FRULE-L2-001", "季节 OR 冬季 OR 夏季 OR 湿度 OR 高湿",
             "【L2环境耦合】温度/湿度影响量化："
             "(1)温度补偿：T_a每降5°C→料温+3~4°C；T_a每升5°C→料温-2~3°C+促进剂-10%。"
             "散热时间常数：τ_heat=ρVc_p/(UA) → T_a↑→散热推动力↓→蓄热加重→批次漂移。"
             "(2)RH判据：>60%需关注、>75%碳粉吸湿(接枝水解)、>85%必须除湿/预烘干。"
             "判定冷凝：IF T_表面 < T_露点(T_a,RH) → 表面结露 → 吸水加速。"
             "(3)冬季配方必须单独设计体积比对照表（每5°C一档），不能全年同一配方。",
             "L2环境耦合", 0.85),
            # ── L3 流场规则 ──
            ("FRULE-L3-001", "搅拌 OR 电流 OR 分散 OR 团聚 OR 沉降",
             "【L3流体流场】关键判据："
             "(1)雷诺数Re=ρND²/μ：Re<10层流(高粘环氧典型)；10<Re<10⁴过渡流；Re>10⁴湍流。容积放大极易流态突变。"
             "(2)桨尖线速度v_tip=πND：量产安全4~6m/s（接枝体系可到7~8m/s）。"
             "(3)幂律粘度μ_app=K·γ̇^(n-1)（假塑性体系）；Herschel-Bulkley τ=τ_y+K·γ̇^n（含填料）。"
             "(4)搅拌电流偏高→粘度过大/碳粉团聚/固含量偏高 → 校验N、μ_app、固含量。"
             "(5)沉降：DLVO V_total<15k_BT → 补分散剂0.2phr或触变剂0.5phr。",
             "L3流场", 0.8),
            # ── L5 放大规则 ──
            ("FRULE-L5-001", "放大 OR 小试 OR 中试 OR 量产 OR 釜",
             "【L5量产放大四大准则（优先级排序）："
             "① 等剪切速率 γ̇=const → N₂=N₁·(D₁/D₂)：保护接枝/高分子链。"
             "② 等桨尖线速度 v_tip=πND=const → N₂=N₁·(D₁/D₂)²：限桨尖过剪切。"
             "③ 等雷诺数 Re=const → N₂=N₁·(D₁/D₂)²：保流态一致。"
             "④ 等单位体积功耗 P/V=const → N₂=N₁·(D₁/D₂)^(2/3)。"
             "放大固有偏移：物料T↑→k_d接枝降解↑(按V^0.8降2~3°C)；停留τ↑→X_d接枝失效↑；剪切场重构→桨尖过剪切+死区并存。"
             "量产安全窗口一定窄于小试窗口！小试转速不能直接平移到量产。",
             "L5放大", 0.9),
            # ── L7 参数敏感性 ──
            ("FRULE-L7-001", "敏感 OR 漂移 OR 批次不稳 OR 波动",
             "【L7参数敏感性矩阵】"
             "S1最高敏感：桨尖线速度v_tip（接枝断裂/碳粉失效）、固化剂纯度/活性（每批必须测胺值）、配比当量比R（±5%容差）。"
             "S2敏感：釜内物料温度T（量产最大风险源）、放大倍数S、雾化压差ΔP、碳粉BET/比表面、偶联剂用量。"
             "S3中等：环境T_a（隐形根源）、RH（超临界后跃变为S1）、真空度P_vac、PCM含量(30~60%窗口)、停留时间τ。"
             "注意：很多现象不是单因子造成，是两个因子交互（如温度×剪切、填料phr×偶联剂）。只调一个参数得不到正确结果。",
             "L7敏感矩阵", 0.85),
            # ── 决策规则 ──
            ("DRULE-001", "新配方 OR 试产 OR 中试",
             "新配方/试产阶段偏差 → 优先调配方(+25~30分)：配方本身未定型，不能把配方缺陷误判为工艺问题。",
             "决策矩阵", 0.9),
            ("DRULE-002", "连续 OR 批次性 OR 多缸多批",
             "连续/批次性系统性异常 → 高度指向配方缺陷(+35~40分)。工艺漂移通常渐进，配方缺陷才导致连续相同问题。",
             "决策矩阵", 0.88),
            ("DRULE-003", "单缸 OR 偶发 OR 自限",
             "单缸偶发自限 → 工艺波动(+30~35分)：温控过冲/搅拌瞬时异常/真空破空/加料波动。",
             "决策矩阵", 0.85),
            ("DRULE-004", "渐进 OR 衰减 OR 漂移 OR 逐渐",
             "性能渐进衰减 → 设备慢变量/工艺漂移(+25~30分)：泵磨损/传感器漂移/桨结垢/滤网堵塞。",
             "决策矩阵", 0.78),
            ("DRULE-005", "换批 OR 换料 OR 新批次",
             "换原料批次后异常 → 配方适配性(+15~20分)：关键指标(EEW/AHEW/包覆率/BET)超出兼容窗口。",
             "决策矩阵", 0.75),
        ]
        for sid, trigger, content, source, conf in seeds:
            insert_rule(sid, trigger, content, source, conf)

    # ═══════════════════════════════════════════════════════════
    # 二、知识文本（knowledge_text）：向量化供RAG检索
    # ═══════════════════════════════════════════════════════════
    existing_kt = conn.execute("SELECT COUNT(*) FROM knowledge_text").fetchone()[0]
    if existing_kt == 0:
        kt = [
            # ── L0 配方知识 ──
            ("formula_knowledge", "K-L0-001",
             "化学计量链(配方核心)：① EEW=100/环氧值 ② AHEW=M_amine/n_H ③ phr=(AHEW/EEW)×100 "
             "④ 理论质量比 m_B/m_A=AHEW/EEW ⑤ 理论体积比 V_B/V_A=(AHEW/EEW)×(ρ_A/ρ_B) "
             "⑥ r_actual=(Q_B·ρ_B/AHEW_B)/(Q_A·ρ_A/EEW_A) ⑦ Δr=(r_actual-1)×100%。"
             "名义1:1体积比≠化学计量1:1，必须用EEW/AHEW/密度/纯度换算。|Δr|≤5%可接受，>10%高危。"),
            ("formula_knowledge", "K-L0-002",
             "固化动力学公式集：Carothers凝胶点 p_c=2/f_avg；Flory p_c=1/√[(f_A-1)(f_B-1)]；"
             "交联密度 E=3ν_eRT；Fox Tg混合律 1/T_g=Σw_i/T_g,i；Gordon-Taylor T_g=[w_1T_g1+k·w_2T_g2]/[w_1+k·w_2]；"
             "Kamal动力学 dα/dt=(k_1+k_2·α^m)(1-α)^n；放热温升 ΔT_ad=ΔH_rxn/c_p；"
             "凝胶时间 t_gel=A·exp(E_a/RT)（用温度调凝胶时间最直接）。"),
            ("formula_knowledge", "K-L0-003",
             "相容分散界面公式：Hansen溶解度 R_a²=4(δ_d1-δ_d2)²+(δ_p1-δ_p2)²+(δ_h1-δ_h2)²；"
             "DLVO V_total=V_vdW+V_edl+V_steric（>15k_BT稳定）；分散剂用量 m_disp≈k·A_s·m_p；"
             "接枝率 G=(m_graft/m_carbon)×100%（目标0.5~2%）；"
             "接枝密度 σ_graft=G·N_A/(M_graft·A_s)（>0.1链/nm²稳定）；"
             "泄漏率 L_leak=m_leak/m_0×100%（<5%合格）。"),
            ("formula_knowledge", "K-L0-004",
             "粘度/流变公式：Einstein η=η_0(1+2.5φ)（φ<5%）；"
             "Krieger-Dougherty η=η_0(1-φ/φ_m)^(-[η]·φ_m)（高φ碳粉）；"
             "Maxwell-E导热 k_eff=k_m·[k_f+2k_m+2φ(k_f-k_m)]/[k_f+2k_m-φ(k_f-k_m)]（低中φ）；"
             "PCM有效潜热 ΔH_eff=ΔH_PCM·w_PCM·η_capsule·(1-L_leak)。"
             "Arrhenius：每降5°C粘度×2~3倍，必须按当量比调泵而非体积比。"),
            # ── L4 材料反应知识 ──
            ("formula_knowledge", "K-L4-001",
             "接枝控制判定：G_actual=G_theo·η_graft·f(T)·f(γ̇)·f(t)。"
             "f(T)：过高→链转移多，过低→引发不足；f(γ̇)：过高→已接枝链断裂；f(t)：后期扩散控制。"
             "接枝失效(X_d)分级：<10%轻微(沉降/刮涂略降)；10~25%中度(团聚/界面脱粘/导热↓↓)；>25%重度(脱落/PCM泄漏/报废)。"),
            ("formula_knowledge", "K-L4-002",
             "相变偏移判据：DSC相变峰偏移>±3°C → 真空系统异常或配方变更；"
             "潜热<标称90% → PCM原料ΔH↓或碳粉含水吸热；"
             "DSC出现双峰 → 不同组分分别相变 → 基体不均匀。"),
            # ── L3 流场知识 ──
            ("process_knowledge", "K-L3-001",
             "雷诺数 Re=ρND²/μ：层流<10，过渡10~10⁴，湍流>10⁴。"
             "桨尖线速度 v_tip=πND：小试4~8m/s，中试4~7m/s，量产4~6m/s。"
             "超上限→接枝断裂；低于下限→分散不足/团聚。"
             "容积放大极易流态突变（小试层流→量产湍流）。"),
            ("process_knowledge", "K-L3-002",
             "四大放大准则优先级：①等γ̇(保护接枝) ②等v_tip(限过剪切) ③等Re(保流态) ④等P/V(能耗)。"
             "放大偏移：物料T↑→k_d降解(按V^0.8降2~3°C)；τ↑→X_d失效；剪切场重构→过剪切+死区并存。"),
            # ── L5 放大工艺窗口速查 ──
            ("process_knowledge", "K-L5-001",
             "耦合工艺窗口(示意值需按配方标定)："
             "v_tip：小试4~8/中试4~7/量产4→6m/s | 物料T：小试55~75/中试55~72/量产55~70°C | "
             "剪切γ̇：小试5~40/中试5~35/量产5→30s⁻¹ | 真空P_vac：-0.05~-0.08MPa | "
             "停留τ：小试20~40/中试20~35/量产20→30min。"
             "关键规则：配方组分改动必须同步刷新整套工艺窗口！"),
            # ── L2 环境知识 ──
            ("process_knowledge", "K-L2-001",
             "碳粉含水→气泡速查：搅拌初期大量小气泡上翻→延长搅拌；"
             "排出料含气翻滚→检查真空系统密封性；表面冒泡>5min→真空度不足+含气量超标。"
             "高碳粉体系脱泡三步：预烘干80°C/4h+润湿20min+真空(三步缺一不可)。"
             "RH>75%必须启动除湿+预烘干程序。"),
        ]
        for stype, sid, content in kt:
            index_text(stype, sid, content)

    # ═══════════════════════════════════════════════════════════
    # 三、失败教训（新增独立表：failure_lesson）
    # ═══════════════════════════════════════════════════════════
    fl_count = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='failure_lesson'"
    ).fetchone()[0]
    if fl_count == 0:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS failure_lesson (
                lesson_id TEXT PRIMARY KEY,
                action_desc TEXT NOT NULL,
                consequence TEXT,
                redline TEXT NOT NULL,
                loss_level TEXT
            )
        """)
        lessons = [
            ("FAIL-001", "一次性大量加稀释剂+8phr",
             "交联网络被稀释破坏、Tg大幅下降、冷热循环大批量开裂",
             "单次稀释剂≤5phr，必须分2~3次递进验证", "重大批量报废"),
            ("FAIL-002", "小试转速直接等比例平移到量产釜",
             "桨尖过剪切、碳粉接枝大面积断裂、整批团聚报废",
             "放大必须执行四大放大准则，禁止照搬转速", "重大批量报废"),
            ("FAIL-003", "为解决固化慢单方面大幅上调固化剂",
             "交联度过高、内应力爆炸、成品批量开裂、漏芯增多",
             "调固化剂必须评估脆性，禁止单点暴力调参", "批量报废"),
            ("FAIL-004", "高RH车间未预烘干直接投料",
             "接枝水解、粉体团聚、气泡超标",
             "RH>75%必须先预烘干80°C/4h+真空润湿", "批次不良"),
            ("FAIL-005", "冬季按体积比调泵不按当量比",
             "管路两侧粘度不对称→配比漂移→不固化/发粘",
             "冬季必须计算当量比R·ΔT↓5°C→μ↑3~4倍，按当量比调泵", "批次不良"),
            ("FAIL-006", "单一组分急救式调整",
             "单点改配比→Tg漂移→性能顾此失彼",
             "必须做组分互补配对(2~3组分联动)，禁止单点暴力调参", "中"),
        ]
        conn.executemany("INSERT OR IGNORE INTO failure_lesson VALUES(?,?,?,?,?)", lessons)
        conn.commit()

    # ═══════════════════════════════════════════════════════════
    # 四、工艺窗口样例（用于展示查询能力）
    # ═══════════════════════════════════════════════════════════
    pw_count = conn.execute("SELECT COUNT(*) FROM process_window").fetchone()[0]
    if pw_count == 0:
        # 为某通用配方注入一套示意工艺窗口
        upsert_formula("FORMULA-DEMO-001", "通用环氧树脂导热灌封胶", "灌封胶系列",
                      "A树脂+固化剂+碳粉+稀释剂+导热填料",
                      "搅拌/温控/真空/保温", "导热≥0.8W/mK", "demo")
        demo_windows = [
            ("FORMULA-DEMO-001", "桨尖线速度v_tip", 4.0, 6.0, 5.0, "m/s"),
            ("FORMULA-DEMO-001", "物料温度T_mat", 55.0, 70.0, 62.0, "°C"),
            ("FORMULA-DEMO-001", "搅拌转速N", 100.0, 300.0, 200.0, "rpm"),
            ("FORMULA-DEMO-001", "真空度P_vac", -0.08, -0.05, -0.06, "MPa"),
            ("FORMULA-DEMO-001", "停留时间τ", 20.0, 30.0, 25.0, "min"),
            ("FORMULA-DEMO-001", "环境RH上限", 0.0, 75.0, 60.0, "%"),
        ]
        for fid, pn, lo, hi, nominal, unit in demo_windows:
            add_process_window(fid, pn, lo, hi, nominal, unit)

    # ═══════════════════════════════════════════════════════════
    # 五、附录 I~S 知识入库
    # ═══════════════════════════════════════════════════════════
    # --- I. 测试指标-产线工序映射 ---
    tpm_count = conn.execute("SELECT COUNT(*) FROM test_process_mapping").fetchone()[0]
    if tpm_count == 0:
        test_mappings = [
            ("DSC相变峰偏移>±3°C", "釜温控系统、真空脱泡工段", "原料PCM批次", "复测2份平行样；确认DSC校准有效"),
            ("潜热ΔH低于标称90%", "接枝分散搅拌工段、投料称量", "PCM原料来料", "排除制样时PCM渗漏造成测试假象"),
            ("Tg明显漂移", "配比投料工段（AHEW/EEW计量）", "固化温控工段", "校验当量比r_actual；排除样品固化不完全"),
            ("剪切强度断崖下跌", "偶联剂分散搅拌工段", "储存-老化（RH水汽侵入）", "区分失效模式：界面剥离/内聚破坏"),
            ("粒度Span>1.5", "雾化工段、搅拌分散工段", "物料粘度漂移", "排除取样、制样造成粒度测试假象"),
            ("凝胶时间t_gel严重偏离", "投料配比工段、环境温控工段", "助剂投料（促进剂/抑制剂）", "确认测试时样品温度条件"),
        ]
        conn.executemany(
            "INSERT INTO test_process_mapping (test_indicator, priority_process, secondary_process, exclusion_note) VALUES(?,?,?,?)",
            test_mappings
        )

    # --- J. 材料相容性矩阵 ---
    mc_count = conn.execute("SELECT COUNT(*) FROM material_compatibility").fetchone()[0]
    if mc_count == 0:
        compat_data = [
            ("铂催化有机硅", "含硫助剂", "看似稳定", "微量硫即造成铂催化剂中毒", "胶体永久不固化", "严禁混合", "绝对禁止：任何含硫物质接触铂催化硅体系"),
            ("环氧胺固化剂", "水汽", "短期共存", "高RH储存固化剂吸潮预反应", "活性下降", "条件相容（需干燥密封）", "必须干燥密封保存，RH>60%需除湿"),
            ("接枝碳粉", "强酸性助剂", "尚可", "高剪切+高温接枝界面水解断裂", "粉体脱附团聚", "谨慎使用", "需提前实验室验证相容性"),
            ("铝粉/活泼金属粉", "环氧胺体系+湿气", "缓慢反应", "高温下界面氧化", "导热层氧化钝化", "条件相容（需表面钝化）", "使用前需表面处理或偶联剂包覆"),
            ("导热氧化铝粉", "含水偶联剂", "粘度异常", "流变非线性剧变", "粉体架桥", "条件相容（需预烘干）", "80°C/4h预烘干后使用"),
        ]
        conn.executemany(
            "INSERT INTO material_compatibility (material_a, material_b, static_ok, dynamic_high_tell, long_term_aging, conclusion, redline) VALUES(?,?,?,?,?,?,?)",
            compat_data
        )

    # --- K. 经验规则触发边界 ---
    rb_count = conn.execute("SELECT COUNT(*) FROM rule_trigger_boundary").fetchone()[0]
    if rb_count == 0:
        boundaries = [
            ("冬季环境温度每下降5°C，釜料预升温3~4°C",
             "高填料环氧/PCM体系；环境Ta<15°C；非低填料透明体系",
             "低填料<20%体系；UV光固化体系",
             "若和接枝温度上限冲突，优先服从接枝失效温度红线",
             "高", "历史售后统计，同类故障发生4次"),
            ("高RH>75%必须预烘干碳粉80°C/4h",
             "碳粉/竹粉体系",
             "不含吸湿填料体系",
             "服从安全红线（人员安全>生产效率）",
             "高", "多批次气泡/团聚事故复盘"),
            ("放大时优先等剪切速率",
             "含接枝/高分子链体系",
             "纯低粘混合",
             "与等桨尖线速度并存权衡（通常取更严的）",
             "中", "老工程师经验+流场模拟验证"),
            ("旧料混入>5%必须重做首件",
             "所有双组份产品",
             "单组份产品",
             "服从客户验收标准",
             "高", "客户投诉案例复盘"),
        ]
        conn.executemany(
            "INSERT INTO rule_trigger_boundary (rule_text, preconditions, not_applicable, conflict_resolution, confidence_level, source) VALUES(?,?,?,?,?,?)",
            boundaries
        )

    # --- M. 行业标准约束 ---
    std_count = conn.execute("SELECT COUNT(*) FROM industry_standard").fetchone()[0]
    if std_count == 0:
        standards = [
            ("PCM相变潜热", "导热灌封胶/TIM凝胶", "ASTM C1784-14；GB/T 19466.1", "升降温速率5°C/min", "按标称值±10%", "升降温速率影响结果", "不达标将导致客户拒收"),
            ("PCM相变温度", "导热灌封胶/TIM凝胶", "DSC按GB/T 19466.1", "升降温速率5°C/min", "±3°C", "与ASTM测试条件可能不同", "影响产品认证和终端使用"),
            ("环氧剪切强度", "结构胶/灌封胶", "GB/T 7124-2008", "搭接面积25×12.5mm，加载速率5mm/min", "≥8MPa", "加载速率和搭接面积影响数值", "客户验收关键指标"),
            ("导热系数", "导热灌封胶/TIM凝胶", "GB/T 10297(稳态)/ASTM D5470(界面)", "25°C稳态", ">0.8W/(m·K)", "稳态法与界面垫片法数值不可直接换算", "直接影响客户选型"),
            ("玻璃化转变温度Tg", "固化环氧树脂", "GB/T 19466.2 (DSC)", "二次扫描为准", "按产品规格", "Tg1(首次)≠Tg2(二次)，差值>5°C说明固化不完全", "耐热性关键指标"),
            ("胶粘剂粘度", "所有液态胶粘剂", "GB/T 2794", "指定转子/转速", "按产品规格(cP)", "转子/转速影响读数", "影响施工工艺设计"),
        ]
        conn.executemany(
            "INSERT INTO industry_standard (performance_indicator, product_scope, standard_code, test_condition, threshold_value, notes, business_consequence) VALUES(?,?,?,?,?,?,?)",
            standards
        )

    # --- N. DOE交互效应 ---
    doe_count = conn.execute("SELECT COUNT(*) FROM doe_interaction").fetchone()[0]
    if doe_count == 0:
        interactions = [
            ("温度", "剪切速率", "强交互", "高温下高剪切接枝失效急剧放大，必须联动调整（不能只降一个）"),
            ("填料phr", "偶联剂phr", "强交互", "高填料必须匹配更高偶联剂用量，否则界面失效→团聚→强度下降"),
            ("固含量", "雾化压差", "中交互", "高固含需更高雾化压差补偿雾化效果"),
            ("环境RH", "储存时间", "强交互", "高RH+长储存接枝水解累积效应，时间越长劣化越严重"),
            ("物料温度T", "环境温度Ta", "中交互", "厂房温度变化→散热推动力→表观温度漂移"),
            ("搅拌转速N", "物料填充量", "中交互", "相同转速下填充量影响实际剪切场（尤其非满釜）"),
        ]
        conn.executemany(
            "INSERT INTO doe_interaction (factor_a, factor_b, interaction_level, description) VALUES(?,?,?,?)",
            interactions
        )

    # --- O. 原料隐性风险 ---
    hr_count = conn.execute("SELECT COUNT(*) FROM material_hidden_risk").fetchone()[0]
    if hr_count == 0:
        hidden_risks = [
            ("碳粉/竹基粉体", "粒度、BET比表面积", "表面羟基含量、微量吸附水分、接枝残留杂质", "粘度漂移、接枝水解、团聚"),
            ("氧化铝/氮化铝导热粉", "粒度、纯度", "粉体表面羟基、吸附水", "体系粘度剧烈变化、研磨效率下降"),
            ("环氧树脂", "EEW环氧当量", "微量小分子杂质、水解氯含量", "固化速度漂移、后期金属腐蚀"),
            ("固化剂胺类", "AHEW活泼氢当量", "微量水分、氧化变质组分", "储存后预反应活性下降、配比失效"),
        ]
        conn.executemany(
            "INSERT INTO material_hidden_risk (material_category, routine_checks, hidden_risks, process_consequence) VALUES(?,?,?,?)",
            hidden_risks
        )

    # --- P. 设备老化漂移 ---
    ed_count = conn.execute("SELECT COUNT(*) FROM equipment_drift").fetchone()[0]
    if ed_count == 0:
        equipment_data = [
            ("搅拌桨", "桨叶表面结垢粘附物料；桨有效直径变大", "同等转速下实际剪切上升、功耗失控、接枝断裂", "停机开釜测桨外形对比新桨；实测功耗与基线偏差"),
            ("双组份计量泵", "屏幕配比读数正常但实际流量偏移", "r_actual悄悄偏移无报警→批次性配比漂移", "定期称重法标定输出流量"),
            ("温度传感器", "设定温度不变但釜内多点温差逐步扩大", "局部蓄热、接枝热降解、局部过固化", "多点热电偶交叉比对校验"),
            ("烘箱/干燥风道", "风速分布随积灰逐渐畸变", "局部干燥过快/过慢、多孔材料孔隙坍塌", "风道风速分布定期检测"),
            ("喷嘴/雾化头", "出口磨损，压力读数不变", "D_32变粗、Span雾化均匀性变差", "更换同型号喷嘴对照测试"),
        ]
        conn.executemany(
            "INSERT INTO equipment_drift (equipment_part, drift_symptom, real_consequence, detection_method) VALUES(?,?,?,?)",
            equipment_data
        )

    # --- Q. 成本-性能权衡方案 ---
    cp_count = conn.execute("SELECT COUNT(*) FROM cost_performance_plan").fetchone()[0]
    if cp_count == 0:
        cp_plans = [
            # 团聚整改
            ("粉体团聚整改", "上调偶联剂0.3phr", "中：分散状态改善", "原料成本小幅上升", "小试验证2批次，无需停机", "低", 1),
            ("粉体团聚整改", "换更高BET碳粉", "高：分散性大幅提升", "原料成本+15~20%", "完整DOE实验，周期2~4周", "中", 3),
            ("粉体团聚整改", "延长分散时间10min", "中低：分散更均匀", "几乎无额外成本", "当天可验证", "低", 2),
            # 粘度异常
            ("体系粘度过高", "小幅升温3~5°C", "中：Arrhenius效应明显", "几乎无额外成本", "当批验证", "低（需验接枝热降解风险）", 1),
            ("体系粘度过高", "活性稀释剂+2~3phr", "中：快速降粘", "原料成本小幅下降", "当批验证+留样观察", "中（过量影响Tg）", 2),
            ("体系粘度过高", "降固含量1~2%", "低：降粘但可能影响性能", "无额外成本", "小试验证性能达标", "中（规格可能不达标）", 3),
            # 固化异常
            ("固化不良", "升温2~3°C", "中高：Arrhenius效应", "几乎无额外成本", "当批验证DSC确认", "低", 1),
            ("固化不良", "微调固化剂±0.5phr", "高：直接矫正配比", "原料成本几乎不变", "小试验证", "中（过量→脆性）", 2),
            # 气泡超标
            ("气泡超标", "真空<-0.08MPa延长5min", "中高：提升脱泡效率", "能耗小幅增加", "当批验证", "低", 1),
            ("气泡超标", "预烘干80°C/4h", "高：根除湿源", "需要烘干设备+工时", "提前半天准备", "低", 2),
        ]
        conn.executemany(
            "INSERT INTO cost_performance_plan (scenario, candidate_plan, performance_gain, cost_change, verification_cycle, risk_level, priority_rank) VALUES(?,?,?,?,?,?,?)",
            cp_plans
        )

    # --- R. 客户工况约束 ---
    cc_count = conn.execute("SELECT COUNT(*) FROM customer_constraint").fetchone()[0]
    if cc_count == 0:
        constraints = [
            ("冬施户外施工客户", "无恒温棚、加热手段有限", "冬季混合不均、配比漂移、粘度上升", "冬季体积比对照表+桶预热规范+施工温度下限提示"),
            ("高节拍产线客户要求快干", "供料速度高、易私改混合管节数", "微观混合不均、层流不充分", "出厂前混合管节数锁死、防拆提醒+定期巡检"),
            ("高湿地区客户（华南/东南亚）", "露天存放、无除湿设备", "接枝水解、碳粉吸湿、微孔", "RH>75%强制密封+干燥剂+到货即用"),
            ("低成本敏感客户", "对价格极度敏感，不接受涨价", "降本空间有限，需避免偷工减料", "替代原料验证后替换，并提供对等规格说明"),
        ]
        conn.executemany(
            "INSERT INTO customer_constraint (customer_type, inherent_limits, high_freq_failures, exclusive_constraints) VALUES(?,?,?,?)",
            constraints
        )

    # --- S. 工程师思维链案例 ---
    tcc_count = conn.execute("SELECT COUNT(*) FROM thinking_chain_case").fetchone()[0]
    if tcc_count == 0:
        thinking_cases = [
            ("TC-001", "批次潜热忽高忽低，配方未动，来料合格",
             "DSC设备校准问题/原料批次波动",
             "取芯材原料直接DSC→原料稳定→排除原料问题",
             "A投料总量微小浮动→体积变化→散热面积/体积比偏移→蓄热温度不同→接枝失效；B搅拌桨磨损→v_tip下降→分散漂移",
             "非必要不动配方，先查工艺/设备隐性漂移",
             "调取各批实际投料总重量；拆桨测桨尖直径对比基线",
             "操作工投料允许±3kg浮动→料高变化→换热条件改变→部分批次蓄热超标→碳粉接枝轻微降解→潜热不稳",
             "锁定配方≠锁死工艺环境；投料总量微小浮动常被忽略"),
            ("TC-002", "小试完美、中试漏芯（PCM包封率）",
             "配方放大效应未考虑",
             "中试釜降到小试体积投料→漏芯消失→锁定不是配方/原料问题",
             "A满釜桨尖局部过剪切断接枝；B体积放大蓄热降解接枝；C死区停留过长",
             "不要盲目降转速（会带来分散不足团聚），两难问题需权衡",
             "釜内多点测温；取桨尖附近物料做界面表征",
             "中试满釜立方-平方悖论生效→热蓄积→接枝失效Xd↑→界面脱粘→漏芯",
             "放大后窗口整体收缩，量产安全窗口一定窄于小试窗口"),
        ]
        conn.executemany(
            "INSERT INTO thinking_chain_case VALUES(?,?,?,?,?,?,?,?,?)",
            thinking_cases
        )
        conn.commit()

    conn.close()


def list_failure_lessons(keyword: str = "") -> list:
    """检索失败教训"""
    conn = get_conn()
    if keyword:
        rows = conn.execute(
            "SELECT * FROM failure_lesson WHERE action_desc LIKE ? OR consequence LIKE ?",
            (f"%{keyword}%", f"%{keyword}%")
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM failure_lesson").fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════
# 附录 I~S 查询函数
# ═══════════════════════════════════════════════════════════

def list_test_process_mappings(keyword=""):
    conn = get_conn()
    if keyword:
        rows = conn.execute(
            "SELECT * FROM test_process_mapping WHERE test_indicator LIKE ?",
            (f"%{keyword}%",)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM test_process_mapping").fetchall()
    return [dict(r) for r in rows]


def search_compatibility(material_keyword):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM material_compatibility WHERE material_a LIKE ? OR material_b LIKE ?",
        (f"%{material_keyword}%", f"%{material_keyword}%")
    ).fetchall()
    return [dict(r) for r in rows]


def list_doe_interactions():
    conn = get_conn()
    return [dict(r) for r in conn.execute("SELECT * FROM doe_interaction").fetchall()]


def list_cost_performance_plans(scenario=""):
    conn = get_conn()
    if scenario:
        rows = conn.execute(
            "SELECT * FROM cost_performance_plan WHERE scenario LIKE ? ORDER BY priority_rank",
            (f"%{scenario}%",)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM cost_performance_plan ORDER BY priority_rank").fetchall()
    return [dict(r) for r in rows]


def list_thinking_chain_cases():
    conn = get_conn()
    return [dict(r) for r in conn.execute("SELECT * FROM thinking_chain_case").fetchall()]


def list_customer_constraints():
    conn = get_conn()
    return [dict(r) for r in conn.execute("SELECT * FROM customer_constraint").fetchall()]


def list_industry_standards(keyword=""):
    conn = get_conn()
    if keyword:
        rows = conn.execute(
            "SELECT * FROM industry_standard WHERE performance_indicator LIKE ?",
            (f"%{keyword}%",)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM industry_standard").fetchall()
    return [dict(r) for r in rows]


def list_rule_boundaries():
    conn = get_conn()
    return [dict(r) for r in conn.execute("SELECT * FROM rule_trigger_boundary").fetchall()]


def list_material_hidden_risks():
    conn = get_conn()
    return [dict(r) for r in conn.execute("SELECT * FROM material_hidden_risk").fetchall()]


def list_equipment_drift():
    conn = get_conn()
    return [dict(r) for r in conn.execute("SELECT * FROM equipment_drift").fetchall()]
