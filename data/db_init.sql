-- ============================================================
-- 新材料工艺偏差排查AI系统 - 数据库初始化脚本
-- 技术栈: PostgreSQL + pgvector
-- 执行方法: psql -U postgres -d material_db -f db_init.sql
-- ============================================================

-- 启用向量插件
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 表1：配方工艺档案
-- ============================================================
CREATE TABLE IF NOT EXISTS formula_archive (
    formula_id VARCHAR(32) PRIMARY KEY,
    formula_name VARCHAR(128) DEFAULT '',
    material_list TEXT NOT NULL,        -- 成分+规格+供应商+批次(JSON文本)
    process_params TEXT NOT NULL,       -- 温度/时间/压力/转速等(JSON文本)
    env_params TEXT,                    -- 温湿度/季节/洁净度(JSON文本)
    theory_expected TEXT,               -- AI预测输出(JSON文本)
    source_tag TEXT DEFAULT 'manual',   -- manual / ai_generated
    create_time TIMESTAMP DEFAULT NOW(),
    update_time TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE formula_archive IS 'AI生成的配方工艺或人工设计配方档案';

-- ============================================================
-- 表2：试验结果记录
-- ============================================================
CREATE TABLE IF NOT EXISTS test_record (
    test_id VARCHAR(32) PRIMARY KEY,
    formula_id VARCHAR(32) REFERENCES formula_archive(formula_id),
    core_index TEXT NOT NULL,           -- 核心收率/性能数据(JSON文本)
    deviation_desc TEXT,                -- 偏差自然语言描述
    deviation_magnitude FLOAT,          -- 偏差幅度百分比，用于阈值判断
    attach_link TEXT,                   -- 原始记录附件路径
    real_condition TEXT,                -- 实际试验条件(JSON文本)
    create_time TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE test_record IS '试验结果填报，偏差超阈值自动触发排查';

-- ============================================================
-- 表3：经验规则库（一点课）
-- ============================================================
CREATE TABLE IF NOT EXISTS experience_rule (
    rule_id VARCHAR(32) PRIMARY KEY,
    rule_name VARCHAR(128) DEFAULT '',
    trigger_condition TEXT NOT NULL,    -- 触发条件(JSON结构化描述)
    exp_content TEXT NOT NULL,          -- 经验结论
    verify_count INT DEFAULT 0,         -- 总验证次数
    hit_count INT DEFAULT 0,            -- 命中次数
    source_info TEXT,                   -- 来源（谁写的、哪个场景）
    confidence FLOAT DEFAULT 0.5,       -- 置信度 0~1，贝叶斯动态更新
    create_time TIMESTAMP DEFAULT NOW(),
    update_time TIMESTAMP DEFAULT NOW(),
    CONSTRAINT chk_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

COMMENT ON TABLE experience_rule IS '高经验人员的一点课，置信度动态更新';

-- ============================================================
-- 表4：排查案例库（核心资产）
-- ============================================================
CREATE TABLE IF NOT EXISTS case_library (
    case_id VARCHAR(32) PRIMARY KEY,
    problem_desc TEXT NOT NULL,         -- 归一化后的问题描述
    problem_original TEXT,              -- 用户原始输入
    hypothesis_list JSONB,              -- [{hypo_id, hypo_text, init_prob, final_prob, status}]
    check_process JSONB,                -- [{round, question, answer, eliminate_list, confirm_list}]
    root_cause TEXT,                    -- 确认的根因
    root_cause_rule_ids TEXT[],         -- 关联的经验规则ID
    improve_solution_id VARCHAR(32),    -- 关联改善方案
    improve_result TEXT,                -- 改善效果回授
    check_round INT DEFAULT 0,          -- 找到根因的排查轮次
    status VARCHAR(20) DEFAULT 'open',  -- open/closed/expert_referral
    create_time TIMESTAMP DEFAULT NOW(),
    update_time TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE case_library IS '完整排查案例，假设列表含初始和最终概率';

-- ============================================================
-- 表5：改善方案知识库
-- ============================================================
CREATE TABLE IF NOT EXISTS improve_solution (
    solution_id VARCHAR(32) PRIMARY KEY,
    solution_name VARCHAR(128) DEFAULT '',
    root_cause_type VARCHAR(64) NOT NULL, -- raw_material/equipment/process/environment/operator
    plan_short TEXT,                    -- 短期措施
    plan_mid TEXT,                      -- 中期措施
    plan_long TEXT,                     -- 长期措施
    applicable_condition TEXT,          -- 适用前提
    expect_improve TEXT,                -- 预期改善效果
    risk_warning TEXT,                  -- 风险提示
    source_case_ids TEXT[],             -- 来源案例
    create_time TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE improve_solution IS '分层改善方案模板库，按根因类型索引';

-- ============================================================
-- 向量检索表：全知识库embedding
-- ============================================================
CREATE TABLE IF NOT EXISTS knowledge_vector (
    id SERIAL PRIMARY KEY,
    source_type VARCHAR(20) NOT NULL,   -- formula / rule / case / solution
    source_id VARCHAR(32) NOT NULL,     -- 关联业务表ID
    content TEXT NOT NULL,              -- 文本片段
    content_hash VARCHAR(64),           -- 去重hash
    embedding vector(1024),             -- 豆包embedding维度1024
    create_time TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE knowledge_vector IS '知识库全文向量索引，支持语义检索';

-- 向量索引（HNSW加速余弦相似度检索）
CREATE INDEX IF NOT EXISTS idx_knowledge_vector_embedding
    ON knowledge_vector USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 来源ID检索索引
CREATE INDEX IF NOT EXISTS idx_knowledge_vector_source
    ON knowledge_vector (source_type, source_id);

-- 去重索引
CREATE INDEX IF NOT EXISTS idx_knowledge_vector_hash
    ON knowledge_vector (content_hash);

-- ============================================================
-- 变更追踪表：置信度更新历史
-- ============================================================
CREATE TABLE IF NOT EXISTS confidence_history (
    id SERIAL PRIMARY KEY,
    rule_id VARCHAR(32) NOT NULL,
    case_id VARCHAR(32),
    old_confidence FLOAT,
    new_confidence FLOAT,
    update_reason VARCHAR(20),          -- hit / miss / decay
    create_time TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE confidence_history IS '置信度更新历史，用于审计和回滚';

-- ============================================================
-- 完成提示
-- ============================================================
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE ' 新材料工艺偏差排查AI系统 - 数据库初始化完成';
    RAISE NOTICE ' 表数量: 6张业务表 + 1张向量表';
    RAISE NOTICE ' 向量维度: 1024 (豆包企业API)';
    RAISE NOTICE '========================================';
END $$;
