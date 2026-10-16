"""演示引擎v2：主引擎 + 审计Agent双系统联动"""
import sys, types, os

# Mock psycopg2（演示模式不需要PG）
for m in ["psycopg2", "psycopg2.extras", "psycopg2.extensions", "pgvector", "pgvector.psycopg2"]:
    sys.modules[m] = types.ModuleType(m)
sys.modules["psycopg2"].extras = sys.modules["psycopg2.extras"]
sys.modules["psycopg2"].extensions = sys.modules["psycopg2.extensions"]
sys.modules["psycopg2.extras"].RealDictCursor = None
sys.modules["psycopg2.extras"].Json = lambda x, **kw: x
sys.modules["pgvector.psycopg2"].register_vector = lambda *a, **k: None

# 设置超时（审计每次调用模型都需要较长时间）
os.environ["DOUBAO_TIMEOUT"] = "300"

import config
import db_sqlite as db
from model_manager import get_model
from bayesian_engine import calculate_prior as bayes_calc
from auditor_engine import get_auditor

import json
import re
import uuid
from datetime import datetime


def _gen_id(prefix):
    return f"{prefix}-{datetime.now().strftime('%Y%m%d')}-{uuid.hex()[:6]}"


def _safe_extract_json(text):
    try:
        return json.loads(text)
    except:
        pass
    raw = text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    s = raw.find("{")
    e = raw.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(raw[s:e + 1])
        except:
            pass
    return {}


def start_diagnosis_with_audit(problem_text: str) -> dict:
    """排查全流程——带审计"""
    model = get_model()
    auditor = get_auditor()

    print("\n" + "=" * 65)
    print("  全品类材料 — AI 排查 + 审计 双系统")
    print("=" * 65)
    print(f"\n📋 问题: {problem_text}")

    full_audit_data = {"problem": problem_text, "eliminated": []}

    # ====== 主引擎：Step1 RAG召回 ======
    print("\n── 主引擎 Step1: 知识库检索 ──")
    retrieved = db.retrieve(problem_text, top_k=5)
    print(f"  召回: 规则{len([r for r in retrieved if r['source_type']=='rule'])} 案例{len([r for r in retrieved if r['source_type']=='case'])} 知识{len([r for r in retrieved if r['source_type']=='knowledge'])}")

    full_audit_data["retrieved"] = retrieved

    # ====== 审计：召回质量 ======
    print("\n── 审计Agent: 召回质量 ──")
    rag_audit = auditor.audit_rag(problem_text, retrieved)
    print(f"  召回得分: {rag_audit.get('recall_score', '?')}/10")
    for issue in rag_audit.get("issues", []):
        print(f"  ⚠️ [{issue.get('type')}] {issue.get('description', '')[:60]}")
    if rag_audit.get("missing_kinds"):
        print(f"  缺失知识类别: {', '.join(rag_audit['missing_kinds'])}")

    full_audit_data["rag_audit"] = rag_audit

    # ====== 主引擎：Step2 假设生成 ======
    print("\n── 主引擎 Step2: 豆包AI头脑风暴 ──")
    prompt_sys = config.load_prompt("hypothesis")
    know_text = "\n".join(f"- {r['content'][:60]}" for r in retrieved[:2])
    prompt_user = f"偏差描述：{problem_text}\n相关知识库片段：\n{know_text}"
    hypo_resp = model.chat_completion(prompt_sys, prompt_user, temperature=0.2)
    hypo_data = _safe_extract_json(hypo_resp)
    hypo_list_raw = hypo_data.get("hypothesis_list", [])

    hypo_from_cases = []
    for r in retrieved:
        if r["source_type"] == "case":
            rc = db.get_case_by_id(r["source_id"])
            if rc and rc.get("root_cause"):
                hypo_from_cases.append(f"[案例{rc['case_id'] if 'case_id' in rc else r['source_id']}] {rc['root_cause']}")

    hypo_list = list(dict.fromkeys(hypo_from_cases + hypo_list_raw))[:5]
    hypo_prob = bayes_calc(hypo_list, [{"content": r["content"], "confidence": 0.5} for r in retrieved if r["source_type"] == "rule"])

    print("\n  编号 | 置信度假设")
    print("  " + "-" * 52)
    for i, (h, p) in enumerate(hypo_prob.items(), 1):
        bar = "█" * int(p * 30)
        print(f"  H{i}   | {p:5.1%} {bar} {h[:40]}...")

    full_audit_data["hypotheses"] = hypo_prob

    # ====== 审计：假设覆盖度 ======
    print("\n── 审计Agent: 假设覆盖度 ──")
    hypo_audit = auditor.audit_hypotheses(problem_text, retrieved, hypo_prob)
    print(f"  覆盖得分: {hypo_audit.get('coverage_score', '?')}/10")
    for blind in hypo_audit.get("blind_spots", []):
        print(f"  🔍 盲区 [{blind.get('dimension')}]: {blind.get('possible_cause', '')[:50]}")
    if hypo_audit.get("overlooked_dimensions"):
        print(f"  未考虑维度: {', '.join(hypo_audit['overlooked_dimensions'])}")

    full_audit_data["hypotheses_audit"] = hypo_audit

    # ====== 主引擎：Step3 追问 ======
    print("\n── 主引擎 Step3: 生成追问 ──")
    prompt_sys_q = config.load_prompt("question")
    prompt_user_q = f"问题：{problem_text}\n当前假设:\n" + "\n".join(f"  {i+1}. {h} ({p:.0%})" for i, (h, p) in enumerate(hypo_prob.items()))
    q_resp = model.chat_completion(prompt_sys_q, prompt_user_q, temperature=0.1)
    q_info = _safe_extract_json(q_resp)
    questions = q_info.get("question_list", [])

    for i, q in enumerate(questions[:3], 1):
        opts = q.get("options", [])
        print(f"\n  Q{i}: {q['question']}")
        for o in opts:
            print(f"     {o}")

    full_audit_data["questions"] = questions

    # ====== 审计：追问效率 ======
    print("\n── 审计Agent: 追问效率 ──")
    q_audit = auditor.audit_questions(hypo_prob, questions)
    print(f"  效率得分: {q_audit.get('efficiency_score', '?')}/10")
    if q_audit.get("killer_question"):
        print(f"  💡 遗漏关键问题: {q_audit['killer_question'][:60]}")
    for aq in q_audit.get("alternative_questions", [])[:2]:
        print(f"  + 建议追问: {aq[:60]}")

    full_audit_data["questions_audit"] = q_audit

    # ====== 模拟排查过程（假设用户在Q1选了A等情况）======
    print("\n── 主引擎 Step4: 模拟用户回答 → 概率更新 ──")
    # 用户选A = 有明显料层差异（支持H1传热不均）
    eliminated_h = []
    remaining = dict(hypo_prob)
    # 模拟：保留TOP2（案例推理），排除末位猜想
    if len(remaining) > 2:
        sorted_h = sorted(remaining.items(), key=lambda x: x[1])
        eliminate = sorted_h[0][0]
        remaining.pop(eliminate, None)
        eliminated_h.append(eliminate)
        total = sum(remaining.values())
        remaining = {k: v / total for k, v in remaining.items()}
        print(f"  排除: {eliminate[:40]}")
        print(f"  剩余假设概率更新:")
        for h, p in remaining.items():
            print(f"    {p:.0%} | {h[:50]}")

    full_audit_data["remaining_hypotheses"] = remaining
    full_audit_data["eliminated"] = eliminated_h

    # ====== 审计：概率逻辑 ======
    print("\n── 审计Agent: 概率逻辑 ──")
    prob_audit = auditor.audit_probability(hypo_prob, eliminated_h, remaining)
    print(f"  逻辑得分: {prob_audit.get('logic_score', '?')}/10")
    for issue in prob_audit.get("issues", []):
        print(f"  ⚠️ [{issue.get('type')}] {issue.get('description', '')[:60]}")

    # ====== 根因 + 方案 ======
    root_cause = max(remaining, key=remaining.get) if remaining else "需更多排查"
    print(f"\n── 🎯 锁定根因: {root_cause[:60]} ──")

    print("\n── 主引擎 Step5: 改善方案 ──")
    prompt_sys_s = config.load_prompt("solution")
    prompt_user_s = f"确认根因: {root_cause}\n生成标准改善方案卡片"
    sol_resp = model.chat_completion(prompt_sys_s, prompt_user_s, temperature=0.2, max_tokens=1024)
    if sol_resp and not sol_resp.startswith("{"):
        print(sol_resp)

    full_audit_data["root_cause"] = root_cause
    full_audit_data["solution"] = sol_resp

    # ====== 审计：方案质量 ======
    print("\n── 审计Agent: 方案质量 ──")
    sol_audit = auditor.audit_solution(root_cause, sol_resp)
    print(f"  针对性得分: {sol_audit.get('targeting_score', '?')}/10")
    if sol_audit.get("verification_metric"):
        print(f"  📏 验证指标: {sol_audit['verification_metric']}")
    for gap in sol_audit.get("missing_layers", [])[:2]:
        print(f"  ⚠️ 缺失层次: {gap}")

    # ====== 范式级审计 ======
    print("\n── 审计Agent: 范式审视 ──")
    paradigm_audit = auditor.audit_paradigm(full_audit_data)
    print(f"  范式得分: {paradigm_audit.get('paradigm_score', '?')}/10")
    for flaw in paradigm_audit.get("flow_flaws", [])[:2]:
        print(f"  ⚠️ [{flaw.get('stage')}] {flaw.get('flaw', '')[:50]} → {flaw.get('fix', '')[:40]}")
    for alt in paradigm_audit.get("paradigm_alternatives", [])[:2]:
        print(f"  💡 替代范式: {alt[:60]}")

    full_audit_data["paradigm_audit"] = paradigm_audit

    # ====== 总结报告 ======
    print("\n" + "=" * 65)
    print("  📊 审计总结")
    print("=" * 65)
    audit_summary = {
        "召回质量": rag_audit.get("recall_score", "?"),
        "假设覆盖": hypo_audit.get("coverage_score", "?"),
        "追问效率": q_audit.get("efficiency_score", "?"),
        "概率逻辑": prob_audit.get("logic_score", "?"),
        "方案质量": sol_audit.get("targeting_score", "?"),
        "范式健康": paradigm_audit.get("paradigm_score", "?"),
    }
    for k, v in audit_summary.items():
        print(f"  {k}: {v}/10")

    # 找出最大风险
    min_item = min(audit_summary.items(), key=lambda x: x[1] if isinstance(x[1], int) else 10)
    print(f"\n  ⚠️ 最弱环节: {min_item[0]}({min_item[1]}/10) — 建议优先改进")

    return full_audit_data


if __name__ == "__main__":
    test = ("竹基碳/PEG复合PCM：500ml小试真空浸渍包封率85%；"
            "放大至50L釜同配方同参数，包封率跌至58%，热循环50次底部渗出PEG液体")
    start_diagnosis_with_audit(test)
