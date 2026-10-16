"""演示版排查引擎——完整业务流程，零外部API依赖
使用SQLite + 本地Embedding + 规则模板驱动
生产环境替换为 diagnosis_engine.py 即可"""

import json
import logging
import random
import re
import uuid
from datetime import datetime

import config
import db_sqlite as db
from model_manager import get_model
from bayesian_engine import calculate_prior as bayes_calc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _safe_extract_json(text: str) -> dict:
    """从AI输出中安全提取JSON（容错）"""
    try:
        return json.loads(text)
    except Exception:
        pass
    # 尝试去除 markdown 代码块
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])
    # 尝试找到第一个 { 到最后一个 }
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass
    return {}


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"


def start_diagnosis(problem_text: str, **kw) -> dict:
    """完整排查流程"""
    print("\n" + "=" * 65)
    print("  PCM 相变材料 — 偏差排查系统")
    print("=" * 65)
    print(f"\n📋 问题描述: {problem_text}")

    # --- Step 1: RAG召回 ---
    print("\n🔍 Step1: 知识库检索...")
    retrieved = db.retrieve(problem_text, top_k=5)
    rules = [r for r in retrieved if r["source_type"] == "rule"]
    cases = [r for r in retrieved if r["source_type"] == "case"]
    knowledge = [r for r in retrieved if r["source_type"] == "knowledge"]

    print(f"  召回: 规则{len(rules)}条 / 案例{len(cases)}个 / 通用知识{len(knowledge)}条")
    for i, r in enumerate(retrieved, 1):
        print(f"    [{i}] {r['source_type']}/{r['source_id']} 相似度:{r['similarity']:.3f} {r['content'][:60]}...")

    # --- Step 2: AI头脑风暴（真实豆包API）---
    print("\n🧠 Step2: 豆包AI头脑风暴（doubao-seed-2-1-lite）...")
    model = get_model()
    prompt_sys = config.load_prompt("hypothesis")
    # 知识片段精简：Top2，每条≤60字（保快）
    know_text = "\n".join(f"- {r['content'][:60]}" for r in retrieved[:2])
    prompt_user = f"偏差描述：{problem_text}\n相关知识库片段：\n{know_text}"
    hypo_resp = model.chat_completion(prompt_sys, prompt_user, temperature=0.2)
    hypo_json = _safe_extract_json(hypo_resp)
    hypo_list_raw = hypo_json.get("hypothesis_list", [])

    # 从召回结果中补强假设（如果LLM生成数量不够）
    hypo_from_cases = []
    for c in cases:
        root = db.get_case_by_id(c["source_id"])
        if root and root.get("root_cause"):
            hypo_from_cases.append(f"[案例{c['source_id']}] {root['root_cause']}")

    # 合并去重，最多5条
    hypo_list = list(dict.fromkeys(hypo_from_cases + hypo_list_raw))[:config.MAX_HYPOTHESIS]

    # --- Step 3: 贝叶斯计算概率 ---
    print("\n📊 Step3: 贝叶斯置信度计算...")
    # 把SQLite规则转为bayesian_engine需要的格式
    bayes_rules = [{"content": r["content"], "confidence": 0.5} for r in rules]
    hypo_prob = bayes_calc(hypo_list, bayes_rules)

    print("\n  编号 | 置信度 | 假设")
    print("  " + "-" * 55)
    for i, (h, p) in enumerate(hypo_prob.items(), 1):
        bar = "█" * int(p * 30)
        print(f"  H{i}   | {p:5.1%} | {bar} {h[:40]}...")

    # --- Step 4: 生成追问（真实豆包API）---
    print("\n❓ Step4: 生成追问清单...")
    prompt_sys_q = config.load_prompt("question")
    prompt_user_q = f"问题：{problem_text}\n当前假设列表(按置信度排序):\n" + "\n".join(f"  {i+1}. {h} ({p:.0%})" for i, (h, p) in enumerate(hypo_prob.items()))
    q_resp = model.chat_completion(prompt_sys_q, prompt_user_q, temperature=0.1)
    q_info = _safe_extract_json(q_resp)

    questions = q_info.get("question_list", [])
    for i, q in enumerate(questions, 1):
        print(f"\n  Q{i}: {q['question']}")
        for opt in q["options"]:
            print(f"     {opt}")

    # --- Step 5: 持久化案例 ---
    case_id = _gen_id("CASE")
    db.insert_case(case_id, problem_text)
    db.update_case(case_id, check_round=0, status="open")

    return {
        "case_id": case_id,
        "hypothesis_with_prob": hypo_prob,
        "questions": questions,
        "retrieved": retrieved,
    }


def run_diagnosis_interactive():
    """交互式排查: 自动模拟用户回答"""
    problem = input("\n📝 输入偏差描述 (回车用默认): ").strip()
    if not problem:
        problem = "50L真空浸渍釜生产竹基PEG复合PCM，包封率从85%降至58%，热循环50次底部渗出PEG"
        print(f"[默认] {problem}")

    result = start_diagnosis(problem)

    print("\n" + "=" * 65)
    print("  🔬 模拟排查回答（演示模式）")
    print("=" * 65)

    # 模拟"用户"回答第一个问题的第一个选项（积极线索）
    hypo_prob = result["hypothesis_with_prob"]
    questions = result["questions"]

    round_num = 0
    while questions and round_num < config.MAX_ROUNDS:
        round_num += 1
        print(f"\n--- 第{round_num}轮追问 ---")
        # 模拟用户总是选A（意思是"有问题"）
        answers = []
        for q in questions:
            ans = "A"  # 模拟选A=有问题
            answers.append({"selected": ans, "question": q["question"]})
            print(f"  ✓ {q['question'][:30]}... → 选A")

        # 排除概率最低的假设
        if hypo_prob:
            sorted_h = sorted(hypo_prob.items(), key=lambda x: x[1])
            if len(sorted_h) > 1:
                eliminated = sorted_h[0][0]
                elim_prob = sorted_h[0][1]
                print(f"\n  ❌ 排除: {eliminated[:40]}... (置信度{elim_prob:.1%})")
                hypo_prob.pop(eliminated)
                # 重新归一化
                total = sum(hypo_prob.values())
                if total > 0:
                    hypo_prob = {k: v / total for k, v in hypo_prob.items()}

        # 检查是否锁定
        if hypo_prob and max(hypo_prob.values()) > 0.55:
            break

        # 生成下一轮追问（演示版直接熔断）
        questions = []  # 简化：只演示1-2轮

    # 锁定根因
    if hypo_prob:
        root_cause = max(hypo_prob, key=hypo_prob.get)
        root_prob = hypo_prob[root_cause]

        print("\n" + "=" * 65)
        print(f"  🎯 锁定根因 (第{round_num}轮)")
        print(f"  {root_cause}")
        print(f"  置信度: {root_prob:.1%}")
        print("=" * 65)

        # 生成改善方案（真实豆包API）
        print("\n📋 Step5: 生成改善方案卡片（豆包AI）...")
        model = get_model()
        prompt_sys_s = config.load_prompt("solution")
        prompt_user_s = f"确认根因: {root_cause}\n关联案例: {case_id}\n请生成标准三层改善方案卡片"
        sol_resp = model.chat_completion(prompt_sys_s, prompt_user_s, temperature=0.2, max_tokens=1024)

        # 输出豆包生成的方案卡片
        if sol_resp and not sol_resp.startswith("{"):
            print("\n" + sol_resp)
        else:
            # 从知识库匹配改善方案兜底
            solutions = db.retrieve(root_cause, top_k=2)
            real_solutions = [s for s in solutions if s["source_type"] == "solution"]
            if real_solutions:
                print("\n  📎 匹配知识库改善方案:")
                for s in real_solutions:
                    print(f"    - {s['content'][:80]}...")
            else:
                print(f"\n  📎 通用改善:")
                print(f"    A.【短期】排查工艺参数记录/SOP符合性")
                print(f"    B.【中期】更新SOP，建立原料-工艺联动规则")
                print(f"    C.【长期】建立数据驱动的参数自适应系统")
    else:
        print("  ⚠️ 未锁定根因，建议转专家介入")


if __name__ == "__main__":
    run_diagnosis_interactive()
