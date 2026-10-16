"""核心排查引擎：串联RAG→大模型→贝叶斯→追问→方案生成的完整业务流程"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

import config
import database as db
from doubao_api import get_doubao
from rag_engine import retrieve, build_full_context, RAGContext, index_text_to_vector
from bayesian_engine import calculate_prior, update_confidence

logger = logging.getLogger(__name__)


def _gen_id(prefix: str) -> str:
    """生成业务ID: prefix-年月日-4位随机"""
    date_str = datetime.now().strftime("%Y%m%d")
    rand = uuid.uuid4().hex[:6]
    return f"{prefix}-{date_str}-{rand}"


# ============================================================
# 阶段1：偏差排查主入口
# ============================================================

def start_diagnosis(problem_text: str, formula_id: str = "",
                    deviation_magnitude: float = 0.0) -> dict:
    """
    偏差排查完整流程入口

    输入：
    - problem_text: 偏差自然语言描述
    - formula_id: 关联配方ID（可选）
    - deviation_magnitude: 偏差幅度

    输出：包含假设列表、置信度、首轮追问的完整排查会话
    """
    case_id = _gen_id("CASE")
    logger.info(f"===== 启动排查: {case_id} =====")
    logger.info(f"问题: {problem_text[:80]}")

    # Step1: RAG检索知识库
    rag, full_context = build_full_context(problem_text)
    retrieved_rules = rag.get_rules()

    # Step2: 调用豆包API生成假设（只输出文本，不含概率）
    doubao = get_doubao()
    hypo_raw = doubao.chat_completion(
        system_prompt=config.load_prompt("hypothesis"),
        user_content=full_context,
        temperature=0.1,
    )

    # Step3: 解析大模型输出
    try:
        hypo_json = json.loads(hypo_raw.strip().strip("```").strip())
        hypothesis_list = hypo_json.get("hypothesis_list", [])
    except Exception as e:
        logger.error(f"假设生成失败: {e}\n原始返回: {hypo_raw[:300]}")
        return {
            "case_id": case_id,
            "status": "error",
            "message": "假设生成失败，请重试或转人工"
        }

    # 截断到最多5条
    hypothesis_list = hypothesis_list[:config.MAX_HYPOTHESIS]

    # Step4: 后端贝叶斯计算概率
    hypo_prob = calculate_prior(hypothesis_list, retrieved_rules)

    # Step5: 持久化案例
    hypo_records = []
    for i, (hypo, prob) in enumerate(hypo_prob.items()):
        hypo_records.append({
            "hypo_id": f"H{i+1}",
            "hypo_text": hypo,
            "init_prob": round(prob, 4),
            "final_prob": round(prob, 4),
            "status": "pending",  # pending / suspected / confirmed / eliminated
        })

    db.insert_case(
        case_id=case_id,
        problem_desc=problem_text[:200],
        problem_original=problem_text,
        hypothesis_list=hypo_records,
    )

    # Step6: 生成首轮追问
    question_result = _generate_questions(case_id, problem_text, hypo_prob)

    return {
        "case_id": case_id,
        "status": "active",
        "hypothesis_with_prob": hypo_prob,
        "hypothesis_records": hypo_records,
        "rag_retrieved_count": len(rag.items),
        "question_info": question_result,
    }


# ============================================================
# 阶段2：动态追问引擎
# ============================================================

def _generate_questions(case_id: str, problem_text: str,
                        hypo_prob: dict) -> dict:
    """生成追问问题"""
    doubao = get_doubao()

    # 构建追问上下文
    hypo_text = "\n".join(
        f"  [{i+1}] {h} (置信度:{p:.1%})"
        for i, (h, p) in enumerate(hypo_prob.items())
    )

    user_input = (
        f"问题描述: {problem_text}\n"
        f"当前假设列表(按置信度排序):\n{hypo_text}\n"
        f"已收集线索: 暂无（首轮追问）"
    )

    raw = doubao.chat_completion(
        system_prompt=config.load_prompt("question"),
        user_content=user_input,
        temperature=0.1,
    )

    try:
        result = json.loads(raw.strip().strip("```").strip())
        return result
    except Exception as e:
        logger.error(f"追问生成失败: {e}")
        return {"root_cause_confirmed": False, "question_list": []}


def submit_answer(case_id: str, answers: list) -> dict:
    """
    提交本轮追问的答案，进入下一轮或锁定根因

    answers格式:
    [
      {"question_idx": 0, "selected_option": "A"},
      {"question_idx": 1, "selected_option": "B"},
    ]
    """
    case = db.get_case_by_id(case_id)
    if not case:
        return {"status": "error", "message": f"案例{case_id}不存在"}

    # 更新排查过程记录
    check_process = case.get("check_process") or []
    if isinstance(check_process, str):
        check_process = json.loads(check_process)

    round_num = len(check_process) + 1
    round_record = {
        "round": round_num,
        "answers": answers,
        "timestamp": datetime.now().isoformat(),
    }
    check_process.append(round_record)

    # 基于答案更新假设状态
    hypothesis_list = case.get("hypothesis_list") or []
    if isinstance(hypothesis_list, str):
        hypothesis_list = json.loads(hypothesis_list)

    # 分析答案，尝试eliminate或suspect某些假设
    eliminate_list, suspect_list = _analyze_answers(answers, hypothesis_list)

    round_record["eliminate_list"] = eliminate_list
    round_record["suspect_list"] = suspect_list

    # 更新假设状态
    for h in hypothesis_list:
        if h["hypo_text"] in eliminate_list:
            h["status"] = "eliminated"
            h["final_prob"] = 0.0
        elif h["hypo_text"] in suspect_list:
            h["status"] = "suspected"

    # 重新归一化剩余假设的概率
    active_hypo = {h["hypo_text"]: h["final_prob"]
                   for h in hypothesis_list if h["status"] != "eliminated"}
    if active_hypo:
        total = sum(active_hypo.values())
        if total > 0:
            for h in hypothesis_list:
                if h["status"] != "eliminated":
                    h["final_prob"] = round(h["final_prob"] / total, 4)

    # 判断是否可以锁定根因
    confirmed = _check_root_cause_confirmed(hypothesis_list)

    if confirmed:
        # 锁定根因
        root_cause_text = confirmed["hypo_text"]
        db.update_case(
            case_id=case_id,
            check_process=check_process,
            hypothesis_list=hypothesis_list,
            root_cause=root_cause_text,
            check_round=round_num,
            status="closed",
        )
        # 根因关联的经验规则 → 置信度提升
        _reward_associated_rules(case_id, root_cause_text, hypothesis_list)

        # 生成改善方案卡片
        solution = generate_solution_card(case_id, root_cause_text)
        return {
            "case_id": case_id,
            "status": "root_cause_confirmed",
            "root_cause": root_cause_text,
            "solution": solution,
            "total_rounds": round_num,
        }

    # 检查熔断条件
    if round_num >= config.MAX_ROUNDS:
        db.update_case(
            case_id=case_id,
            check_process=check_process,
            hypothesis_list=hypothesis_list,
            status="expert_referral",
        )
        return {
            "case_id": case_id,
            "status": "expert_referral",
            "message": f"已达最大追问轮次({config.MAX_ROUNDS}轮)，请专家介入",
            "hypothesis_list": hypothesis_list,
        }

    # 继续下一轮追问
    next_hypo_prob = {
        h["hypo_text"]: h["final_prob"]
        for h in hypothesis_list if h["status"] != "eliminated"
    }
    q_result = _generate_questions(
        case_id,
        case.get("problem_desc", ""),
        next_hypo_prob,
    )

    # 持久化
    db.update_case(
        case_id=case_id,
        check_process=check_process,
        hypothesis_list=hypothesis_list,
        check_round=round_num,
    )

    return {
        "case_id": case_id,
        "status": "continue",
        "round": round_num,
        "eliminated": eliminate_list,
        "remaining_hypothesis": next_hypo_prob,
        "question_info": q_result,
    }


def _analyze_answers(answers: list, hypothesis_list: list) -> tuple:
    """
    分析答案，决定哪些假设被排除/怀疑
    简化实现：基于规则硬匹配
    """
    eliminate_list = []
    suspect_list = []

    for ans in answers:
        opt = ans.get("selected_option", "")
        if opt == "D" or "排除" in opt or "不是" in opt:
            # 排除对应的最高概率假设
            for h in hypothesis_list:
                if h["status"] == "pending":
                    eliminate_list.append(h["hypo_text"])
                    break
        elif opt in ("A", "B", "C") and "未注意" not in opt:
            # 可能是确认方向
            for h in hypothesis_list:
                if h["status"] == "pending":
                    suspect_list.append(h["hypo_text"])
                    break

    return eliminate_list, suspect_list


def _check_root_cause_confirmed(hypothesis_list: list) -> Optional[dict]:
    """
    检查是否可以确认根因
    条件：某假设被标记为confirmed，或只剩1个活跃假设且概率>0.6
    """
    active = [h for h in hypothesis_list if h["status"] != "eliminated"]
    if len(active) == 1 and active[0]["final_prob"] > 0.5:
        return active[0]

    confirmed = [h for h in hypothesis_list if h["status"] == "confirmed"]
    if confirmed:
        return confirmed[0]

    return None


def _reward_associated_rules(case_id: str, root_cause_text: str, hypothesis_list: list):
    """根因确认后，给关联的经验规则提升置信度"""
    # 通过语义相似度匹配关联规则
    rag = retrieve(root_cause_text)
    for rule_item in rag.get_rules():
        rule_id = rule_item["source_id"]
        if rule_item.get("distance", 1.0) < 0.3:  # 高相似度
            update_confidence(rule_id, is_hit=True, case_id=case_id)


# ============================================================
# 阶段3：改善方案生成
# ============================================================

def generate_solution_card(case_id: str, root_cause: str) -> str:
    """根因确认后，生成改善方案卡片"""
    doubao = get_doubao()

    # 检索历史相似改善方案
    rag = retrieve(root_cause)
    ref_solutions = rag.get_solutions()
    ref_text = "\n".join(f"- {s['content']}" for s in ref_solutions) if ref_solutions else "暂无相似历史方案"

    solution_id = _gen_id("IM")
    user_input = (
        f"确认根因: {root_cause}\n"
        f"关联案例: {case_id}\n"
        f"参考历史改善方案:\n{ref_text}\n"
        f"方案卡片ID: {solution_id}"
    )

    card = doubao.chat_completion(
        system_prompt=config.load_prompt("solution"),
        user_content=user_input,
        temperature=0.2,
    )

    # 异步索引方案到向量库
    try:
        index_text_to_vector("solution", solution_id, card)
    except Exception as e:
        logger.warning(f"方案向量索引失败: {e}")

    return card


# ============================================================
# 阶段4：知识录入
# ============================================================

def ingest_experience_rule(raw_text: str, source_info: str = "") -> dict:
    """
    一点课知识录入入口
    1. 调用大模型从自然语言抽取结构化规则
    2. 写入经验规则表
    3. 为规则生成向量索引
    """
    doubao = get_doubao()
    raw_json = doubao.chat_completion(
        system_prompt=config.load_prompt("extract"),
        user_content=raw_text,
        temperature=0.05,
    )

    try:
        data = json.loads(raw_json.strip().strip("```").strip())
    except Exception as e:
        logger.error(f"经验抽取失败: {e}")
        return {"status": "error", "message": "经验抽取失败"}

    rule_id = _gen_id("EXP")
    trigger_condition = data.get("trigger_condition", raw_text[:100])
    exp_content = data.get("exp_content", raw_text[:200])

    # 写入业务表
    db.insert_experience_rule(
        rule_id=rule_id,
        trigger_condition=trigger_condition,
        exp_content=exp_content,
        source_info=source_info,
        rule_name=data.get("source_suggestion", ""),
    )

    # 同步索引到向量库
    index_text_to_vector("rule", rule_id, f"{trigger_condition}\n{exp_content}")

    return {
        "status": "success",
        "rule_id": rule_id,
        "trigger_condition": trigger_condition,
        "exp_content": exp_content,
        "initial_confidence": config.CONF_INIT,
    }


if __name__ == "__main__":
    # 端到端测试
    print("=" * 60)
    print("  新材料工艺偏差排查AI系统 - 端到端测试")
    print("=" * 60)

    # 测试1：录入一条经验
    print("\n[1] 录入经验规则")
    result = ingest_experience_rule(
        "供应商甲的原料在梅雨季节生产的那批，做出来的产物颜色会发黄，可能是原料吸水了，含水量偏高",
        source_info="张工 车间经验 2024-06"
    )
    print(f"  -> {result}")

    # 测试2：启动排查
    print("\n[2] 启动偏差排查")
    result = start_diagnosis(
        "高分子聚合反应，收率低于预期30%，原料外观颜色偏深，有结块，存放超过一个月",
        deviation_magnitude=30.0,
    )
    print(f"  case_id: {result.get('case_id')}")
    print(f"  假设数量: {len(result.get('hypothesis_with_prob', {}))}")
    for h, p in result.get("hypothesis_with_prob", {}).items():
        print(f"    [{p:.1%}] {h}")
    print(f"  追问: {result.get('question_info', {})}")
