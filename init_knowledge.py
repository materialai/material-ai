"""知识库初始化：灌入种子数据，构建一期可用的经验库和案例库"""

import json
import logging
import sys

import config
import database as db
from rag_engine import index_text_to_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ====================== 种子数据 ======================

SEED_RULES = [
    {
        "rule_id": "EXP-SEED-001",
        "rule_name": "原料梅雨季吸湿",
        "trigger_condition": "原料供应商在6-8月梅雨季节供货，且原料具有吸湿性（如无机盐、某些高分子单体）",
        "exp_content": "梅雨季节空气湿度高，原料在运输存放过程中易吸湿含水量增加2-5%，导致水解副反应增多，产物收率下降3-8个百分点，颜色可能偏黄",
        "source_info": "张工，碳基导热垫研发，2024年6月-8月三次偏差后总结",
    },
    {
        "rule_id": "EXP-SEED-002",
        "rule_name": "原料结块=有效成分降解",
        "trigger_condition": "原料外观出现结块、颜色不均匀，且存放时间超过30天",
        "exp_content": "原料结块说明已吸湿并发生部分团聚，有效单体含量下降5-15%，必须做水分检测和含量复核后方可投料。长期使用结块原料是收率持续偏低的重要原因",
        "source_info": "李工，预混料母粒生产线，2024年Q4",
    },
    {
        "rule_id": "EXP-SEED-003",
        "rule_name": "温度积分饱和",
        "trigger_condition": "聚合反应温度控制出现持续偏离设定值超过±3°C，且持续超过10分钟",
        "exp_content": "PID温控器的积分时间Ti设置过短会导致积分饱和，实际温度持续偏高或偏低。典型表现是产物分子量分布变宽、收率下降、批次间差异大。建议将积分时间从当前值增大30-50%",
        "source_info": "王工，反应釜温控系统调试，2025年3月",
    },
    {
        "rule_id": "EXP-SEED-004",
        "rule_name": "搅拌转速-分散不均",
        "trigger_condition": "高粘度反应体系（粘度>5000cP）搅拌转速低于临界分散转速",
        "exp_content": "当体系粘度超过5000cP而搅拌转速未相应提高时，会出现分散不均，局部浓度过高导致副产物增多、收率下降。应将转速提高20-40%或改用锚式搅拌桨",
        "source_info": "陈工，高分子聚合工艺放大，2025年1月",
    },
    {
        "rule_id": "EXP-SEED-005",
        "rule_name": "催化剂失活判定",
        "trigger_condition": "相同配方下收率连续3批下降超过5%，且原料已确认无问题",
        "exp_content": "很可能是催化剂活性失活。检查催化剂存储条件（是否需要氮气保护、是否过期）、添加时机（是否过早加入导致在反应完成前已失活）。建议做催化剂活性滴定确认有效含量",
        "source_info": "刘工，催化体系优化，2025年5月",
    },
]


SEED_FORMULAS = [
    {
        "formula_id": "FM-SEED-001",
        "formula_name": "碳基导热垫基础配方V3",
        "material_list": json.dumps({
            "组分A": "氧化铝微球 D50=20μm 供应商甲 100份",
            "组分B": "硅油 粘度1000cSt 供应商乙 30份",
            "组分C": "偶联剂KH-560 供应商丙 2份",
            "组分D": "催化剂铂金 供应商丁 0.1份",
        }, ensure_ascii=False),
        "process_params": json.dumps({
            "混合温度": "60°C",
            "混合转速": "200rpm",
            "混合时间": "60min",
            "固化温度": "150°C",
            "固化时间": "30min",
        }, ensure_ascii=False),
        "env_params": json.dumps({
            "车间温度": "25±3°C",
            "洁净度": "万级",
        }, ensure_ascii=False),
        "theory_expected": json.dumps({
            "导热系数": "≥3.0 W/mK",
            "硬度": "40±5 Shore OO",
            "收率": "≥95%",
        }, ensure_ascii=False),
        "source_tag": "manual",
    },
]


SEED_CASES = [
    {
        "case_id": "CASE-SEED-001",
        "problem_desc": "碳基导热垫收率连续3批下降8%，原料外观正常但存放超6周，环境为梅雨季",
        "hypothesis_list": json.dumps([
            {"hypo_id": "H1", "hypo_text": "原料吸湿导致有效成分含量下降", "init_prob": 0.40, "final_prob": 0.85, "status": "confirmed"},
            {"hypo_id": "H2", "hypo_text": "梅雨季环境湿度影响反应平衡", "init_prob": 0.25, "final_prob": 0.10, "status": "eliminated"},
            {"hypo_id": "H3", "hypo_text": "催化剂存储条件不当导致失活", "init_prob": 0.20, "final_prob": 0.05, "status": "eliminated"},
        ]),
        "root_cause": "原料吸湿导致有效成分含量下降（梅雨季+存放超6周双重因素）",
        "check_round": 2,
        "status": "closed",
    },
    {
        "case_id": "CASE-SEED-002",
        "problem_desc": "高分子聚合反应收率低于预期25%，反应过程温度波动±5°C",
        "hypothesis_list": json.dumps([
            {"hypo_id": "H1", "hypo_text": "PID温控器积分时间设置不当导致温度漂移", "init_prob": 0.50, "final_prob": 0.90, "status": "confirmed"},
            {"hypo_id": "H2", "hypo_text": "加热功率不足或加热带老化", "init_prob": 0.20, "final_prob": 0.05, "status": "eliminated"},
            {"hypo_id": "H3", "hypo_text": "反应热未及时移除导致局部过热", "init_prob": 0.30, "final_prob": 0.05, "status": "eliminated"},
        ]),
        "root_cause": "PID温控器积分时间Ti偏小导致积分饱和，需增大Ti参数30%",
        "check_round": 1,
        "status": "closed",
    },
]


# ====================== 初始化入口 ======================

def init_all():
    """执行全量知识库初始化"""
    print("=" * 60)
    print("  知识库种子数据灌入")
    print("=" * 60)

    # 1. 灌入经验规则
    print(f"\n[1/3] 灌入 {len(SEED_RULES)} 条经验规则")
    for rule in SEED_RULES:
        result = db.insert_experience_rule(**{k: v for k, v in rule.items()})
        # 索引到向量库
        index_text_to_vector(
            "rule",
            rule["rule_id"],
            f"{rule['trigger_condition']}\n{rule['exp_content']}"
        )
        print(f"  ✓ {rule['rule_id']} - {rule['rule_name']}")

    # 2. 灌入配方档案
    print(f"\n[2/3] 灌入 {len(SEED_FORMULAS)} 个配方")
    for formula in SEED_FORMULAS:
        db.insert_formula(**formula)
        # 配方也入向量库
        index_text_to_vector(
            "formula",
            formula["formula_id"],
            f"{formula['formula_name']} 材料: {formula['material_list']} 工艺: {formula['process_params']}"
        )
        print(f"  ✓ {formula['formula_id']} - {formula['formula_name']}")

    # 3. 灌入历史案例
    print(f"\n[3/3] 灌入 {len(SEED_CASES)} 个历史案例")
    for case in SEED_CASES:
        db.insert_case(**{k: v for k, v in case.items() if k != "status"})
        # 如果案例已关闭，补充root_cause
        if case.get("status") == "closed":
            db.update_case(case["case_id"], status="closed", root_cause=case.get("root_cause", ""))
        # 案例入向量库
        index_text_to_vector(
            "case",
            case["case_id"],
            f"{case['problem_desc']} 根因: {case.get('root_cause', '')}"
        )
        print(f"  ✓ {case['case_id']}")

    # 统计
    vec_count = db.vector_count()
    print(f"\n✅ 知识库初始化完成！当前向量库共 {vec_count} 条记录")


def reset_and_init():
    """清空后重新初始化（危险！）"""
    confirm = input("确认清空所有数据并重新初始化？输入'RESET'执行: ")
    if confirm != "RESET":
        print("已取消")
        return
    # TODO: 执行截断（此处省略，安全起见需手动确认）
    init_all()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset_and_init()
    else:
        init_all()
