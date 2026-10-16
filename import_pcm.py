"""PCM相变材料知识库导入脚本
读取 C:\\Users\\T480\\Downloads\\PCM相变材料知识库.xlsx，
将数据灌入5张业务表 + 向量索引表"""

import json
import re
import logging
import sys
from pathlib import Path

import pandas as pd

# 项目根目录加入路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import database as db
from rag_engine import index_text_to_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# xlsx路径
PCM_XLSX = Path(r"C:\Users\T480\Downloads\PCM相变材料知识库.xlsx")


def clean_json(val):
    """安全解析JSON字符串为空则返回{}"""
    if pd.isna(val) or not val:
        return {}
    try:
        return json.loads(val) if isinstance(val, str) else val
    except Exception:
        return {}


def extract_hypothesis_id(text):
    """从假设文本提取①②③编号"""
    m = re.match(r"[①②③④⑤]\s*", text)
    if m:
        return m.group(0).strip()
    return None


def parse_xlsx():
    """解析xlsx所有工作表为dict"""
    xls = pd.ExcelFile(PCM_XLSX)

    # --------- Sheet 1: 核心知识总览 -> 提取知识点当作文本向量 ---------
    df1 = pd.read_excel(xls, sheet_name="1-核心知识总览")

    # --------- Sheet 2: 排查案例库 -> case_library ---------
    df2 = pd.read_excel(xls, sheet_name="2-排查案例库", header=1)
    df2 = df2.dropna(subset=["案例ID"])

    # --------- Sheet 3: 经验规则库 -> experience_rule ---------
    df3 = pd.read_excel(xls, sheet_name="3-经验规则库", header=1)
    df3 = df3.dropna(subset=["规则ID"])

    # --------- Sheet 4: 改善方案库 -> improve_solution ---------
    df4 = pd.read_excel(xls, sheet_name="4-改善方案库", header=1)
    df4 = df4.dropna(subset=["方案ID"])

    return df1, df2, df3, df4


def import_rules(df):
    """导入经验规则库"""
    count = 0
    for _, row in df.iterrows():
        rule_id = str(row.get("规则ID", "")).strip()
        if not rule_id or rule_id == "nan":
            continue

        trigger = str(row.get("触发条件", ""))
        content = str(row.get("经验内容", ""))
        source = str(row.get("来源标注", ""))
        confidence = float(row.get("置信度(初值)", 0.5))

        # 写入数据库
        db.insert_experience_rule(
            rule_id=rule_id,
            trigger_condition=trigger,
            exp_content=content,
            source_info=source,
        )

        # 固定初始置信度
        with db.db_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE experience_rule SET confidence=%s WHERE rule_id=%s",
                (confidence, rule_id),
            )

        # 向量索引
        index_text_to_vector("rule", rule_id, f"{trigger}\n{content}")

        count += 1
        logger.info(f"  规则入库: {rule_id}")
    return count


def import_solutions(df):
    """导入改善方案库"""
    count = 0
    for _, row in df.iterrows():
        sid = str(row.get("方案ID", "")).strip()
        if not sid or sid == "nan":
            continue

        root_type = str(row.get("根因类型", ""))
        template = str(row.get("方案模板", ""))
        condition = str(row.get("适用前提", ""))
        expect = str(row.get("预期改善幅度", ""))
        risk = str(row.get("风险提示", ""))

        with db.db_cursor(commit=True) as cur:
            cur.execute("""
                INSERT INTO improve_solution (solution_id, root_cause_type, plan_short,
                    applicable_condition, expect_improve, risk_warning)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (solution_id) DO UPDATE SET
                    root_cause_type=EXCLUDED.root_cause_type,
                    plan_short=EXCLUDED.plan_short,
                    applicable_condition=EXCLUDED.applicable_condition,
                    expect_improve=EXCLUDED.expect_improve,
                    risk_warning=EXCLUDED.risk_warning
            """, (sid, root_type, template, condition, expect, risk))

        index_text_to_vector("solution", sid, f"{root_type}\n{template}")
        count += 1
        logger.info(f"  方案入库: {sid}")
    return count


def import_cases(df):
    """导入排查案例库"""
    count = 0
    for _, row in df.iterrows():
        cid = str(row.get("案例ID", "")).strip()
        if not cid or cid == "nan":
            continue

        problem = str(row.get("问题描述", ""))
        hypo_str = str(row.get("假设清单(AI)", ""))
        check_str = str(row.get("排查过程", ""))
        root = str(row.get("最终根因", ""))
        improve = str(row.get("改善方案", ""))
        result = str(row.get("改善效果", ""))
        round_str = str(row.get("排查轮次", "0"))
        # 提取轮次数字
        round_num = int(re.search(r"\d+", round_str).group()) if re.search(r"\d+", round_str) else 0

        # 解析假设清单
        hypo_list = []
        if hypo_str and hypo_str != "nan":
            # 分割①②③格式
            parts = re.split(r"(?=[①②③④⑤])", hypo_str)
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                # 提取文本和概率
                prob_m = re.search(r"\((\d+)%\)", part)
                prob = int(prob_m.group(1)) / 100 if prob_m else 1.0 / len(parts)
                # 去掉概率部分
                hypo_text = re.sub(r"\(\d+%\)", "", part).lstrip("①②③④⑤ ").strip()
                hypo_list.append({
                    "hypo_id": f"H{i+1}",
                    "hypo_text": hypo_text,
                    "init_prob": round(prob, 4),
                    "final_prob": round(prob, 4),
                    "status": "confirmed" if hypo_text in root else "eliminated",
                })

        # 假设根因对应的那个status改成confirmed
        for h in hypo_list:
            if root and h["hypo_text"][:10] in root:
                h["status"] = "confirmed"
                break

        check_process = [{
            "round": i + 1,
            "description": check_str[:100] if i == 0 else "",
        } for i in range(max(round_num, 1))]

        check_json = json.dumps(check_process, ensure_ascii=False)

        db.insert_case(
            case_id=cid,
            problem_desc=problem[:200],
            problem_original=problem,
            hypothesis_list=hypo_list,
        )

        db.update_case(
            case_id=cid,
            root_cause=root,
            check_round=round_num,
            status="closed",
            improve_result=result,
        )

        # 向量索引
        index_text_to_vector("case", cid, f"{problem} 根因: {root}")

        count += 1
        logger.info(f"  案例入库: {cid}")
    return count


def import_core_knowledge(df):
    """核心知识总览 -> 文本向量库（不写入配方表，而是作为纯知识片段）"""
    count = 0
    current_section = ""

    for _, row in df.iterrows():
        cells = [str(v) for v in row.values if pd.notna(v) and str(v).strip()]
        if not cells:
            continue

        # 合并多个单元格为一行文本
        line = " | ".join(cells)

        # 跳过纯表头行
        if line.startswith("指标") or line.startswith("PCM 相变"):
            continue

        # 检测章节标题
        if line.startswith("一、") or line.startswith("二、") or line.startswith("三、") or \
           line.startswith("四、") or line.startswith("五、") or line.startswith("六、") or \
           line.startswith("七、"):
            current_section = line.split("、")[1][:20] if "、" in line else line[:20]
            continue

        if not current_section:
            continue

        # 只导入有实质内容的行
        if len(line) < 10:
            continue

        kid = f"KNOW-PCM-{count+1:03d}"
        index_text_to_vector("knowledge", kid, f"{current_section}: {line}")
        count += 1

    return count


def run_full_import():
    """全量导入入口"""
    if not PCM_XLSX.exists():
        logger.error(f"知识库文件不存在: {PCM_XLSX}")
        logger.error("请确认文件路径或修改 PCM_XLSX 变量")
        return

    logger.info("=" * 60)
    logger.info("  PCM 相变材料知识库 数据导入")
    logger.info(f"  数据源: {PCM_XLSX}")
    logger.info("=" * 60)

    try:
        df1, df2, df3, df4 = parse_xlsx()
    except Exception as e:
        logger.error(f"xlsx读取失败: {e}")
        return

    logger.info(f"\n解析结果: 核心知识{len(df1)}行 / 案例{len(df2)}条 / 规则{len(df3)}条 / 方案{len(df4)}条")

    # 按序导入
    logger.info("\n--- 1. 经验规则库 ---")
    n_rules = import_rules(df3)

    logger.info("\n--- 2. 改善方案库 ---")
    n_solutions = import_solutions(df4)

    logger.info("\n--- 3. 排查案例库 ---")
    n_cases = import_cases(df2)

    logger.info("\n--- 4. 核心知识总览 ---")
    n_knowledge = import_core_knowledge(df1)

    # 最终统计
    vec_count = db.vector_count()

    summary = f"""
导入完成！
  经验规则: {n_rules} 条
  改善方案: {n_solutions} 套
  排查案例: {n_cases} 个
  核心知识: {n_knowledge} 条
  向量总数: {vec_count} 条
"""
    logger.info(summary)
    print(summary)


if __name__ == "__main__":
    run_full_import()
