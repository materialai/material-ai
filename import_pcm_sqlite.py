"""PCM相变材料知识库导入 → SQLite"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

# 项目根目录
sys.path.insert(0, str(Path(__file__).parent))

import db_sqlite as db

PCM_XLSX = Path(r"C:\Users\T480\Downloads\PCM相变材料知识库.xlsx")


def parse_xlsx():
    xls = pd.ExcelFile(PCM_XLSX)
    df_cases = pd.read_excel(xls, sheet_name="2-排查案例库", header=1).dropna(subset=["案例ID"])
    df_rules = pd.read_excel(xls, sheet_name="3-经验规则库", header=1).dropna(subset=["规则ID"])
    df_sols = pd.read_excel(xls, sheet_name="4-改善方案库", header=1).dropna(subset=["方案ID"])
    df_know = pd.read_excel(xls, sheet_name="1-核心知识总览")
    return df_cases, df_rules, df_sols, df_know


def import_all():
    if not PCM_XLSX.exists():
        print(f"❌ 知识库文件不存在: {PCM_XLSX}")
        return

    db.init_db()
    df_cases, df_rules, df_sols, df_know = parse_xlsx()

    print("=" * 60)
    print("  PCM相变材料知识库 → SQLite 导入")
    print("=" * 60)

    # 1. 经验规则
    print(f"\n[1/4] 经验规则 ({len(df_rules)} 条)")
    for _, row in df_rules.iterrows():
        rid = str(row["规则ID"]).strip()
        db.insert_rule(
            rule_id=rid,
            trigger=str(row.get("触发条件", "")),
            content=str(row.get("经验内容", "")),
            source=str(row.get("来源标注", "")),
            confidence=float(row.get("置信度(初值)", 0.5)),
        )
        db.index_text("rule", rid, f"{row.get('触发条件', '')} {row.get('经验内容', '')}")
        print(f"  ✓ {rid}")

    # 2. 改善方案
    print(f"\n[2/4] 改善方案 ({len(df_sols)} 套)")
    for _, row in df_sols.iterrows():
        sid = str(row["方案ID"]).strip()
        db.insert_solution(
            sid=sid,
            root_type=str(row.get("根因类型", "")),
            plan=str(row.get("方案模板", "")),
        )
        all_text = f"{row.get('根因类型', '')} {row.get('方案模板', '')} {row.get('适用前提', '')} {row.get('风险提示', '')}"
        db.index_text("solution", sid, all_text)
        print(f"  ✓ {sid}")

    # 3. 排查案例
    print(f"\n[3/4] 排查案例 ({len(df_cases)} 个)")
    for _, row in df_cases.iterrows():
        cid = str(row["案例ID"]).strip()
        root = str(row.get("最终根因", ""))
        check_round = str(row.get("排查轮次", "0"))
        round_m = re.search(r"\d+", check_round)
        rnum = int(round_m.group()) if round_m else 0

        db.insert_case(
            case_id=cid,
            problem=str(row.get("问题描述", "")),
            root_cause=root,
            check_round=rnum,
        )
        all_text = f"{row.get('问题描述', '')} {root} {row.get('改善方案', '')}"
        db.index_text("case", cid, all_text)
        print(f"  ✓ {cid}")

    # 4. 核心知识
    print(f"\n[4/4] 核心工艺知识 ({len(df_know)} 行)")
    count = 0
    current_section = ""

    for _, row in df_know.iterrows():
        cells = [str(v) for v in row.values if pd.notna(v) and str(v).strip()]
        if not cells:
            continue
        line = " | ".join(cells)
        if line.startswith("PCM 相变") or line.startswith("指标"):
            continue
        m = re.match(r"([一二三四五六七])、(.+)", line)
        if m:
            current_section = m.group(2)[:20]
            continue
        if current_section and len(line) > 10:
            db.index_text("knowledge", f"KNOW-{count+1:03d}", f"{current_section}: {line}")
            count += 1
    print(f"  ✓ 有效知识条目: {count}")

    stats = db.get_stats()
    print(f"\n{'=' * 60}")
    print(f"  ✅ 导入完成")
    print(f"  规则: {stats['rules']} | 方案: {stats['solutions']} | 案例: {stats['cases']} | 向量: {stats['vectors']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import_all()
