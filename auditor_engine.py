"""
Audit Agent：独立于诊断流程的审计引擎
职责：复核主引擎每一步输出，发现漏洞、盲区、范式缺陷
方法：用独立的提示词审计召回质量 / 假设覆盖度 / 追问效率 / 概率逻辑 / 方案有效性
"""

import json
import logging
import re
from pathlib import Path

import config
from model_manager import get_model

logger = logging.getLogger(__name__)


# ====================== 审计提示词 ====================

AUDIT_RAG_PROMPT = """你是知识检索质量审计专家。
输入：用户的原始问题 + RAG 系统召回的知识片段列表。
任务：评估召回质量，找出遗漏和冗余。

请输出 JSON {{
  "recall_score": 0-10 分,
  "relevance": "高/中/低",
  "issues": [
    {{
      "type": "missing/irrelevant/noise",
      "description": "具体问题描述",
      "suggested_keywords": ["建议补充的检索关键词"]
    }}
  ],
  "missing_kinds": ["应该出现但没出现的知识类别，如历史案例/工艺参数/经验规则等"],
  "diagnosis": "总体评估一句话"
}}
禁止除 JSON 外的任何文字。
"""

AUDIT_HYPOTHESIS_PROMPT = """你是工艺诊断逻辑审计专家。
输入：用户问题 + 知识库召回片段 + AI 生成的假设列表。
任务：评估假设的覆盖度、排他性和逻辑严谨性。

请输出 JSON {{
  "coverage_score": 0-10 分,
  "blind_spots": [
    {{
      "dimension": "工艺/设备/原料/环境/检测/人员",
      "possible_cause": "未覆盖的可能原因",
      "why_missing": "为什么AI可能遗漏了这条"
    }}
  ],
  "classification": {{
    "total_hypothesis": 5,
    "case_based": 0,
    "knowledge_based": 0,
    "pure_speculation": 0,
    "rebalance_suggestion": "建议的比例调整"
  }},
  "overlooked_dimensions": ["AI未考虑的分析维度，如热力学/动力学/经济成本等"],
  "diagnosis": "假设集合的整体完整性和逻辑性评价"
}}
禁止除 JSON 外的任何文字。
"""

AUDIT_QUESTIONS_PROMPT = """你是工艺流程追问审计专家。
输入：当前假设列表 + AI 生成的追问问题。
任务：评估追问的信息增益效率和排查顺序。

请输出 JSON {{
  "efficiency_score": 0-10 分,
  "information_gain": "高/中/低",
  "issues": [
    {{
      "type": "redundant/ambiguous/unverifiable",
      "description": "问题描述"
    }}
  ],
  "ordering": "当前追问顺序是否合理？如何调整？",
  "killer_question": "当前最有诊断价值但被遗漏的关键问题是什么？",
  "alternative_questions": ["建议补充的追问"],
  "diagnosis": "追问策略总体评价"
}}
禁止除 JSON 外的任何文字。
"""

AUDIT_PROBABILITY_PROMPT = """你是概率审计专家。
输入：假设列表 + 各假设的置信度 + 已收集的证据链。
任务：评估贝叶斯概率更新逻辑是否合理。

请输出 JSON {{
  "logic_score": 0-10 分,
  "issues": [
    {{
      "type": "prior_bias/evidence_miscount/update_error",
      "description": "具体问题",
      "suggested_correction": "建议修正"
    }}
  ],
  "confidence_distribution": "当前分布是否合理？有没有过度集中或过度分散？",
  "diagnosis": "概率更新机制整体评价"
}}
禁止除 JSON 外的任何文字。
"""

AUDIT_SOLUTION_PROMPT = """你是工艺改善方案审计专家。
输入：确认的根因 + AI 生成的改善方案卡。
任务：评估方案的针对性、可行性和完整性。

请输出 JSON {{
  "targeting_score": 0-10 分,
  "feasibility": "高/中/低",
  "issues": [
    {{
      "type": "off_target/unmeasurable/missing_layer",
      "description": "具体问题"
    }}
  ],
  "missing_layers": ["缺失的措施层次，如检测/监控/培训/备选方案"],
  "risk_gaps": ["未覆盖的潜在风险"],
  "verification_metric": "建议用哪个量化指标验证改善效果？",
  "diagnosis": "改善方案整体评价"
}}
禁止除 JSON 外的任何文字。
"""

AUDIT_PARADIGM_PROMPT = """你是系统范式审计专家。
输入：完整诊断流程的用户问题→召回→假设→追问→根因→方案全记录。
任务：从更高维度审视整体流程范式本身。

请输出 JSON {{
  "paradigm_score": 0-10 分,
  "flow_flaws": [
    {{
      "stage": "流程阶段名",
      "flaw": "流程缺陷描述",
      "impact": "高/中/低",
      "fix": "具体修复建议"
    }}
  ],
  "paradigm_bottlenecks": ["当前范式的瓶颈，如过度依赖文本知识/缺少量化推理/未闭环验证等"],
  "paradigm_alternatives": ["可选的替代范式，如量化建模/自动实验设计/根因数据库等"],
  "fundamental_question": "当前诊断逻辑的根本前提假设是什么？这个假设是否总成立？",
  "diagnosis": "整体范式级别的改进建议"
}}
禁止除 JSON 外的任何文字。
"""


# ====================== 审计执行器 ====================

class AuditorAgent:
    """独立审计Agent，检查诊断流程的每一步"""

    def __init__(self):
        self.model = get_model()

    def _extract_json(self, text: str) -> dict:
        """鲁棒JSON提取：处理markdown包裹、嵌套括号、尾逗号、截断等"""
        # Step 1: Strip markdown code blocks
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        # Step 2: Find the first '{' and use bracket matching to find its partner
        start = text.find("{")
        if start < 0:
            return {"audit_error": "无JSON输出", "raw": text[:300]}

        depth = 0
        end = -1
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and (i == 0 or text[i-1] != '\\'):
                # Toggle string mode (simple heuristic)
                in_string = not in_string
                continue
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break

        if end < 0:
            # Try to find the last '}' as fallback
            end = text.rfind("}")
            if end <= start:
                return {"audit_error": "JSON结构截断", "raw": text[:300]}

        json_str = text[start:end + 1]

        # Step 3: Fix trailing commas before } or ]
        # Fix ,] -> ] and ,} -> }
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        # Step 4: Parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing single-quoted strings -> double-quoted
            try:
                fixed = json_str.replace("'", '"')
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
            # Last resort: return error with the fixed string for manual inspection
            return {"audit_error": "JSON解析失败", "raw": json_str[:300]}

    def _audit(self, system_prompt: str, user_input: str) -> dict:
        """通用审计调用"""
        try:
            raw = self.model.chat_completion(system_prompt, user_input, temperature=0.1, max_tokens=1500)
            return self._extract_json(raw)
        except Exception as e:
            logger.error(f"审计调用失败: {e}")
            return {"audit_error": f"调用异常: {e}"}

    # ---------- 各步骤审计 ----------

    def audit_rag(self, problem: str, retrieved: list) -> dict:
        """审计召回质量"""
        rag_ctx = "\n".join(f"- [{r['source_type']}] {r['content'][:100]}" for r in retrieved[:5])
        user_input = f"问题: {problem}\n\n召回片段:\n{rag_ctx}"
        return self._audit(AUDIT_RAG_PROMPT, user_input)

    def audit_hypotheses(self, problem: str, retrieved: list, hypotheses: dict) -> dict:
        """审计假设覆盖度"""
        rag_ctx = "\n".join(f"- {r['content'][:80]}" for r in retrieved[:3])
        hypo_ctx = "\n".join(f"- {h} ({p:.0%})" for h, p in hypotheses.items())
        user_input = f"问题: {problem}\n\n知识召回:\n{rag_ctx}\n\n生成假设:\n{hypo_ctx}"
        return self._audit(AUDIT_HYPOTHESIS_PROMPT, user_input)

    def audit_questions(self, hypotheses: dict, questions: list) -> dict:
        """审计追问效率"""
        hypo_ctx = "\n".join(f"- {h}" for h in hypotheses.keys())
        q_ctx = "\n".join(f"Q{i+1}: {q['question']}" for i, q in enumerate(questions))
        user_input = f"假设:\n{hypo_ctx}\n\n追问:\n{q_ctx}"
        return self._audit(AUDIT_QUESTIONS_PROMPT, user_input)

    def audit_probability(self, hypotheses: dict, eliminated: list, remaining: dict) -> dict:
        """审计概率逻辑"""
        status = "\n".join(f"- {h}: {p:.0%} ({'已排除' if h in eliminated else '活跃'})"
                          for h, p in {**hypotheses, **remaining}.items())
        user_input = f"假设状态:\n{status}"
        return self._audit(AUDIT_PROBABILITY_PROMPT, user_input)

    def audit_solution(self, root_cause: str, solution_card: str) -> dict:
        """审计改善方案"""
        user_input = f"根因: {root_cause}\n\n改善方案:\n{solution_card[:500]}"
        return self._audit(AUDIT_SOLUTION_PROMPT, user_input)

    def audit_paradigm(self, full_record: dict) -> dict:
        """审计整体范式"""
        record_str = json.dumps(full_record, ensure_ascii=False, indent=2, default=str)
        return self._audit(AUDIT_PARADIGM_PROMPT, record_str)

    # ---------- 一键全量审计 ----------

    def full_audit(self, audit_data: dict) -> dict:
        """对完整诊断流程执行全量审计"""
        print("\n" + "=" * 65)
        print("  🔍 审计 Agent — 全流程独立审查")
        print("=" * 65)

        results = {}

        # Audit 1: RAG 召回
        if "retrieved" in audit_data:
            print("\n📋 审计1: RAG召回质量 → ", end="", flush=True)
            r = self.audit_rag(audit_data.get("problem", ""), audit_data["retrieved"])
            results["rag"] = r
            score = r.get("recall_score", "?")
            diag = r.get("diagnosis", "")[:50]
            print(f"得分 {score}/10 | {diag}")

        # Audit 2: 假设覆盖度
        if "hypotheses" in audit_data and "retrieved" in audit_data:
            print("🧠 审计2: 假设覆盖度 → ", end="", flush=True)
            r = self.audit_hypotheses(
                audit_data.get("problem", ""),
                audit_data["retrieved"],
                audit_data["hypotheses"]
            )
            results["hypotheses"] = r
            score = r.get("coverage_score", "?")
            blind_count = len(r.get("blind_spots", []))
            print(f"得分 {score}/10 | 盲区 {blind_count} 个")

        # Audit 3: 追问效率
        if "questions" in audit_data and "hypotheses" in audit_data:
            print("❓ 审计3: 追问效率   → ", end="", flush=True)
            r = self.audit_questions(audit_data["hypotheses"], audit_data["questions"])
            results["questions"] = r
            score = r.get("efficiency_score", "?")
            killer = r.get("killer_question", "")[:40]
            print(f"得分 {score}/10 | 遗漏关键问题: {killer or '无'}")

        # Audit 4: 概率逻辑
        if "hypotheses" in audit_data:
            print("📊 审计4: 概率逻辑   → ", end="", flush=True)
            r = self.audit_probability(
                audit_data.get("hypotheses", {}),
                audit_data.get("eliminated", []),
                audit_data.get("remaining_hypotheses", {})
            )
            results["probability"] = r
            score = r.get("logic_score", "?")
            print(f"得分 {score}/10")

        # Audit 5: 方案质量
        if "root_cause" in audit_data and "solution" in audit_data:
            print("📋 审计5: 方案质量   → ", end="", flush=True)
            r = self.audit_solution(audit_data["root_cause"], audit_data["solution"])
            results["solution"] = r
            score = r.get("targeting_score", "?")
            verify = r.get("verification_metric", "")[:40]
            print(f"得分 {score}/10 | 验证指标: {verify or '未指定'}")

        # Audit 6: 范式级
        print("🏗️ 审计6: 范式审视   → ", end="", flush=True)
        r = self.audit_paradigm(audit_data)
        results["paradigm"] = r
        score = r.get("paradigm_score", "?")
        flaws = len(r.get("flow_flaws", []))
        print(f"得分 {score}/10 | 流程缺陷 {flaws} 个")

        return results


# ====================== 便捷入口 ======================

_auditor = None


def get_auditor() -> AuditorAgent:
    global _auditor
    if _auditor is None:
        _auditor = AuditorAgent()
    return _auditor


if __name__ == "__main__":
    # 单机测试
    a = AuditorAgent()
    test_data = {
        "problem": "竹基碳/PEG复合PCM浸渍放大包封率暴跌",
        "retrieved": [
            {"source_type": "case", "content": "CASE-PCM-001包封率下降"},
        ],
        "hypotheses": {
            "真空度不足": 0.3,
            "传热不均": 0.25,
            "原料问题": 0.15,
        },
        "questions": [
            {"question": "真空表校验了吗？"},
            {"question": "温度曲线正常吗？"},
        ],
        "root_cause": "放大釜真空不足+传热差",
        "solution": "A.分段抽真空 B.增加真空泵 C.降低料层厚度",
    }
    a.full_audit(test_data)
