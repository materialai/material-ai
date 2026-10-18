"""工艺偏差诊断决策引擎 — 配方vs工艺优先级决策 + 多轮追问引导

这个模块让 AI 从"被动回答"变成"主动诊断"：
1. 根据已收集信息动态决定下一步追问什么
2. 评估当前信息充分度，足够时输出决策
3. 内置配方vs工艺优先级的决策矩阵
4. 生成给豆包的增强 prompt（带推理链指令）

设计原则：AI主导对话，工程师只需如实回答，不需要懂排查方法论。
"""

from __future__ import annotations

import json
import time
import logging
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class DiagnosticProfile:
    """一轮诊断会话的结构化画像（逐步填充）"""
    # Phase 1 收集
    product_name: str = ""           # 产品名称
    formula_name: str = ""           # 配方名称/编号
    test_phase: str = ""             # 测试环节：试产 / 量产 / 客户反馈 / 型式试验
    failure_mode: str = ""           # 失效模式关键词
    batch_scope: str = ""            # 影响范围：单缸/连续多缸/批次性
    is_self_limiting: bool = False   # 自限性（后续恢复正常）
    
    # Phase 2 排查
    batch_changed: bool = False      # 近期换过原料批次
    env_changed: bool = False        # 环境变化（温湿度）
    new_formula: bool = False        # 是否新配方试产
    
    # Phase 3 深度（按需追问）
    temp_curve_ok: Optional[bool] = None     # 温控曲线是否异常
    mix_speed_ok: Optional[bool] = None     # 搅拌转速是否异常
    pressure_ok: Optional[bool] = None      # 真空/压力是否异常
    material_lot: str = ""                  # 原料批次号
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def completeness(self) -> float:
        """信息完整度 0.0~1.0"""
        core_fields = [
            self.product_name, self.formula_name, self.test_phase,
            self.failure_mode, self.batch_scope
        ]
        filled = sum(1 for f in core_fields if f)
        return filled / len(core_fields)
    
    def is_decision_ready(self) -> bool:
        """是否足够做出配方vs工艺优先级判断"""
        return all([
            self.product_name, self.failure_mode,
            self.test_phase, self.batch_scope
        ])


@dataclass
class DecisionResult:
    """最终决策输出"""
    primary_direction: str = ""       # "formula" or "process"
    confidence: float = 0.0           # 0.0~1.0
    formula_score: int = 0
    process_score: int = 0
    reasoning: str = ""               # 推理链
    hypothesis_list: list = field(default_factory=list)
    action_checklist: list = field(default_factory=list)
    risk_warning: str = ""            # 风险提示
    similar_cases: list = field(default_factory=list)


# ============================================================
# 决策矩阵（配方 vs 工艺优先级）
# ============================================================

# 规则格式: (条件回调函数, formula加分, process加分, 规则说明)
# 当条件满足时，给对应方向加分。最终比较总分决定优先级。

DECISION_RULES = [
    # ── 产品信息类 ──
    ("new_formula",       30,  0,  "新配方试产阶段 → 配方本身未定型"),
    ("pilot_phase",       25,  0,  "中试阶段 → 配方需确认"),
    ("batch_material_changed", 20,  5,  "近期换原料批次 → 配方适配性问题"),
    
    # ── 失效模式类 ──
    ("single_self_limiting",  5, 35,  "单缸偶发+自限 → 工艺波动概率高"),
    ("continuous_multi",     40,  5,  "连续/多缸出现 → 系统性问题，配方缺陷"),
    ("batch_wide",           35, 10,  "批次性出现 → 配方问题"),
    ("gradual_decay",         5, 30,  "渐进衰减 → 工艺漂移/设备磨损"),
    
    # ── 失效模式指向 ──
    ("failure_color",        15, 10,  "颜色异常 → 配方配比或颜料分散"),
    ("failure_viscosity",    20, 15,  "粘度异常 → 可能配比或温度"),
    ("failure_cure",         25, 10,  "固化问题 → 配方或固化条件"),
    ("failure_mechanical",   10, 20,  "力学性能 → 工艺条件或配方"),
    ("failure_appearance",   10, 20,  "外观缺陷 → 工艺参数或配方"),
]


def evaluate_rules(profile: DiagnosticProfile) -> tuple[int, int, list[str]]:
    """
    对诊断画像应用决策矩阵。
    返回: (formula总分, process总分, 命中的规则说明列表)
    """
    f_score = 0
    p_score = 0
    fired_rules: list[str] = []
    
    # 条件判断
    conds = {}
    
    # 产品信息
    conds["new_formula"] = profile.new_formula
    conds["pilot_phase"] = profile.test_phase in ("pilot", "中试", "试产", "小批")
    conds["batch_material_changed"] = profile.batch_changed
    
    # 失效特征
    conds["single_self_limiting"] = (
        profile.batch_scope in ("single", "单缸", "单批") and profile.is_self_limiting
    )
    conds["continuous_multi"] = profile.batch_scope in ("continuous", "连续", "multi", "多缸多批")
    conds["batch_wide"] = profile.batch_scope in ("batch", "批次", "batch_wide")
    conds["gradual_decay"] = "渐进" in profile.failure_mode or "衰减" in profile.failure_mode or "逐渐" in profile.failure_mode
    
    # 失效模式指向
    fm = profile.failure_mode.lower()
    conds["failure_color"] = any(k in fm for k in ["色", "色差", "发黄", "发黑", "变色"])
    conds["failure_viscosity"] = any(k in fm for k in ["粘度", "稠", "稀", "流动性"])
    conds["failure_cure"] = any(k in fm for k in ["固化", "干燥", "硫化", "交联", "凝胶", "结晶"])
    conds["failure_mechanical"] = any(k in fm for k in ["强度", "硬度", "拉伸", "剥离", "冲击", "伸长"])
    conds["failure_appearance"] = any(k in fm for k in ["气泡", "针眼", "颗粒", "杂质", "划痕", "流平"])
    
    # 应用规则
    for rule_name, f_add, p_add, rule_desc in DECISION_RULES:
        if conds.get(rule_name, False):
            f_score += f_add
            p_score += p_add
            fired_rules.append(f"[+] {rule_desc} (F+{f_add}/P+{p_add})")
    
    return f_score, p_score, fired_rules


def decide_priority(profile: DiagnosticProfile) -> DecisionResult:
    """
    综合决策：配方优先 or 工艺优先
    这是核心决策函数，输出给移动端展示。
    """
    f, p, rules = evaluate_rules(profile)
    
    total = f + p
    if total == 0:
        # 无规则命中 → 不确定
        confidence = 0.0
        primary = "undetermined"
    else:
        if f > p:
            primary = "formula"
            confidence = (f - p) / total
        elif p > f:
            primary = "process"
            confidence = (p - f) / total
        else:
            primary = "both"
            confidence = 0.5
    
    # 推理链
    reasoning = _build_reasoning(profile, f, p, rules)
    
    # 假设列表
    hypotheses = _build_hypotheses(profile, primary)
    
    # 行动清单
    checklist = _build_checklist(profile, primary)
    
    # 风险提示
    risk = _build_risk_warning(profile, primary, confidence)
    
    return DecisionResult(
        primary_direction=primary,
        confidence=confidence,
        formula_score=f,
        process_score=p,
        reasoning=reasoning,
        hypothesis_list=hypotheses,
        action_checklist=checklist,
        risk_warning=risk,
    )


def _build_reasoning(profile: DiagnosticProfile, f: int, p: int, rules: list[str]) -> str:
    """构建推理链文本"""
    lines = []
    lines.append(f"产品信息：{profile.product_name or '未知'} / {profile.formula_name or '未知'}")
    lines.append(f"测试环节：{profile.test_phase or '未知'}")
    lines.append(f"失效现象：{profile.failure_mode or '未知'}")
    lines.append(f"影响范围：{profile.batch_scope or '未知'}（{'自限' if profile.is_self_limiting else '非自限'}）")
    lines.append("")
    lines.append(f"【配方调整得分: {f}  vs  工艺调整得分: {p}】")
    lines.append("")
    if rules:
        lines.append("命中规则：")
        for r in rules:
            lines.append(f"  {r}")
    return "\n".join(lines)


def _build_hypotheses(profile: DiagnosticProfile, primary: str) -> list[str]:
    """基于决策方向生成假设列表"""
    hypotheses = []
    fm = profile.failure_mode
    
    if primary == "formula":
        hypotheses.append(f"① 配方体系中可能某种组分比例不当（与{fm}相关）")
        if profile.batch_changed:
            hypotheses.append(f"② 更换原料批次后，原料批次间差异导致配方偏离最优窗口")
        if not profile.is_self_limiting:
            hypotheses.append(f"③ 配方核心组分配比需要重新校准")
        hypotheses.append(f"④ 可能缺少某种功能性助剂（如分散剂、稳定剂）")
    elif primary == "process":
        hypotheses.append(f"① 工艺参数（温度/时间/搅拌/真空）偏离工艺窗口")
        hypotheses.append(f"② 设备执行精度下降或传感器漂移")
        if profile.is_self_limiting:
            hypotheses.append(f"③ 瞬时工艺波动（如加料顺序/温度过冲）已恢复，根因仍存在")
        hypotheses.append(f"④ 环境监测（温湿度/洁净度）超出控制范围")
    else:
        hypotheses.append(f"① 配方或工艺均有可能，需进一步验证")
    
    return hypotheses


def _build_checklist(profile: DiagnosticProfile, primary: str) -> list[str]:
    """生成排查行动清单（按优先级排序）"""
    checklist = []
    
    if primary == "formula":
        checklist.extend([
            "【立即确认】对比当前配方与上次成功批次配方的差异（特别是关键组分用量）",
            "【原料追溯】查该批次原料的批次号、供应商、关键指标检测报告",
            "【小试验证】用相同原料做50mL~200mL小试，重现问题",
            "【配方微调】如确认配比问题，先做±5%微调实验，验证后再上线",
        ])
        if profile.batch_changed:
            checklist.insert(1, "【原料差异】将新旧批次原料的关键指标（固含量/羟值/粘度等）做对比检测")
    elif primary == "process":
        checklist.extend([
            "【立即确认】调取该批次工艺参数全记录（温控曲线/搅拌/真空/压力/速度）",
            "【窗口对比】将当前参数与工艺卡窗口值对比，找出偏移项",
            "【趋势分析】拉取前10批次同参数趋势，判断是突发异常还是渐变漂移",
            "【设备点检】对相关温控/搅拌/传动部件做运行精度验证",
            "【验证实验】在工艺窗口边界做DOE，定位最优窗口中心",
        ])
        if profile.is_self_limiting:
            checklist.insert(0, "【特征确认】查该缸次前后30min的工艺参数波动曲线，定位异常时间点")
    else:
        checklist.extend([
            "【并行排查】同时进行配方对比和工艺参数审核",
            "【先易后难】优先排查工艺（成本低、见效快），如无效再回溯配方",
        ])
    
    return checklist


def _build_risk_warning(profile: DiagnosticProfile, primary: str, confidence: float) -> str:
    """风险提示"""
    warnings = []
    
    if confidence < 0.2:
        warnings.append("⚠️ 置信度偏低：建议补充更多信息后再做决策")
    
    if primary == "formula" and profile.test_phase in ("量产", "mass"):
        warnings.append("⚠️ 量产阶段调整配方风险较高：需确认原料批次 + 小试验证 + 留样观察")
    
    if primary == "process" and profile.is_self_limiting:
        warnings.append("⚠️ 自限性故障的根因可能未被根除：建议排查后留1~3批次强化监测")
    
    if profile.batch_scope in ("continuous", "连续", "multi", "多缸多批"):
        warnings.append("🔴 批次性异常：建议立即隔离问题批次并通知品质部门")
    
    return "\n".join(warnings) if warnings else "✅ 常规风险可控，按计划执行"


# ============================================================
# 追问引擎：决定下一步问什么
# ============================================================

@dataclass
class DiagnosticSession:
    """管理一次诊断会话的状态和追问逻辑"""
    session_id: str = ""
    profile: DiagnosticProfile = field(default_factory=DiagnosticProfile)
    question_history: list[dict] = field(default_factory=list)
    current_phase: int = 1
    turn_count: int = 0
    decision_result: Optional[DecisionResult] = None
    created_at: float = 0.0
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"DX{int(time.time())}"
        if not self.created_at:
            self.created_at = time.time()
        self._pending_field = ""  # 当前追问对应的字段名
        self._asked_self_limiting = False
    
    def generate_next_question(self) -> Optional[str]:
        """
        根据当前画像信息完整度，决定下一个追问。
        返回 None 表示信息已充分、可以输出决策。
        """
        p = self.profile
        questions = []
        
        # Phase 1: 必须回答的核心信息（0→100%）
        if not p.product_name:
            questions.append({
                "q": "请告诉我是哪个产品（产品名称或编号）？",
                "field": "product_name",
                "phase": 1,
            })
        if not p.formula_name:
            questions.append({
                "q": "这次是哪个配方在用？（可填名称、编号或大致类别，如'A胶-标准型'）",
                "field": "formula_name",
                "phase": 1,
            })
        if not p.test_phase:
            questions.append({
                "q": "目前产品处于哪个阶段？\n1-新配方试产 / 2-中试放大 / 3-量产 / 4-客户反馈",
                "field": "test_phase",
                "phase": 1,
            })
        if not p.failure_mode:
            questions.append({
                "q": "遇到的偏差现象是什么？（一句话描述，如'结晶'、'粘度升高'、'固化不良'）",
                "field": "failure_mode",
                "phase": 1,
            })
        if not p.batch_scope:
            questions.append({
                "q": "问题影响范围是？\n1-仅单缸/单批 / 2-连续多缸 / 3-整批物料 / 4-批次间突变",
                "field": "batch_scope",
                "phase": 1,
            })
        if not p.batch_scope:
            pass  # scope not yet answered
        elif not hasattr(self, '_asked_self_limiting') or not self._asked_self_limiting:
            self._asked_self_limiting = True
            questions.append({
                "q": "后续批次是否还有同样问题？\n1-后续恢复正常 / 2-后续仍出现 / 3-不确认",
                "field": "is_self_limiting",
                "phase": 1,
            })
        
        if questions:
            q = questions[0]
            self.current_phase = q["phase"]
            self._pending_field = q["field"]
            return self._format_question(q)
        
        # Phase 2: 辅助诊断信息（提高置信度）
        self._pending_field = ""
        if not p.batch_changed:
            questions.append({
                "q": "最近有没有换过原料批次或供应商？\n1-没有 / 2-换了批次 / 3-换了供应商 / 4-不确认",
                "field": "batch_changed",
                "phase": 2,
            })
        if not p.env_changed and self.turn_count >= 2:
            questions.append({
                "q": "车间环境（温度/湿度）近期是否有明显变化？\n1-正常 / 2-有变化 / 3-不确认",
                "field": "env_changed",
                "phase": 2,
            })
        if not p.new_formula:
            questions.append({
                "q": "这个配方是否属于新开发或首次试产？\n1-是，新配方 / 2-不是，量产配方 / 3-经常规微调",
                "field": "new_formula",
                "phase": 2,
            })
        
        if questions and self.turn_count < 5:  # 最多3个辅助问题
            q = questions[0]
            self.current_phase = 2
            self._pending_field = q["field"]
            return self._format_question(q)
        
        # 信息充分 → 可以决策
        self.current_phase = 3
        self._pending_field = ""
        return None
    
    def _format_question(self, q: dict) -> str:
        """格式化追问，附带进度提示"""
        progress = ""
        if q["phase"] == 1:
            filled = sum(1 for v in [
                self.profile.product_name, self.profile.formula_name,
                self.profile.test_phase, self.profile.failure_mode,
                self.profile.batch_scope
            ] if v)
            progress = f"\n\n📋 基本信息 ({filled}/5)"
        elif q["phase"] == 2:
            progress = "\n🔍 深度追问中..."
        
        return f"【第{self.turn_count + 1}问】{q['q']}{progress}"
    
    def record_answer(self, field: str, answer: str):
        """记录用户回答，更新画像"""
        self.turn_count += 1
        self.question_history.append({"field": field, "answer": answer})
        
        # 将回答映射到画像字段
        if field == "product_name":
            self.profile.product_name = answer.strip()
        elif field == "formula_name":
            self.profile.formula_name = answer.strip()
        elif field == "test_phase":
            self.profile.test_phase = self._normalize_phase(answer)
        elif field == "failure_mode":
            self.profile.failure_mode = answer.strip()
        elif field == "batch_scope":
            self.profile.batch_scope = self._normalize_batch(answer)
        elif field == "is_self_limiting":
            self.profile.is_self_limiting = answer.strip() in ("1", "后续恢复正常", "是", "yes")
        elif field == "batch_changed":
            # "换了批次"或"换了供应商" → True
            self.profile.batch_changed = any(k in answer for k in ["2", "换了"])
        elif field == "env_changed":
            self.profile.env_changed = "2" in answer or "有变化" in answer
        elif field == "new_formula":
            self.profile.new_formula = "1" in answer or "新配方" in answer
    
    @staticmethod
    def _normalize_phase(text: str) -> str:
        """标准化测试环节"""
        t = text.strip()
        if t in ("1", "新配方试产", "试产", "小批"):
            return "pilot"
        if t in ("2", "中试放大", "中试"):
            return "pilot"
        if t in ("3", "量产"):
            return "mass"
        if t in ("4", "客户反馈", "客诉"):
            return "customer"
        return t
    
    @staticmethod
    def _normalize_batch(text: str) -> str:
        """标准化影响范围"""
        t = text.strip()
        if t in ("1", "单缸", "单批"):
            return "single"
        if t in ("2", "连续多缸", "连续"):
            return "continuous"
        if t in ("3", "整批物料", "批次"):
            return "batch"
        if t in ("4", "批次间突变", "突发"):
            return "batch"
        return t
    
    def finalize(self) -> DecisionResult:
        """信息收集完毕，输出最终决策"""
        result = decide_priority(self.profile)
        self.decision_result = result
        return result


# ============================================================
# Prompt 构造：给豆包的增强提示
# ============================================================

def build_diagnosis_prompt(profile: DiagnosticProfile, retrieved_context: list[dict]) -> tuple[str, str]:
    """
    构建诊断 prompt。
    返回 (system_prompt, user_message)
    
    核心思路：把人设+规则+检索到的知识+画像全部注入给AI，
    让AI按照我们规定的推理框架输出结构化结论。
    """
    system = """你是「材料工艺诊断大师」，一位拥有15年经验的复合型材料工程师。

【你的职责】
主导工艺偏差诊断的全流程：分析现象→建立假设→给出优先级决策→输出行动清单。

【诊断方法论 - 必须遵循】
1. **现象分类**：先准确定性失效模式（结晶/凝胶/相分离/变色/发泡/机械性能异常...）
2. **根因分层**：区分配方缺陷、工艺参数偏移、设备故障、环境干扰、原料批次差异
3. **优先级决策矩阵**：
   - 新配方/试产阶段 → 优先调配方
   - 量产+连续出现 → 优先调配方
   - 单缸偶发自限 → 优先查工艺
   - 渐进衰减 → 优先查工艺/设备
4. **排除法**：每条假设必须标注可执行的验证动作
5. **风险管控**：配方调整需小试验证，工艺调整需工艺窗口确认

【输出格式 - 严格按此结构】
你会收到"诊断画像"和"知识库检索结果"，输出必须包含以下6个段落：

一、现象定性与关键特征
二、可能根因（最多4条，按概率排序）
三、决策结论（配方优先 / 工艺优先 / 参照历史）
四、核查清单（按优先级列出3~5个立即动作）
五、验证方案（明确怎么做小试）
六、风险提示

【语言风格】
简洁、专业、可操作。避免推测性语句，每个结论必须说明依据。
"""
    
    # 构造用户消息（画像 + 检索结果）
    user_lines = []
    user_lines.append("═" * 40)
    user_lines.append("【诊断画像 - 已确认信息】")
    user_lines.append("═" * 40)
    
    p = profile
    fields = [
        ("产品名称", p.product_name),
        ("配方", p.formula_name),
        ("测试环节", p.test_phase),
        ("失效模式", p.failure_mode),
        ("影响范围", p.batch_scope),
        ("自限性", "是" if p.is_self_limiting else "否"),
        ("原料批次变更", "是" if p.batch_changed else "否"),
        ("环境变化", "是" if p.env_changed else "否"),
        ("新配方", "是" if p.new_formula else "否"),
    ]
    
    for label, val in fields:
        if val:
            user_lines.append(f"  · {label}：{val}")
    
    if retrieved_context:
        user_lines.append("")
        user_lines.append("═" * 40)
        user_lines.append("【知识库检索 - 相关参考】")
        user_lines.append("═" * 40)
        for i, ctx in enumerate(retrieved_context[:5], 1):
            user_lines.append(f"\n  [{i}] 来源：{ctx.get('source_type', '知识库')} | 相似度：{ctx.get('similarity', 0):.2f}")
            content = ctx.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            user_lines.append(f"      {content}")
    
    user_lines.append("")
    user_lines.append("═" * 40)
    user_lines.append("请根据以上信息，按照规定的6段式格式输出诊断报告。")
    
    return system, "\n".join(user_lines)


# ============================================================
# 便捷入口
# ============================================================

def create_session() -> DiagnosticSession:
    """创建新的诊断会话"""
    return DiagnosticSession()


def run_full_diagnosis(product: str, formula: str, phase: str,
                       failure_mode: str, scope: str, self_limiting: bool,
                       batch_changed: bool, env_changed: bool, new_formula: bool,
                       retrieved: list[dict] | None = None) -> tuple[str, DecisionResult]:
    """
    一次性诊断（用于API直接传入信息的场景）。
    返回 (llm诊断报告, 决策结果)
    """
    p = DiagnosticProfile(
        product_name=product,
        formula_name=formula,
        test_phase=phase,
        failure_mode=failure_mode,
        batch_scope=scope,
        is_self_limiting=self_limiting,
        batch_changed=batch_changed,
        env_changed=env_changed,
        new_formula=new_formula,
    )
    result = decide_priority(p)
    sys_prompt, user_msg = build_diagnosis_prompt(p, retrieved or [])
    return user_msg, result
