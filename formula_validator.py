"""
配方硬校验闸门引擎（Hard VETO System）
═══════════════════════════════════════════════════
通用LLM配方幻觉的结构性防线——5道硬性校验闸门。
每条规则带"HARD VETO"执行效力，命中即拦截。
参考架构：guards-and-laws / Agent Assembler AutoCraft 四级校验。

设计理念：LLM负责发散和解释，规则引擎负责否决和拦截。
═══════════════════════════════════════════════════════════
"""

from __future__ import annotations
import re
import json
import logging
import importlib
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 第五道阀门：L0-L7框外根因逃逸口
# ═══════════════════════════════════════════════════

def gate_out_of_framework_escape(hypotheses: list[str]) -> list[VetoViolation]:
    """
    确保假设列表包含框外根因的兜底项。
    L0-L7框架无法覆盖所有根因 — 必须有一个逃逸口让工程师知道
    "有些根因不在AI分析框架内"。
    """
    violations = []

    # 检查假设列表中是否有"框架外"/"其他"/"未知"关键词
    has_escape = any(
        any(kw in h for kw in ["其他根因", "框架外", "L0-L7之外", "未覆盖", "超出范围", "需人工"])
        for h in hypotheses
    )

    if not has_escape:
        violations.append(VetoViolation(
            gate="框外根因",
            severity="WARNING",
            rule_id="ESCAPE-001",
            message="假设列表未包含L0-L7框架外的兜底根因。AI分析范围有限，建议工程师额外考虑：原料供应商工艺变更/设备交叉污染/质检取样偏差等非典型根因",
        ))

    return violations


# ═══════════════════════════════════════════════════
# 综合入口：诊断模式的全量校验
# ═══════════════════════════════════════════════════

def validate_diagnosis_comprehensive(
    profile: DiagnosticProfile,
    llm_output: str,
    hypotheses: list[str] | None = None,
    rag_results: list[dict] | None = None,
) -> dict:
    """
    诊断模式全量校验入口：
    1. 配方结构闸门（原有5道）
    2. 工艺安全窗口闸门（新增#1）
    3. 证据充分性闸门（新增#2）
    4. 产品体系匹配闸门（新增#3）
    5. 失败教训红线闸门（新增#4）
    6. 框外根因逃逸口（新增#5）

    返回：{
        "channel": "rejected" | "candidate" | "executable" | "abstain",
        "executable": bool,
        "violations": [{...}, ...],
        "engineer_text": str,
        "hypotheses": list[str],    # 可能追加框外兜底假设
        "abstain": bool,            # 是否触发信息不足弃权
        "abstain_reason": str,      # 弃权原因
    }
    """
    violations = []

    # ── 闸门1+2+4: 从LLM文本解析校验 ──
    violations.extend(gate_process_safety_window(profile, llm_output))
    violations.extend(gate_failure_lesson_intercept(profile, llm_output))

    # ── 阀门3: 产品体系匹配 ──
    if rag_results:
        violations.extend(gate_product_scope_match(profile, rag_results))

    # ── 闸门5: 框外根因逃逸口 ──
    if hypotheses is None:
        hypotheses = []
    violations.extend(gate_out_of_framework_escape(hypotheses))

    # ── 闸门2: 证据充分性（最强约束 — 可触发abstain）──
    evidence_violations = gate_evidence_sufficiency(profile, rag_results or [])

    # 判断是否有HARD_VETO级别的证据不足
    hard_evidence_veto = any(v.severity == "HARD_VETO" for v in evidence_violations)
    violations.extend(evidence_violations)

    # ── 配方结构闸门（如果有配方建议）──
    # DiagnosticProfile 没有 process_temp，用 env_temp 或默认25°C
    _proc_temp = getattr(profile, 'env_temp', None) or 25
    _shear = getattr(profile, 'shear_rate', None) or 0
    recipe = parse_formula_from_text(llm_output, default_temp=_proc_temp,
                                      default_shear=_shear)
    has_formula = len(recipe.components) >= 2
    if has_formula:
        formula_report = validate_formula(recipe)
        violations.extend(formula_report.violations)

    # ── 决策逻辑 ──
    hard_vetos = [v for v in violations if v.severity == "HARD_VETO"]

    # 最高优先级：证据严重不足 → 弃权（abstain）
    if hard_evidence_veto and len(rag_results or []) < 1:
        return {
            "channel": "abstain",
            "executable": False,
            "violations": [v.__dict__ for v in violations],
            "engineer_text": "⛔ 信息严重不足，AI拒绝输出建议。\n" +
                "\n".join(f"  • {v.message}" for v in violations if v.severity == "HARD_VETO"),
            "abstain": True,
            "abstain_reason": "证据不足以支撑任何可靠建议。请补充失效模式、产品名、测试阶段等关键信息后再试。",
            "hypotheses": hypotheses,
        }

    if hard_vetos:
        channel = "rejected"
    elif [v for v in violations if v.severity == "WARNING"]:
        channel = "candidate"
    else:
        channel = "executable"

    # 工程师视角的汇总报告
    summary_lines = []
    if channel == "rejected":
        summary_lines.append("🚫 硬校验闸门 — 建议已被拦截")
    elif channel == "candidate":
        summary_lines.append("⚠️ 硬校验闸门 — 候选方案（需人工确认）")
    elif channel == "abstain":
        summary_lines.append("⛔ 信息不足，无法给出建议")
    else:
        summary_lines.append("✅ 硬校验闸门 — 通过（可执行）")

    if violations:
        summary_lines.append("")
        for v in violations:
            icon = "🚫" if v.severity == "HARD_VETO" else "⚠️"
            summary_lines.append(f"{icon} [{v.rule_id}] {v.message}")

    return {
        "channel": channel,
        "executable": channel == "executable",
        "violations": [{"gate": v.gate, "severity": v.severity, "rule_id": v.rule_id,
                        "message": v.message, "evidence": v.evidence} for v in violations],
        "engineer_text": "\n".join(summary_lines),
        "hypotheses": hypotheses,
        "abstain": False,
        "abstain_reason": "",
    }


# ═══════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════

@dataclass
class FormulaComponent:
    """配方中的一个组分"""
    name: str          # 组分名称
    role: str          # 角色：树脂/固化剂/填料/稀释剂/助剂/PCM/偶联剂/其他
    phr: float = 0.0   # 份数（per hundred resin，相对于100份树脂）
    wt_pct: float = 0.0  # 质量百分比
    density: float = 1.0  # 密度（g/cm³），用于体积分数计算
    source: str = ""   # 证据来源（知识库ID或实验ID）
    eew: float = 0.0   # 环氧当量（树脂用）
    ahew: float = 0.0  # 活泼氢当量（固化剂用）


@dataclass
class FormulaRecipe:
    """一个完整配方（开发输出或诊断调整建议）"""
    components: list[FormulaComponent] = field(default_factory=list)
    product_name: str = ""
    process_temp: float = 25.0   # 工艺温度°C
    process_shear_rate: float = 0.0  # 剪切速率 1/s
    notes: str = ""

    def get_by_role(self, role: str) -> list[FormulaComponent]:
        return [c for c in self.components if c.role == role]

    def total_phr(self) -> float:
        return sum(c.phr for c in self.components)

    def resin_mass(self) -> float:
        """树脂总质量（基准=100phr）"""
        resins = self.get_by_role("树脂")
        return sum(r.phr for r in resins) if resins else 100.0


@dataclass
class VetoViolation:
    """一条校验违规"""
    gate: str          # 闸门名称
    severity: str      # "HARD_VETO" / "WARNING" / "INFO"
    rule_id: str       # 规则编号
    message: str       # 人类可读的违规说明
    evidence: str = ""  # 证据来源


@dataclass
class ValidationReport:
    """配方校验完整报告"""
    recipe: FormulaRecipe
    violations: list[VetoViolation] = field(default_factory=list)
    passed: bool = False
    executable: bool = False
    channel: str = "rejected"   # "rejected"/"candidate"/"executable"
    audit_log: list[str] = field(default_factory=list)

    @property
    def hard_vetos(self) -> list[VetoViolation]:
        return [v for v in self.violations if v.severity == "HARD_VETO"]

    @property
    def warnings(self) -> list[VetoViolation]:
        return [v for v in self.violations if v.severity == "WARNING"]

    def to_engineer_text(self) -> str:
        lines = []
        if self.executable:
            lines.append("✅ 配方已通过全部硬校验闸门 → 标记为「可执行」")
        elif self.channel == "candidate":
            lines.append("⚠️ 配方未通过硬校验 → 归入「候选假设」通道")
        else:
            lines.append("❌ 配方被硬校验闸门拒绝")
        lines.append("")

        if self.hard_vetos:
            lines.append("🚫 硬违规项（不可绕过）：")
            for v in self.hard_vetos:
                lines.append(f"   [{v.rule_id}] {v.message}")
                if v.evidence:
                    lines.append(f"   📎 证据: {v.evidence}")
            lines.append("")

        if self.warnings:
            lines.append("⚠️ 警告项（需人工确认）：")
            for v in self.warnings:
                lines.append(f"   [{v.rule_id}] {v.message}")
            lines.append("")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════
# 闸门 1：化学计量约束
# ═══════════════════════════════════════════════════

# 当量比容差
R_TOLERANCE = 0.05  # 5%容差
# 默认密度表（简化）
DENSITY_TABLE = {
    "环氧树脂": 1.16,
    "胺类固化剂": 1.05,
    "酸酐固化剂": 1.20,
    "碳粉": 2.0,
    "氧化铝": 3.9,
    "氮化铝": 3.3,
    "PCM": 0.9,
    "稀释剂": 0.85,
    "偶联剂": 0.95,
}

# 逾渗阈值（体积分数上限，超过则分散体系崩溃）
PERCOLATION_LIMITS = {
    "碳粉": 0.25,       # 碳粉逾渗约20-30%
    "氧化铝": 0.60,     # 氧化铝 packing limit ~60 vol%
    "氮化铝": 0.55,
    "导热粉": 0.65,     # 通用导热粉上限
    "PCM": 0.70,        # 相变材料上限
    "填料": 0.50,       # 通用填料安全上限（保守值）
}

# 单位换算常数
PHR_TO_WT_CTE = 1.0  # 简化：phr ≈ wt% 当总重≈100phr时


def _compute_r_actual(recipe: FormulaRecipe) -> Optional[float]:
    """计算实际当量比 r_actual = n_active_B / n_active_A"""
    resins = recipe.get_by_role("树脂")
    curatives = recipe.get_by_role("固化剂")

    # 找含EEW的树脂和含AHEW的固化剂
    resin_with_eew = [r for r in resins if r.eew > 0]
    cur_with_ahew = [c for c in curatives if c.ahew > 0]

    if resin_with_eew and cur_with_ahew:
        # n_A = mass_A / EEW, n_B = mass_B / AHEW
        total_n_A = sum(r.phr / r.eew for r in resin_with_eew)
        total_n_B = sum(c.phr / c.ahew for c in cur_with_ahew)
        if total_n_A > 0:
            return total_n_B / total_n_A
    return None


def gate_chemical_stoichiometry(recipe: FormulaRecipe) -> list[VetoViolation]:
    """闸门1：当量比约束"""
    violations = []
    r_actual = _compute_r_actual(recipe)

    if r_actual is not None:
        if abs(r_actual - 1.0) > R_TOLERANCE:
            violations.append(VetoViolation(
                gate="化学计量",
                severity="HARD_VETO",
                rule_id="STOICH-001",
                message=f"当量比 r_actual = {r_actual:.3f}，超出 1±{R_TOLERANCE} 容差范围",
                evidence=f"理论r=1.0；实测r={r_actual:.3f}；偏差{abs(r_actual-1)*100:.1f}%"
            ))
    return violations


# ═══════════════════════════════════════════════════
# 闸门 2：体积分数约束（逾渗限制）
# ═══════════════════════════════════════════════════

def gate_volume_fraction(recipe: FormulaRecipe) -> list[VetoViolation]:
    """闸门2：填料/PCM不得超过逾渗阈值或Krieger-Dougherty极限"""
    violations = []

    for comp in recipe.components:
        limit = None
        for key, val in PERCOLATION_LIMITS.items():
            if key in comp.name or key in comp.role:
                limit = val
                break

        if limit is not None and comp.density > 0:
            # 估算体积分数（简化：基于phr / (phr + 100) * density_factor）
            total_mass = recipe.total_phr() or 100.0
            mass_frac = comp.phr / total_mass
            # 粗略体积分数 = (phr/ρ) / Σ(phr_i/ρ_i)
            vol_comp = comp.phr / comp.density
            vol_total = sum(c.phr / max(c.density, 0.1) for c in recipe.components)
            vol_frac = vol_comp / vol_total if vol_total > 0 else 0

            if vol_frac > limit:
                violations.append(VetoViolation(
                    gate="体积分数",
                    severity="HARD_VETO",
                    rule_id="VOLFRAC-001",
                    message=f"{comp.name}体积分数≈{vol_frac:.1%}，超过{key}逾渗上限{limit:.0%}",
                    evidence=f"Krieger-Dougherty逾渗限={limit:.0%}；配方中={comp.name} phr={comp.phr}"
                ))
    return violations


# ═══════════════════════════════════════════════════
# 闸门 3：组分互斥检查
# ═══════════════════════════════════════════════════

# 禁止组合清单（配方中不得同时出现）
INCOMPATIBLE_COMBOS = [
    (["铂催化", "铂催化剂"], ["含硫", "硫"], "铂催化剂与含硫物质不可共存（催化中毒）"),
    (["铂催化", "铂催化剂"], ["含磷", "磷"], "铂催化剂与含磷物质不可共存（催化中毒）"),
    (["酸酐固化", "酸酐"], ["胺类", "胺"], "酸酐与胺类固化剂不可混用（竞争反应）"),
    (["异氰酸酯", "MDI", "HDI"], ["水", "湿气"], "异氰酸酯体系严禁含水（产气→气泡）"),
    (["阳离子UV"], ["碱性", "胺类"], "阳离子UV体系遇碱性物质阻聚失效"),
]

def gate_component_compatibility(recipe: FormulaRecipe) -> list[VetoViolation]:
    """闸门3：禁用组分组合"""
    violations = []
    names = [c.name + "|" + c.role for c in recipe.components]

    for group_a, group_b, reason in INCOMPATIBLE_COMBOS:
        has_a = any(any(kw in n for kw in group_a) for n in names)
        has_b = any(any(kw in n for kw in group_b) for n in names)
        if has_a and has_b:
            violations.append(VetoViolation(
                gate="组分互斥",
                severity="HARD_VETO",
                rule_id="COMPAT-001",
                message=reason,
                evidence=f"命中黑名单: {group_a} × {group_b}"
            ))
    return violations


# ═══════════════════════════════════════════════════
# 闸门 4：量纲与单位一致性
# ═══════════════════════════════════════════════════

def gate_unit_consistency(recipe: FormulaRecipe) -> list[VetoViolation]:
    """闸门4：量纲检查（phr累加、单位混用检出）"""
    violations = []

    # 检查phr累加异常（正常配方总phr在100-500之间）
    total = recipe.total_phr()
    if total > 0:
        if total > 1000:
            violations.append(VetoViolation(
                gate="单位一致性",
                severity="HARD_VETO",
                rule_id="UNIT-001",
                message=f"配方总phr={total:.0f}，远超合理范围（通常100-500），疑似单位错误或重复添加",
            ))
        elif total < 50:
            violations.append(VetoViolation(
                gate="单位一致性",
                severity="WARNING",
                rule_id="UNIT-002",
                message=f"配方总phr={total:.0f}，低于合理范围，是否遗漏主组分？",
            ))

    # 检查wt%usage异常（如果有多个组分的wt%>0，总wt%应≈100%）
    wt_components = [c for c in recipe.components if c.wt_pct > 0]
    if len(wt_components) > 1:
        total_wt = sum(c.wt_pct for c in wt_components)
        if abs(total_wt - 100.0) > 5:
            violations.append(VetoViolation(
                gate="单位一致性",
                severity="WARNING",
                rule_id="UNIT-003",
                message=f"质量百分比总和={total_wt:.1f}%，与100%偏差{abs(total_wt-100):.1f}%",
            ))
    return violations


# ═══════════════════════════════════════════════════
# 闸门 5：放大可行性约束
# ═══════════════════════════════════════════════════

def gate_scaleup_feasibility(recipe: FormulaRecipe) -> list[VetoViolation]:
    """闸门5：放大可行性 —— 小试配方在量产剪切下是否必然失效"""
    violations = []

    # 经验法则：高填料体系在高剪切下易接枝断裂
    fillers = [c for c in recipe.components if c.role in ("填料", "导热粉", "PCM")]
    total_filler_phr = sum(f.phr for f in fillers)

    if total_filler_phr > 200 and recipe.process_shear_rate > 50:
        violations.append(VetoViolation(
            gate="放大可行性",
            severity="HARD_VETO",
            rule_id="SCALEUP-001",
            message=f"高填料(total={total_filler_phr:.0f}phr) + 高剪切(γ={recipe.process_shear_rate:.0f}/s) "
                    f"→ 接枝大面积断裂风险极高",
            evidence=f"接枝安全窗口通常γ<50/s当填料>200phr；当前配方直接放大必然报废"
        ))
    elif total_filler_phr > 150 and recipe.process_shear_rate > 100:
        violations.append(VetoViolation(
            gate="放大可行性",
            severity="WARNING",
            rule_id="SCALEUP-002",
            message=f"中高填料({total_filler_phr:.0f}phr)在高剪切下需注意接枝热降解",
        ))

    # 高温+高填料蓄热风险
    if recipe.process_temp > 80 and total_filler_phr > 180:
        violations.append(VetoViolation(
            gate="放大可行性",
            severity="WARNING",
            rule_id="SCALEUP-003",
            message=f"高温({recipe.process_temp}°C)+高填料蓄热风险，局部温度可能超过Tg偏移阈值",
        ))
    return violations


# ═══════════════════════════════════════════════════
# 新增五道闸门 — 覆盖 TOP 5 高风险场景
# ═══════════════════════════════════════════════════

# ── 安全窗口常量 (material_type → safety limits) ──
SAFETY_WINDOWS = {
    "环氧树脂-胺": {"temp_max": 180, "temp_min": 40, "pressure_max": 0.1, "time_max_min": 240},
    "环氧树脂-酸酐": {"temp_max": 200, "temp_min": 60, "pressure_max": 0.1, "time_max_min": 300},
    "聚氨酯": {"temp_max": 120, "temp_min": 20, "pressure_max": 0.05, "time_max_min": 120},
    "有机硅-铂催化": {"temp_max": 150, "temp_min": 15, "pressure_max": -1, "time_max_min": 180},
    "PCM复合": {"temp_max": 90, "temp_min": 10, "pressure_max": -1, "time_max_min": 120},
    "_default": {"temp_max": 150, "temp_min": 15, "pressure_max": 0.1, "time_max_min": 180},
}

# 体系识别关键词
MATERIAL_TYPE_KEYWORDS = {
    "环氧树脂-胺": ["环氧", "胺固化", "双组份环氧", "环氧-胺"],
    "环氧树脂-酸酐": ["酸酐", "环氧-酸酐", "单组份环氧"],
    "聚氨酯": ["聚氨酯", "MDI", "HDI", "异氰酸酯"],
    "有机硅-铂催化": ["有机硅", "铂催化", "硅橡胶", "加成型"],
    "PCM复合": ["PCM", "相变", "导热灌封", "导热凝胶", "thermal"],
    "_通用胶粘": ["电子胶", "电子胶粘剂", " adhes", "密封胶"],
}

# 感知阈值（认知偏差警示）
# 用户描述的工艺参数在安全窗口的 80-100% 区间内时，需要提示风险
WARNING_THRESHOLD_RATIO = 0.8


def _detect_material_type(text: str) -> str:
    """从用户描述+LLM输出中识别材料体系"""
    for mtype, kws in MATERIAL_TYPE_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return mtype
    return "_default"


# ──────────────────────────────────────────────────
# 阀门1: 工艺安全窗口校验（针对场景#2）
# ──────────────────────────────────────────────────

def gate_process_safety_window(profile: DiagnosticProfile, suggestion_text: str) -> list[VetoViolation]:
    """
    从LLM建议文本中解析数值+单位，对比材料安全窗口。
    HARD_VETO: 超出材料temp_max / pressure_max
    WARNING:  超出80%安全窗口但仍在上限内
    """
    violations = []
    material_type = _detect_material_type(profile.failure_mode + " " + (profile.product_name or ""))
    window = SAFETY_WINDOWS.get(material_type, SAFETY_WINDOWS["_default"])

    import re

    # 解析温度建议
    for m in re.finditer(r'(?:升温|降温|加热|冷却|温度|T|t)\s*(?:至|到|至|约)?\s*(\d+)\s*[°℃]?C?', suggestion_text):
        temp = int(m.group(1))
        if temp > window["temp_max"]:
            violations.append(VetoViolation(
                gate="工艺安全窗口",
                severity="HARD_VETO",
                rule_id="SAFETY-TEMP-001",
                message=f"建议温度{temp}°C超过{material_type}安全上限{window['temp_max']}°C，会导致材料热降解",
                evidence=f"体系={material_type}；安全上限={window['temp_max']}°C；建议={temp}°C"
            ))
        elif temp > window["temp_max"] * WARNING_THRESHOLD_RATIO:
            violations.append(VetoViolation(
                gate="工艺安全窗口",
                severity="WARNING",
                rule_id="SAFETY-TEMP-002",
                message=f"建议温度{temp}°C接近{material_type}安全上限({window['temp_max']}°C)的{WARNING_THRESHOLD_RATIO:.0%}",
                evidence=f"安全上限={window['temp_max']}°C；建议={temp}°C；距离上限{window['temp_max']-temp}°C"
            ))

    # 解析压力建议
    for m in re.finditer(r'(?:压力|真空|加压)\s*(?:至|到|提高|降低|升到|降至|提至|提到|至)?\s*(\d+\.?\d*)\s*(MPa|mpa|Pa|kPa|Mpa)?', suggestion_text):
        pressure = float(m.group(1))
        unit = m.group(2) or "MPa"
        if unit.lower() in ("mpa", "Mpa"):
            pressure_mpa = pressure
        elif unit.lower() in ("kpa",):
            pressure_mpa = pressure / 1000
        elif unit.lower() in ("pa",):
            pressure_mpa = pressure / 1e6
        else:
            pressure_mpa = pressure

        if window["pressure_max"] > 0 and pressure_mpa > window["pressure_max"]:
            violations.append(VetoViolation(
                gate="工艺安全窗口",
                severity="HARD_VETO",
                rule_id="SAFETY-PRES-001",
                message=f"建议压力{pressure}{unit}超过{material_type}安全上限{window['pressure_max']}MPa",
                evidence=f"体系={material_type}；建议={pressure}{unit}；安全上限={window['pressure_max']}MPa"
            ))

    # 长时间高温（超过安全窗口的 4h 阈值）
    for m in re.finditer(r'(?:保温|保持|反应|固化|延长)\s*(\d+)\s*(分钟|min|小时|h|hour)', suggestion_text):
        duration = int(m.group(1))
        unit = m.group(2)
        if unit in ("小时", "h", "hour"):
            duration_min = duration * 60
        else:
            duration_min = duration

        if duration_min > window["time_max_min"]:
            violations.append(VetoViolation(
                gate="工艺安全窗口",
                severity="HARD_VETO",
                rule_id="SAFETY-TIME-001",
                message=f"建议时间{duration}{unit}超过{material_type}安全窗口{window['time_max_min']}min，热蓄积风险极高",
                evidence=f"安全窗口={window['time_max_min']}min；建议={duration_min}min"
            ))

    return violations


# ──────────────────────────────────────────────────
# 阀门2: 证据充分性门槛 + 拒答弃权（针对场景#1）
# ──────────────────────────────────────────────────

# 证据充分性阈值
MIN_RAG_RESULTS_FOR_CONCLUSION = 2   # 最少需要RAG结果数
MIN_RAG_AVG_SCORE = 0.5              # RAG平均相关性分数
MIN_PROFILE_FIELDS_FOR_CONCLUSION = 3 # 最少需要已填写的profile字段数

def gate_evidence_sufficiency(
    profile: DiagnosticProfile,
    rag_results: list[dict],
    rag_scores: list[float] | None = None,
) -> list[VetoViolation]:
    """
    信息不足时强制触发"信息不足请补充"的追问，而非允许输出结论。
    """
    violations = []

    # 统计 Profile 中已填写的有效字段数
    filled_fields = sum(1 for v in [
        profile.product_name, profile.failure_mode, profile.test_phase,
        profile.batch_scope, profile.formula_name
    ] if v and str(v).strip())

    if filled_fields < MIN_PROFILE_FIELDS_FOR_CONCLUSION:
        violations.append(VetoViolation(
            gate="证据充分性",
            severity="HARD_VETO",
            rule_id="EVID-NOINFO-001",
            message=f"基础信息仅有{filled_fields}项，需至少{MIN_PROFILE_FIELDS_FOR_CONCLUSION}项才可输出建议。请补充: 失效模式/产品名/测试阶段/批次范围",
            evidence=f"已填: {filled_fields}/{MIN_PROFILE_FIELDS_FOR_CONCLUSION}；failure_mode='{profile.failure_mode}'；product='{profile.product_name}'"
        ))

    # RAG 检索结果评估
    if len(rag_results) < MIN_RAG_RESULTS_FOR_CONCLUSION:
        violations.append(VetoViolation(
            gate="证据充分性",
            severity="HARD_VETO",
            rule_id="EVID-NORAG-001",
            message=f"知识库匹配度不足（{len(rag_results)}条相关记录，需≥{MIN_RAG_RESULTS_FOR_CONCLUSION}条）。AI可能依赖训练数据而非领域知识→幻觉风险极高",
            evidence=f"RAG命中: {len(rag_results)}条；需求: ≥{MIN_RAG_RESULTS_FOR_CONCLUSION}条"
        ))

    # 精确数字幻觉检测 + 无来源引用
    if rag_results:
        has_numeric_ref = any("phr" in str(r).lower() or "份" in str(r) or "%" in str(r) for r in rag_results)
        if not has_numeric_ref:
            violations.append(VetoViolation(
                gate="证据充分性",
                severity="WARNING",
                rule_id="EVID-NOREF-001",
                message="知识库中无精确数值引用。若AI输出具体数字（如'加2.5phr'），该数字为推测值，需工程师独立验证",
                evidence="RAG记录中无phr/%数值字段"
            ))

    return violations


# ──────────────────────────────────────────────────
# 阀门3: 产品体系匹配校验（针对场景#3：知识迁移错误）
# ──────────────────────────────────────────────────

# 知识库 rule_id 前缀 → 适用体系（防止在美缝剂体系上套用电子胶规则）
RAG_PRODUCT_SCOPE = {
    "FRULE-L0": ["环氧", "聚氨酯", "通用"],
    "FRULE-L1": ["环氧", "聚氨酯", "有机硅", "通用"],
    "FRULE-L2": ["环氧", "聚氨酯", "有机硅", "PCM", "通用"],
    "FRULE-L3": ["环氧", "聚氨酯", "通用"],
    "FRULE-L4": ["环氧", "聚氨酯", "通用"],
    "FRULE-L5": ["环氧", "聚氨酯", "通用"],
    "FRULE-L7": ["环氧", "通用"],
    "DRULE": ["通用"],     # 决策规则通用
    "FAIL": ["通用"],       # 失败教训通用
    "K-L0": ["环氧", "美缝剂"],
    "K-L1": ["环氧", "聚氨酯", "通用"],
    "K-L2": ["环氧", "通用"],
    "K-L3": ["环氧", "聚氨酯", "导热", "通用"],
    "K-L4": ["环氧", "通用"],
    "K-L5": ["环氧", "通用"],
}


def gate_product_scope_match(
    profile: DiagnosticProfile,
    rag_results: list[dict],
) -> list[VetoViolation]:
    """
    防止知识迁移：RAG检索到的规则是否适用于用户描述的产品体系？
    不匹配时发出WARNING，提醒工程师该建议可能来自不同体系。
    """
    violations = []
    user_material = _detect_material_type(profile.failure_mode + " " + (profile.product_name or ""))

    if user_material == "_default":
        violations.append(VetoViolation(
            gate="体系匹配",
            severity="WARNING",
            rule_id="SCOPE-UNKNOWN-001",
            message=f"无法识别产品体系（关键词未匹配）。知识库规则可能来自不同材料体系，输出建议需人工验证适用性",
            evidence=f"failure_mode='{profile.failure_mode}'；product='{profile.product_name}'；未匹配任何已知体系"
        ))
        return violations

    # 检查 RAG 命中的规则是否匹配体系
    for result in rag_results:
        rule_id = result.get("rule_id", "") or result.get("source_id", "")
        if not rule_id:
            continue
        prefix = rule_id.split("-")[0] + "-" + rule_id.split("-")[1] if "-" in rule_id else rule_id[:4]
        allowed_scopes = RAG_PRODUCT_SCOPE.get(prefix, ["通用"])
        if "通用" not in allowed_scopes and user_material not in allowed_scopes:
            violations.append(VetoViolation(
                gate="体系匹配",
                severity="WARNING",
                rule_id="SCOPE-MISMATCH-001",
                message=f"规则{rule_id}的适用体系为{allowed_scopes}，但当前产品体系为{user_material}→知识迁移风险",
                evidence=f"规则前缀={prefix}；适用={allowed_scopes}；当前体系={user_material}"
            ))

    return violations


# ──────────────────────────────────────────────────
# 阀门4: 失败教训红线代码级硬拦截（针对场景#4）
# ──────────────────────────────────────────────────

def gate_failure_lesson_intercept(profile: DiagnosticProfile, suggestion_text: str) -> list[VetoViolation]:
    """
    从LLM建议文本中搜索是否违反失败教训库红线。
    仅靠LLM prompt 遵守已被证明不够（上下文可能淹没规则）→ 代码级兜底。
    """
    violations = []
    import re

    try:
        import db_sqlite as _db
        lessons = _db.list_failure_lessons()
    except Exception:
        lessons = []

    if not lessons:
        return violations

    text = suggestion_text.lower()

    # ── 数值型红线兜底（直接解析文本中的数值+单位，与红线限值比较）──
    import re as _re
    for lesson in lessons:
        rl = lesson.get('redline', '') or ''
        _desc = lesson.get('action_desc', '') or ''
        _cons = lesson.get('consequence', '') or ''
        # 例子：'+Xphr' 提取X并与红线 '≤Yphr' 的Y比较
        rl_limit_m = _re.search(r'(\d+\.?\d*)\s*(phr|份)', rl)
        if rl_limit_m and ('稀释' in _desc or '加固化剂' in _desc or '上调' in _desc):
            limit_val = float(rl_limit_m.group(1))
            # 从建议文本中找操作数值
            desc_kw = '稀释剂' if '稀释' in _desc else ('固化剂' if '固化剂' in _desc else '调整')
            sug_m = _re.search(desc_kw + r'.*?(\d+\.?\d*)\s*(phr|份)?', suggestion_text)
            if sug_m:
                sug_val = float(sug_m.group(1))
                if sug_val > limit_val:
                    violations.append(VetoViolation(
                        gate='失败教训红线',
                        severity='HARD_VETO',
                        rule_id=f'LESSON-{lesson.get("lesson_id","")}',
                        message=f'建议{sug_val}phr超过红线{limit_val}phr。历史教训：{_cons}',
                        evidence=f'红线: {rl}；实际: {sug_val}phr；来自: {_desc}'
                    ))
                    break  # 已拦截一条，无需继续

    # ── 关键词单字分词匹配（fallback）──
        redline = lesson.get("redline", "") or ""
        action = lesson.get("action_desc", "") or ""
        consequence = lesson.get("consequence", "") or ""

        # 提取红线中的关键禁用词
        # 提取动作中的关键操作词
        # 关键修复：单字分词（每CJK字符独立token）避免整短语无法匹配
        action_keywords = re.findall(r'[\u4e00-\u9fff]|[A-Za-z0-9]+', action)

        # 检查建议中是否包含红线操作
        # 规则：若建议中出现了 action_desc 中的关键操作（>2个关键词匹配）即触发
        if action_keywords:
            # 单字分词后：每个CJK字符独立匹配，去重后≥2即触发
            matched = [kw for kw in set(action_keywords) if kw in text]
            match_count = len(matched)
            if match_count >= 2:
                violations.append(VetoViolation(
                    gate="失败教训红线",
                    severity="HARD_VETO",
                    rule_id=f"LESSON-{lesson.get('lesson_id','')}",
                    message=f"建议可能违反红线：{redline}。历史教训：{consequence}",
                    evidence=f"违反规则: {lesson.get('lesson_id','')}；命中: {matched[:5]}；禁止: {redline}"
                ))

    return violations


# ═══════════════════════════════════════════════════
# 集成校验入口（扩展版）
# ═══════════════════════════════════════════════════

def validate_formula(recipe: FormulaRecipe) -> ValidationReport:
    """
    对配方执行全部5道闸门校验（原有）。
    返回 ValidationReport，明确标记 executable/rejected/candidate + 全部违规项。
    """
    report = ValidationReport(recipe=recipe)

    all_violations = []
    gates = [
        gate_chemical_stoichiometry,
        gate_volume_fraction,
        gate_component_compatibility,
        gate_unit_consistency,
        gate_scaleup_feasibility,
    ]

    for gate_fn in gates:
        try:
            v = gate_fn(recipe)
            all_violations.extend(v)
        except Exception as e:
            logger.warning(f"闸门 {gate_fn.__name__} 异常: {e}")

    report.violations = all_violations

    # 决策逻辑
    hard_vetos = [v for v in all_violations if v.severity == "HARD_VETO"]
    if hard_vetos:
        report.channel = "rejected"
        report.passed = False
        report.executable = False
    elif any(v.severity == "WARNING" for v in all_violations):
        report.channel = "candidate"
        report.passed = True
        report.executable = False  # 候选≠可执行，需人工确认
    else:
        report.channel = "executable"
        report.passed = True
        report.executable = True

    # 审计日志
    report.audit_log = [
        f"[{report.channel.upper()}] {v.gate}/{v.rule_id}: {v.message}"
        for v in all_violations
    ] if all_violations else ["[PASS] 全部闸门通过"]

    return report


# ═══════════════════════════════════════════════════
# 自由文本配方解析（从LLM非结构化输出中提取组分）
# ═══════════════════════════════════════════════════

# 常见组分关键词映射
ROLE_KEYWORDS = {
    "树脂": ["树脂", "环氧", "聚氨酯", "丙烯酸", "有机硅", "预聚物", "基料"],
    "固化剂": ["固化剂", "交联剂", "胺", "酸酐", "异氰酸酯", "催化剂", "引发剂"],
    "填料": ["填料", "碳粉", "竹粉", "石墨", "碳纳米管", "炭黑"],
    "导热粉": ["氧化铝", "氮化铝", "氮化硼", "BN", "AlN", "Al2O3", "导热"],
    "PCM": ["PCM", "相变", "石蜡", "脂肪酸"],
    "稀释剂": ["稀释剂", "活性稀释", "溶剂"],
    "偶联剂": ["偶联剂", "硅烷", "钛酸酯"],
}

def parse_formula_from_text(text: str, default_temp: float = 25.0,
                            default_shear: float = 0.0) -> FormulaRecipe:
    """
    从LLM自由文本输出中解析配方结构。
    支持格式：
    - "环氧树脂 100phr"
    - "固化剂: 45phr"
    - "树脂 环氧树脂 100"
    """
    recipe = FormulaRecipe()
    recipe.process_temp = default_temp
    recipe.process_shear_rate = default_shear

    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip().lstrip("•-*").strip()
        if not line:
            continue

        # 尝试匹配 "组分名 phr值" 或 "组分名: phr值"
        m = re.match(
            r"[\s•\-*]*([^\d:：]+?)[\s:：]*(\d+\.?\d*)\s*(phr|ph|份)?",
            line, re.IGNORECASE
        )
        if m:
            name = m.group(1).strip()
            phr_val = float(m.group(2))

            # 判断role
            role = "其他"
            for r, kws in ROLE_KEYWORDS.items():
                if any(kw in name for kw in kws):
                    role = r
                    break

            density = DENSITY_TABLE.get(name, 1.0)
            for k, v in DENSITY_TABLE.items():
                if k in name:
                    density = v
                    break

            recipe.components.append(FormulaComponent(
                name=name, role=role, phr=phr_val, density=density
            ))

    return recipe


# ═══════════════════════════════════════════════════
# 双通道工厂
# ═══════════════════════════════════════════════════

def dual_channel_output(recipe: FormulaRecipe, llm_rationale: str = "") -> dict:
    """
    双通道输出：
    - channel: "rejected" | "candidate" | "executable"
    - 自动拆分候选 vs 可执行
    - 附加审计可追溯证据
    """
    report = validate_formula(recipe)

    result = {
        "channel": report.channel,
        "executable": report.executable,
        "passed": report.passed,
        "violations": [
            {
                "gate": v.gate,
                "severity": v.severity,
                "rule_id": v.rule_id,
                "message": v.message,
                "evidence": v.evidence,
            }
            for v in report.violations
        ],
        "engineer_text": report.to_engineer_text(),
        "audit_log": report.audit_log,
        "component_count": len(recipe.components),
    }

    return result
