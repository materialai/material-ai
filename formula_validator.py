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
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


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
# 集成校验入口
# ═══════════════════════════════════════════════════

def validate_formula(recipe: FormulaRecipe) -> ValidationReport:
    """
    对配方执行全部5道闸门校验。
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
