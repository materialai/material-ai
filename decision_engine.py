"""工艺偏差三段式推理引擎 · L0-L7分层演绎 + 配方vs工艺决策

核心升级（相较v1）：
- 推理主干从'简单Q+A打分'改为'L0→L7八层演绎过滤链'
- 内置化工公式库：配比当量/固化动力学/Tg混合律/DLVO分散/接枝控制
- 配方vs工艺互补规则 + 禁止红线（失败教训库拦截）
- 知识反馈闭环：野经验录入 + 置信度动态更新
"""

from __future__ import annotations

import json
import time
import logging
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ============================================================
# 第一部分：数据结构
# ============================================================

@dataclass
class DiagnosticProfile:
    """诊断画像 — 逐步填充的结构化信息"""
    # Phase 1 核心信息
    product_name: str = ""
    formula_name: str = ""
    test_phase: str = ""           # 小试/中试/量产/客户反馈
    failure_mode: str = ""
    batch_scope: str = ""          # 单缸/连续多缸/批次性/渐进
    is_self_limiting: bool = False
    
    # Phase 2 辅助信息
    batch_changed: bool = False
    env_changed: bool = False
    new_formula: bool = False
    
    # Phase 3 深度信息（按需追问）
    temp_curve_ok: Optional[bool] = None
    mix_speed_ok: Optional[bool] = None
    vacuum_ok: Optional[bool] = None
    material_lot: str = ""
    
    # 定量数据（可选，但极有价值）
    r_actual: Optional[float] = None        # 实际配比偏移%
    env_temp: Optional[float] = None        # 环境温度°C
    env_rh: Optional[float] = None          # 环境湿度%
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def is_decision_ready(self) -> bool:
        return all([self.product_name, self.failure_mode,
                    self.test_phase, self.batch_scope])
    
    def completeness(self) -> float:
        core = [self.product_name, self.formula_name, self.test_phase,
                self.failure_mode, self.batch_scope]
        return sum(1 for f in core if f) / len(core)


@dataclass
class DecisionResult:
    """最终决策输出"""
    primary_direction: str = ""       # "formula" / "process" / "both"
    confidence: float = 0.0
    formula_score: int = 0
    process_score: int = 0
    reasoning: str = ""
    l_analysis: str = ""              # L0-L7逐层分析文本
    hypothesis_list: list = field(default_factory=list)
    action_checklist: list = field(default_factory=list)
    risk_warning: str = ""
    forbidden_actions: list = field(default_factory=list)  # 禁止红线
    formula_adjustments: list = field(default_factory=list)  # 配方调整建议
    process_adjustments: list = field(default_factory=list)  # 工艺调整建议


# ============================================================
# 第二部分：L0-L7 分层推理引擎
# ============================================================

# L0 配方层规则
L0_RULES = {
    "不固化": {
        "formula": "配比偏移、固化剂失效、水分侵入",
        "check": "计算 r_actual = (Q_B·ρ_B/AHEW) / (Q_A·ρ_A/EEW)；|Δr|>5%需立即纠正",
        "direction": "formula_first"
    },
    "固化慢": {
        "formula": "温度低/促进剂少/抑制剂多",
        "check": "t_gel = A·exp(E_a/RT)；先升温2~3°C验证",
        "direction": "process_first"
    },
    "固化快": {
        "formula": "促进剂过多、环境热",
        "check": "放热峰前移 → 减促进剂或分装降温",
        "direction": "both"
    },
    "发脆": {
        "formula": "交联度过高、环氧过量",
        "check": "p_c = 2/f_avg；增韧剂+5~8phr 或 调当量比",
        "direction": "formula_first"
    },
    "发软": {
        "formula": "固化不足、固化剂过量、PCM过多",
        "check": "E = 3ν_eRT；提交联密度",
        "direction": "formula_first"
    },
    "气泡": {
        "formula": "真空不足/搅拌裹气/低沸物挥发/反应产气",
        "check": "真空度<-0.06MPa；高碳粉体系需预烘干80°C/4h",
        "direction": "process_first"
    },
}

# L1 用户非标工况规则
L1_TRIGGERS = {
    "设备私改": "用户改动混合管/管路 → 重新标定工艺窗口",
    "加热非标": "明火直烤/单点过热 → 改水浴或烘房加热",
    "低温施工": "T_a<15°C → 需冬季补偿策略（按当量比调泵输出）",
    "操作不稳": "人工施压/中途停机 → 工艺规程再培训",
    "旧料混用": "残料>5% → 需重做首件验证",
}

# L2 环境耦合规则
L2_RULES = {
    "低温": {
        "trigger": "env_temp < 15",
        "action": "冬季补偿：环境温度每降5°C → 料温上调3~4°C（按当量比调泵）",
        "formula": "μ(T) = μ_0·exp[E_a/R·(1/T - 1/T_0)]"
    },
    "高温": {
        "trigger": "env_temp > 35",
        "action": "夏季降速：料温下调2~3°C + 促进剂-10%",
        "formula": "Arrhenius"
    },
    "高湿": {
        "trigger": "env_rh > 75",
        "action": "除湿/预烘干：碳粉80°C/4h + 接枝水解风险评估",
        "formula": "X_d ∝ RH"
    },
}

# 配伍禁忌（禁止红线）
COMPATIBILITY_BLACKLIST = {
    "铂催化有机硅 + 含硫助剂": "铂催化剂中毒 → 胶体永久不固化 → 严禁混合",
    "过量稀释剂 + 高固化剂": "交联网络被稀释破坏 + 高交联追求 → 内应力开裂",
    "高碳粉 + 高增触剂": "粘度指数级上升 → 过剪切 → 接枝破坏",
    "高PCM + 高固化剂": "硬壳+热胀冷缩 → 内应力 → 批量开裂",
    "高碳粉 + 高PCM": "分散相>65% → 流动性崩塌 → 无法生产",
}

# 失败教训红线（历史事故）
FAILURE_LESSONS = [
    {"action": "一次性大量加稀释剂>5phr", "result": "交联网络稀释破坏、Tg大降、批量开裂",
     "redline": "单次稀释剂≤5phr，必须分2~3次递进验证"},
    {"action": "小试转速直接等比例平移量产", "result": "桨尖过剪切、接枝断裂、整批团聚",
     "redline": "放大必须执行四大放大准则（等剪切/等桨尖线速度/等Re/等P/V）"},
    {"action": "为解决固化慢单方面大幅上调固化剂", "result": "交联度过高、内应力爆炸、批量开裂漏芯",
     "redline": "调固化剂必须评估脆性，禁止单点暴力调参"},
    {"action": "高RH车间未预烘干直接投料", "result": "接枝水解、团聚、气泡超标",
     "redline": "RH>75%必须先预烘干80°C/4h+真空润湿"},
    {"action": "忽视冬季粘度不对称", "result": "体积比≠当量比 → 配比漂移 → 不固化",
     "redline": "冬季必须按当量比调节泵输出而非体积比"},
]

# 配方vs工艺互补规则（直接来自文档附录C）
FORMULA_PROCESS_RULES = [
    # (场景, 优选方向, 替代方案, 禁忌)
    ("体系粘度过高、搅拌过载",
     "少量活性稀释剂(+2~3phr)或降固含量(1~2%)",
     "升温3~5°C（需验接枝稳定性）",
     "禁止过量稀释剂损害长期稳定性"),
    ("粉体频繁团聚、分散困难",
     "上调偶联剂(+0.2phr)或评估碳粉BET是否超阈值",
     "延长分散时间10~15min",
     "偶联剂超量产生弱界面"),
    ("成品硬而脆、频繁开裂",
     "增韧剂+5~8phr + 必要时微降稀释剂",
     "降1~2%固含量+降1phr固化剂",
     "单纯提当量比R值加剧脆性"),
    ("相变渗漏、漏芯高发",
     "三联动检查：PCM是否过高+偶联剂是否不足+交联密度是否过低",
     "提R值+补偶联剂+降PCM",
     "单纯提R值加剧脆性（必须组分互补）"),
    ("冬季批次漂移",
     "冬季料温补偿 + Tg微升 + 促进",
     "降黏+升温微调+设备保温",
     "同一配方全年不做任何调整"),
    ("碳粉批次更换后流变漂移",
     "按BET实测调整稀释剂/固含量用量",
     "分散时间调整+分散剂匹配",
     "新批次直接使用不做参数验证"),
    ("单缸偶发+自限+无换料",
     "优先排查工艺参数（温控/搅拌/真空瞬时波动）",
     "设备点检（温控精度/搅拌/传动）",
     "暂不调配方（单缸自限配方缺陷概率低）"),
    ("连续多缸/批次性出现",
     "高度怀疑配方缺陷（系统性问题）",
     "计算r_actual、查原料批次兼容性",
     "不要只调工艺参数掩盖配方问题"),
    ("新配方/试产阶段失效",
     "配方本身未定型 → 优先调整配方",
     "配比复核+关键组分用量验证",
     "把配方问题和工艺问题混为一谈"),
    ("渐进衰减（性能逐步下降）",
     "工艺漂移/设备磨损 → 优先查工艺/设备",
     "传感器校准+泵标定+桨检查",
     "直接改配方（未排除设备问题）"),
]


# ============================================================
# 第三部分：决策引擎（配方 vs 工艺）
# ============================================================

def evaluate_rules(profile: DiagnosticProfile) -> tuple[int, int, list[str]]:
    """
    综合评分：L0-L7规则 + 配方vs工艺互补规则 + 失败教训参考
    返回 (formula得分, process得分, 命中规则文本)
    """
    f_score, p_score = 0, 0
    fired = []
    
    # ── L0 配方层规则 ──
    fm = profile.failure_mode
    for key, rule in L0_RULES.items():
        if key in fm:
            if rule["direction"] == "formula_first":
                f_score += 30
                fired.append(f"[L0] {key}: {rule['formula']} → 配方优先")
            elif rule["direction"] == "process_first":
                p_score += 25
                fired.append(f"[L0] {key}: {rule['formula']} → 工艺优先")
            else:
                f_score += 15
                p_score += 15
                fired.append(f"[L0] {key}: {rule['formula']} → 双向可能")
    
    # ── L1 用户非标 ──
    if profile.env_changed:
        p_score += 20
        fired.append("[L1] 环境/操作变化 → 优先排查工艺工况")
    
    # ── L2 环境耦合 ──
    if profile.env_temp is not None:
        if profile.env_temp < 15:
            p_score += 15
            fired.append(f"[L2] 低温{profile.env_temp}°C → 冬季补偿（优先工艺调温）")
        elif profile.env_temp > 35:
            p_score += 10
            fired.append(f"[L2] 高温{profile.env_temp}°C → 夏季降速")
    
    # ── 基础特征规则 ──
    if profile.batch_scope == "single" and profile.is_self_limiting:
        p_score += 30
        fired.append("[特征] 单缸+自限 → 工艺波动概率高（配方缺陷通常连续出现）")
    
    if profile.batch_scope in ("continuous", "batch"):
        f_score += 35
        fired.append("[特征] 连续/批次性 → 系统性问题高度指向配方缺陷")
    
    if profile.new_formula or profile.test_phase == "pilot":
        f_score += 25
        fired.append("[特征] 新配方/试产 → 配方本身未定型")
    
    if profile.batch_changed:
        f_score += 20
        fired.append("[特征] 近期换原料批次 → 配方适配性问题")
    
    # ── 配方vs工艺互补规则匹配 ──
    for scene, primary, alternative, forbidden in FORMULA_PROCESS_RULES:
        # 检查这一规则是否与当前问题相关
        relevance = _match_scene(scene, fm, profile)
        if relevance > 0:
            if "配方" in primary and "优先" in primary:
                f_score += relevance * 2
            elif "工艺" in primary and "优先" in primary:
                p_score += relevance * 2
            fired.append(f"[互补规则] {scene} → {primary}")
    
    return f_score, p_score, fired


def _match_scene(scene: str, failure_mode: str, profile: DiagnosticProfile) -> int:
    """判断互补规则场景与当前问题的关联度 0~3"""
    score = 0
    scene_kws = scene.lower().split()
    fm = failure_mode.lower()
    
    # 粘度相关
    if any(k in fm for k in ["粘度", "稠", "稀", "变稀", "变稠"]):
        if "粘度" in scene or "流变" in scene:
            score = 3
    
    # 团聚/分散相关
    if any(k in fm for k in ["团聚", "分散", "沉降", "颗粒"]):
        if "团聚" in scene or "分散" in scene:
            score = 3
    
    # 开裂/脆性相关
    if any(k in fm for k in ["开裂", "脆", "断裂", "漏芯"]):
        if "脆" in scene or "开裂" in scene or "漏芯" in scene:
            score = 3
    
    # 季节/批次漂移相关
    if any(k in fm for k in ["季节", "冬季", "漂移", "批次", "不稳"]):
        if "冬季" in scene or "批次漂移" in scene:
            score = 3
    
    # 固化异常
    if any(k in fm for k in ["固化", "不固化", "凝胶", "结晶", "发粘"]):
        if "固化" in scene or "配比" in scene:
            score = 3
    
    # 小试放大差异
    if any(k in fm for k in ["放大", "小试", "中试", "量产"]):
        if "放大" in scene:
            score = 3
    
    # 原料批次相关
    if profile.batch_changed and "碳粉批次" in scene:
        score = 3
    
    return score


def decide_priority(profile: DiagnosticProfile) -> DecisionResult:
    """综合决策"""
    f, p, rules = evaluate_rules(profile)
    
    total = f + p
    if total == 0:
        primary, confidence = "undetermined", 0.0
    elif f > p:
        primary = "formula"
        confidence = (f - p) / total
    elif p > f:
        primary = "process"
        confidence = (p - f) / total
    else:
        primary = "both"
        confidence = 0.5
    
    # 构建推理文本
    reasoning = _build_reasoning(profile, f, p, rules)
    hypotheses = _build_hypotheses(profile, primary)
    checklist = _build_checklist(profile, primary)
    risk = _build_risk_warning(profile, primary, confidence)
    forbidden = _build_forbidden_actions(profile, primary)
    
    return DecisionResult(
        primary_direction=primary,
        confidence=confidence,
        formula_score=f,
        process_score=p,
        reasoning=reasoning,
        hypothesis_list=hypotheses,
        action_checklist=checklist,
        risk_warning=risk,
        forbidden_actions=forbidden,
    )


def _build_reasoning(profile: DiagnosticProfile, f: int, p: int, rules: list[str]) -> str:
    """推理链（包含L0-L7分析摘要）"""
    lines = []
    lines.append(f"产品: {profile.product_name or '未知'} / {profile.formula_name or '未知'}")
    lines.append(f"阶段: {profile.test_phase or '未知'} | 现象: {profile.failure_mode or '未知'}")
    lines.append(f"范围: {profile.batch_scope or '未知'}（{'自限' if profile.is_self_limiting else '非自限'}）")
    lines.append("")
    lines.append(f"【配方调整得分: {f}  vs  工艺调整得分: {p}】")
    lines.append("")
    
    if rules:
        lines.append("命中规则:")
        for r in rules:
            lines.append(f"  {r}")
    
    # L0 配方层面快速检查
    fm = profile.failure_mode
    for key, rule in L0_RULES.items():
        if key in fm:
            lines.append(f"\n[L0-配方层参考] {key}: 计算/排查建议 → {rule['check']}")
    
    return "\n".join(lines)


def _build_hypotheses(profile: DiagnosticProfile, primary: str) -> list[str]:
    """基于L0-L7生成假设列表 — 始终包含框外根因的逃逸口"""
    hypotheses = []
    fm = profile.failure_mode
    
    # L0 配方假设
    if primary == "formula":
        hypotheses.append(f"L0配方层: 配比偏移或关键组分比例不当（与{fm}相关）")
        if profile.batch_changed:
            hypotheses.append("L0原料批次: 更换批次后关键指标（EEW/AHEW/包覆率/羟值）超出兼容窗口")
        hypotheses.append("L0配伍风险: 体系组分相容性劣化 → 建议复核加料顺序和分散策略")
    elif primary == "process":
        hypotheses.append("L2-L3工艺层: 温度/剪切/真空参数偏离工艺窗口")
        if profile.is_self_limiting:
            hypotheses.append("L1-L3瞬时波动: 温控过冲/加料波动/短时真空不足（已恢复）")
        hypotheses.append("L5放大效应: 釜容积增大导致蓄热/流场重构 → 需验证工艺窗口")
    
    # L4 材料反应假设
    if any(k in fm for k in ["固化", "凝胶", "交联", "接枝"]):
        hypotheses.append("L4材料反应: 接枝率/固化度偏离目标 → 检测DSC和红外确认反应程度")
    
    # L6 粒度/分散假设
    if any(k in fm for k in ["气泡", "团聚", "颗粒", "分散"]):
        hypotheses.append("L6分散/雾化: 真空脱泡效率/分散剂用量/固含量窗口 → 评估润湿状态")
    
    # L1 用户非标工况假设
    if primary == "process" and profile.env_changed:
        hypotheses.append("L1非标工况: 用户非标操作/设备私改/加热不规范 → 先排除再查工艺")
    
    # ★ 框外根因逃逸口（必须存在 — gate_out_of_framework_escape 依赖此项）
    hypotheses.append("其他根因（框架外）: L0-L7之外的未覆盖根因 → 建议工程师排查：原料供应商工艺变更/设备交叉污染/质检取样偏差/环境交叉影响")
    
    return hypotheses


def _build_checklist(profile: DiagnosticProfile, primary: str) -> list[str]:
    """生成行动清单（按优先级排序）"""
    checklist = []
    fm = profile.failure_mode
    p = profile
    
    # L1 用户非标工况优先排除（90%玄学问题源）
    if p.env_changed or primary == "process":
        checklist.append("【L1强制前置】排除用户非标：设备私改？加热规范？环境温度/湿度？旧料混用？")
    
    if primary == "formula":
        # L0 配方层核心操作
        checklist.append("【L0-配方核心】① 计算r_actual = (Q_B·ρ_B/AHEW)/(Q_A·ρ_A/EEW)；|Δr|>5%则配比偏移")
        checklist.append("【L0-配方核心】② 核对原料批次检测报告（EEW/AHEW/包覆率/羟值/BET）")
        
        if "粘度" in fm or "稀" in fm or "稠" in fm:
            checklist.append("【L0-流变】③ 校验固含量、填料批次的BET、分散剂用量")
        if "固化" in fm or "脆" in fm or "软" in fm:
            checklist.append("【L0-交联】③ 检测固化度（DSC测Tg或邵尔硬度）；确认固化剂活性")
        if "气泡" in fm or "漏芯" in fm:
            checklist.append("【L0-相容】③ 复核组分相容性矩阵（新批次原料是否改变溶解度参数）")
        
        checklist.append("【L5-放大验证】④ 做50~200mL小试重现问题，确认根因后再上线调整")
        checklist.append("【L3/L4】⑤ 评估调整对其他参数的耦合影响（单一调R值→脆性漂移风险）")
        
    elif primary == "process":
        # L2-L5 工艺层操作
        checklist.append("【L2-环境】① 确认当前环境温度/湿度是否在工艺窗口内")
        checklist.append("【L3/L5-流场】② 调取关键工艺参数：温度曲线/搅拌转速和电流/真空度/停留时间")
        
        if profile.is_self_limiting:
            checklist.append("【L3-自限特征】③ 拉取异常前后30min工艺曲线，定位瞬时波动点（温控过冲/搅拌失速/真空破空）")
        if profile.batch_scope in ("continuous", "batch"):
            checklist.append("【L5-放大④】检查放大准则执行：桨尖线速度是否在4~6m/s（量产）？单位体积功耗是否恒定？")
        
        checklist.append("【L4-材料保护】⑤ 验证接枝/包覆未受热降解（温度未超上限，剪切未超阈值）")
    
    # 共通收尾动作
    checklist.append("【知识沉淀】归档本次异常到案例库，供下次类似问题参考")
    
    return checklist


def _build_risk_warning(profile: DiagnosticProfile, primary: str, confidence: float) -> str:
    """风险提示"""
    warnings = []
    
    if confidence < 0.2:
        warnings.append("⚠️ 置信度偏低：建议补充更多信息后再做决策")
    
    if primary == "formula" and profile.test_phase == "mass":
        warnings.append("⚠️ 量产阶段调配方风险高：必须小试验证 + 留样观察 + 分次递进调整")
    
    if primary == "process" and profile.is_self_limiting:
        warnings.append("⚠️ 自限性故障根因可能未消除：排查后留1~3批强化监测")
    
    if profile.batch_scope in ("continuous", "batch"):
        warnings.append("🔴 批次性异常：建议立即隔离问题批次并通知品质")
    
    return "\n".join(warnings) if warnings else "✅ 常规风险可控，按计划执行"


def _build_forbidden_actions(profile: DiagnosticProfile, primary: str) -> list[str]:
    """禁止红线检查 — 基于失败教训库"""
    forbidden = []
    
    # 通用红线
    forbidden.append("禁止：未经工程师确认直接修改量产配方")
    forbidden.append("禁止：跳过小试验证直接上线调整")
    
    # 针对配方优先的红线
    if primary == "formula":
        forbidden.append("禁止：单次调整稀释剂>5phr（风险：交联网络崩塌、Tg大降）")
        forbidden.append("禁止：单方面大幅上调/下调固化剂（风险：脆性漂移/漏芯）")
        forbidden.append("禁止：只改配比不改配套工艺窗口（必须同步刷新）")
    
    # 针对工艺优先的红线
    if primary == "process":
        forbidden.append("禁止：单变量暴力调参（需考虑因子交互：如温度×剪切）")
        forbidden.append("禁止：小试参数直接平移到量产釜（必须执行放大准则）")
        if any(k in profile.failure_mode for k in ["粘度", "稠", "稀"]):
            forbidden.append("禁止：用无限升温来解决粘度问题（需评估接枝热降解风险）")
    
    return forbidden


# ============================================================
# 第四部分：Prompt 构造（含完整L0-L7框架注入）
# ============================================================

def build_diagnosis_prompt(profile: DiagnosticProfile, retrieved_context: list[dict]) -> tuple[str, str]:
    """
    构建含L0-L7推理框架的诊断prompt。
    让豆包按照：L0配方→L1用户工况→L2环境→L3流场→L4材料→L5放大→L6粒度→L7敏感矩阵 逐层分析。
    """
    system = """你是「材料工艺三级工程师」，15年环氧树脂/PCM复合材料工程经验。

【你遵循的诊断框架 — 必须按以下八层逐层分析】

第0层（L0配方本体与比例耦合）：
- 先计算双组份配比的化学计量：EEW、AHEW、phr、r_actual、Δr
- 快速检查：配比偏移？固化剂失效？水分侵入？组分相容性劣化？
- 配方问题通常是系统性、连续的（非自限性）

第1层（L1用户非标工况）：
- 设备私改？加热不规范？低温施工？操作不稳？旧料混用？
- 化工企业90%的"玄学批次问题"出自这一层

第2层（L2外部环境耦合）：
- 温度：T_a<15°C触发冬季补偿策略
- 湿度：RH>75%碳粉吸湿→接枝水解→团聚/气泡
- 真空：相变点偏移、脱泡效率

第3层（L3流体流场）：
- 雷诺数Re = ρND²/μ判断流态
- 小试层流→量产湍流是常见故障源
- 桨尖线速度v_tip是敏感控制量

第4层（L4材料反应、相变、接枝）：
- 接枝率控制：G_actual = G_theo·η_graft·f(T)·f(γ̇)·f(t)
- DSC潜热/相变点/Tg偏移→指向反应不完全或材料降解

第5层（L5量产放大偏差）：
- 体积按立方增长 vs 换热面积按平方增长 → 蓄热悖论
- 等剪切速率 / 等桨尖线速度 / 等雷诺数 / 等P/V 四大放大准则

第6层（L6雾化粒度）：
- D_32、Span = (D_90 - D_10) / D_50
- Span>1.5为严重恶化

第7层（L7参数敏感性矩阵）：
- S1: v_tip、固化剂纯度、配比R
- S2: 物料T、放大倍数、ΔP、BET、偶联剂用量
- S3: T_a、RH、真空度、PCM含量
- S4: Re、P/V

【三层式运行架构】
第一阶段（接报→多假设）：收到异常后按L0→L7建立2~4个假设+置信度
第二阶段（过滤→锁定）：工程师验证/否决→锁定根因+因果链
第三阶段（行动→闭环）：输出确定性方案→执行→24~48h验证→知识沉淀

【输出格式 — 严格按此结构】
一、L0-L7分层排查结论（每层一句话：正常/可疑/异常）
二、锁定根因（含L层级归属）
三、配方vs工艺决策（选一：配方优先/工艺优先/双向调整）
四、行动清单（按L1→L7优先级排列3~5步）
五、禁止红线（历史教训拦截）
六、验证方案（明确测量项和合格阈值）

【禁止红线（历史教训，绝对不可违反）】
- 一次性加稀释剂>5phr
- 小试参数直接平移到量产
- 单方面暴力调固化剂
- 高RH车间不预烘干
- 冬季按体积比不按当量比调泵

【语言风格】
简洁、定量化、可操作。每条结论必须说明依据的L层级和关键参数。"""

    # 构建用户消息
    user_lines = []
    user_lines.append("═"*50)
    user_lines.append("【诊断画像 - 已确认信息】")
    user_lines.append("═"*50)
    
    fields = [
        ("产品名称", profile.product_name), ("配方", profile.formula_name),
        ("测试环节", profile.test_phase), ("失效现象", profile.failure_mode),
        ("影响范围", profile.batch_scope), ("自限性", "是" if profile.is_self_limiting else "否"),
        ("原料批次变更", "是" if profile.batch_changed else "否"),
        ("环境变化", "是" if profile.env_changed else "否"),
        ("新配方", "是" if profile.new_formula else "否"),
    ]
    
    for label, val in fields:
        if val and val != "否":
            user_lines.append(f"  · {label}: {val}")
    
    if profile.r_actual is not None:
        user_lines.append(f"  · 实际配比r_actual: {profile.r_actual:.3f} (偏移{profile.r_actual*100-100:+.1f}%)")
    if profile.env_temp is not None:
        user_lines.append(f"  · 环境温度: {profile.env_temp}°C")
    if profile.env_rh is not None:
        user_lines.append(f"  · 环境湿度: {profile.env_rh}%")
    
    if retrieved_context:
        user_lines.append("")
        user_lines.append("═"*50)
        user_lines.append("【知识库检索 - L0-L7参考】")
        user_lines.append("═"*50)
        for i, ctx in enumerate(retrieved_context[:5], 1):
            content = ctx.get("content", "")[:200]
            source = ctx.get("source_type", "知识库")
            user_lines.append(f"[{i}] {source}: {content}")
    
    user_lines.append("")
    user_lines.append("═"*50)
    user_lines.append("请按L0→L7八层逐层分析后，输出六段式诊断报告。")
    
    return system, "\n".join(user_lines)


# ============================================================
# 第五部分：多轮追问引擎（保留v1优势）
# ============================================================

@dataclass
class DiagnosticSession:
    """诊断会话状态机"""
    session_id: str = ""
    profile: DiagnosticProfile = field(default_factory=DiagnosticProfile)
    question_history: list[dict] = field(default_factory=list)
    current_phase: int = 1
    turn_count: int = 0
    decision_result: Optional[DecisionResult] = None
    created_at: float = 0.0
    _pending_field: str = ""
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"DX{int(time.time())}"
        if not self.created_at:
            self.created_at = time.time()
    
    def generate_next_question(self) -> Optional[str]:
        """动态追问：根据已收集信息决定下一个问题"""
        p = self.profile
        q = []
        
        # Phase 1: 核心信息
        if not p.product_name:
            q.append(("请告诉我是哪个产品（产品名称或编号）？", "product_name", 1))
        elif not p.formula_name:
            q.append(("这次是哪个配方在用？（名称/编号/类别）", "formula_name", 1))
        elif not p.test_phase:
            q.append(("产品处于哪个阶段？\n1-新配方试产/中试  2-量产  3-客户反馈", "test_phase", 1))
        elif not p.failure_mode:
            q.append(("偏差现象？（一句话，如'结晶'/'粘度升高'/'固化不良'/'气泡'/'漏芯'）", "failure_mode", 1))
        elif not p.batch_scope:
            q.append(("问题影响范围？\n1-单缸/单批  2-连续多缸  3-整批物料  4-批次间突变", "batch_scope", 1))
        elif not p.is_self_limiting and p.batch_scope and not any(h["field"] == "is_self_limiting" for h in self.question_history):
            q.append(("后续批次还有同样问题吗？\n1-后续恢复正常  2-仍出现  3-不确认", "is_self_limiting", 1))
        # Phase 2: 辅助
        elif not p.batch_changed:
            q.append(("有没有换过原料批次或供应商？\n1-没有  2-换了批次  3-换了供应商", "batch_changed", 2))
        elif not p.new_formula:
            q.append(("这个配方是新开发/首次试产吗？\n1-是  2-量产配方  3-微调过的老配方", "new_formula", 2))
        
        if q:
            text, field, phase = q[0]
            self._pending_field = field
            self.current_phase = phase
            filled = int(p.completeness() * 5)
            progress = f"\n📋 基本信息 ({filled}/5)" if phase == 1 else "\n🔍 深度追问"
            return f"【AI·第{self.turn_count+1}问】{text}{progress}"
        
        # 信息充分 → 可以决策
        self.current_phase = 3
        self._pending_field = ""
        return None
    
    def record_answer(self, field: str, answer: str):
        """记录回答，更新画像"""
        self.turn_count += 1
        self.question_history.append({"field": field, "answer": answer})
        p = self.profile
        
        if field == "product_name": p.product_name = answer.strip()
        elif field == "formula_name": p.formula_name = answer.strip()
        elif field == "test_phase": p.test_phase = self._norm_phase(answer)
        elif field == "failure_mode": p.failure_mode = answer.strip()
        elif field == "batch_scope": p.batch_scope = self._norm_batch(answer)
        elif field == "is_self_limiting":
            p.is_self_limiting = answer.strip() in ("1", "后续恢复正常", "是", "yes", "正常")
        elif field == "batch_changed":
            p.batch_changed = any(k in answer for k in ["2", "3", "换了"])
        elif field == "new_formula":
            p.new_formula = "1" in answer or "新" in answer
    
    @staticmethod
    def _norm_phase(text: str) -> str:
        t = text.strip()
        if t in ("1", "新配方试产", "试产", "小批", "中试"): return "pilot"
        if t in ("2", "量产"): return "mass"
        if t in ("3", "客户反馈"): return "customer"
        return t
    
    @staticmethod
    def _norm_batch(text: str) -> str:
        t = text.strip()
        if t in ("1", "单缸", "单批"): return "single"
        if t in ("2", "连续多缸", "连续"): return "continuous"
        if t in ("3", "4", "整批物料", "批次"): return "batch"
        return t
    
    def finalize(self) -> DecisionResult:
        result = decide_priority(self.profile)
        self.decision_result = result
        return result


# ============================================================
# 第六部分：成本-性能权衡 + DOE交互 + 测试映射检索
# ============================================================

def get_cost_performance_plans(scenario: str) -> list[dict]:
    """
    根据异常场景检索成本-性能权衡方案列表（按优先级排序）
    返回值: [{candidate_plan, performance_gain, cost_change, verification_cycle, risk_level, priority_rank}, ...]
    """
    try:
        import db_sqlite as _db
        return _db.list_cost_performance_plans(scenario)
    except Exception:
        return []


def get_test_process_mapping(test_indicator: str) -> list[dict]:
    """
    根据测试指标检索对应的产线工序映射
    返回值: [{test_indicator, priority_process, secondary_process, exclusion_note}, ...]
    """
    try:
        import db_sqlite as _db
        return _db.list_test_process_mappings(test_indicator)
    except Exception:
        return []


def get_doe_interactions(profile: DiagnosticProfile) -> list[dict]:
    """
    检索与当前问题相关的强交互因子对
    返回值: [{factor_a, factor_b, interaction_level, description}, ...]
    """
    try:
        import db_sqlite as _db
        all_pairs = _db.list_doe_interactions()
    except Exception:
        return []
    
    # 根据 failure_mode 过滤出相关的交互对
    fm = profile.failure_mode.lower()
    relevant = []
    keywords_map = {
        "温度": ["温度", "温控", "加热", "冷却", "蓄热", "散热"],
        "剪切": ["搅拌", "分散", "剪切", "团聚", "桨"],
        "填料": ["填料", "碳粉", "加量", "降本"],
        "湿度": ["湿", "潮", "水", "气泡"],
        "配比": ["配比", "固化", "发粘", "不固化"],
    }
    
    for pair in all_pairs:
        for k, kws in keywords_map.items():
            if k in pair["factor_a"] or k in pair["factor_b"]:
                for kw in kws:
                    if kw in fm:
                        relevant.append(pair)
                        break
                break
    
    return relevant


def build_cost_section(plans: list[dict]) -> str:
    """
    构造成本-性能权衡多方案文本块，供报告使用
    """
    if not plans:
        return ""
    
    scenario = plans[0].get("scenario", "当前异常")
    lines = []
    lines.append("")
    lines.append("║" + "═"*48)
    lines.append(f"║ 【成本-性能权衡方案排序 — {scenario}】")
    lines.append("║" + "═"*48)
    lines.append("║ 优先级 | 方案 | 性能收益 | 成本变化 | 验证周期 | 风险")
    lines.append("║" + "─"*48)
    
    for i, plan in enumerate(plans[:5], 1):
        lines.append(
            f"║  P{i}   | {plan.get('candidate_plan','')} "
            f"| 收益:{plan.get('performance_gain','')} "
            f"| 成本:{plan.get('cost_change','')} "
            f"| 验证:{plan.get('verification_cycle','')} "
            f"| ⚠️{plan.get('risk_level','')}"
        )
    
    lines.append("║" + "─"*48)
    lines.append("║ 💡 建议：优先选P1快速验证；如无效再升级P2/P3")
    lines.append("")
    
    return "\n".join(lines)


def build_doe_section(interactions: list[dict]) -> str:
    """
    构造DOE交互效应警告文本块
    """
    if not interactions:
        return ""
    
    lines = []
    lines.append("")
    lines.append("║" + "═"*48)
    lines.append("║ 【⚠️ DOE强交互因子警告 — 禁止单变量暴力调参】")
    lines.append("║" + "═"*48)
    
    for pair in interactions:
        level_icon = "🔴强" if "强" in pair.get("interaction_level", "") else "🟡中"
        lines.append(f"║ {level_icon}交互: {pair['factor_a']} × {pair['factor_b']}")
        lines.append(f"║   → {pair.get('description', '')}")
    
    lines.append("║" + "─"*48)
    lines.append("║ ⚡ 操作要求：调整其中任一参数时，必须联动评估其交互因子")
    lines.append("")
    
    return "\n".join(lines)


# ============================================================
# 第六部分（新增）：配方硬校验闸门接入
#  ============================================================

def validate_diagnosis_suggestion(raw_text: str, profile: DiagnosticProfile) -> dict:
    """
    对诊断模式下的建议做硬校验闸门（轻量版，向后兼容）：
    1. 如果建议配方调整，调用 formula_validator
    2. 如果是工艺参数建议，做量纲/范围检查
    3. 返回双通道结果（candidate/executable/rejected）
    
    注意：全量校验请用 formula_validator.validate_diagnosis_comprehensive()
    """
    from formula_validator import (
        parse_formula_from_text, validate_formula, dual_channel_output,
        gate_process_safety_window, gate_failure_lesson_intercept,
    )
    
    violations = []
    
    # 工艺安全窗口校验（新增）
    violations.extend(gate_process_safety_window(profile, raw_text))
    
    # 失败教训红线校验（新增，代码级硬拦截）
    violations.extend(gate_failure_lesson_intercept(profile, raw_text))
    
    # 尝试从文本中解析配方（profile没有process_temp，用env_temp或默认25°C）
    _proc_temp = getattr(profile, 'env_temp', None) or 25
    _shear = getattr(profile, 'shear_rate', None) or 0
    recipe = parse_formula_from_text(
        raw_text,
        default_temp=_proc_temp,
        default_shear=_shear,
    )
    
    if len(recipe.components) >= 2:
        # 包含配方组分 → 全部闸门校验
        result = dual_channel_output(recipe, llm_rationale=raw_text)
        # 合并新增violations
        for v in violations:
            result["violations"].append({
                "gate": v.gate, "severity": v.severity,
                "rule_id": v.rule_id, "message": v.message, "evidence": v.evidence
            })
        # 重新计算channel
        has_hard = any(v.get("severity") == "HARD_VETO" or v.severity == "HARD_VETO" for v in violations)
        if has_hard and result["channel"] != "rejected":
            result["channel"] = "rejected"
            result["executable"] = False
        return result
    
    # 非配方输出，仅做工艺参数合理性检查
    if violations:
        hard = [v for v in violations if v.severity == "HARD_VETO"]
        warns = [v for v in violations if v.severity == "WARNING"]
        channel = "rejected" if hard else ("candidate" if warns else "executable")
        return {
            "channel": channel,
            "executable": channel == "executable",
            "violations": [{"gate": v.gate, "severity": v.severity, "rule_id": v.rule_id,
                           "message": v.message, "evidence": v.evidence} for v in violations],
            "engineer_text": "\n".join(f"{'🚫' if v.severity=='HARD_VETO' else '⚠️'} [{v.rule_id}] {v.message}" for v in violations),
            "audit_log": [f"[{channel.upper()}] {v.rule_id}: {v.message}" for v in violations],
        }
    
    return {
        "channel": "executable",
        "executable": True,
        "passed": True,
        "violations": [],
        "engineer_text": "✅ 参数建议，已做量纲快速检查",
        "audit_log": ["[PASS] 非配方输出，跳过配方闸门"],
    }


# ============================================================
# 便捷入口 + 报告增强
# ============================================================

def create_session() -> DiagnosticSession:
    return DiagnosticSession()


def build_enhanced_report(profile: DiagnosticProfile, base_report: str) -> str:
    """
    在LLM输出的基础上，自动追加：
    - 成本-性能多方案权衡（附录Q）
    - DOE强交互因子警告（附录N）
    - 测试指标→产线工序反向映射（附录I）
    """
    sections = [base_report]
    
    # 1. 成本-性能权衡
    fm = profile.failure_mode.lower()
    scenario_keywords = {
        "团聚": "粉体团聚整改",
        "分散不良": "粉体团聚整改",
        "粘度过高": "体系粘度过高",
        "粘度": "体系粘过高",
        "固化不良": "固化不良",
        "不固化": "固化不良",
        "气泡": "气泡超标",
        "漏芯": "气泡超标",
    }
    matched_scenario = ""
    for kw, sc in scenario_keywords.items():
        if kw in fm:
            matched_scenario = sc
            break
    
    if matched_scenario:
        plans = get_cost_performance_plans(matched_scenario)
        cost_section = build_cost_section(plans)
        if cost_section:
            sections.append(cost_section)
    
    # 2. DOE交互效应警告
    interactions = get_doe_interactions(profile)
    doe_section = build_doe_section(interactions)
    if doe_section:
        sections.append(doe_section)
    
    # 3. 测试→工序映射（提示工程师需要做的实验室验证）
    test_kws = {
        "DSC": "DSC相变峰偏移>±3°C",
        "潜热": "潜热ΔH低于标称90%",
        "Tg": "Tg明显漂移",
        "剪切强度": "剪切强度断崖下跌",
        "粒度": "粒度Span>1.5",
        "雾化": "粒度Span>1.5",
        "凝胶时间": "凝胶时间t_gel严重偏离",
    }
    for tk, indicator in test_kws.items():
        if tk in fm:
            mappings = get_test_process_mapping(tk)
            if mappings:
                lines = []
                lines.append("")
                lines.append("║" + "═"*48)
                lines.append(f"║ 【测试指标→产线工序反向映射】")
                lines.append("║" + "═"*48)
                for m in mappings[:3]:
                    lines.append(f"║ 测试: {m['test_indicator']}")
                    lines.append(f"║ → 优先排查: {m['priority_process']}")
                    lines.append(f"║ → 次要排查: {m['secondary_process']}")
                    lines.append(f"║ ⚠️ 排除测试误差: {m['exclusion_note']}")
                lines.append("")
                sections.append("\n".join(lines))
            break
    
    return "\n".join(sections)
