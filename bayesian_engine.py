"""贝叶斯置信度计算引擎——全部概率运算在Python代码中完成，大模型不参与概率计算"""

import logging
from typing import Optional

import config
import database as db

logger = logging.getLogger(__name__)


def calculate_prior(hypothesis_list: list, retrieved_rules: list) -> dict:
    """
    为每个假设计算先验概率（基于检索到的经验规则置信度加权）

    算法：
    1. 每个假设基础权重 = 0.2
    2. 遍历召回的规则，如果规则文本与假设存在关键词重叠或因果关联，权重 += rule.confidence
    3. 未匹配任何规则的假设保持基础权重（标记为【未验证猜想】的理论依据）
    4. 所有权重归一化为概率分布
    """
    hypo_weights = {}

    for hypo in hypothesis_list:
        weight = 0.2  # 基础先验

        for rule in retrieved_rules:
            rule_conf = rule.get("confidence", config.CONF_INIT)
            rule_content = rule.get("content", "")

            # 匹配规则：规则内容出现在假设中，或假设关键词出现在规则中
            if _is_related(hypo, rule_content):
                weight += rule_conf

        hypo_weights[hypo] = weight

    # 归一化
    total = sum(hypo_weights.values())
    if total == 0:
        # 极其异常情况：均匀分布
        n = len(hypothesis_list)
        return {h: 1.0 / n for h in hypothesis_list}

    hypo_prob = {h: w / total for h, w in hypo_weights.items()}

    # 按概率降序排列
    sorted_hypo = dict(sorted(hypo_prob.items(), key=lambda x: x[1], reverse=True))
    return sorted_hypo


def _is_related(hypothesis: str, rule_content: str) -> bool:
    """判断假设与经验规则是否关联（关键词重叠检测）"""
    # 提取规则中的关键词（去停用词后的名词性片段）
    rule_keywords = _extract_keywords(rule_content)
    hypo_keywords = _extract_keywords(hypothesis)

    # 任意关键词匹配即视为关联
    for rk in rule_keywords:
        for hk in hypo_keywords:
            if rk in hk or hk in rk:
                return True
    return False


def _extract_keywords(text: str) -> list:
    """简易关键词提取：按停用词切分"""
    stopwords = {"的", "了", "和", "与", "或", "在", "是", "有",
                 "着", "被", "把", "从", "到", "对", "为", "以"}
    # 简单按字符片段提取（实际可用jieba分词增强）
    words = []
    buf = ""
    for ch in text:
        if ch in stopwords or ch in "，。、；：！？ \n\t":
            if buf:
                words.append(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        words.append(buf)
    return words


def update_confidence(rule_id: str, is_hit: bool, case_id: str = None) -> float:
    """
    更新经验规则置信度

    命中：conf += CONF_HIT_DELTA（默认+0.05）
    未命中：conf -= CONF_MISS_DELTA（默认-0.1）
    范围钳位在[CONF_MIN, CONF_MAX]

    同时写入confidence_history表用于审计
    """
    rule = db.get_rule_by_id(rule_id)
    if not rule:
        logger.warning(f"规则{rule_id}不存在，跳过更新")
        return 0.0

    old_conf = float(rule["confidence"])
    hit = int(rule["hit_count"])
    verify = int(rule["verify_count"])

    if is_hit:
        new_conf = old_conf + config.CONF_HIT_DELTA
        hit += 1
    else:
        new_conf = old_conf - config.CONF_MISS_DELTA

    # 钳位
    new_conf = max(config.CONF_MIN, min(config.CONF_MAX, new_conf))
    verify += 1

    # 写入经验规则表
    import database
    with database.db_cursor(commit=True) as cur:
        cur.execute("""
            UPDATE experience_rule
            SET confidence=%s, hit_count=%s, verify_count=%s, update_time=NOW()
            WHERE rule_id=%s
        """, (new_conf, hit, verify, rule_id))

        # 写入历史记录
        cur.execute("""
            INSERT INTO confidence_history (rule_id, case_id, old_confidence, new_confidence, update_reason)
            VALUES (%s, %s, %s, %s, %s)
        """, (rule_id, case_id, old_conf,
              new_conf, "hit" if is_hit else "miss"))

    logger.info(f"规则{rule_id}置信度更新: {old_conf:.3f} -> {new_conf:.3f} ({'命中' if is_hit else '排除'})")
    return new_conf


def batch_update_confidence(hypothesis_status: dict, case_id: str = None) -> dict:
    """
    批量更新置信度

    hypothesis_status格式: {
        "hypothesis_text": {"rule_id": "xxx", "status": "confirmed"/"eliminated"},
        ...
    }
    返回每个规则的最新置信度
    """
    results = {}
    for hypo_text, info in hypothesis_status.items():
        rule_id = info.get("rule_id")
        status = info.get("status")
        if not rule_id:
            continue
        is_hit = (status == "confirmed")
        results[rule_id] = update_confidence(rule_id, is_hit, case_id)
    return results


def reduce_confidence_by_time(rule_id: str, decay_factor: float = 0.95):
    """
    长时间未验证的置信度衰减
    （如果规则超过一定周期被验证次数不增加，触发衰减机制）
    """
    rule = db.get_rule_by_id(rule_id)
    if not rule:
        return 0.0

    verify_count = int(rule["verify_count"])
    update_time = rule.get("update_time")

    # 如果verify_count没有增长，降低置信率
    if verify_count > 0:
        old_conf = float(rule["confidence"])
        new_conf = old_conf * decay_factor
        new_conf = max(config.CONF_MIN, new_conf)

        import database
        with database.db_cursor(commit=True) as cur:
            cur.execute("""
                UPDATE experience_rule
                SET confidence=%s, update_time=NOW()
                WHERE rule_id=%s
            """, (new_conf, rule_id))

            cur.execute("""
                INSERT INTO confidence_history (rule_id, old_confidence, new_confidence, update_reason)
                VALUES (%s, %s, %s, 'decay')
            """, (rule_id, old_conf, new_conf))

        logger.info(f"规则{rule_id}置信度衰减: {old_conf:.3f} -> {new_conf:.3f}")
        return new_conf
    return float(rule["confidence"])


if __name__ == "__main__":
    # 单元测试：贝叶斯概率计算
    test_hypotheses = [
        "原料供应商夏季批次含水量偏高导致收率下降",
        "反应温度PID参数积分时间设置不当引起温控漂移",
        "催化剂活性降低未及时更换",
    ]
    test_rules = [
        {"content": "供应商夏季批次含水量高", "confidence": 0.75},
        {"content": "温度控制偏差大", "confidence": 0.60},
    ]
    priors = calculate_prior(test_hypotheses, test_rules)
    print("=== 假设先验概率 ===")
    for h, p in priors.items():
        print(f"  {p:.3f} | {h}")
