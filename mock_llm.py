"""模拟LLM客户端（演示版，不调用API，靠模板+检索驱动推理）
生产环境换成 doubao_api.DoubaoAPI 即可，接口签名一致"""


class MockLLM:
    """伪大模型：用检索 + 模板填充替代真实推理
    接口与 DoubaoAPI 一致，便于生产无缝切换"""

    def chat_completion(self, system_prompt: str, user_content: str,
                        temperature: float = 0.1, max_tokens: int = 2048) -> str:
        # 在演示版中，提示词调用主要发生在4个环节：
        # 这里用关键词匹配模拟大模型输出
        if "数据抽取专家" in system_prompt or "抽取" in system_prompt:
            return self._extract(user_content)
        elif "排查专家" in system_prompt or "假设" in system_prompt:
            return self._hypothesis(user_content)
        elif "追问" in system_prompt or "调查工程师" in system_prompt:
            return self._question(user_content)
        elif "改善文档" in system_prompt or "方案工程师" in system_prompt:
            return self._solution(user_content)
        return {"status": "noop"}

    def get_embedding(self, text: str):
        from local_embedding import embed_text
        return embed_text(text)

    # ========== 各提示词场景的模拟逻辑 ==========

    def _extract(self, prompt: str) -> str:
        """模拟一点课抽取：从自由文本提取规则"""
        import re, json
        lines = prompt.strip().split("\n")
        # 取含有中文的行作为触发条件和经验
        chinese_lines = [l for l in lines if len(l) > 10 and not l.startswith("{")]
        trigger = chinese_lines[0][:100] if chinese_lines else ""
        content = chinese_lines[1][:200] if len(chinese_lines) > 1 else trigger
        return json.dumps({
            "trigger_condition": trigger,
            "exp_content": content,
            "source_suggestion": "process"
        }, ensure_ascii=False)

    def _hypothesis(self, prompt: str) -> str:
        """模拟头脑风暴：输出假设JSON"""
        import json
        # 从上下文检索结果中提取假设文本
        import re, json
        # 用简单启发：如果有被召回的case根因，直接用
        hypo_templates = self._extract_hypothesis_text(prompt)
        if not hypo_templates:
            hypo_templates = [
                "原料批次差异导致关键参数偏移",
                "工艺条件/参数偏离小试最优区间",
                "设备传热传质限制在大放大工艺中暴露",
                "环境因素（温湿度/季节）影响物料性质",
            ]
        hypo_list = hypo_templates[:5]
        return json.dumps({
            "hypothesis_list": hypo_list,
            "knowledge_coverage": "high" if len(hypo_list) >= 3 else "medium"
        }, ensure_ascii=False)

    def _question(self, prompt: str) -> str:
        """模拟动态追问"""
        import json
        return json.dumps({
            "root_cause_confirmed": False,
            "question_list": [
                {"question": "本次异常批次的外观/颜色/气味是否与合格批次存在可感知差异？",
                 "target_hypothesis": "原料批次差异",
                 "options": ["A.明显异常", "B.轻微差异", "C.无差异", "D.未注意"]},
                {"question": "设备运行记录（温度/压力/转速）是否存在异常跳动或趋势性漂移？",
                 "target_hypothesis": "工艺参数偏离",
                 "options": ["A.明显跳动", "B.轻微漂移", "C.完全正常", "D.未记录"]},
                {"question": "是否为小试放大至中试时出现的差异（重现性与规模相关）？",
                 "target_hypothesis": "放大工艺设备限制",
                 "options": ["A.放大才出现", "B.小试就偶发", "C.规模无关联", "D.不确定"]},
            ]
        }, ensure_ascii=False)

    def _solution(self, prompt: str) -> str:
        """模拟改善方案卡片"""
        import re, json
        root_m = re.search(r"确认根因:\s*(.+?)(?:\n|$)", prompt)
        root = root_m.group(1).strip() if root_m else "工艺参数偏移"
        sid_m = re.search(r"方案卡片ID:\s*(IM-\S+)", prompt)
        sid = sid_m.group(1) if sid_m else "IM-TEMPLATE"

        return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# {sid} 改善方案卡
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 根因
{root}

证据链：
- 排查过程多轮锁定为该根因
- 有知识库历史案例支撑

## 改善方案

A.【短期】立即校验当日批次原料并记录；复核实际工艺参数与SOP一致性
- 成本影响：可忽略
- 执行难度：低
- 责任岗位：工艺工程师
- 完成时限：7天内

B.【中期】更新SOP，增加原料批次与工艺参数联动调整规则
- 成本影响：可忽略
- 执行难度：低
- 责任岗位：工艺工程师
- 完成时限：4周内

C.【长期】建立原料入厂全检数据驱动的工艺自动匹配系统
- 成本影响：一次性改造约5万元
- 执行难度：中
- 责任岗位：设备/IT
- 完成时限：3个月内

## 预期效果
基于历史经验，改善后指标恢复至理论预期±5%范围内，连续5批稳定

## 风险提示
短期调整可能引入新变异，建议改善后持续监控3个批次

## 执行跟踪
- 方案启动日期：___月___日
- 第1次效果验证（+7天）：自动提醒
- 第2次效果验证（+14天）：自动提醒
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    def _extract_hypothesis_text(self, user_content: str) -> list:
        """从召回上下文中提取可能的假设"""
        import re
        hypos = []
        # 搜索历史案例根因中的关键词作为假设
        for line in user_content.split("\n"):
            # 寻找含根因描述的行
            if "根因" in line and len(line) > 15:
                cleaned = line.replace("根因:", "").replace("根因：", "").strip()
                if cleaned and len(cleaned) < 100:
                    hypos.append(cleaned)
            # 匹配知识库中的经验片段
            elif "内容:" in line and any(k in line for k in ["料层", "温度", "真空", "氧化", "孔隙", "分散", "过冷", "分相", "膨胀"]):
                cleaned = line.split("内容:")[-1].strip()
                if cleaned and len(cleaned) < 80:
                    hypos.append(cleaned)
        return hypos[:5]


# 单称：与 doubao_api.get_doubao() 同接口
_mock_llm = None

def get_llm():
    global _mock_llm
    if _mock_llm is None:
        _mock_llm = MockLLM()
    return _mock_llm
