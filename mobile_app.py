"""
新材料工艺偏差排查AI — 移动端 (Flet)
兼容 Android / 鸿蒙 (HarmonyOS APK安装)
"""
import sys, os, json, asyncio, threading
sys.path.insert(0, os.path.dirname(__file__))

import flet as ft

# ============================================================
# 移动端 SQLite 路径：Flet打包后工作目录会变，需复制DB到可写路径
# ============================================================
def get_db_path():
    """获取移动端可用的SQLite路径（Android/Mobile安全写入）"""
    db_name = "pcm_knowledge.db"

    if sys.platform == "android":
        # Android: use CWD which Flutter sets to /data/data/<pkg>/files/
        cwd = os.getcwd()
        if os.access(cwd, os.W_OK):
            return os.path.join(cwd, db_name)
        # Fallback: try /data/local/tmp (always writable)
        tmp = "/data/local/tmp/material_ai"
        os.makedirs(tmp, exist_ok=True)
        return os.path.join(tmp, db_name)

    # Desktop / development
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, db_name)


def init_mobile_db():
    """确保移动端数据库已就绪"""
    db_path = get_db_path()

    # 同步给 db_sqlite 模块
    try:
        import db_sqlite as _db
        _db.DB_PATH = db_path
    except Exception:
        pass

    if os.path.exists(db_path):
        return db_path
    # 首次运行：直接创建表（不需要 import_pcm_sqlite 脚本）
    try:
        import db_sqlite as _db
        _db.init_db()
        # 注入核心知识（配方失效模式、决策规则、经验规则）
        try:
            _db.seed_default_knowledge()
        except Exception as e:
            print(f"知识库注入警告: {e}")
    except Exception as e:
        print(f"移动端DB初始化警告: {e}")
    return db_path


def load_saved_api_settings():
    """从SQLite加载已保存的API配置到config"""
    try:
        import db_sqlite as _db
        import config as _cfg
        _db.DB_PATH = get_db_path()
        saved_key = _db.load_setting("DOUBAO_API_KEY", "")
        saved_mid = _db.load_setting("DOUBAO_MODEL_ID", "")
        if saved_key:
            _cfg.DOUBAO_API_KEY = saved_key
        if saved_mid:
            _cfg.DOUBAO_MODEL_ID = saved_mid
    except Exception:
        pass


# ============================================================
# 业务逻辑 — 调用现有引擎
# ============================================================
def import_diagnosis_modules():
    """延迟导入诊断引擎（避免Flet打包时循环依赖）"""
    import config
    import db_sqlite as db
    from model_manager import get_model
    from bayesian_engine import calculate_prior as bayes_calc
    from auditor_engine import get_auditor
    
    # 覆盖DB路径
    db.DB_PATH = get_db_path()
    
    return config, db, get_model, bayes_calc, get_auditor


def _safe_extract_json(text):
    try:
        return json.loads(text)
    except:
        pass
    raw = text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    s = raw.find("{")
    e = raw.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(raw[s:e + 1])
        except:
            pass
    return {}


def run_diagnosis(problem_text: str, progress_cb=None) -> dict:
    """在移动端运行完整诊断（含审计），通过progress_cb回传进度"""
    def emit(stage, msg):
        if progress_cb:
            progress_cb({"stage": stage, "message": msg})
    
    config, db, get_model, bayes_calc, get_auditor = import_diagnosis_modules()
    
    # 安全校验：API Key 是否已配置
    if not config.DOUBAO_API_KEY or config.DOUBAO_API_KEY in ("YOUR_API_KEY_HERE", ""):
        emit("error", "❌ API Key 未配置。请检查 .env 文件或联系开发者。")
        return {"error": "API Key 未配置"}
    
    model = get_model()
    auditor = get_auditor()
    
    result = {
        "problem": problem_text, "eliminated": [],
        "hypotheses": [], "questions": [],
        "root_cause": "", "solution": "",
        "audit_scores": {}, "full_audit": {}
    }
    
    # Step1: RAG
    emit("rag", "🔍 知识库检索中...")
    retrieved = db.retrieve(problem_text, top_k=5)
    result["retrieved"] = retrieved
    emit("rag", f"📚 召回 {len(retrieved)} 条知识")
    
    # Step2: 假设生成
    emit("hypothesis", "🧠 AI头脑风暴生成假设...")
    prompt_sys = config.load_prompt("hypothesis")
    know_text = "\n".join(f"- {r['content'][:60]}" for r in retrieved[:2])
    hypo_resp = model.chat_completion(prompt_sys, f"偏差描述：{problem_text}\n相关知识：\n{know_text}", temperature=0.2)
    hypo_data = _safe_extract_json(hypo_resp)
    hypo_list_raw = hypo_data.get("hypothesis_list", [])
    
    # 案例推理
    hypo_from_cases = []
    for r in retrieved:
        if r["source_type"] == "case":
            rc = db.get_case_by_id(r["source_id"])
            if rc and rc.get("root_cause"):
                hypo_from_cases.append(rc["root_cause"])
    
    hypo_list = list(dict.fromkeys(hypo_from_cases + hypo_list_raw))[:5]
    emit("hypothesis", f"🎯 生成 {len(hypo_list)} 条假设")
    
    # Step3: 追问
    emit("questions", "❓ 生成精准追问...")
    prompt_sys_q = config.load_prompt("question")
    hypo_with_prob = bayes_calc(hypo_list, [{"content": r["content"], "confidence": 0.5} for r in retrieved if r["source_type"] == "rule"])
    result["hypotheses"] = hypo_with_prob
    
    prompt_user_q = f"问题：{problem_text}\n当前假设:\n" + "\n".join(f"  {i+1}. {h} ({p:.0%})" for i, (h, p) in enumerate(hypo_with_prob.items()))
    q_resp = model.chat_completion(prompt_sys_q, prompt_user_q, temperature=0.1)
    q_info = _safe_extract_json(q_resp)
    questions = q_info.get("question_list", [])
    result["questions"] = questions
    emit("questions", f"✅ 生成 {len(questions)} 道追问")
    
    # Step4: 贝叶斯更新 (简化版)
    emit("probability", "📊 置信度更新...")
    remaining = dict(hypo_with_prob)
    if len(remaining) > 2:
        sorted_h = sorted(remaining.items(), key=lambda x: x[1])
        eliminate = sorted_h[0][0]
        remaining.pop(eliminate, None)
        total = sum(remaining.values())
        remaining = {k: v / total for k, v in remaining.items()}
        result["eliminated"] = [eliminate]
    result["remaining"] = remaining
    
    # Step5: 锁定根因 + 方案
    root_cause = max(remaining, key=remaining.get) if remaining else "需更多排查"
    result["root_cause"] = root_cause
    emit("solution", "📋 生成改善方案...")
    
    prompt_sys_s = config.load_prompt("solution")
    sol_resp = model.chat_completion(prompt_sys_s, f"确认根因: {root_cause}", temperature=0.2, max_tokens=1024)
    result["solution"] = sol_resp
    emit("solution", "✅ 改善方案已生成")
    
    # Step6: 审计
    emit("audit", "🔍 独立审计Agent启动...")
    audit_data = {"problem": problem_text, "eliminated": result["eliminated"], 
                  "retrieved": result["retrieved"], "hypotheses": hypo_with_prob,
                  "questions": questions, "root_cause": root_cause, "solution": sol_resp,
                  "remaining_hypotheses": remaining}
    
    rag_audit = auditor.audit_rag(problem_text, retrieved)
    full_audit_score = rag_audit.get("recall_score", "?")
    emit("audit", f"📊 召回审计: {full_audit_score}/10")
    
    sol_audit = auditor.audit_solution(root_cause, sol_resp)
    sol_score = sol_audit.get("targeting_score", "?")
    emit("audit", f"📊 方案审计: {sol_score}/10")
    
    result["audit_scores"] = {
        "召回质量": rag_audit.get("recall_score", "?"),
        "方案质量": sol_audit.get("targeting_score", "?"),
    }
    result["full_audit"] = rag_audit
    emit("audit", "✅ 全流程完成！")
    
    return result


# ============================================================
# Flet 移动端 UI
# ============================================================
def build_main_ui(page: ft.Page):
    page.title = "材料AI诊断"
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 16
    page.bgcolor = ft.Colors.GREY_50

    # 语音录音器 + 播放器（Flet 1.0.x 兼容初始化）
    audio_recorder = None
    player = None
    try:
        audio_recorder = ft.AudioRecorder(
            audio_encoder=ft.AudioEncoder.OPUS,
            on_state_changed=lambda e: print(f"Recorder: {e.data}"),
        )
        player = ft.Audio(autoplay=False)
        # 优先新版 services，回退旧版 overlay
        try:
            page.services.append(audio_recorder)
            page.services.append(player)
        except (AttributeError, TypeError):
            try:
                page.overlay.append(audio_recorder)
                page.overlay.append(player)
            except Exception:
                audio_recorder = None
                player = None
    except (AttributeError, TypeError, NameError):
        audio_recorder = None
        player = None

    # 如果是窄屏，设置合适宽度
    if page.width < 600:
        page.window.width = page.width
    
    # ====== 导航栏 ======
    nav_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.SEARCH, label="排查"),
            ft.NavigationBarDestination(icon=ft.Icons.ANALYTICS, label="结果"),
            ft.NavigationBarDestination(icon=ft.Icons.SHIELD, label="审计"),
            ft.NavigationBarDestination(icon=ft.Icons.BOOK, label="知识库"),
            ft.NavigationBarDestination(icon=ft.Icons.SETTINGS, label="设置"),
        ],
        on_change=lambda e: on_nav_change(e.control.selected_index)
    )
    
    # ====== 内容页容器 ======
    content_area = ft.Column([], scroll=ft.ScrollMode.AUTO, expand=True)
    
    def on_nav_change(index):
        content_area.controls.clear()
        if index == 0:
            content_area.controls.append(build_diagnosis_page())
        elif index == 1:
            content_area.controls.append(build_results_page())
        elif index == 2:
            content_area.controls.append(build_audit_page())
        elif index == 3:
            content_area.controls.append(build_knowledge_page())
        elif index == 4:
            content_area.controls.append(build_settings_page())
        page.update()
    
    # ====== 排查页 ======
    problem_input = ft.TextField(
        label="描述工艺偏差现象",
        hint_text="例：XX材料小试OK，放大后性能下降...",
        multiline=True, min_lines=3, max_lines=6,
        filled=True,
    )
    
    progress_ring = ft.ProgressRing(visible=False)
    progress_text = ft.Text("", size=14, color=ft.Colors.GREY_700)
    result_container = ft.Column([], spacing=12, scroll=ft.ScrollMode.AUTO)
    diagnosis_result = {}  # 保存最新结果
    
    # ══ AI主导多轮诊断对话状态 ══
    diag_session = [None]        # DiagnosticSession 实例
    chat_mode = [False]          # 是否已进入对话模式
    chat_messages = []           # (role, text) 消息列表
    

    def add_message(role: str, text: str):
        """往对话区添加一条气泡消息"""
        chat_messages.append((role, text))
        if role == "user":
            result_container.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text("👤 你", size=11, color=ft.Colors.GREY_500),
                        ft.Text(text, size=14, color=ft.Colors.WHITE),
                    ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.END),
                    bgcolor=ft.Colors.BLUE_500,
                    border_radius=border_radius_rtl(12),
                    padding=10, width=280,
                    margin=ft.margin.only(left=40, top=4, bottom=4),
                    alignment=ft.alignment.center_right,
                )
            )
        elif role == "ai":
            result_container.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text("🔬 AI诊断助理", size=11, color=ft.Colors.GREY_500),
                        ft.Text(text, size=14),
                    ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.START),
                    bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.ON_SURFACE),
                    border_radius=border_radius_ltr(12),
                    padding=10, width=320,
                    margin=ft.margin.only(right=40, top=4, bottom=4),
                )
            )
        elif role == "system":
            result_container.controls.append(
                ft.Container(
                    content=ft.Text(text, size=12, color=ft.Colors.GREY_600, italic=True),
                    padding=8,
                    alignment=ft.alignment.center,
                )
            )
        page.update()
    

    def on_run_diagnosis(e):
        """主按钮点击：首次=开始排查, 后续=发送回答"""
        if not chat_mode[0]:
            _start_diagnosis_chat()
        else:
            _send_chat_answer()
    

    def _start_diagnosis_chat():
        """首次点击：用输入的文本启动多轮对话"""
        initial_text = problem_input.value.strip()
        if not initial_text:
            page.show_snack_bar(ft.SnackBar(content=ft.Text("请先输入偏差描述")))
            return
        
        # 导入决策引擎
        try:
            from decision_engine import DiagnosticSession
        except ImportError:
            page.show_snack_bar(ft.SnackBar(content=ft.Text("⚠️ 决策引擎未就绪，请检查安装")))
            return
        
        # 创建对话会话
        diag_session[0] = DiagnosticSession()
        
        # 尝试从初始文本提取失效模式
        diag_session[0].profile.failure_mode = initial_text
        
        # 生成第一个追问
        first_q = diag_session[0].generate_next_question()
        
        if first_q is None:
            # 信息已足够（不太可能发生）
            _finalize_diagnosis()
            return
        
        # 切换到对话模式
        chat_mode[0] = True
        problem_input.value = ""
        problem_input.label = "✎ 输入你的回答..."
        problem_input.hint_text = "直接输入文字，完成后点右侧 →"
        problem_input.min_lines = 1
        problem_input.max_lines = 3
        
        # 清空结果区，显示对话气泡
        result_container.controls.clear()
        add_message("system", f"已记录问题描述：{initial_text[:60]}{'...' if len(initial_text) > 60 else ''}")
        add_message("ai", first_q)
        
        # 更新按钮
        _update_main_button("发送 →", ft.Colors.GREEN)
        page.update()
    

    def _send_chat_answer():
        """对话模式：处理用户回答 -> 生成下一个追问或最终报告"""
        user_text = problem_input.value.strip()
        if not user_text:
            return
        
        add_message("user", user_text)
        problem_input.value = ""
        
        session = diag_session[0]
        
        # 把回答记录到画像的对应字段
        field = session._pending_field
        if field:
            session.record_answer(field, user_text)
        
        # 生成下一个追问
        next_q = session.generate_next_question()
        
        if next_q:
            add_message("ai", next_q)
            page.update()
        else:
            # 信息充分 -> 输出最终诊断
            _finalize_diagnosis()
    

    def _finalize_diagnosis():
        """信息收集完毕：调用豆包 + 决策引擎输出完整报告"""
        progress_ring.visible = True
        progress_text.value = "⏳ AI正在综合分析，请稍候（约15~30秒）..."
        _update_main_button("分析中...", ft.Colors.GREY)
        page.update()
        
        def _work():
            try:
                _run_llm_diagnosis()
            except Exception as exc:
                err = str(exc)[:200]
                page.run_thread(lambda: add_message("system", f"❌ 分析过程出错: {err}"))
            finally:
                page.run_thread(lambda: setattr(progress_ring, 'visible', False))
                page.run_thread(lambda: setattr(progress_text, 'value', ""))
        
        threading.Thread(target=_work, daemon=True).start()
    

    def _run_llm_diagnosis():
        """后台线程：调用豆包 + 决策引擎生成报告 + 配方硬校验闸门"""
        from decision_engine import build_diagnosis_prompt, build_enhanced_report, validate_diagnosis_suggestion
        import db_sqlite as _db
        import model_manager as _mm
        
        session = diag_session[0]
        profile = session.profile
        
        # 知识库检索
        retrieved = _db.retrieve(profile.failure_mode, top_k=5)
        
        # 构造增强 prompt
        sys_prompt, user_msg = build_diagnosis_prompt(profile, retrieved)
        
        # 获取模型
        model = _mm.get_model()
        
        # 调用豆包
        llm_report = model.chat_completion(
            sys_prompt, user_msg,
            temperature=0.3, max_tokens=2000,
        )
        
        # 关闭模型（释放连接）
        try:
            model.close()
        except Exception:
            pass
        
        # 决策引擎输出
        decision = session.finalize()
        
        # ★ 增强报告：自动追加成本-性能权衡 + DOE交互警告 + 测试映射
        enhanced_report = build_enhanced_report(profile, llm_report)
        
        # ★ 硬校验闸门：配方/建议合规性检查
        veto_result = validate_diagnosis_suggestion(enhanced_report, profile)
        
        # 归档到数据库（含VETO审计日志）
        try:
            full_report = enhanced_report + "\n\n" + (veto_result.get("engineer_text") or "")
            _db.save_diagnosis_session(
                session.session_id,
                profile.to_dict(),
                decision.primary_direction,
                decision.confidence,
                full_report,
            )
        except Exception:
            pass
        
        # 回到主线程显示结果（含VETO状态）
        def _show():
            _render_decision_report(enhanced_report, decision, veto_result)
        
        page.run_thread(_show)
    

    def _render_decision_report(llm_report: str, decision, veto_result=None):
        """主线程：渲染完整的诊断决策报告（含硬校验闸门结果）"""
        result_container.controls.clear()
        chat_messages.clear()
        
        # ══ 决策头部卡片 ══
        if decision.primary_direction == "formula":
            deco = ("🔧", "优先调整配方", ft.Colors.ORANGE_100, ft.Colors.ORANGE_800)
        elif decision.primary_direction == "process":
            deco = ("⚙️", "优先调整工艺", ft.Colors.BLUE_100, ft.Colors.BLUE_800)
        else:
            deco = ("🔍", "需进一步验证", ft.Colors.GREY_200, ft.Colors.GREY_800)
        
        icon, title, bg, fg = deco
        result_container.controls.append(
            ft.Container(
                content=ft.Row([
                    ft.Text(icon, size=32),
                    ft.Column([
                        ft.Text(title, size=20, weight=ft.FontWeight.BOLD, color=fg),
                        ft.Text(
                            f"  置信度: {decision.confidence:.0%}  |  配方调整得分: {decision.formula_score}  vs  工艺调整得分: {decision.process_score}",
                            size=13, color=ft.Colors.GREY_700),
                    ], spacing=4, expand=True),
                ], alignment=ft.MainAxisAlignment.START, spacing=12),
                bgcolor=bg, border_radius=12, padding=16,
                margin=ft.margin.only(bottom=12),
            )
        )
        
        # ══ 决策推理 ══
        result_container.controls.append(ft.Text("📊 决策推理依据", size=16, weight=ft.FontWeight.BOLD))
        result_container.controls.append(ft.Text(decision.reasoning, size=13))
        
        # ══ 可能根因 ══
        if decision.hypothesis_list:
            result_container.controls.append(ft.Text("\n💡 可能根因", size=16, weight=ft.FontWeight.BOLD))
            for h in decision.hypothesis_list:
                result_container.controls.append(
                    ft.Container(content=ft.Text(h, size=13), padding=8,
                                 bgcolor=ft.Colors.BLUE_50, border_radius=8)
                )
        
        # ══ 行动清单 ══
        if decision.action_checklist:
            result_container.controls.append(ft.Text("\n✅ 行动清单（按优先级）", size=16, weight=ft.FontWeight.BOLD))
            for i, item in enumerate(decision.action_checklist, 1):
                result_container.controls.append(
                    ft.Row([
                        ft.Container(content=ft.Text(str(i), size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                            bgcolor=ft.Colors.GREEN_600, border_radius=12, width=24, height=24,
                            alignment=ft.alignment.center),
                        ft.Text(item, size=13, expand=True),
                    ], alignment=ft.MainAxisAlignment.START)
                )
        
        # ══ 风险提示 ══
        if decision.risk_warning:
            result_container.controls.append(ft.Text("\n⚠️ 风险提示", size=16, weight=ft.FontWeight.BOLD))
            result_container.controls.append(
                ft.Container(content=ft.Text(decision.risk_warning, size=13, color=ft.Colors.RED_700),
                             bgcolor=ft.Colors.RED_50, border_radius=8, padding=10)
            )
        
        # ══ 硬校验闸门结果（VETO） ══
        if veto_result:
            channel = veto_result.get("channel", "executable")
            violations = veto_result.get("violations", [])
            engineer_text = veto_result.get("engineer_text", "")
            
            if channel == "rejected":
                veto_deco = ("🚫", "硬校验闸门 — 建议已被拦截", ft.Colors.RED_100, ft.Colors.RED_800)
            elif channel == "candidate":
                veto_deco = ("⚠️", "硬校验闸门 — 候选方案（需人工确认）", ft.Colors.YELLOW_100, ft.Colors.YELLOW_800)
            else:
                veto_deco = ("✅", "硬校验闸门 — 通过（可执行）", ft.Colors.GREEN_100, ft.Colors.GREEN_800)
            
            v_icon, v_title, v_bg, v_fg = veto_deco
            veto_controls = [
                ft.Text(v_title, size=16, weight=ft.FontWeight.BOLD, color=v_fg),
            ]
            
            if violations:
                for v in violations:
                    sev = v.get("severity", "")
                    icon = "🚫" if sev == "HARD_VETO" else "⚠️"
                    rule = v.get("rule_id", "")
                    msg = v.get("message", "")
                    evi = v.get("evidence", "")
                    
                    bg = ft.Colors.RED_50 if sev == "HARD_VETO" else ft.Colors.YELLOW_50
                    veto_controls.append(
                        ft.Container(
                            content=ft.Column([
                                ft.Text(f"{icon} [{rule}] {msg}", size=13, color=ft.Colors.GREY_800),
                                ft.Text(f"   证据: {evi}", size=11, color=ft.Colors.GREY_500) if evi else ft.Text(""),
                            ], spacing=2),
                            bgcolor=bg, border_radius=6, padding=8,
                            margin=ft.margin.only(top=4)
                        )
                    )
            
            result_container.controls.append(
                ft.Container(
                    content=ft.Column(veto_controls, spacing=6),
                    bgcolor=v_bg, border_radius=12, padding=14,
                    margin=ft.margin.only(bottom=12)
                )
            )
            
            # 候选模式：工程师确认环节
            if channel == "candidate":
                confirm_btn = ft.Button(
                    "👷 工程师确认 — 接受此候选方案",
                    on_click=lambda e: _confirm_candidate(veto_result),
                    style=ft.ButtonStyle(bgcolor=ft.Colors.ORANGE, color=ft.Colors.WHITE),
                    height=40
                )
                result_container.controls.append(confirm_btn)
                
        # ══ AI深度报告 ══
        result_container.controls.append(ft.Text("\n📋 AI深度分析报告", size=16, weight=ft.FontWeight.BOLD))
        result_container.controls.append(ft.Text(llm_report, size=13, color=ft.Colors.GREY_700))
        
        # ══ 重新开始按钮 ══
        result_container.controls.append(
            ft.Button("🔄 开始新一轮排查", on_click=_reset_chat,
                style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN, color=ft.Colors.WHITE),
                height=44)
        )
        
        page.update()
    
    
    def _confirm_candidate(veto_result):
        """工程师确认候选方案 — 接受为可执行，记录审计日志"""
        try:
            import db_sqlite as _db
            # 写入审计日志：工程师确认时间、VETO结果、接受原因
            audit_entry = {
                "action": "engineer_confirm_candidate",
                "channel": veto_result.get("channel"),
                "violations": veto_result.get("violations", []),
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }
            _db.index_text("audit_log", f"confirm_{id(veto_result)}", str(audit_entry))
        except Exception:
            pass
        
        # UI反馈：标记为工程师已接受的执行方案
        result_container.controls.clear()
        result_container.controls.append(
            ft.Container(
                content=ft.Column([
                    ft.Text("✅ 工程师已确认接受此候选方案", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_800),
                    ft.Text("该方案已被记录为「工程师审核通过可追溯决策」", size=12, color=ft.Colors.GREY_600),
                    ft.Text(f"违规项数: {len(veto_result.get('violations', []))}（已人工审核豁免）", size=11, color=ft.Colors.GREY_500),
                ], spacing=4),
                bgcolor=ft.Colors.GREEN_50, border_radius=12, padding=16,
            )
        )
        page.update()
    
    def _reset_chat(e=None):
        """重置对话，回到初始模式"""
        diag_session[0] = None
        chat_mode[0] = False
        chat_messages.clear()
        problem_input.value = ""
        problem_input.label = "描述工艺偏差现象"
        problem_input.hint_text = "例：XX材料小试OK，放大后性能下降..."
        problem_input.min_lines = 3
        problem_input.max_lines = 6
        progress_ring.visible = False
        progress_text.value = ""
        result_container.controls.clear()
        # 显示快捷案例行（通过重建页面）
        try:
            nav_bar.selected_index = 0
            on_nav_change(0)
        except Exception:
            pass
    
    # 辅助函数：圆角（左右区分）
    def border_radius_rtl(radius):
        return ft.BorderRadius(top_left=radius, top_right=2, bottom_left=radius, bottom_right=radius)
    
    def border_radius_ltr(radius):
        return ft.BorderRadius(top_left=2, top_right=radius, bottom_left=radius, bottom_right=radius)
    
    # 主按钮的引用（在 build_diagnosis_page 中初始化）
    _main_btn_ref = [None]  # mutable container for ft.Ref
    
    def _update_main_button(text, color):
        """更新主按钮的文字和颜色（由 build_diagnosis_page 内的闭包覆盖）"""
        pass
    
    def build_diagnosis_page():
        """AI主导多轮诊断对话页：初始输入 + 追问交互"""
        voice_status = ft.Text("", size=13, color=ft.Colors.GREY_600)
        is_recording = [False]

        def toggle_recording(e):
            if audio_recorder is None:
                voice_status.value = "⚠️ 当前设备不支持录音"
                page.update()
                return
            if is_recording[0]:
                is_recording[0] = False
                voice_status.value = "🔴 录音中..."
                page.update()
                try:
                    try:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        try:
                            rec_path = loop.run_until_complete(audio_recorder.stop_recording())
                        finally:
                            loop.close()
                    except Exception:
                        rec_path = audio_recorder.stop_recording()
                    voice_status.value = "⏹ 录音结束，识别中..."
                    page.update()
                    import voice_api
                    audio_bytes = open(rec_path, "rb").read()
                    text = voice_api.recognize_speech(audio_bytes, fmt="opus", rate=16000)
                    if text.strip():
                        problem_input.value = text.strip()
                        voice_status.value = f"✅ 已识别: {text[:30]}..."
                    else:
                        voice_status.value = "⚠️ 未识别到文字，请重试"
                except Exception as ex:
                    voice_status.value = f"❌ 识别失败: {str(ex)[:40]}"
                page.update()
            else:
                rec_path = os.path.join(os.path.dirname(__file__), "temp_rec.opus")
                try:
                    try:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(audio_recorder.start_recording(rec_path))
                        finally:
                            loop.close()
                    except Exception:
                        audio_recorder.start_recording(rec_path)
                    is_recording[0] = True
                    voice_status.value = "🔴 正在录音..."
                except Exception as ex:
                    voice_status.value = f"❌ 录音启动失败: {str(ex)[:30]}"
                page.update()

        # 快捷案例按钮
        quick_cases = [
            ("PCM放大包封率跌", "竹基碳/PEG复合PCM：500ml小试包封率85%；放大至50L釜包封率跌至58%"),
            ("PVC型材变形", "硬质PVC型材：同一配方不同批次，热变形温度相差15°C"),
            ("环氧气泡", "环氧地坪涂料：施工后表面出现大量针孔气泡，固化温度25°C"),
            ("胶粘剂不干", "双组份聚氨酯胶粘剂：混合后24h仍未完全固化"),
        ]
        
        quick_btns = [ft.Button(label, on_click=lambda e, t=text: set_and_fill(t),
            height=40, style=ft.ButtonStyle(text_style=ft.TextStyle(size=12))) for label, text in quick_cases]
        
        def set_and_fill(text):
            problem_input.value = text
            page.update()
        
        # 主按钮
        main_btn_ref = ft.Ref[ft.Button]()
        
        def rebuild_main_btn():
            btn = main_btn_ref.current
            if btn is None:
                page.update()
                return
            if not chat_mode[0]:
                btn.content = ft.Row([
                    ft.Icon(ft.Icons.PLAY_ARROW, size=18),
                    ft.Text("开始AI排查", size=16, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=6)
                btn.style = ft.ButtonStyle(bgcolor=ft.Colors.BLUE, color=ft.Colors.WHITE,
                    padding=ft.Padding.symmetric(horizontal=32, vertical=16),
                    shape=ft.RoundedRectangleBorder(radius=20))
            elif diag_session[0] and diag_session[0].current_phase == 3:
                btn.content = ft.Row([
                    ft.Icon(ft.Icons.HOURGLASS_EMPTY, size=18),
                    ft.Text("分析中...", size=16, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=6)
                btn.style = ft.ButtonStyle(bgcolor=ft.Colors.GREY_400, color=ft.Colors.WHITE,
                    shape=ft.RoundedRectangleBorder(radius=20))
            else:
                btn.content = ft.Row([
                    ft.Text("发送", size=16, weight=ft.FontWeight.BOLD),
                    ft.Icon(ft.Icons.SEND, size=18),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=6)
                btn.style = ft.ButtonStyle(bgcolor=ft.Colors.GREEN, color=ft.Colors.WHITE,
                    padding=ft.Padding.symmetric(horizontal=24, vertical=12),
                    shape=ft.RoundedRectangleBorder(radius=20))
            try:
                btn.update()
            except Exception:
                page.update()
        
        # 覆盖外层的 _update_main_button
        globals()['_update_main_button'] = rebuild_main_btn
        # 初始化按钮状态
        # (按钮还未创建，等 Column 渲染后再 rebuild)
        
        # 延迟重建：Flet 需要在控件渲染后调用 ref
        def _schedule_rebuild():
            page.run_thread(lambda: (rebuild_main_btn(), None))
        
        main_btn = ft.Button(
            ref=main_btn_ref,
            content=ft.Row([
                ft.Icon(ft.Icons.PLAY_ARROW, size=18),
                ft.Text("开始AI排查", size=16, weight=ft.FontWeight.BOLD),
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=6),
            on_click=on_run_diagnosis, height=50,
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE, color=ft.Colors.WHITE,
                padding=ft.Padding.symmetric(horizontal=32, vertical=16),
                shape=ft.RoundedRectangleBorder(radius=20)),
        )
        
        # 渲染后自动初始化按钮样式
        page.run_thread(lambda: rebuild_main_btn())
        
        return ft.Column([
            ft.Text("🔬 工艺偏差排查", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("AI主导诊断：你只需回答问题，AI综合分析后给出决策", size=14, color=ft.Colors.GREY_700),
            problem_input,
            *([ft.Row([
                ft.Button("🎤 语音输入", icon=ft.Icons.MIC, on_click=toggle_recording,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.RED_100, color=ft.Colors.RED)),
                voice_status,
            ], alignment=ft.MainAxisAlignment.START)] if audio_recorder is not None else
               [ft.Text("ℹ️ 当前设备不支持语音输入，请使用文字描述", size=12, color=ft.Colors.GREY_500)]),
            ft.Row([
                ft.Text("💡 快速案例", size=16, weight=ft.FontWeight.W_600, expand=True),
            ]),
            ft.Row(quick_btns, wrap=True, spacing=8),
            main_btn,
            ft.Container(content=ft.Row([progress_ring, progress_text], alignment=ft.MainAxisAlignment.CENTER), padding=ft.Padding.symmetric(vertical=8)),
            ft.Container(content=result_container, height=360, border_radius=12,
                         border=ft.border.all(1, ft.Colors.GREY_300), padding=10),
        ], spacing=12)
    
    def build_results_page():
        """结果页：方案卡 + 语音播报"""
        if not diagnosis_result:
            return ft.Column([
                ft.Text("📋 改善方案", size=20, weight=ft.FontWeight.BOLD),
                ft.Text("请先完成排查步骤", color=ft.Colors.GREY_500),
            ])

        sol = diagnosis_result.get("solution", "")
        speak_status = ft.Text("", size=13, color=ft.Colors.GREY_600)

        def speak_solution(e):
            if player is None:
                speak_status.value = "⚠️ 当前设备不支持播放"
                page.update()
                return
            speak_status.value = "🔊 正在合成语音..."
            page.update()
            try:
                import voice_api
                sol_text = sol[:200] if sol else "暂无方案"
                audio_data = voice_api.text_to_speech(sol_text)
                audio_path = voice_api.save_audio_tmp(audio_data, ".mp3")
                if audio_path:
                    player.src = audio_path
                    player.play()
                    speak_status.value = "🔊 播放中..."
                else:
                    speak_status.value = "❌ 播放失败"
            except Exception as ex:
                speak_status.value = f"❌ {str(ex)[:40]}"
            page.update()

        def stop_speak(e):
            if player is None:
                return
            player.pause()
            speak_status.value = "⏸ 已停止"
            page.update()

        return ft.Column([
            ft.Text("📋 改善方案卡", size=20, weight=ft.FontWeight.BOLD),
            ft.Container(
                content=ft.Text(sol, size=13, color=ft.Colors.GREY_800),
                padding=16, bgcolor=ft.Colors.WHITE,
                border_radius=12, border=ft.border.all(1, ft.Colors.GREY_300)
            ),
            ft.Row([
                ft.Button("🔊 语音播报", icon=ft.Icons.VOLUME_UP, on_click=speak_solution,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_100, color=ft.Colors.GREEN)),
                ft.Button("⏹ 停止", icon=ft.Icons.STOP, on_click=stop_speak),
                speak_status,
            ], alignment=ft.MainAxisAlignment.START),
        ], spacing=12, scroll=ft.ScrollMode.AUTO)
    
    def build_knowledge_page():
        """知识库/案例录入页"""
        import db_sqlite as kb_db
        kb_db.DB_PATH = get_db_path()
        stats = kb_db.get_stats()

        # 案例录入
        case_input = ft.TextField(label="问题描述", multiline=True, min_lines=2, max_lines=4)
        cause_input = ft.TextField(label="根因分析", multiline=True, min_lines=2, max_lines=3)
        solution_input = ft.TextField(label="改善方案", multiline=True, min_lines=2, max_lines=3)
        case_status = ft.Text("", size=13, color=ft.Colors.GREEN)

        def save_case(e):
            prob = case_input.value.strip()
            cause = cause_input.value.strip()
            sol = solution_input.value.strip()
            if not prob:
                case_status.value = "❌ 问题描述不能为空"
                case_status.color = ft.Colors.RED
                page.update()
                return
            import uuid
            case_id = f"case-{uuid.uuid4().hex[:8]}"
            kb_db.insert_case(case_id, prob, cause)
            if sol:
                kb_db.insert_solution(f"sol-{uuid.uuid4().hex()[6:8]}", cause or prob[:20], sol)
            # 把案例文本也索引到向量知识库
            try:
                kb_db.index_text("manual_case", case_id, f"问题:{prob}\n根因:{cause}\n方案:{sol}")
            except Exception:
                pass
            case_status.value = "✅ 案例已保存，可用于后续排查"
            case_status.color = ft.Colors.GREEN
            case_input.value = ""
            cause_input.value = ""
            solution_input.value = ""
            page.update()

        # 经验规则录入
        rule_trigger = ft.TextField(label="触发条件（关键词）")
        rule_content = ft.TextField(label="经验内容", multiline=True, min_lines=2, max_lines=4)
        rule_status = ft.Text("", size=13, color=ft.Colors.GREEN)

        def save_rule(e):
            trigger = rule_trigger.value.strip()
            content = rule_content.value.strip()
            if not trigger or not content:
                rule_status.value = "❌ 两项都不能为空"
                rule_status.color = ft.Colors.RED
                page.update()
                return
            import uuid
            rule_id = f"rule-{uuid.uuid4().hex()[6:8]}"
            kb_db.insert_rule(rule_id, trigger, content, source="mobile")
            try:
                kb_db.index_text("experience_rule", rule_id, f"触发:{trigger}\n经验:{content}")
            except Exception:
                pass
            rule_status.value = "✅ 经验规则已保存"
            rule_status.color = ft.Colors.GREEN
            rule_trigger.value = ""
            rule_content.value = ""
            page.update()

        # Tab切换：案例录入 / 规则录入 / 教训+野经验
        tab_case = ft.Column([
            ft.Text("录入新案例，沉淀经验供AI参考", size=14, color=ft.Colors.GREY_700),
            case_input,
            cause_input,
            solution_input,
            ft.Button("💾 保存案例", icon=ft.Icons.SAVE, on_click=save_case),
            case_status,
        ], spacing=10)

        tab_rule = ft.Column([
            ft.Text("录入经验规则，提升排查精度", size=14, color=ft.Colors.GREY_700),
            rule_trigger,
            rule_content,
            ft.Button("💾 保存规则", icon=ft.Icons.SAVE, on_click=save_rule),
            rule_status,
        ], spacing=10)

        # ══ 野经验/教训录入 Tab ══
        lesson_action = ft.TextField(label='触发操作/动作（如"一次性加稀释剂8phr"）')
        lesson_consequence = ft.TextField(label="连锁次生后果", multiline=True, min_lines=2, max_lines=3)
        lesson_redline = ft.TextField(label='禁止红线（如"单次<=5phr，必须分次"）')
        lesson_status = ft.Text("", size=13, color=ft.Colors.GREEN)

        def save_lesson(e):
            action = lesson_action.value.strip()
            consequence = lesson_consequence.value.strip()
            redline = lesson_redline.value.strip()
            if not action or not redline:
                lesson_status.value = "❌ 操作动作和禁止红线不能为空"
                lesson_status.color = ft.Colors.RED
                page.update()
                return
            import uuid
            lid = f"FAIL-{uuid.uuid4().hex[:6]}"
            try:
                kb_db.execute(
                    "INSERT INTO failure_lesson (lesson_id, action_desc, redline, consequence, loss_level) VALUES(?,?,?,?,?)",
                    (lid, action, redline, consequence, "待评估")
                )
                kb_db.commit()
                # 也索引到向量知识库
                kb_db.index_text("failure_lesson", lid, f"失败教训:{action} 后果:{consequence} 红线:{redline}")
            except Exception as ex:
                lesson_status.value = f"❌ 保存失败: {str(ex)[:60]}"
                lesson_status.color = ft.Colors.RED
                page.update()
                return
            lesson_status.value = "✅ 教训已入库，AI后续方案输出将自动拦截此类操作"
            lesson_status.color = ft.Colors.GREEN
            lesson_action.value = ""
            lesson_consequence.value = ""
            lesson_redline.value = ""
            page.update()

        tab_lesson = ft.Column([
            ft.Text("录入历史失败教训，AI输出方案前自动拦截红线操作", size=14, color=ft.Colors.GREY_700),
            ft.Text("工程师野经验是最宝贵的知识——让AI不再重蹈覆辙", size=12, color=ft.Colors.GREY_500),
            lesson_action,
            lesson_consequence,
            lesson_redline,
            ft.Button("🚫 保存教训红线", icon=ft.Icons.SAVE, on_click=save_lesson),
            lesson_status,
        ], spacing=10)

        # Flet 1.0.x 改用 TabBar + TabBarView; Tab 只接受 label
        page_tabs = ft.Tabs(
            length=3,
            selected_index=0,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        expand=True,
                        tabs=[
                            ft.Tab(label="📋 案例录入"),
                            ft.Tab(label="📐 规则录入"),
                            ft.Tab(label="🚫 教训红线"),
                        ],
                    ),
                    ft.TabBarView(
                        expand=True,
                        controls=[
                            ft.Container(content=tab_case, padding=10, expand=True),
                            ft.Container(content=tab_rule, padding=10, expand=True),
                            ft.Container(content=tab_lesson, padding=10, expand=True),
                        ],
                    ),
                ],
            ),
        )

        return ft.Column([
            ft.Text("📚 知识库管理", size=20, weight=ft.FontWeight.BOLD),
            ft.Card(ft.Container(
                content=ft.Row([
                    ft.Column([
                        f"经验规则: {stats.get('rules', 0)}",
                        f"向量索引: {stats.get('vectors', 0)}",
                    ], spacing=4),
                    ft.Column([
                        f"案例数量: {stats.get('cases', 0)}",
                        f"方案数量: {stats.get('solutions', 0)}",
                    ], spacing=4),
                ], alignment=ft.MainAxisAlignment.SPACE_AROUND),
                padding=16
            )),
            ft.Container(content=page_tabs, padding=ft.Padding.symmetric(vertical=8)),
        ], spacing=12)

    def build_audit_page():
        """审计报告页"""
        if not diagnosis_result or not diagnosis_result.get("audit_scores"):
            return ft.Column([
                ft.Text("🛡️ 审计报告", size=20, weight=ft.FontWeight.BOLD),
                ft.Text("请先完成排查步骤", color=ft.Colors.GREY_500),
            ])
        
        scores = diagnosis_result.get("audit_scores", {})
        audit_cards = []
        for k, v in scores.items():
            score_val = v if isinstance(v, (int, float)) else 5
            audit_cards.append(
                ft.Card(ft.Container(
                    content=ft.Column([
                        ft.Text(k, size=14, weight=ft.FontWeight.W_600),
                        ft.ProgressBar(value=score_val/10, color=ft.Colors.GREEN, bar_height=10),
                        ft.Text(f"{v}/10", size=18, weight=ft.FontWeight.BOLD),
                    ], spacing=6),
                    padding=12
                ), elevation=2)
            )
        
        issues = []
        full_audit = diagnosis_result.get("full_audit", {})
        for issue in full_audit.get("issues", [][:5]):
            issues.append(
                ft.Chip(ft.Text(f"⚠️ {issue.get('type')}: {issue.get('description', '')[:30]}"))
            )
        
        return ft.Column([
            ft.Text("🛡️ AI审计报告", size=20, weight=ft.FontWeight.BOLD),
            ft.Text("独立审计Agent对诊断流程的客观评估", size=13, color=ft.Colors.GREY_600),
            ft.Row(audit_cards, wrap=True),
            ft.Text("发现问题", size=16, weight=ft.FontWeight.W_600),
            ft.Row(issues, wrap=True, spacing=8),
        ], spacing=12)
    
    def build_settings_page():
        import db_sqlite as _db
        saved_key = _db.load_setting("DOUBAO_API_KEY", "")
        saved_mid = _db.load_setting("DOUBAO_MODEL_ID", "")

        api_key_field = ft.TextField(
            label="豆包 API Key",
            value=saved_key,
            password=True,
            hint_text="从火山引擎方舟平台获取",
        )
        model_id_field = ft.TextField(
            label="豆包 Model ID (推理接入点)",
            value=saved_mid,
            hint_text="形如 ep-202xxxxxxxx-xxxxx",
        )
        api_save_status = ft.Text("", size=14, color=ft.Colors.GREEN)

        def save_api(e):
            key = api_key_field.value.strip()
            mid = model_id_field.value.strip()
            if not key or not mid:
                api_save_status.value = "❌ 两项都不能为空"
                api_save_status.color = ft.Colors.RED
                page.update()
                return
            _db.save_setting("DOUBAO_API_KEY", key)
            _db.save_setting("DOUBAO_MODEL_ID", mid)
            import config as _cfg
            _cfg.DOUBAO_API_KEY = key
            _cfg.DOUBAO_MODEL_ID = mid
            api_save_status.value = "✅ 已保存到本地，立即生效"
            api_save_status.color = ft.Colors.GREEN
            page.update()

        return ft.Column([
            ft.Text("⚙️ 设置", size=20, weight=ft.FontWeight.BOLD),
            ft.Card(ft.Container(
                content=ft.Column([
                    ft.Text("豆包 API 配置", size=16, weight=ft.FontWeight.W_600),
                    api_key_field,
                    model_id_field,
                    ft.Button("保存配置", icon=ft.Icons.SAVE, on_click=save_api),
                    api_save_status,
                    ft.Text("信息本地保存，重启后依然有效", size=11, color=ft.Colors.GREY),
                ], spacing=8),
                padding=16
            )),
            ft.Card(ft.Container(
                content=ft.Column([
                    ft.Text("知识库状态", size=16, weight=ft.FontWeight.W_600),
                    ft.Text("PCM规则+案例+全品类知识 已加载", size=14),
                    ft.Text(f"数据库: {get_db_path()}", size=11, color=ft.Colors.GREY),
                ], spacing=8),
                padding=16
            )),
            ft.Card(ft.Container(
                content=ft.Column([
                    ft.Text("关于", size=16, weight=ft.FontWeight.W_600),
                    ft.Text("新材料工艺偏差排查AI v1.0", size=14),
                    ft.Text("兼容 Android / 鸿蒙 (HarmonyOS)", size=11, color=ft.Colors.GREY),
                ], spacing=8),
                padding=16
            )),
        ], spacing=12)
    
    # ====== 页面组装 ======
    page.add(
        ft.Container(content=content_area, expand=True, padding=ft.Padding.symmetric(horizontal=0, vertical=8))
    )
    page.add(nav_bar)
    
def main(page: ft.Page):
    """Flet入口包装"""
    build_main_ui(page)

if __name__ == "__main__":
    init_mobile_db()
    load_saved_api_settings()
    
    import config as cfg
    has_key = bool(getattr(cfg, "DOUBAO_API_KEY", ""))
    has_mid = bool(getattr(cfg, "DOUBAO_MODEL_ID", ""))
    
    if has_key and has_mid:
        ft.run(main)
    else:
        def login_page(page: ft.Page):
            page.title = "豆包登录"
            page.theme_mode = ft.ThemeMode.LIGHT
            page.window.width = 400
            page.window.height = 500
            page.vertical_alignment = ft.MainAxisAlignment.CENTER
            
            api_input = ft.TextField(label="豆包 API Key", password=True, width=350)
            mid_input = ft.TextField(label="Model ID (推理接入点)", width=350)
            status = ft.Text("", size=13, color=ft.Colors.RED)
            
            def do_login(e):
                key = api_input.value.strip()
                mid = mid_input.value.strip()
                if not key or not mid:
                    status.value = "❌ 请填写两项后点登录"
                    status.color = ft.Colors.RED
                    page.update()
                    return
                import db_sqlite as _db
                _db.DB_PATH = get_db_path()
                _db.save_setting("DOUBAO_API_KEY", key)
                _db.save_setting("DOUBAO_MODEL_ID", mid)
                import os as _os
                _os.environ["DOUBAO_API_KEY"] = key
                _os.environ["DOUBAO_MODEL_ID"] = mid
                cfg.DOUBAO_API_KEY = key
                cfg.DOUBAO_MODEL_ID = mid
                # 重置模型单例，使其下次调用时读取新的环境变量
                try:
                    import model_manager as _mm
                    _mm._instance = None
                except Exception:
                    pass
                # 清除登录页并绘制主界面
                page.controls.clear()
                page.vertical_alignment = None
                page.padding = 16
                build_main_ui(page)

            page.add(
                ft.Column([
                    ft.Text("🔑 豆包AI 登录", size=22, weight=ft.FontWeight.BOLD),
                    ft.Text("首次使用需要输入API凭证", size=13, color=ft.Colors.GREY),
                    ft.Container(height=16),
                    api_input,
                    mid_input,
                    ft.Container(height=8),
                    ft.Button("登录", width=350, on_click=do_login),
                    status,
                    ft.Container(height=16),
                    ft.Text("凭证保存在手机本地，只需输入一次", size=11, color=ft.Colors.GREY),
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            )
        
        ft.run(login_page)
