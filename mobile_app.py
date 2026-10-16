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
    """获取移动端可用的SQLite路径"""
    if getattr(sys, 'frozen', False):
        # PyInstaller / Flet packaged
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(__file__)
    
    db_name = "pcm_knowledge.db"
    src = os.path.join(base, db_name)
    
    # Android: 复制到 app 私有目录
    if sys.platform == "android" or "android" in sys.platform:
        try:
            from android.storage import app_private_dir
            dst_dir = app_private_dir
        except ImportError:
            dst_dir = os.path.expanduser("~/.material_ai")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, db_name)
        if not os.path.exists(dst) and os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
        return dst
    
    return src


def init_mobile_db():
    """确保移动端数据库已就绪"""
    db_path = get_db_path()
    if os.path.exists(db_path):
        return db_path
    # 首次运行：用脚本初始化
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "import_pcm_sqlite",
            os.path.join(os.path.dirname(__file__), "import_pcm_sqlite.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
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
def main(page: ft.Page):
    page.title = "材料AI诊断"
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 16
    page.bgcolor = ft.Colors.GREY_50

    # 语音录音器 + 播放器 — Flet 0.7.0+ services 模式
    try:
        audio_recorder = ft.AudioRecorder(
            audio_encoder=ft.AudioEncoder.OPUS,
            on_state_changed=lambda e: print(f"Recorder: {e.data}"),
        )
        player = ft.Audio(autoplay=False)
        try:
            page.services.append(audio_recorder)
            page.services.append(player)
        except AttributeError:
            try:
                page.overlay.append(audio_recorder)
                page.overlay.append(player)
            except Exception:
                pass
    except (AttributeError, TypeError):
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
    result_container = ft.Column([], spacing=12)
    diagnosis_result = {}  # 保存最新结果
    
    def on_run_diagnosis(e):
        if not problem_input.value.strip():
            page.show_snack_bar(ft.SnackBar(content=ft.Text("请先输入偏差描述")))
            return
        
        result_container.controls.clear()
        progress_ring.visible = True
        progress_text.value = "🚀 启动诊断引擎..."
        page.update()
        
        def progress_cb(data):
            progress_text.value = f"[{data['stage']}] {data['message']}"
            page.update()
        
        def thread_work():
            try:
                nonlocal diagnosis_result
                diagnosis_result = run_diagnosis(problem_input.value.strip(), progress_cb)
                # 回到主线程更新UI
                page.run_thread(lambda: show_result(diagnosis_result))
            except Exception as exc:
                err_msg = str(exc)
                page.run_thread(lambda msg=err_msg: setattr(progress_text, 'value', f"❌ 错误: {msg}"))
            finally:
                page.run_thread(lambda: setattr(progress_ring, 'visible', False))
        
        t = threading.Thread(target=thread_work, daemon=True)
        t.start()
    
    def show_result(result):
        """诊断结果展示"""
        result_container.controls.clear()
        result_container.controls.append(
            ft.Text("🎯 诊断结果", size=20, weight=ft.FontWeight.BOLD)
        )
        
        # 假设列表
        if result.get("hypotheses"):
            for i, (h, p) in enumerate(result["hypotheses"].items(), 1):
                result_container.controls.append(
                    ft.Card(ft.Container(
                        content=ft.Column([
                            ft.Text(f"H{i}: {h}", size=14, weight=ft.FontWeight.W_500),
                            ft.ProgressBar(value=p, color=ft.Colors.BLUE, bar_height=8),
                            ft.Text(f"置信度: {p:.0%}", size=12, color=ft.Colors.GREY),
                        ], spacing=6),
                        padding=12
                    ), elevation=2)
                )
        
        # 根因
        if result.get("root_cause"):
            result_container.controls.append(
                ft.Card(ft.Container(
                    content=ft.Column([
                        ft.Text("🎯 锁定根因", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.RED_700),
                        ft.Text(result["root_cause"], size=14),
                    ], spacing=8),
                    padding=16, bgcolor=ft.Colors.RED_50
                ), elevation=3)
            )
        
        # 跳转按钮
        result_container.controls.append(
            ft.Row([
                ft.Button("查看改善方案", icon=ft.Icons.ARTICLE,
                    on_click=lambda _: nav_bar.selected_index == 1 and on_nav_change(1)),
                ft.Button("查看审计报告", icon=ft.Icons.SHIELD,
                    on_click=lambda _: (setattr(nav_bar, 'selected_index', 2), on_nav_change(2))),
            ], alignment=ft.MainAxisAlignment.CENTER)
        )
        
        page.update()
    
    def build_diagnosis_page():
        """输入 + 问题描述页"""
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
                    # 新版 Flet: stop_recording 是 async
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
                # 开始录音
                rec_path = os.path.join(os.path.dirname(__file__), "temp_rec.opus")
                try:
                    # 新版 Flet: start_recording 是 async
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
        
        quick_btns = []
        for label, text in quick_cases:
            quick_btns.append(
                ft.Button(label, 
                    on_click=lambda e, t=text: set_and_fill(t),
                    height=40, style=ft.ButtonStyle(text_style=ft.TextStyle(size=12)))
            )
        
        def set_and_fill(text):
            problem_input.value = text
            page.update()
        
        return ft.Column([
            ft.Text("🔬 工艺偏差排查", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("描述您遇到的材料/工艺问题，AI将综合分析", size=14, color=ft.Colors.GREY_700),
            problem_input,
            ft.Row([
                ft.Button("🎤 语音输入", icon=ft.Icons.MIC, on_click=toggle_recording,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.RED_100, color=ft.Colors.RED)),
                voice_status,
            ], alignment=ft.MainAxisAlignment.START),
            ft.Text("💡 快速案例", size=16, weight=ft.FontWeight.W_600),
            ft.Row(quick_btns, wrap=True, spacing=8),
            ft.Button("🚀 开始AI排查", icon=ft.Icons.PLAY_ARROW,
                on_click=on_run_diagnosis,
                style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE, color=ft.Colors.WHITE,
                    padding=ft.Padding.symmetric(horizontal=32, vertical=16),
                    shape=ft.RoundedRectangleBorder(radius=20)),
                height=50),
            ft.Container(content=ft.Row([progress_ring, progress_text], alignment=ft.MainAxisAlignment.CENTER), padding=ft.Padding.symmetric(vertical=8)),
            result_container,
        ], spacing=16)
    
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

        # Tab切换：案例录入 / 规则录入
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

        page_tabs = ft.Tabs(
            selected_index=0,
            tabs=[
                ft.Tab(text="📋 案例录入", content=tab_case),
                ft.Tab(text="📐 规则录入", content=tab_rule),
            ]
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
    
    # 默认显示排查页
    on_nav_change(0)


# ============================================================
# Flet 入口
# ============================================================
if __name__ == "__main__":
    init_mobile_db()
    load_saved_api_settings()
    
    import config as cfg
    has_key = bool(getattr(cfg, "DOUBAO_API_KEY", ""))
    has_mid = bool(getattr(cfg, "DOUBAO_MODEL_ID", ""))
    
    if has_key and has_mid:
        ft.app(main)
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
                cfg.DOUBAO_API_KEY = key
                cfg.DOUBAO_MODEL_ID = mid
                page.window.close()

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
        
        ft.app(login_page)
