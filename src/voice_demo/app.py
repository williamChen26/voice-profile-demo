from functools import wraps


def build_app(service):
    import gradio as gr

    def choices():
        return [(f"{p['name']} · {p['profile_id'][:8]}", p["profile_id"]) for p in service.profiles.list()]

    def guarded(function):
        @wraps(function)
        def run(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
        return run

    @guarded
    def enroll(name, files, recording, confirmed):
        if not confirmed:
            raise ValueError("请确认录入音频仅包含本人正常说话")
        paths = list(files or []) + ([recording] if recording else [])
        profile = service.enroll(name, paths)
        summary = {k: v for k, v in profile.items() if k not in ("centroid", "vectors")}
        return summary, gr.Dropdown(choices=choices(), value=profile["profile_id"])

    @guarded
    def verify(identifier, audio, confirmed):
        if not audio:
            raise ValueError("请先上传待验证音频")
        if not confirmed:
            raise ValueError("请确认这是单人录音；多人录音请使用整段分析")
        result = service.verify(identifier, audio)
        label = {"self": "本人", "other": "他人", "unknown": "无法确定"}[result["identity"]]
        score = result["similarity"]
        return f"**{label}** · 相似度：{score:.4f}" if score is not None else f"**{label}** · 没有足够的有效语音", result

    @guarded
    def analyze(identifier, audio, with_asr, count):
        if not audio:
            raise ValueError("请先上传录音")
        if count and (count < 0 or count != int(count)):
            raise ValueError("说话人数必须为非负整数；0 表示自动估计")
        result, report, archive = service.analyze(identifier, audio, with_asr, int(count) if count else None)
        rows = [[w["start"], w["end"], w["speaker_id"], w["identity"], w["text"], w["overlap"]] for w in result["words"]]
        clips = {f"{i:04d} · {s['start']:.2f}–{s['end']:.2f}s · {s['speaker']} · {s['identity']}": str(report.parent / s["clip"])
                 for i, s in enumerate(result["segments"])}
        return result, rows, str(report), str(archive), clips, gr.Dropdown(choices=list(clips), value=None), None

    @guarded
    def delete(identifier, confirmed):
        if not confirmed:
            raise ValueError("请勾选确认删除")
        service.profiles.delete(identifier)
        return gr.Dropdown(choices=choices(), value=None), "已删除声纹模板。历史分析结果和界面临时文件需另行清理。"

    with gr.Blocks(title="Voice Profile Demo", analytics_enabled=False) as demo:
        gr.Markdown("# 声纹验证实验室\n先录入本人声音，再验证另一段录音。模型本地推理；首次运行需下载权重。")
        gr.Markdown(f"**实验阈值，尚未校准：** 本人 ≥ {service.settings.accept}，他人 ≤ {service.settings.reject}；中间或有效语音不足 {service.settings.min_speech:g} 秒时无法确定。相似度不是概率。")
        with gr.Row():
            profile = gr.Dropdown(label="当前本人声纹", choices=choices(), interactive=True)
            refresh = gr.Button("刷新声纹列表")
        with gr.Tab("1 · 录入声纹"):
            name = gr.Textbox(label="声纹名称", placeholder="例如：William · 日常说话")
            files = gr.File(label="上传本人单独说话的录音，可多选", file_count="multiple", type="filepath")
            recording = gr.Audio(label="或直接录制本人声音", sources=["microphone"], type="filepath", format="wav")
            gr.Markdown("建议 3 段，每段 10～20 秒，使用不同内容。有效连续语音累计至少 10 秒。第一版不自动确认录入者身份或可靠检查多人混入。")
            confirmed = gr.Checkbox(label="我确认这些录音仅包含本人正常说话，且有权使用这些声音")
            enroll_button = gr.Button("建立声纹", variant="primary")
            enrollment_result = gr.JSON(label="录入信息（不展示声纹向量）")
        with gr.Tab("2 · 单段声纹对比"):
            verify_audio = gr.File(label="上传一段单人录音", type="filepath")
            verify_confirm = gr.Checkbox(label="我确认待测音频是单人录音")
            verify_button = gr.Button("与当前本人声纹比较", variant="primary")
            verdict = gr.Markdown()
            verification_result = gr.JSON(label="相似度、各窗口分数与判定原因")
        with gr.Tab("3 · 整段录音分析"):
            gr.Markdown("首次使用需先在 Hugging Face 接受 Community-1 条件，并在项目 .env 配置 HF_TOKEN。重叠发言标记为无法确定，导出的片段仍是原混音。")
            meeting = gr.File(label="上传多人录音", type="filepath")
            with_asr = gr.Checkbox(label="同时运行 ASR", value=True)
            count = gr.Number(label="实际说话人数（0 = 自动估计，不是本人/他人类别数）", value=0, precision=0)
            analyze_button = gr.Button("开始分析", variant="primary")
            transcript = gr.Dataframe(headers=["开始", "结束", "说话人", "身份", "文字", "重叠"], interactive=False)
            analysis_result = gr.JSON(label="完整分析结果")
            with gr.Row():
                report = gr.File(label="JSON 报告")
                archive = gr.File(label="报告与音频片段 ZIP")
            clip_paths = gr.State({})
            clip_selector = gr.Dropdown(label="选择片段回听", choices=[], interactive=True)
            player = gr.Audio(label="片段回听", interactive=False)
        with gr.Tab("声纹管理"):
            delete_confirm = gr.Checkbox(label="确认删除当前选中的声纹模板")
            delete_button = gr.Button("删除声纹")
            delete_message = gr.Markdown()
        refresh.click(lambda: gr.Dropdown(choices=choices(), value=None), outputs=profile)
        shared = {"concurrency_id": "local_models", "concurrency_limit": 1}
        enroll_button.click(enroll, [name, files, recording, confirmed], [enrollment_result, profile], **shared)
        verify_button.click(verify, [profile, verify_audio, verify_confirm], [verdict, verification_result], **shared)
        analyze_button.click(analyze, [profile, meeting, with_asr, count],
                             [analysis_result, transcript, report, archive, clip_paths, clip_selector, player], **shared)
        clip_selector.change(lambda label, paths: paths.get(label), [clip_selector, clip_paths], player)
        delete_button.click(delete, [profile, delete_confirm], [profile, delete_message], **shared)
    return demo.queue(default_concurrency_limit=1, max_size=8)
