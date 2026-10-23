# Voice Profile Demo

用 Python 在本机完成声纹录入、本人/他人验证、说话人分段和 ASR。当前只接入 WeSpeaker，不接 ECAPA 或商业 API。技术设计见 [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md)，验证记录见 [VALIDATION.md](VALIDATION.md)。

## 已实现

- 多段本人语音录入，保存版本化声纹模板。
- 单人录音与指定模板比较：`self` / `other` / `unknown`，显示窗口分数。
- Community-1 多人分段，排除重叠部分后聚合声纹证据。
- faster-whisper 转写及词级说话人对齐；可关闭 ASR。
- JSON 报告、WAV 片段和 ZIP 下载、片段回听。
- 命令行批量评测固定阈值下的 FAR、本人漏识率与覆盖率。
- 本地 Gradio 界面、模板列表和删除。

**本 Demo 是实验工具。默认相似度阈值没有经过业务数据校准；不是商用准确率承诺。**

## 安装

建议 Python 3.12；当前项目支持 Python 3.10～3.13。需要可访问 PyPI 和 Hugging Face 的网络。依赖和模型可能占用数 GB，首次运行会下载模型。macOS 默认使用 CPU；CUDA 可用时 `DEVICE=auto` 会使用 CUDA。

在项目文件夹打开终端：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
cp .env.example .env
voice-demo doctor
voice-demo web
```

Windows 激活命令为 `.venv\Scripts\activate`。打开 <http://127.0.0.1:7860>。

也可以在 macOS 双击 `start.command`：它优先使用项目 `.venv`，其次使用此次开发时创建的 `../../work/venv`，没有环境时创建并安装。

主要依赖版本固定在 `pyproject.toml`；`requirements-tested-macos.txt` 是本次 Apple Silicon / Python 3.12 的完整依赖快照，仅供复现该环境，不保证适用于其他平台。系统 FFmpeg 是 pyannote 官方建议的环境依赖；本 Demo 的文件解码使用 soundfile / PyAV，并将内存波形传给 pyannote，尽量避免依赖系统 FFmpeg 的文件解码路径。

## Hugging Face 设置

整段录音分析依赖受限访问的 Community-1 权重，需要你本人完成：

1. 登录并接受 [Community-1 模型使用条件](https://huggingface.co/pyannote/speaker-diarization-community-1)。
2. 在 [Hugging Face token 页面](https://huggingface.co/settings/tokens)创建具备相应模型读取权限的 token。
3. 只在本地 `.env` 填写 `HF_TOKEN=...`，重启 Demo。不要在聊天中发送 token。

WeSpeaker 是公开模型，通常不需要完成 Community-1 的授权就可以先做声纹录入与单段对比。首次下载仍需要联网。推理不调用收费 API，也不上传音频。

## 第一次测试

1. 在「录入声纹」上传 3 段你单独说话的录音，每段建议 10～20 秒，也可直接录音。填写名称并确认单人录音后建立声纹。累计可用连续语音至少 10 秒。
2. 在「单段声纹对比」先上传你另一时间录制的声音，再上传其他人的声音。使用独立录音，不要仅用录入文件验证准确率。
3. 在「整段录音分析」上传多人录音。实际说话人数不确定时保留 0，**不要因为只有本人/他人两类就填 2**。
4. 查看每个匿名说话人的身份结果、窗口分数和重叠标记，回听导出的片段。

默认 ASR 为 `large-v3`，CPU 上可能较慢。在 `.env` 中设置 `ASR_MODEL=small` 可加快体验；`tiny` 可用于快速验证流程。不同模型的转写效果需要分别评测。默认语言中文，设置 `ASR_LANGUAGE=` 启用自动语言识别。改配置后重启。

默认判定规则：

| 条件 | 结果 |
|---|---|
| 可用连续语音不足 3 秒 | unknown |
| 窗口分数同时出现明确高匹配与低匹配 | unknown |
| 聚合相似度 ≥ 0.70 | self |
| 聚合相似度 ≤ 0.45 | other |
| 介于两个阈值之间 | unknown |

这些数字只是实验起点。相似度 0.8 不代表 80% 概率。不同录音条件、模型或模板策略下阈值都需要重新验证。

## 命令行

```bash
# 新建模板；命令行输入也必须由你确认是本人单独说话
voice-demo enroll --name '本人' enroll1.wav enroll2.wav enroll3.wav

# 获取 profile_id
voice-demo profiles

# 将 PROFILE_ID 换成录入结果中的 ID
voice-demo verify --profile PROFILE_ID test.wav
voice-demo analyze --profile PROFILE_ID meeting.wav
voice-demo analyze --profile PROFILE_ID meeting.wav --no-asr
voice-demo analyze --profile PROFILE_ID meeting.wav --num-speakers 3

# 删除模板
voice-demo delete-profile PROFILE_ID
```

批量评测清单为 JSONL，一行一条测试。`audio` 的相对路径以清单所在目录为基准：

```jsonl
{"profile_id":"替换为真实ID","audio":"test/me_next_day.wav","expected":"self","condition":"phone"}
{"profile_id":"替换为真实ID","audio":"test/other_person.wav","expected":"other","condition":"phone"}
```

```bash
voice-demo evaluate trials.jsonl --output data/evaluation.json
```

评测必须包含本人和他人。静音/短音频会计入 `unknown`，本人被判 `unknown` 会计入本人漏识。坏文件或模型运行错误会让整次评测失败，不会静默删除困难样本来美化指标。

## 数据与清理

默认写入当前项目的 `data/`，可用 `DATA_DIR` 修改。请始终从项目目录启动以保持相同路径。

```text
data/
  profiles/<id>.json       声纹向量、聚合模板、录入质量和模型版本
  jobs/<job-id>/           分析报告、音频片段和 ZIP
  cache/                  下载模型和运行缓存
  tmp/                    界面上传、录音和下载的临时副本
```

录入原文件不会复制到声纹库，但浏览器上传和录音会产生临时副本。删除声纹只删除对应模板，不自动删除历史分析结果、模型缓存或临时文件。关闭程序后，可按需删除对应的 `data/jobs` 和 `data/tmp`；声纹向量本身也应视为敏感数据。

界面仅监听 `127.0.0.1`，关闭公网分享。当前无用户鉴权、数据库加密、并发多租户隔离；不要直接部署为公网服务。

## 已知边界

- 录入和单段验证使用 Silero VAD 去除非语音，**VAD 不能确认身份，也不能可靠验证只有一个说话人**。多人音频应使用整段分析。
- 只使用长度至少 1.5 秒的连续语音窗口；窗口最长 5 秒，不跨静音或另一个说话人拼接。过多短促应答可能导致 unknown。
- 削波比例超过 2% 的窗口跳过；没有完整 SNR 估计、去噪或自动录入者校验。
- 整段身份来自说话人簇；局部分段污染即使通过窗口一致性检查，仍可能导致误认。
- 重叠区域不用于声纹提取。含重叠的导出 turn 保守标为 unknown，词级结果仅对实际重叠的词标 unknown。原始分段、簇级结果都保留，便于诊断。
- 不做人声分离、不做实时流式识别、不做反重放或合成声音检测。
- 当前没有 ECAPA 横向对比、自动阈值训练、EER/DER/CER 计算、置信区间或商用验收自动化。

运行测试：`python -m pytest -q`。测试中的模拟模型只验证流程和边界，不代表真实模型准确率。
