# 验证记录

日期：2026-10-23。环境：macOS Apple Silicon、Python 3.12.4、CPU。

## 已验证

- `pip check`：没有损坏的依赖关系。
- `voice-demo doctor`：pyannote Model/Pipeline 可导入，Silero ONNX VAD 可加载，CUDA 不可用。
- 自动测试：26 项通过。覆盖余弦归一化、无效向量、阈值边界、短语音拒识、窗口冲突、重叠剔除、说话人时间对齐、立体声重采样、录音时长限制、声纹版本不匹配、模板隔离删除、ZIP/WAV 导出、关闭 ASR、无语音结果及 unknown 统计口径。
- Gradio 界面成功构建并在 `127.0.0.1:7860` 启动；浏览器检查录入、单段对比、整段分析、报告下载及片段回听控件；未上传文件时会提示输入错误。
- 真实 WeSpeaker 权重加载和推理通过，输出 256 维向量。权重 revision：`837717ddb9ff5507820346191109dc79c958d614`。
- 真实 VoiceService 录入、模板保存和再次加载验证通过。为达到录入最低时长，仅在冒烟测试中重复拼接公开短样本：同录音得到 self，另一公开样本得到 other。重复录音不构成独立测试，不能作为泛化准确率证据。
- 真实 Silero VAD 检测公开样本得到约 2.84 秒语音。
- 真实 Whisper **tiny** 英文样本转写通过，得到 7 个带时间戳的词。这里只用 tiny 检查调用链；默认 large-v3 尚未下载和实际验证。

真实模型测试使用 SpeechBrain 仓库公开的 `tests/samples/ASR/spk1_snt1.wav` 和 `spk2_snt1.wav`，未使用用户私人音频。公开样本仅保存在开发工作目录，不打包进 Demo。

同一向量与自身比较约为 1；两段不同公开样本的向量相似度约为 0.129。**这些数字仅证明模型与比较代码能够工作，不是识别准确率，也不能用于选择生产阈值。**

## 尚未验证

- Community-1 的真实权重推理：当前未配置 HF_TOKEN，模型访问还需要用户本人接受 Hugging Face 使用条件。分段后的流程使用模拟后端进行了集成测试，不等同于完成真实分段测试。
- 用户本人录入、跨会话识别、中文转写准确率、噪声和跨设备泛化。
- large-v3、CUDA、Windows/Linux 和其他 Python 版本的实际运行。
- 商用 FAR/FRR、DER/CER、统计置信区间、吞吐和成本。
- 麦克风录音权限与设备采集：当前嵌入式浏览器显示无可用麦克风；可先上传文件，或用系统浏览器打开 localhost 并由用户授权录音。

自动测试中的 FakeBackend 明确只是流程测试替身。应用运行时始终使用真实 LocalBackend，不会用模拟分数冒充模型结果。

## 复现

```bash
python -m pip install -e '.[test]'
python -m pytest -q
voice-demo doctor
voice-demo web
```

本次完整依赖快照见 `requirements-tested-macos.txt`。模型下载及真实测试需要网络/缓存；Community-1 还需要账号模型访问权限。真实用户的准确率验证应按 `TECHNICAL_DESIGN.md` 的数据隔离和评测方法开展。
