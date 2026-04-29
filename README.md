# voicebridge

把 Claude Code 接到麦克风和扬声器,让你不用敲键盘就能跟它对话。
专属 VSCode 实例后台跑,你说话 → 转写 → 多模态分析 → Claude 回 → 朗读。

---

## 功能

### 对话流

- **唤醒词触发**:可自定义任意词(默认 `小助手` / `嘿 Claude` / `hey claude`)。识别到自动进入对话状态
- **多轮对话**:进入对话后,你说话 → Claude 回 → 你接着说,中间不需要重新唤醒
- **结束词回 idle**:说自定义结束词(默认 `再见` / `结束对话` / `拜拜`),Claude 简短告别,然后回到等唤醒词的状态。下次说话又要先唤醒
- **静默切片**:silero-VAD 检测语音端点,2 秒静默切一段,4 秒静默把当前一轮提交给 Claude
- **Wake pre-roll**:把唤醒词检测前 2.5 秒的音频接到指令前面,避免"小助手 + 紧接的指令头"被吞
- **快捷键打断**:TTS 朗读期间按 Ctrl+Shift+空格(可改),立刻停 TTS,你下一句话会带 carry-over 上下文重提交给 Claude

### 多模态分析

每一轮录音除了转写,还会跑这些信号一起送给 Claude(以 `[副语音:…]` 块附在转写后):

- **语速** 字/秒、**音量** RMS — librosa
- **音高** F0 平均/范围/标准差、**jitter** / **shimmer** — parselmouth(Praat 包装)
- **背景声分类** — YAMNet 521 类(室内安静/键盘敲击/狗叫/音乐 等),带常见标签的中文翻译
- **情绪标签** — emotion2vec(中文)或 superb wav2vec2(英文),按 ASR 检测到的语言自动选

### TTS

可插拔后端,用户挂自己的模型也行:

- `edge`:Microsoft Edge TTS,在线、免费、自然中文女/男声(默认 `zh-CN-XiaoyiNeural`)
- `openai`:任何兼容 OpenAI `/v1/audio/speech` 的 HTTP 端点(本地 vLLM、ElevenLabs OpenAI 模式、xtts-api-server …)
- `python`:用户自己写一个 Python 文件实现 `make_backend(**kwargs) -> TTSBackend`,挂任意本地模型(GPT-SoVITS、XTTS、Piper、Bark …)。模板见 `examples/my_tts_template.py`

### 系统集成

- **专属 VSCode 实例**:程序启动时如果还没在跑,自动开一个隔离的 VSCode 窗口(独立 user-data-dir + extensions-dir,完全不影响你平时用的 VSCode);Claude Code 跑在它的集成终端里
- **VSCode ↔ 程序双向联动**:你关掉那个 VSCode 窗口,程序自动退出;你从托盘退出程序,VSCode 也跟着关
- **回声消除(可选)**:WASAPI loopback 拿扬声器输出,频域块 NLMS 自适应滤波,提升 TTS 期间录入信噪比
- **API 错误兜底**:Claude API 返回 content filter / 4xx / 5xx 时不会朗读 raw 错误文本,而是说一句友好回复,对话继续
- **配置热加载**:GUI 改完配置(唤醒词、热键、TTS 后端、emotion 模型等)即时生效,无需重启
- **系统托盘**:状态彩色指示(绿=空闲/橙=录入/紫=提交/蓝=朗读),双击托盘 → 弹 GUI,右键 → 设置/退出

### 启动方式

`autostart.py install-all` 一键装三个入口:

- 开机自动启动(隐藏窗口、跳过 GUI、用上次保存的配置)
- 桌面快捷方式(双击 → GUI 弹 → 保存并启动)
- 开始菜单条目(Win 键搜索 voicebridge 找到)

---

## 环境要求

| 项 | 版本 / 说明 |
| --- | --- |
| 操作系统 | Windows 10 / 11 |
| Python | 3.12+(实测 3.12.10) |
| Node.js | 18+(VSCode 扩展用) |
| ffmpeg | 任意近期版本,`ffplay` 必须在 `PATH`(`winget install Gyan.FFmpeg`) |
| VSCode | 任意近期版本,`code` CLI 在 PATH |
| Claude Code 扩展 | 用户级安装并登录(主 VSCode 装一次即可;程序会复制到隔离实例) |
| GPU | NVIDIA + CUDA 12.x(推荐 sm_80+;CPU 模式也能跑但会卡顿) |
| 显存 | ≥ 4 GB(同时驻 small + large-v3-turbo 约 3 GB) |
| 麦克风 | 任意输入设备 |
| 扬声器 | 输出 + 可选 loopback 设备(用于 AEC,如启用) |

### Python 依赖

```text
faster-whisper
silero-vad
sounddevice
onnxruntime-gpu
torch torchaudio
nvidia-cublas-cu12
nvidia-cudnn-cu12
nvidia-cuda-nvrtc-cu12
edge-tts
librosa
praat-parselmouth
tensorflow-cpu
tensorflow-hub
transformers
funasr           # 中文情绪 emotion2vec
pywin32          # Windows 快捷方式 + COM
pystray          # 系统托盘
pillow           # 托盘图标
keyboard         # 备用 hotkey 库(实际用 Win32 API)
```

一行装好(国内可换 `-i https://pypi.tuna.tsinghua.edu.cn/simple`):

```bash
pip install faster-whisper silero-vad sounddevice onnxruntime-gpu torch torchaudio \
  nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 \
  edge-tts librosa praat-parselmouth tensorflow-cpu tensorflow-hub transformers \
  funasr pywin32 pystray pillow keyboard
```

---

## 初次使用

### 1. 准备主 VSCode 的 Claude Code(必做一次)

把 voicebridge 解压到任意位置(下面假设 `D:\apps\voicebridge\`),它需要从你的主 VSCode 复制 Claude Code 扩展:

- 在你常用的 VSCode 里安装 `Anthropic.claude-code` 扩展
- 用它登录你的 Claude 账号(扩展会引导浏览器登录)
- 至少打开一次 Claude Code panel,确认能用(避免后面在隔离实例里再走一遍登录)

### 2. 安装依赖

```bash
cd D:\apps\voicebridge
pip install -r 上面那串包
```

如果 NVIDIA 驱动已就绪,`ctranslate2` 会自动找到 CUDA。第一次运行时它检测不到 cuBLAS DLL 会报错;装完上面的 `nvidia-cublas-cu12` 等 wheel 后 voicebridge 启动时会自动把 DLL 路径加到 `PATH`。

### 3. 第一次启动

```bash
python src/main.py
```

或者直接双击 GUI 启动器(下一节)。

启动后:

1. **GUI 配置窗弹出** — 选 workspace 路径(Claude Code 要操作哪个项目目录),填唤醒/结束词,选麦克风,选 TTS 声音,等等。点"保存并启动"
2. **隔离 VSCode 自动打开**(如果没在跑) — 这个 VSCode 实例只为程序服务,跟你日常用的那个 VSCode 是两个隔离实例
3. **隔离 VSCode 里 Claude Code 首次 onboarding**(只此一次) — 在它的集成终端里手动走完:
    - 选主题(Dark / Light 等):任意,Enter 确认
    - 登录:`Sign in with Anthropic`(自动开浏览器)
    - "Use recommended terminal setup?":**必选 1 (Yes)**(不选会让程序推送 prompt 时无法自动提交)
    - "Trust this folder?":选 1(Yes)
4. **你的程序就 ready** — 任务栏托盘有绿色 W 图标,意味着在等唤醒词

如果 emotion 中文打开了,第一次会从 modelscope 下载 ~1.4 GB 的 emotion2vec 模型(异步、不阻塞主流程,但下载完之前 emotion 字段为空)。

### 4. 装快捷方式(推荐)

**双击 `install.bat`** —— 一键装好下面三个入口:

- 桌面 `voicebridge.lnk`(双击 → 弹 GUI → 保存并启动)
- 开始菜单 `voicebridge`(Win 键搜索找到)
- 启动文件夹 `voicebridge.lnk`(开机自动后台启动,跳过 GUI 用上次配置)

要解除:**双击 `uninstall.bat`** 或者手动到对应目录删 `.lnk`。

也支持命令行:

```bash
python autostart.py status              # 查看当前安装状态
python autostart.py install-all         # 装全部
python autostart.py desktop             # 只装桌面
python autostart.py uninstall-all       # 全部移除
```

### 5. 日常使用

- **说唤醒词**(例如"小助手") → 等托盘图标变橙色 → 接着说指令
- **多轮对话**:Claude 回完你直接说下一句,不必再唤醒
- **打断 TTS**:按 Ctrl+Shift+空格(可在 GUI 改)
- **结束对话**:说"再见"(或你配的结束词) → Claude 简短告别 → 回空闲
- **改配置**:双击托盘 → GUI 改完保存 → 大多数字段(唤醒词/热键/TTS 等)立即生效,极个别字段(麦克风设备、ASR 模型)要重启程序

---

## 运行原理

### 整体架构

```
       ┌─────────────────────────────────────────────┐
       │  voicebridge/src/main.py                    │
       │  ─ asyncio 主进程,常驻                     │
       │  ─ 状态机 idle ↔ active ↔ submitting ↔     │
       │       speaking                              │
       └────────┬─────────┬─────────┬────────────┬──┘
                │         │         │            │
        ┌───────▼──┐ ┌────▼────┐ ┌──▼──────┐ ┌──▼──────────┐
        │ 麦克风    │ │ 监听    │ │ 主转写  │ │ TTS 朗读    │
        │ +AEC     │ │ small  │ │ turbo  │ │ ffplay      │
        │ silero   │ │ +分析  │ │+元数据  │ │             │
        └──────────┘ └─────────┘ └────┬────┘ └─────────────┘
                                      │
                                ┌─────▼─────────────────────┐
                                │ HTTP+SSE :43117           │
                                │ POST /prompt              │
                                │ GET  /events              │
                                │ POST /shutdown            │
                                │ GET  /status              │
                                └─────┬─────────────────────┘
                                      │
                ┌─────────────────────▼─────────────────────┐
                │  隔离 VSCode 实例                         │
                │  ─ user-data-dir + extensions-dir 隔离    │
                │  ─ voice-bridge 扩展(本仓提供)         │
                │  ─ Claude Code 扩展(从主用户复制)      │
                │  ─ 集成终端跑 `claude` CLI               │
                │  ─ 监听 ~/.claude/projects/*/*.jsonl     │
                │    把 Claude 回复 SSE 推回主进程         │
                └─────────────────────────────────────────────┘
```

### 状态机

```
            wake word
   IDLE ────────────────► ACTIVE  ◄──┐
    ▲                       │       │
    │                       │ 4s 静默 / 8s 静默
    │                       ▼       │
    │                   SUBMITTING  │
    │                       │       │ 用户接着说话
    │                       │       │
    │                  收到 SSE     │
    │                       │       │
    │                       ▼       │
    │                   SPEAKING ───┘  TTS 自然结束
    │                       │
    │                       │ 用户说结束词 → claude 回 farewell → TTS
    └───────────────────────┘
```

- IDLE 状态下 watcher 监听唤醒词
- ACTIVE 状态下 watcher 监听结束词,segmenter 录用户音频
- SPEAKING 状态下 watcher 静音(避免 Claude 自己 TTS 出来的"再见"误触结束词)
- 任何 ACTIVE/SPEAKING 状态下按打断热键 → 立即停 TTS,记下 carry-over 上下文,下一轮提交时拼到 prompt 前

### Prompt 组装

每一轮提交给 Claude 的 prompt 长这样:

```
小助手帮我看一下今天有什么会议要开

[副语音:语速 2.4 字/秒,音量 中 · 音高 中,起伏自然 · 嗓音稍紧 · 环境声 安静(0.62) · 人声(0.31) · 室内/小空间(0.04) · 情绪标签:hap (0.85) (zh)]
```

主要是转写文本,`[副语音:…]` 是给 Claude 的次要参考。Claude 系统提示里没硬塞规则,但实测它会根据这些信号自动调整回复(比如 jitter/shimmer 异常时主动关心、情绪 sad 时回得轻、急切时回得短)。

### 多模型并行

提交一段录音后(比如 6 秒),后台并行跑这几件事(都在 GPU 或 CPU 多核):

- **faster-whisper large-v3-turbo** 转写(GPU,fp16,典型 6 秒音频 → 0.3 秒)
- **librosa + parselmouth** 计算 RMS / 谱重心 / F0 / jitter / shimmer(CPU,毫秒级)
- **YAMNet** 背景声分类(TF Lite-ish,CPU 几十毫秒)
- **emotion2vec(中)/ wav2vec2-superb(英)** 情绪分类(按 ASR 语言选,CPU)

转写完成 → 拼 prompt → POST 给 voice-bridge 的 HTTP 端点。

### 桥接到 Claude Code

`voice-bridge` 是一个 VSCode 扩展(本仓 `vscode-ext/voice-bridge/`):

- `activate()` 时启动 HTTP+SSE server `127.0.0.1:43117`
- 在隔离 VSCode 里 `vscode.window.createTerminal()` 起一个集成终端,自动跑 `claude` CLI
- `POST /prompt {text}` → `terminal.sendText(text)` 把字符串塞进 Claude CLI 的输入框
  - Claude CLI 用 bracketed paste 处理多字符输入,直接发 `\n` 不算 Enter,所以扩展先发文本,延迟 120 ms 再发独立 `\r` 才算提交
- `terminal.sendText` 是公开 VSCode API,**不需要焦点、不模拟系统键盘**,因此完全不影响你前台的别的窗口
- watcher 监听 `~/.claude/projects/<encoded-cwd>/<sessionId>.jsonl` 的增量,有新的 `type:assistant` 记录就 SSE `event: assistant` 推回主进程
  - 同时识别 `isApiErrorMessage:true` 的 synthetic error 记录,推 `event: error`,主进程把它替换成短句 fallback,不让 TTS 朗读 raw error
- `POST /shutdown` 让扩展执行 `vscode.commands.executeCommand('workbench.action.quit')` 自杀

### VSCode 实例隔离

`launcher.py` 启动 VSCode 用:

```bash
code --user-data-dir <install-dir>/vscode-data \
     --extensions-dir <install-dir>/vscode-ext \
     --new-window <workspace>
```

这两个 flag 让该实例完全跟你日常 VSCode 隔离 — 设置、最近文件、扩展等都在 voicebridge 目录里,不污染你的用户配置。`vscode-ext/` 里同时存放:

- `voice-bridge/` — 我们的 HTTP 桥扩展,源码方式直接被 VSCode 扫描加载
- `anthropic.claude-code-<version>-...` — 第一次启动时从你主用户的 `~/.vscode/extensions/` 复制过来,后续随主版本自动同步(mtime 比较)

用户级 MCP server(在 `~/.claude.json`)在所有 Claude CLI 实例间共享,所以你之前在主 VSCode 里 `claude mcp add` 装的 MCP(比如 computer-control 截屏、各种自定义工具)在隔离实例的 Claude session 里也能用。扩展启动 claude 时会等 3 秒让 PowerShell 完成 profile 加载再发 `claude` 命令,避免 cold-start race 让 user-level MCP 启动失败。

### 回声消除(可选)

如果不戴耳机:扬声器播放 TTS 的声音会被麦克风录回去,影响后续录入和未来的 voice 关键词识别。voicebridge 的 AEC 用:

1. WASAPI loopback 拿扬声器同时刻输出(`audio.loopback_device` 配置一个"立体声混音"或类似设备)
2. 频域块 NLMS 自适应滤波器(`audio_io.py: FBNLMS`)— 学习扬声器到麦克风的房间冲击响应,从麦克风信号里减掉
3. 输出的"clean mic"才送给 silero-VAD 和主转写

NLMS 收敛需要 1-3 秒 TTS 播放,首次效果一般,持续对话越来越好。语音端识别短关键词("停"等)在 cleaned 信号上 whisper-small 仍不稳,因此 voicebridge 用全局热键作为唯一的稳定打断方式。

### 各文件职责

| 文件 | 职责 |
| --- | --- |
| `src/main.py` | 主状态机,串起所有模块,生命周期 |
| `src/audio_io.py` | 麦克风采集(sounddevice),silero VAD,2s/4s 切片,AEC NLMS,WASAPI loopback |
| `src/wakeword.py` | 滑动窗口 + small whisper + 关键词字符串匹配 |
| `src/transcribe.py` | faster-whisper 包装,small + large-v3-turbo 双模型 |
| `src/analyze.py` | librosa + parselmouth 计算副语音特征 |
| `src/sound_tag.py` | YAMNet 521 类背景声 |
| `src/emotion.py` | emotion2vec(zh)+ wav2vec2 superb(en)双模型,按语言路由 |
| `src/tts.py` | 三种 TTS 后端 + ffplay 流式播放 |
| `src/bridge_client.py` | HTTP/SSE 客户端,POST /prompt 和监听 SSE |
| `src/launcher.py` | 启动隔离 VSCode + 复制 Claude Code 扩展 |
| `src/win_hotkey.py` | Win32 RegisterHotKey 全局热键 |
| `src/tray.py` | pystray 系统托盘(双击/右键/状态颜色) |
| `autostart.py` | 三种 .lnk 快捷方式安装/卸载(顶层,自包含,无 src/ 依赖) |
| `install.bat` / `uninstall.bat` | 双击安装 / 卸载快捷方式 |
| `src/startup_gui.py` | tkinter 启动配置界面 |
| `src/cuda_setup.py` | NVIDIA cuBLAS/cuDNN DLL 路径引导(必须在导入 ctranslate2 之前) |
| `src/paths.py` | 所有路径集中管理,使包可移植 |
| `vscode-ext/voice-bridge/extension.js` | VSCode 扩展:HTTP+SSE,Claude 终端,transcript 监听 |

### 配置热加载

main.py 起一个 task 每 1.5 秒看 `config.json` 的 mtime。变了就重新读,对比新旧值,可以热加载的字段直接换:

- 唤醒词 / 结束词 → watcher 字段
- 打断热键 → 卸载旧的、注册新的(Win32)
- TTS 后端 → 重建 player 后端实例
- emotion model → 失效缓存,下一轮自动加载新模型

不能热加载的(麦克风设备、ASR 模型选择)会打印一行提示让用户知道改完要重启程序生效。

### 整段对话生命周期(具体一例)

1. 程序启动 → 检查 `bridge` 是否在线,不在 → 调 `launcher.launch_vscode(workspace)` 启隔离 VSCode → 等 `/status` 返回
2. 启动 monitor whisper(small)→ 启 audio pipeline + watcher + SSE listener + tray + hotkey
3. 用户说"小助手,看一下我屏幕"
4. watcher 滑动窗口里 small whisper 转出 "小助手 看一下..." → 命中"小助手" → 触发 `_on_wake_hit`
5. 状态 idle → active,segmenter 开始录音,把 watcher 缓存的 2.5 秒预录音(包含"小助手 看一下我屏幕"完整句子)塞到 segments 头部
6. 用户停说话 4 秒 → segmenter 触发 submit
7. 状态 active → submitting,turbo 转写整段(0.3 秒)→ 同时 librosa/YAMNet/emotion 跑分析
8. 拼 prompt:`<转写文本>\n\n[副语音:…]` → POST `/prompt`
9. voice-bridge 扩展把 prompt 推给隔离 VSCode 终端里的 claude CLI
10. claude 调 `mcp__computer-control__screenshot` 截屏 → LLM 描述 → 回复写到 transcript jsonl
11. 扩展的文件 watcher 抓到新行 → SSE `event: assistant {text:"…"}` 推回主进程
12. 主进程 `_on_assistant_text` 收到 → 状态 submitting → speaking,启动 edge-tts 流式合成 → 通过 ffplay 边收边播
13. TTS 自然结束 → 状态 speaking → active(多轮),segmenter 重新开始录音
14. 用户说"再见" → watcher 命中结束词 → claude 回 farewell → TTS 朗读 → 状态回 idle

整轮端到端延迟 ≈ silence 4 秒(可调) + 转写 0.3 秒 + claude 思考 + TTS 第一个 chunk 出来的时间。

---

## 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| `cublas64_12.dll is not found` | 没装 NVIDIA pip wheels。`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12` |
| 唤醒词没反应 | 麦克风音量太低,主屏右下角声音设置里把"输入音量"拉满;或在 GUI 里换设备 |
| 唤醒词触发但指令头被吞 | 检查 wake pre-roll 配置;或者说唤醒词后稍微停顿再说指令 |
| TTS 不出声 | 看 `.ffplay.log`,常见是输出设备占用或 ffplay 不在 PATH |
| Hotkey 启动失败(winerr 1409) | 别的应用占了同一组合键,改 GUI 里的"停止朗读快捷键" |
| AEC `loopback failed` (-9996) | 在 GUI 选不同的 loopback 设备(常见是"立体声混音"或扬声器名字),或先在 Windows 声音设置启用立体声混音 |
| API 400 content filtering | Claude 偶发拒答,程序会朗读"这条被系统拦了,换个说法",直接换个问法即可 |
| Claude session 启动慢 / MCP 连不上 | voice-bridge 扩展启动 claude 前等 3 秒让 PowerShell profile 完成,首次启动可能仍偶发 — 等几秒重试或 reload window |
| 隔离 VSCode 关掉后程序没退 | 等约 9 秒(三次 ping /status 失败)程序自动退出 |

---

## 不做的事 / 已知限制

- **语音关键词打断已禁用**:whisper-small 在 NLMS-cleaned 信号上对单字"停"识别不稳;raw 信号又会把 TTS 自身的"停"误识别。改用全局热键打断 100% 稳定
- **macOS / Linux**:当前只测了 Windows,大部分代码跨平台,但 Win32 hotkey、WASAPI loopback、`pythonw.exe` 快捷方式、tray 行为是 Windows-only
- **首次 emotion2vec 下载慢**:modelscope CDN 在国内某些网络下到 1 MB/s 以下,模型 ~1.4 GB。建议第一次启动后等下载完(后台异步),期间 emotion 字段为空但其他功能正常

---

## 二次开发 / 推新版

代码结构松散耦合,核心是 `main.py` 的状态机(`State.IDLE/ACTIVE/SUBMITTING/SPEAKING`)和 `_on_*` 系列事件处理器。常见的修改方向:

- **加一个新的多模态分析维度**:写在 `analyze.py` / 新文件,在 `main.py:_on_submit_signal` 里 await 它,把结果拼进 `[副语音:…]` 块
- **换 ASR**:`transcribe.py` 的 `Transcriber` 类对外只暴露 `transcribe_array`,可以替换为别的 ASR(注意 main.py 启动时已加了 `cuda_setup`,导入顺序别打乱)
- **加一个 TTS 后端**:继承 `tts.TTSBackend`,实现 `audio_format` 和 `async stream(text)`,在 `tts.load_backend()` 加路由
- **改提示词组装格式**:`analyze.format_for_prompt` + `main.py` 里拼 `prompt_with_meta` 的那段
- **VSCode 扩展功能扩展**:`vscode-ext/voice-bridge/extension.js` 加 HTTP endpoint;扩展用 `npx vsce package` 打成 .vsix 发布

任何改完 `extension.js` 都要在隔离 VSCode 里 `Ctrl+Shift+P → Developer: Reload Window` 才会生效;改完任何 `src/*.py` 重启 `main.py`(从托盘退出,双击桌面图标重开)。
