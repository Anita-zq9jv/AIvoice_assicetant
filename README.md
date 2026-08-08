# 🎙️ AI Voice Assistant

> 基于 Whisper + ModelScope + Gradio 的语音助手 —— 语音识别 → 智能对话 → 语音合成

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Whisper](https://img.shields.io/badge/Whisper-Local-green)]()
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)]()

## 📋 项目背景

**目标**：构建一个端到端的语音交互助手，用户通过语音输入问题，AI 识别后进行智能回答，并以语音形式返回结果。

**技术亮点**：
- 语音识别使用 OpenAI Whisper 本地部署，支持中英文混合识别
- 智能对话使用 ModelScope 大模型 API
- 语音合成使用 Edge TTS，支持多种音色
- 前端使用 Gradio 快速搭建交互界面

## ✨ 功能特性

- 🎤 **语音输入** —— 支持实时录音或上传音频文件
- 🗣️ **智能识别** —— Whisper 本地识别，支持中英文混合
- 🤖 **AI 对话** —— 接入 ModelScope 大模型，支持多轮对话
- 🔊 **语音合成** —— Edge TTS 语音输出，支持多种音色
- 🌐 **Web 界面** —— Gradio 搭建，一键部署

## 🏗️ 技术架构

```
用户语音输入
    ↓
[Whisper 本地识别] → 文本
    ↓
[ModelScope API] → AI 回答
    ↓
[Edge TTS] → 语音合成
    ↓
语音输出
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- FFmpeg（Whisper 依赖）
- ModelScope API Key

### 安装

```bash
git clone https://github.com/Anita-zq9jv/AIvoice_assicetant.git
cd AIvoice_assicetant
pip install -r requirements.txt
```

### 配置

1. 注册 [ModelScope](https://modelscope.cn/) 账号并获取 API Key
2. 下载 Whisper 模型到本地
3. 安装 FFmpeg

### 运行

```bash
python main.py

# 浏览器访问 http://localhost:7860
```

## 📁 项目结构

```
.
├── main.py              # 主程序
├── requirements.txt     # 依赖列表
└── README.md            # 项目说明
```

## 🔧 技术细节

### 语音识别（Whisper）
- 使用 OpenAI Whisper 本地部署
- 支持 tiny/base/small/medium/large 多种模型
- 推荐使用 small 模型平衡速度和准确率

### 智能对话（ModelScope）
- 接入通义千问系列模型
- 支持多轮对话上下文管理
- 可自定义 system prompt 调整 AI 人设

### 语音合成（Edge TTS）
- 使用微软 Edge TTS 引擎
- 支持中文（zh-CN）、英文（en-US）等多种语言
- 可选音色：晓晓、云扬、晓萱等

## 📄 License

MIT License
