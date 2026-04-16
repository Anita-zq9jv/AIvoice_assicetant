import gradio as gr
import requests
import os
import whisper
from gtts import gTTS
import tempfile

# ===== API 配置 =====
API_URL = "https://api-inference.modelscope.cn/v1/chat/completions"
API_KEY = os.getenv("SILICONFLOW_API_KEY", "key")  # 请确保环境变量已设置
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528"

# ===== 加载 Whisper 模型（全局加载一次，避免重复加载）=====
print("正在加载 Whisper 模型...")
whisper_model = whisper.load_model("base")  # 可选 "tiny", "base", "small", "medium", "large"
print("模型加载完成。")

# ===== LLM =====
def call_deepseek_api(messages):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": MODEL_ID,
        "messages": messages,
    }
    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"API错误: {response.text}"

# ===== ASR（使用本地 Whisper）=====
def asr(audio_file):
    if audio_file is None:
        return ""
    # 使用 Whisper 转录音频文件
    result = whisper_model.transcribe(audio_file)
    return result["text"]

# ===== TTS =====
def tts(text):
    if not text:
        return None
    tts_obj = gTTS(text, lang="zh")  # 可指定语言，如中文 "zh"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts_obj.save(tmp_file.name)
    return tmp_file.name

# ===== 核心对话（带记忆）=====
def chat(user_input, history):
    if history is None:
        history = []

    messages = [
        {"role": "system", "content": "你是一个有情感记忆的AI助手"}
    ]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": user_input})

    reply = call_deepseek_api(messages)
    history.append((user_input, reply))
    audio = tts(reply)

    # 新messages格式, 符合 gr.Chatbot 要求
    chatbot_history = []
    for h in history:
        chatbot_history.append({"role": "user", "content": h[0]})
        chatbot_history.append({"role": "assistant", "content": h[1]})

    return chatbot_history, history, audio

# ===== UI =====
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ PMAOS语音助手 Demo")

    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath")
        text_input = gr.Textbox(label="识别出的文本 / 手动输入")

    send_btn = gr.Button("发送")
    audio_output = gr.Audio(label="AI 语音回复", autoplay=True)

    # 语音 → 文本
    audio_input.change(
        fn=asr,
        inputs=audio_input,
        outputs=text_input
    )

    # 文本 → 对话 + TTS
    send_btn.click(
        fn=chat,
        inputs=[text_input, state],
        outputs=[chatbot, state, audio_output]
    )

# 本地运行（share=True 可能会因网络问题失败，不影响本地使用）
demo.launch(share=False)  # 若需公网分享可改为 share=True
