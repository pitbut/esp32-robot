import os
import io
import requests
import tempfile
from flask import Flask, Response, request, jsonify
from gtts import gTTS
import static_ffmpeg

static_ffmpeg.add_paths()

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

conversation_history = []

@app.route('/')
def index():
    return "ESP32 AI Robot работает!"

# ===== Полный цикл: WAV → текст → Claude → MP3 =====
@app.route('/talk', methods=['POST'])
def full_pipeline():
    if not request.data:
        return Response(status=400)

    audio_data = request.data
    print(f"[TALK] Аудио: {len(audio_data)} байт")

    # Шаг 1: Whisper STT (бесплатно, без ключа)
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name

        import whisper
        model = whisper.load_model("tiny")  # tiny = быстро, мало RAM
        result = model.transcribe(tmp_path, language="ru")
        user_text = result["text"].strip()
        print(f"[STT] Распознано: {user_text}")
        os.unlink(tmp_path)

    except Exception as e:
        print(f"[STT] Ошибка: {e}")
        user_text = "привет"

    if not user_text:
        user_text = "привет"

    # Шаг 2: Claude AI
    try:
        conversation_history.append({"role": "user", "content": user_text})
        if len(conversation_history) > 10:
            conversation_history.pop(0)

        ai_resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 150,
                "system": "Ты голосовой помощник робота. Отвечай коротко по-русски, максимум 2 предложения.",
                "messages": conversation_history
            },
            timeout=15
        ).json()

        answer = ai_resp["content"][0]["text"]
        print(f"[AI] Ответ: {answer}")
        conversation_history.append({"role": "assistant", "content": answer})

    except Exception as e:
        print(f"[AI] Ошибка: {e}")
        answer = "Извини, не смог ответить."

    # Шаг 3: gTTS → MP3
    try:
        tts = gTTS(text=answer, lang='ru')
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        return Response(mp3_buf.read(), mimetype='audio/mpeg')

    except Exception as e:
        print(f"[TTS] Ошибка: {e}")
        return Response(status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

И обнови `requirements.txt`:
```
flask
gtts
gunicorn
requests
static-ffmpeg
openai-whisper
torch
