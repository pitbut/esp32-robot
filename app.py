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

@app.route('/talk', methods=['POST'])
def full_pipeline():
    if not request.data:
        return Response(status=400)

    audio_data = request.data
    print("[TALK] Аудио: " + str(len(audio_data)) + " байт")

    # Шаг 1: Whisper STT
    user_text = "привет"
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(audio_data)
        tmp.close()

        import whisper
        model = whisper.load_model("tiny")
        result = model.transcribe(tmp.name, language="ru")
        user_text = result["text"].strip()
        print("[STT] Распознано: " + user_text)
        os.unlink(tmp.name)

    except Exception as e:
        print("[STT] Ошибка: " + str(e))

    if not user_text:
        user_text = "привет"

    # Шаг 2: Claude AI
    answer = "Извини, не смог ответить."
    try:
        conversation_history.append({"role": "user", "content": user_text})
        if len(conversation_history) > 10:
            conversation_history.pop(0)

        resp = requests.post(
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
        )

        data = resp.json()
        answer = data["content"][0]["text"]
        print("[AI] Ответ: " + answer)

        conversation_history.append({"role": "assistant", "content": answer})

    except Exception as e:
        print("[AI] Ошибка: " + str(e))

    # Шаг 3: gTTS -> MP3
    try:
        tts = gTTS(text=answer, lang='ru')
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        return Response(mp3_buf.read(), mimetype='audio/mpeg')

    except Exception as e:
        print("[TTS] Ошибка: " + str(e))
        return Response(status=500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
