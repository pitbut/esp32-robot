import os
import io
import time
import requests
from flask import Flask, Response, request
from gtts import gTTS
from pydub import AudioSegment
import static_ffmpeg

static_ffmpeg.add_paths()

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ASSEMBLYAI_KEY    = os.environ.get("ASSEMBLYAI_KEY", "")
conversation_history = []

@app.route('/')
def index():
    return "ESP32 AI Robot v2.0 работает!"

@app.route('/talk', methods=['POST'])
def full_pipeline():
    if not request.data:
        return Response(status=400)

    audio_data = request.data
    print("[TALK] Аудио: " + str(len(audio_data)) + " байт")

    # Шаг 1: AssemblyAI STT
    user_text = "привет"
    try:
        # Загружаем аудио
        headers = {"authorization": ASSEMBLYAI_KEY}
        upload = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            data=audio_data,
            timeout=30
        )
        audio_url = upload.json()["upload_url"]
        print("[STT] Загружено: " + audio_url)

        # Запрашиваем транскрипцию
        transcript = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            headers=headers,
            json={"audio_url": audio_url, "language_code": "ru"},
            timeout=30
        ).json()

        transcript_id = transcript["id"]
        print("[STT] ID: " + transcript_id)

        # Ждём результат
        for _ in range(30):
            result = requests.get(
                "https://api.assemblyai.com/v2/transcript/" + transcript_id,
                headers=headers,
                timeout=10
            ).json()

            if result["status"] == "completed":
                user_text = result.get("text", "привет") or "привет"
                print("[STT] Распознано: " + user_text)
                break
            elif result["status"] == "error":
                print("[STT] Ошибка: " + result.get("error", ""))
                break
            time.sleep(2)

    except Exception as e:
        print("[STT] Ошибка: " + str(e))

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
        ).json()

        answer = resp["content"][0]["text"]
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

        audio = AudioSegment.from_mp3(mp3_buf)
        audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
        return Response(audio.raw_data, mimetype='application/octet-stream')

    except Exception as e:
        print("[TTS] Ошибка: " + str(e))
        return Response(status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
