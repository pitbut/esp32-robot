import os
import io
import random
import requests
from flask import Flask, Response, request, jsonify
from gtts import gTTS
from pydub import AudioSegment
import static_ffmpeg

static_ffmpeg.add_paths()

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_STT_KEY    = os.environ.get("GOOGLE_STT_KEY", "")

# История разговора (последние 10 сообщений)
conversation_history = []

@app.route('/')
def index():
    return "ESP32 AI Robot Server работает!"

# ===== STT: аудио → текст (Google Speech-to-Text) =====
@app.route('/stt', methods=['POST'])
def speech_to_text():
    if not request.data:
        return jsonify({"error": "нет аудио данных"}), 400

    audio_data = request.data
    print(f"[STT] Получено аудио: {len(audio_data)} байт")

    try:
        # Конвертируем WAV в формат для Google STT
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_STT_KEY}"

        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": "ru-RU",
                "model": "default"
            },
            "audio": {
                "content": __import__('base64').b64encode(audio_data).decode('utf-8')
            }
        }

        response = requests.post(url, json=payload, timeout=10)
        result = response.json()

        if "results" in result and result["results"]:
            text = result["results"][0]["alternatives"][0]["transcript"]
            print(f"[STT] Распознано: {text}")
            return jsonify({"text": text})
        else:
            print(f"[STT] Не распознано: {result}")
            return jsonify({"text": ""}), 200

    except Exception as e:
        print(f"[STT] Ошибка: {e}")
        return jsonify({"error": str(e)}), 500

# ===== AI: текст → ответ Claude =====
@app.route('/ai', methods=['POST'])
def ai_response():
    data = request.get_json()
    user_text = data.get("text", "").strip()

    if not user_text:
        return jsonify({"error": "пустой текст"}), 400

    print(f"[AI] Вопрос: {user_text}")

    # Добавляем в историю
    conversation_history.append({
        "role": "user",
        "content": user_text
    })

    # Оставляем только последние 10 сообщений
    if len(conversation_history) > 10:
        conversation_history.pop(0)

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 150,
                "system": "Ты голосовой помощник робота. Отвечай коротко и по-русски, максимум 2-3 предложения. Ты умный, дружелюбный и немного с юмором.",
                "messages": conversation_history
            },
            timeout=15
        )

        result = response.json()
        answer = result["content"][0]["text"]
        print(f"[AI] Ответ: {answer}")

        # Добавляем ответ в историю
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        return jsonify({"text": answer})

    except Exception as e:
        print(f"[AI] Ошибка: {e}")
        return jsonify({"text": "Извини, не смог ответить."}), 500

# ===== TTS: текст → MP3 =====
@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return Response(status=400)

    print(f"[TTS] Говорю: {text}")

    try:
        tts = gTTS(text=text, lang='ru')
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        return Response(mp3_buf.read(), mimetype='audio/mpeg')

    except Exception as e:
        print(f"[TTS] Ошибка: {e}")
        return Response(status=500)

# ===== Полный цикл: аудио → текст → AI → MP3 =====
@app.route('/talk', methods=['POST'])
def full_pipeline():
    if not request.data:
        return Response(status=400)

    audio_data = request.data
    print(f"[TALK] Аудио: {len(audio_data)} байт")

    # Шаг 1: STT
    try:
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_STT_KEY}"
        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": "ru-RU"
            },
            "audio": {
                "content": __import__('base64').b64encode(audio_data).decode('utf-8')
            }
        }
        stt_resp = requests.post(url, json=payload, timeout=10).json()

        if "results" not in stt_resp or not stt_resp["results"]:
            print("[TALK] Речь не распознана")
            user_text = "Привет"
        else:
            user_text = stt_resp["results"][0]["alternatives"][0]["transcript"]
            print(f"[TALK] Распознано: {user_text}")

    except Exception as e:
        print(f"[TALK] STT ошибка: {e}")
        user_text = "Привет"

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
                "system": "Ты голосовой помощник робота. Отвечай коротко и по-русски, максимум 2-3 предложения.",
                "messages": conversation_history
            },
            timeout=15
        ).json()

        answer = ai_resp["content"][0]["text"]
        print(f"[TALK] Ответ AI: {answer}")

        conversation_history.append({"role": "assistant", "content": answer})

    except Exception as e:
        print(f"[TALK] AI ошибка: {e}")
        answer = "Извини, произошла ошибка."

    # Шаг 3: gTTS → MP3
    try:
        tts = gTTS(text=answer, lang='ru')
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        mp3_data = mp3_buf.read()
        print(f"[TALK] MP3: {len(mp3_data)} байт")
        return Response(mp3_data, mimetype='audio/mpeg')

    except Exception as e:
        print(f"[TALK] TTS ошибка: {e}")
        return Response(status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
