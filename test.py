import os
import requests

audio_folder = "dataset/vetc_data/audio"
asr_api_url = "http://10.201.25.13:8000/transcribe"

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        response = requests.post(asr_api_url, files=files)
    if response.status_code == 200:
        data = response.json()
        return data.get("transcript", "")
    else:
        print(f"ASR API error for {audio_path}: {response.status_code}")
        return ""

def create_transcripts_from_folder(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
            audio_path = os.path.join(folder, filename)
            transcript = transcribe_audio(audio_path)
            txt_path = os.path.splitext(audio_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"Created transcript: {txt_path}")

if __name__ == "__main__":
    create_transcripts_from_folder(audio_folder)