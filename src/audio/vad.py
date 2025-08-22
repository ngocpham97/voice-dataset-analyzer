from scipy.io import wavfile
import numpy as np

def read_audio(file_path):
    sample_rate, audio_data = wavfile.read(file_path)
    return sample_rate, audio_data

def compute_speech_ratio(audio_data, threshold=0.01):
    speech_frames = np.sum(np.abs(audio_data) > threshold)
    total_frames = len(audio_data)
    return speech_frames / total_frames if total_frames > 0 else 0

def apply_vad(audio_data, sample_rate, frame_duration=0.02):
    frame_size = int(sample_rate * frame_duration)
    num_frames = len(audio_data) // frame_size
    speech_segments = []

    for i in range(num_frames):
        frame = audio_data[i * frame_size:(i + 1) * frame_size]
        if np.mean(np.abs(frame)) > 0.01:  # Simple energy threshold
            speech_segments.append(frame)

    return np.concatenate(speech_segments) if speech_segments else np.array([])

def get_average_pause_duration(audio_data, sample_rate, threshold=0.01):
    silence_threshold = np.mean(np.abs(audio_data)) * threshold
    pauses = []
    current_pause = 0
    for sample in audio_data:
        if np.abs(sample) < silence_threshold:
            current_pause += 1
        else:
            if current_pause > 0:
                pauses.append(current_pause / sample_rate)
                current_pause = 0
    return np.mean(pauses) if pauses else 0

def analyze_audio(file_path):
    sample_rate, audio_data = read_audio(file_path)
    speech_ratio = compute_speech_ratio(audio_data)
    vad_output = apply_vad(audio_data, sample_rate)
    average_pause_duration = get_average_pause_duration(audio_data, sample_rate)

    return {
        'sample_rate': sample_rate,
        'speech_ratio': speech_ratio,
        'vad_output': vad_output,
        'average_pause_duration': average_pause_duration
    }

def calculate_speech_ratio(audio_data, threshold=0.01):
    return compute_speech_ratio(audio_data, threshold)

