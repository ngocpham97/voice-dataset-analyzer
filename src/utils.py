def calculate_mean(values):
    if not values:
        return 0
    return sum(values) / len(values)

def normalize_value(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def is_audio_format_supported(sample_rate):
    supported_rates = [16000, 22050, 24000, 44100]
    return sample_rate in supported_rates

def check_clipping(audio_data):
    return max(abs(audio_data)) < 1.0

def calculate_speech_ratio(speech_segments, total_duration):
    if total_duration == 0:
        return 0
    return sum(speech_segments) / total_duration

def calculate_overlap_ratio(overlap_segments, total_segments):
    if total_segments == 0:
        return 0
    return sum(overlap_segments) / total_segments

def check_transcript_format(transcript):
    # Implement checks for transcript formatting consistency
    return True  # Placeholder for actual implementation

def calculate_cer(reference, hypothesis):
    # Implement Character Error Rate calculation
    return 0.0  # Placeholder for actual implementation

def calculate_wer(reference, hypothesis):
    # Implement Word Error Rate calculation
    return 0.0  # Placeholder for actual implementation

def is_transcript_available(transcript):
    return bool(transcript)