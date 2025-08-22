import numpy as np

from .audio.acoustic_metrics import analyze_audio_metrics
from .audio.diarization import Diarization
from .audio.vad import analyze_audio, calculate_speech_ratio, read_audio
from .classification.l0_classifier import L0Classifier
from .classification.l1_classifier import L1Classifier
from .classification.l2_classifier import L2Classifier
from .scoring.suitability import SuitabilityEvaluator
from .transcript.metrics import calculate_cer, calculate_wer
from .transcript.normalization import normalize_transcript


def evaluate_pipeline(audio_input, transcript_file=None):
    """Evaluate a single audio+transcript pair.

    audio_input may be either:
    - a filesystem path to a wav file (string), or
    - a dataset sample dict (e.g. from HuggingFace datasets) that contains an
      "audio" key (which can be a dict with 'path' or 'array' / 'sampling_rate').

    For quick testing we only process one datum and return a minimal result.
    """
    # If user passed a dataset sample (dict-like), handle it.
    if isinstance(audio_input, dict):
        sample = audio_input
        # Resolve audio array and sampling rate
        audio_field = sample.get('audio') or sample.get('audio_path') or sample.get('path')
        audio_array = None
        sample_rate = None

        if isinstance(audio_field, dict):
            # HuggingFace streaming dataset often provides {'array': ..., 'sampling_rate': ...}
            if 'array' in audio_field:
                audio_array = np.array(audio_field['array'])
                sample_rate = audio_field.get('sampling_rate')
            elif 'path' in audio_field:
                sample_rate, audio_array = read_audio(audio_field['path'])
        elif isinstance(audio_field, str):
            sample_rate, audio_array = read_audio(audio_field)

        if audio_array is None:
            raise ValueError('Could not resolve audio data from the provided sample')

        # Compute simple metrics
        speech_ratio = calculate_speech_ratio(audio_array)
        acoustic_metrics = analyze_audio_metrics(audio_array, 0)

        transcript = sample.get('transcript') or sample.get('text') or sample.get('sentence')

        # Step 2: Normalize transcript
        if transcript:
            normalized_transcript = normalize_transcript(transcript)

            # Step 3: Calculate metrics
            cer = calculate_cer(transcript, normalized_transcript)
            wer = calculate_wer(transcript, normalized_transcript)

            # Step 4: Classify transcript
            l0_classifier = L0Classifier()
            l1_classifier = L1Classifier()
            l2_classifier = L2Classifier()
            l0_label = l0_classifier.classify(normalized_transcript, None)
            l1_label = l1_classifier.classify(normalized_transcript)
            l2_label = l2_classifier.classify({}, {'transcript': normalized_transcript})

            # Step 5: Calculate suitability score
            audio_metrics = {
                'snr': acoustic_metrics.get('SNR', 0),
                'clipping_pct': acoustic_metrics.get('clipping_percentage', 0),
                'speech_ratio': speech_ratio,
                'overlap_ratio': 0.1  # placeholder
            }
            transcript_metrics = {
                'has_transcript': bool(normalized_transcript),
                'cer_baseline': cer,
                'normalization_consistency': 1,  # Placeholder
                'transcript': normalized_transcript
            }
            suitability = SuitabilityEvaluator(audio_metrics, transcript_metrics)
            suitability_score = suitability.calculate_suitability()

            return {
                'suitability_score': suitability_score,
                'classification': {
                    'L0': l0_label,
                    'L1': l1_label,
                    'L2': l2_label
                },
                'audio': {
                    'speech_ratio': speech_ratio,
                    'acoustic_metrics': acoustic_metrics,
                    'sample_rate': sample_rate
                },
                'transcript': {
                    'original': transcript,
                    'normalized': normalized_transcript,
                    'cer': cer,
                    'wer': wer
                }
            }
        else:
            # No transcript case
            return {
                'suitability_score': 0.0,
                'classification': {'L0': None, 'L1': None, 'L2': None},
                'audio': {
                    'speech_ratio': speech_ratio,
                    'acoustic_metrics': acoustic_metrics,
                    'sample_rate': sample_rate
                },
                'transcript': None
            }

    # Legacy behavior: path-based invocation
    audio_file = audio_input
    # Step 1: Analyze audio
    speech_segments = analyze_audio(audio_file)
    acoustic_metrics = analyze_audio_metrics(audio_file, 0)
    print("Speech segments:", speech_segments)
    print("Acoustic metrics:", acoustic_metrics)

    diarizer = Diarization()
    diarization_results = diarizer.perform_diarization(audio_file)

    # Step 2: Normalize transcript
    if transcript_file:
        with open(transcript_file, encoding='utf-8') as f:
            raw_transcript = f.read()
        normalized_transcript = normalize_transcript(raw_transcript)

        # Step 3: Calculate metrics
        cer = calculate_cer(normalized_transcript, raw_transcript)
        wer = calculate_wer(normalized_transcript, raw_transcript)

        # Step 4: Classify transcript
        l0_classifier = L0Classifier()
        l1_classifier = L1Classifier()
        l2_classifier = L2Classifier()
        l0_label = l0_classifier.classify(normalized_transcript, None)
        l1_label = l1_classifier.classify(normalized_transcript)
        l2_label = l2_classifier.classify({}, {'transcript': normalized_transcript})

        # Step 5: Calculate suitability score
        audio_metrics = {
            'snr': acoustic_metrics.get('SNR', 0),
            'clipping_pct': acoustic_metrics.get('clipping_percentage', 0),
            'speech_ratio': speech_segments.get('speech_ratio', 0),
            'overlap_ratio': diarization_results.get('overlap_ratio', 0) if isinstance(diarization_results, dict) else 0
        }
        transcript_metrics = {
            'has_transcript': bool(normalized_transcript),
            'cer_baseline': cer,
            'normalization_consistency': 1,  # Placeholder
            'transcript': normalized_transcript
        }
        suitability = SuitabilityEvaluator(audio_metrics, transcript_metrics)
        suitability_score = suitability.calculate_suitability()

        return {
            'suitability_score': suitability_score,
            'classification': {
                'L0': l0_label,
                'L1': l1_label,
                'L2': l2_label
            }
        }
    else:
        return {'error': 'No transcript provided for file-based evaluation'}


if __name__ == "__main__":
    evaluate_pipeline(audio_input='/home/ngocpt/Documents/Project/erd_voice_dataset/voice-dataset-evaluator/dataset/speaker1/ref.wav',
                      transcript_file='/home/ngocpt/Documents/Project/erd_voice_dataset/voice-dataset-evaluator/dataset/speaker1/transcript.txt')
    # Note: The above paths should be adjusted based on your actual dataset structure.    # Note: The above paths should be adjusted based on your actual dataset structure.
    # Note: The above paths should be adjusted based on your actual dataset structure.    # Note: The above paths should be adjusted based on your actual dataset structure.
    # Note: The above paths should be adjusted based on your actual dataset structure.    # Note: The above paths should be adjusted based on your actual dataset structure.
    # Note: The above paths should be adjusted based on your actual dataset structure.    # Note: The above paths should be adjusted based on your actual dataset structure.
    # Note: The above paths should be adjusted based on your actual dataset structure.    # Note: The above paths should be adjusted based on your actual dataset structure.
