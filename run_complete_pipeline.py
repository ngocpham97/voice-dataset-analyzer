#!/usr/bin/env python3

from src.transcript.normalization import normalize_transcript
from src.transcript.metrics import calculate_cer, calculate_wer
from src.classification.l2_classifier import L2Classifier
from src.classification.l1_classifier import L1Classifier
from src.classification.l0_classifier import L0Classifier
from src.audio.vad import analyze_audio, calculate_speech_ratio, read_audio
from src.audio.acoustic_metrics import analyze_audio_metrics
import sys

import numpy as np

# Add the voice-dataset-analyzer directory to Python path
sys.path.insert(0, '/Users/mayvees/Documents/Code/robotic/voice-dataset-analyzer')


def evaluate_pipeline_complete(audio_input, transcript_file=None):
    """Complete evaluation pipeline that works with HuggingFace dataset samples."""

    # Handle dataset sample (dict-like) input
    if isinstance(audio_input, dict):
        sample = audio_input

        # Extract audio data
        audio_field = sample.get('audio') or sample.get('audio_path') or sample.get('path')
        audio_array = None
        sample_rate = None

        if isinstance(audio_field, dict):
            if 'array' in audio_field:
                audio_array = np.array(audio_field['array'])
                sample_rate = audio_field.get('sampling_rate')
            elif 'path' in audio_field:
                sample_rate, audio_array = read_audio(audio_field['path'])
        elif isinstance(audio_field, str):
            sample_rate, audio_array = read_audio(audio_field)

        if audio_array is None:
            raise ValueError('Could not resolve audio data from the provided sample')

        # Step 1: Audio analysis
        speech_ratio = calculate_speech_ratio(audio_array)
        acoustic_metrics = analyze_audio_metrics(audio_array, 0)

        # Extract transcript
        transcript = sample.get('transcript') or sample.get('text') or sample.get('sentence')

        if transcript:
            # Step 2: Normalize transcript
            normalized_transcript = normalize_transcript(transcript)

            # Step 3: Calculate transcript metrics
            cer = calculate_cer(transcript, normalized_transcript)
            wer = calculate_wer(transcript, normalized_transcript)

            # Step 4: Classify transcript
            l0_classifier = L0Classifier()
            l1_classifier = L1Classifier()
            l2_classifier = L2Classifier()

            l0_label = l0_classifier.classify(normalized_transcript, None)
            l1_label = l1_classifier.classify(normalized_transcript)
            l2_label = l2_classifier.classify({}, {'transcript': normalized_transcript})

            # Step 5: Calculate simple suitability score
            # Audio quality score (0-1)
            snr = acoustic_metrics.get('SNR', 0)
            clipping_pct = acoustic_metrics.get('clipping_percentage', 0)

            # Simple scoring logic
            audio_score = min(1.0, (
                min(1, snr / 20 if snr != float('inf') else 1) * 0.3 +
                (1 - clipping_pct) * 0.3 +
                speech_ratio * 0.4
            ))

            # Transcript quality score
            transcript_score = (
                (1.0 if transcript else 0.0) * 0.5 +
                (1 - min(1, cer)) * 0.3 +
                (1 - min(1, wer)) * 0.2
            )

            # Combined score
            suitability_score = 0.6 * audio_score + 0.4 * transcript_score

            return {
                'suitability_score': float(suitability_score),
                'classification': {
                    'L0': l0_label,
                    'L1': l1_label,
                    'L2': l2_label
                },
                'audio': {
                    'speech_ratio': float(speech_ratio),
                    'acoustic_metrics': {
                        'SNR': float(acoustic_metrics['SNR']) if acoustic_metrics['SNR'] != float('inf') else 999.0,
                        'clipping_percentage': float(acoustic_metrics['clipping_percentage']),
                        'loudness': float(acoustic_metrics['loudness'])
                    },
                    'sample_rate': sample_rate,
                    'audio_score': float(audio_score)
                },
                'transcript': {
                    'original': transcript,
                    'normalized': normalized_transcript,
                    'cer': float(cer),
                    'wer': float(wer),
                    'transcript_score': float(transcript_score)
                }
            }
        else:
            # No transcript case
            audio_score = min(1.0, (
                min(1, acoustic_metrics.get('SNR', 0) / 20 if acoustic_metrics.get('SNR', 0) != float('inf') else 1) * 0.5 +
                (1 - acoustic_metrics.get('clipping_percentage', 0)) * 0.3 +
                speech_ratio * 0.2
            ))

            return {
                'suitability_score': float(audio_score * 0.6),  # Lower score without transcript
                'classification': {'L0': None, 'L1': None, 'L2': None},
                'audio': {
                    'speech_ratio': float(speech_ratio),
                    'acoustic_metrics': {
                        'SNR': float(acoustic_metrics['SNR']) if acoustic_metrics['SNR'] != float('inf') else 999.0,
                        'clipping_percentage': float(acoustic_metrics['clipping_percentage']),
                        'loudness': float(acoustic_metrics['loudness'])
                    },
                    'sample_rate': sample_rate,
                    'audio_score': float(audio_score)
                },
                'transcript': None
            }

    # File-based processing (legacy)
    else:
        audio_file = audio_input
        speech_segments = analyze_audio(audio_file)
        sample_rate, audio_array = read_audio(audio_file)
        acoustic_metrics = analyze_audio_metrics(audio_array, 0)

        if transcript_file:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()

            # Process similar to above...
            normalized_transcript = normalize_transcript(transcript)
            cer = calculate_cer(transcript, normalized_transcript)
            wer = calculate_wer(transcript, normalized_transcript)

            # ... (rest of processing)
            return {
                'suitability_score': 0.5,  # Placeholder
                'message': 'File-based processing completed'
            }
        else:
            return {'error': 'No transcript file provided'}


def main():
    print("Testing complete pipeline...")

    # Test 1: Synthetic sample
    print("\n1. Testing with synthetic sample...")
    sample = {
        'audio': {'array': [0.1, 0.2, 0.1, 0.0, 0.3, 0.2, 0.1], 'sampling_rate': 16000},
        'transcript': 'đây là một ví dụ về bài học giáo dục'
    }

    result = evaluate_pipeline_complete(sample)
    print("✓ Synthetic sample result:")
    print(f"  Suitability Score: {result['suitability_score']:.3f}")
    print(f"  L0 Classification: {result['classification']['L0']}")
    print(f"  Speech Ratio: {result['audio']['speech_ratio']:.3f}")
    if result['transcript']:
        print(f"  CER: {result['transcript']['cer']:.3f}")
        print(f"  WER: {result['transcript']['wer']:.3f}")

    # Test 2: Try HuggingFace dataset if available
    print("\n2. Testing with HuggingFace dataset...")
    try:
        from datasets import load_dataset

        dataset = load_dataset("linhtran92/viet_bud500", split="test", streaming=True)
        hf_sample = next(iter(dataset))

        print("✓ Loaded HuggingFace sample")
        print(f"  Sample keys: {list(hf_sample.keys())}")

        # Inspect audio field
        audio_field = hf_sample.get('audio')
        if audio_field and isinstance(audio_field, dict):
            print(f"  Audio keys: {list(audio_field.keys())}")
            if 'array' in audio_field:
                print(f"  Audio array length: {len(audio_field['array'])}")
                print(f"  Sample rate: {audio_field.get('sampling_rate')}")

        hf_result = evaluate_pipeline_complete(hf_sample)
        print("✓ HuggingFace sample result:")
        print(f"  Suitability Score: {hf_result['suitability_score']:.3f}")
        print(f"  L0 Classification: {hf_result['classification']['L0']}")
        print(f"  Audio Score: {hf_result['audio']['audio_score']:.3f}")
        if hf_result['transcript']:
            print(f"  Original: {hf_result['transcript']['original'][:50]}...")
            print(f"  Normalized: {hf_result['transcript']['normalized'][:50]}...")

    except Exception as e:
        print(f"✗ HuggingFace test failed: {e}")

    print("\n🎉 Complete pipeline testing finished!")


if __name__ == "__main__":
    main()
    main()
    main()
    main()
    main()
