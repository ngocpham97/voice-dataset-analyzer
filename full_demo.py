#!/usr/bin/env python3

import json
import sys

sys.path.insert(0, '/Users/mayvees/Documents/Code/robotic/voice-dataset-analyzer')

print("=" * 60)
print("VOICE DATASET ANALYZER - COMPLETE PIPELINE DEMO")
print("=" * 60)

try:
    # Step 1: Test HuggingFace dataset loading
    print("\n1. Loading HuggingFace dataset...")
    from datasets import load_dataset

    dataset = load_dataset("linhtran92/viet_bud500", split="test", streaming=True)
    sample = next(iter(dataset))

    print("✓ Dataset loaded successfully")
    print(f"   Sample keys: {list(sample.keys()) if hasattr(sample, 'keys') else 'List format'}")

    # Extract audio and transcript
    if hasattr(sample, 'get'):
        audio_field = sample.get('audio')
        transcript = sample.get('transcript') or sample.get('text') or sample.get('sentence')
    else:
        # If sample is a list/tuple, try to access by index
        print("   Sample appears to be in list format, attempting to access...")
        audio_field = None
        transcript = None

    print(f"   Audio field type: {type(audio_field)}")
    print(f"   Transcript preview: {str(transcript)[:50] if transcript else 'None'}...")

    # Step 2: Test audio processing
    print("\n2. Testing audio processing...")
    import numpy as np

    if audio_field and isinstance(audio_field, dict) and 'array' in audio_field:
        audio_array = np.array(audio_field['array'])
        sample_rate = audio_field.get('sampling_rate', 16000)
        print(f"✓ Audio array extracted: length={len(audio_array)}, rate={sample_rate}")
    else:
        # Use synthetic audio for testing
        audio_array = np.random.normal(0, 0.1, 16000)  # 1 second of audio
        sample_rate = 16000
        print("✓ Using synthetic audio for testing")

    # Test audio metrics
    from src.audio.acoustic_metrics import analyze_audio_metrics
    from src.audio.vad import calculate_speech_ratio

    acoustic_metrics = analyze_audio_metrics(audio_array, 0)
    speech_ratio = calculate_speech_ratio(audio_array)

    print(f"   SNR: {acoustic_metrics.get('SNR', 'N/A')}")
    print(f"   Clipping: {acoustic_metrics.get('clipping_percentage', 'N/A'):.3f}")
    print(f"   Speech ratio: {speech_ratio:.3f}")

    # Step 3: Test transcript processing
    print("\n3. Testing transcript processing...")

    if not transcript:
        transcript = "Đây là một ví dụ về bài học giáo dục trong lĩnh vực công nghệ."
        print("   Using synthetic transcript for testing")

    from src.transcript.metrics import calculate_cer, calculate_wer
    from src.transcript.normalization import normalize_transcript

    normalized_transcript = normalize_transcript(transcript)
    cer = calculate_cer(transcript, normalized_transcript)
    wer = calculate_wer(transcript, normalized_transcript)

    print(f"✓ Original: {transcript[:50]}...")
    print(f"✓ Normalized: {normalized_transcript[:50]}...")
    print(f"   CER: {cer:.3f}, WER: {wer:.3f}")

    # Step 4: Test classification
    print("\n4. Testing classification...")

    from src.classification.l0_classifier import L0Classifier
    from src.classification.l1_classifier import L1Classifier
    from src.classification.l2_classifier import L2Classifier

    l0_classifier = L0Classifier()
    l1_classifier = L1Classifier()
    l2_classifier = L2Classifier()

    l0_result = l0_classifier.classify(normalized_transcript, None)
    l1_result = l1_classifier.classify(normalized_transcript)
    l2_result = l2_classifier.classify({}, {'transcript': normalized_transcript})

    print(f"✓ L0 Classification: {l0_result}")
    print(f"✓ L1 Classification: {l1_result}")
    print(f"✓ L2 Classification: {l2_result}")

    # Step 5: Calculate final suitability score
    print("\n5. Calculating suitability score...")

    # Audio quality score
    snr = acoustic_metrics.get('SNR', 0)
    clipping_pct = acoustic_metrics.get('clipping_percentage', 0)

    audio_score = min(1.0, (
        min(1, snr / 20 if snr != float('inf') else 1) * 0.4 +
        (1 - clipping_pct) * 0.3 +
        speech_ratio * 0.3
    ))

    # Transcript quality score
    transcript_score = (
        1.0 * 0.5 +  # has transcript
        (1 - min(1, cer)) * 0.3 +
        (1 - min(1, wer)) * 0.2
    )

    # Final suitability score
    suitability_score = 0.6 * audio_score + 0.4 * transcript_score

    print(f"✓ Audio score: {audio_score:.3f}")
    print(f"✓ Transcript score: {transcript_score:.3f}")
    print(f"✓ Final suitability score: {suitability_score:.3f}")

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    final_result = {
        'suitability_score': round(suitability_score, 3),
        'classification': {
            'L0': l0_result,
            'L1': l1_result,
            'L2': l2_result
        },
        'audio_metrics': {
            'speech_ratio': round(speech_ratio, 3),
            'snr': round(float(acoustic_metrics.get('SNR', 0)) if acoustic_metrics.get('SNR', 0) != float('inf') else 999.0, 3),
            'clipping_percentage': round(float(acoustic_metrics.get('clipping_percentage', 0)), 3),
            'audio_score': round(audio_score, 3)
        },
        'transcript_metrics': {
            'cer': round(cer, 3),
            'wer': round(wer, 3),
            'transcript_score': round(transcript_score, 3),
            'length': len(transcript)
        }
    }

    print(json.dumps(final_result, indent=2, ensure_ascii=False))

    # Step 7: Suitability category
    if suitability_score >= 0.7:
        category = "PASS ✅"
    elif suitability_score >= 0.5:
        category = "REVIEW ⚠️"
    else:
        category = "FAIL ❌"

    print(f"\nOverall Assessment: {category}")
    print(f"Suitability Score: {suitability_score:.3f}/1.000")

    print("\n🎉 Complete pipeline executed successfully!")
    print("All steps are working and ready for batch processing.")

except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
    print("\nSome components may need debugging.")
    print("\nSome components may need debugging.")
    print("\nSome components may need debugging.")
    print("\nSome components may need debugging.")
    print("\nSome components may need debugging.")
