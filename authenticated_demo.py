#!/usr/bin/env python3

import json
import os
import sys

sys.path.insert(0, '/Users/mayvees/Documents/Code/robotic/voice-dataset-analyzer')

print("=" * 60)
print("VOICE DATASET ANALYZER - AUTHENTICATED DEMO")
print("=" * 60)


def try_authenticate():
    """Try different authentication methods for HuggingFace."""
    try:
        from huggingface_hub import login

        # Method 1: Check if already logged in
        try:
            from huggingface_hub import whoami
            user_info = whoami()
            print(f"✓ Already authenticated as: {user_info.get('name', 'Unknown')}")
            return True
        except:
            pass

        # Method 2: Check environment variable
        token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
        if token:
            print("✓ Using token from environment variable")
            login(token=token)
            return True

        # Method 3: Interactive login
        print("No authentication found. You can:")
        print("1. Set HF_TOKEN environment variable")
        print("2. Run: huggingface-cli login")
        print("3. Get token from: https://huggingface.co/settings/tokens")

        # For demo, we'll skip HF dataset and use synthetic data
        return False

    except Exception as e:
        print(f"Authentication error: {e}")
        return False


def run_pipeline_demo():
    """Run the complete pipeline demo with authentication handling."""

    # Step 1: Try authentication
    print("\n1. Checking HuggingFace authentication...")
    authenticated = try_authenticate()

    # Step 2: Load dataset or use synthetic data
    print("\n2. Loading data...")
    transcript = None
    audio_array = None
    sample_rate = 16000

    if authenticated:
        try:
            from datasets import load_dataset
            print("Trying to load HuggingFace dataset...")
            dataset = load_dataset("linhtran92/viet_bud500", split="test", streaming=True)
            sample = next(iter(dataset))

            print("✓ Dataset loaded successfully")

            # Extract audio and transcript
            if hasattr(sample, 'get'):
                audio_field = sample.get('audio')
                transcript = sample.get('transcript') or sample.get('text') or sample.get('sentence')

                if audio_field and isinstance(audio_field, dict) and 'array' in audio_field:
                    import numpy as np
                    audio_array = np.array(audio_field['array'])
                    sample_rate = audio_field.get('sampling_rate', 16000)
                    print(f"✓ Real audio extracted: length={len(audio_array)}, rate={sample_rate}")

        except Exception as e:
            print(f"Dataset loading failed: {e}")
            authenticated = False

    if not authenticated or audio_array is None:
        print("Using synthetic data for demo...")
        import numpy as np

        # Create synthetic Vietnamese-style audio (1 second)
        audio_array = np.random.normal(0, 0.1, sample_rate)
        # Add some speech-like patterns
        for i in range(0, len(audio_array), 1000):
            audio_array[i:i+500] *= np.random.uniform(0.5, 2.0)

        transcript = "Xin chào, đây là một ví dụ về bài học giáo dục trong lĩnh vực công nghệ thông tin."
        print("✓ Synthetic data created")

    # Step 3: Test audio processing
    print("\n3. Processing audio...")

    from src.audio.acoustic_metrics import analyze_audio_metrics
    from src.audio.vad import calculate_speech_ratio

    acoustic_metrics = analyze_audio_metrics(audio_array, 0)
    speech_ratio = calculate_speech_ratio(audio_array)

    print(f"✓ SNR: {acoustic_metrics.get('SNR', 'N/A')}")
    print(f"✓ Clipping: {acoustic_metrics.get('clipping_percentage', 0):.3f}")
    print(f"✓ Speech ratio: {speech_ratio:.3f}")

    # Step 4: Test transcript processing
    print("\n4. Processing transcript...")

    from src.transcript.metrics import calculate_cer, calculate_wer
    from src.transcript.normalization import normalize_transcript

    normalized_transcript = normalize_transcript(transcript)
    cer = calculate_cer(transcript, normalized_transcript)
    wer = calculate_wer(transcript, normalized_transcript)

    print(f"✓ Original: {transcript}")
    print(f"✓ Normalized: {normalized_transcript}")
    print(f"✓ CER: {cer:.3f}, WER: {wer:.3f}")

    # Step 5: Test classification
    print("\n5. Running classification...")

    from src.classification.l0_classifier import L0Classifier
    from src.classification.l1_classifier import L1Classifier
    from src.classification.l2_classifier import L2Classifier

    l0_classifier = L0Classifier()
    l1_classifier = L1Classifier()
    l2_classifier = L2Classifier()

    l0_result = l0_classifier.classify(normalized_transcript, None)
    l1_result = l1_classifier.classify(normalized_transcript)
    l2_result = l2_classifier.classify({'num_speakers': 1}, {'transcript': normalized_transcript})

    print(f"✓ L0 Classification: {l0_result}")
    print(f"✓ L1 Classification: {l1_result}")
    print(f"✓ L2 Classification: {l2_result}")

    # Step 6: Calculate suitability score
    print("\n6. Calculating suitability score...")

    # Audio quality metrics
    snr = acoustic_metrics.get('SNR', 0)
    clipping_pct = acoustic_metrics.get('clipping_percentage', 0)

    # Audio score calculation
    audio_score = min(1.0, (
        min(1, snr / 20 if snr != float('inf') else 1) * 0.4 +
        (1 - clipping_pct) * 0.3 +
        speech_ratio * 0.3
    ))

    # Transcript score calculation
    transcript_score = (
        1.0 * 0.5 +  # has transcript
        (1 - min(1, cer)) * 0.3 +
        (1 - min(1, wer)) * 0.2
    )

    # Final suitability score
    suitability_score = 0.6 * audio_score + 0.4 * transcript_score

    print(f"✓ Audio score: {audio_score:.3f}")
    print(f"✓ Transcript score: {transcript_score:.3f}")
    print(f"✓ Final suitability: {suitability_score:.3f}")

    # Step 7: Final results
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    result = {
        'dataset_source': 'HuggingFace' if authenticated else 'Synthetic',
        'suitability_score': round(suitability_score, 3),
        'suitability_category': (
            'PASS ✅' if suitability_score >= 0.7 else
            'REVIEW ⚠️' if suitability_score >= 0.5 else
            'FAIL ❌'
        ),
        'classification': {
            'L0': l0_result,
            'L1': l1_result,
            'L2': l2_result
        },
        'audio_metrics': {
            'speech_ratio': round(speech_ratio, 3),
            'snr': round(float(snr) if snr != float('inf') else 999.0, 3),
            'clipping_percentage': round(float(clipping_pct), 3),
            'audio_score': round(audio_score, 3)
        },
        'transcript_metrics': {
            'length': len(transcript),
            'cer': round(cer, 3),
            'wer': round(wer, 3),
            'transcript_score': round(transcript_score, 3)
        }
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n🎯 Overall Assessment: {result['suitability_category']}")
    print(f"📊 Suitability Score: {suitability_score:.3f}/1.000")

    if authenticated:
        print("\n✅ Complete pipeline with real HuggingFace data!")
    else:
        print("\n⚠️  Pipeline tested with synthetic data.")
        print("   To use real data, authenticate with HuggingFace:")
        print("   export HF_TOKEN=your_token_here")

    print("\n🎉 All pipeline steps completed successfully!")
    return result


if __name__ == "__main__":
    try:
        result = run_pipeline_demo()
        print(f"\n✅ Demo completed with score: {result['suitability_score']}")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
