#!/usr/bin/env python3
"""
Demo script to test the complete voice dataset evaluation pipeline
with a single sample from the HuggingFace dataset.
"""

import json

from datasets import load_dataset
from src.pipeline import evaluate_pipeline


def main():
    print("Loading dataset sample...")

    # Load one sample from the Vietnamese dataset
    try:
        # Use streaming to avoid downloading the full dataset
        dataset = load_dataset("linhtran92/viet_bud500", split="test", streaming=True)
        sample = next(iter(dataset))

        print("Sample keys:", list(sample.keys()))
        print("Audio info:", type(sample.get('audio')), sample.get('audio', {}).keys() if isinstance(sample.get('audio'), dict) else 'Not a dict')

        # Run the complete evaluation pipeline
        print("\nRunning evaluation pipeline...")
        result = evaluate_pipeline(sample)

        # Pretty print the results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Suitability Score: {result.get('suitability_score', 'N/A'):.3f}")
        print(f"L0 Classification: {result.get('classification', {}).get('L0', 'N/A')}")
        print(f"L1 Classification: {result.get('classification', {}).get('L1', 'N/A')}")
        print(f"L2 Classification: {result.get('classification', {}).get('L2', 'N/A')}")
        print(f"Speech Ratio: {result.get('audio', {}).get('speech_ratio', 'N/A'):.3f}")

        if result.get('transcript'):
            transcript_info = result['transcript']
            print(f"CER: {transcript_info.get('cer', 'N/A'):.3f}")
            print(f"WER: {transcript_info.get('wer', 'N/A'):.3f}")
            print(f"Original transcript: {transcript_info.get('original', 'N/A')[:100]}...")

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to local test file...")

        # Fallback to local file if available
        local_audio = "dataset/speaker1/ref.wav"
        local_transcript = "dataset/speaker1/transcript.txt"

        try:
            result = evaluate_pipeline(local_audio, local_transcript)
            print("Local file evaluation result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as local_e:
            print(f"Local file test also failed: {local_e}")


if __name__ == "__main__":
    main()
    main()
    main()
    main()
    main()
