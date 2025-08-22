#!/usr/bin/env python3
"""
Voice Dataset Analyzer - HuggingFace Dataset Evaluation Pipeline

This script performs comprehensive analysis of Vietnamese voice datasets from HuggingFace.
It evaluates audio quality, transcript accuracy, content classification, and generates
visualization reports with histograms and JSON outputs.

Main Pipeline Steps:
1. Authentication - Verify HuggingFace access
2. Dataset Loading - Load and cache samples from parquet files  
3. Audio Processing - Extract features, calculate quality metrics
4. Transcript Analysis - Normalize text, compute error rates
5. Classification - Categorize content across L0/L1/L2 levels
6. Suitability Scoring - Combine audio/text quality into overall score
7. Visualization - Generate histograms for data distribution analysis
8. Export Results - Save detailed JSON reports with timestamps

Usage:
    python hf_only_demo.py --samples 10 --clear-output
    python hf_only_demo.py -n 50 -c
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/Users/mayvees/Documents/Code/robotic/voice-dataset-analyzer')

print("=" * 60)
print("VOICE DATASET ANALYZER - HUGGINGFACE ONLY")
print("=" * 60)


def ensure_authentication():
    """Ensure HuggingFace authe        print(f"📝 Avg Metrics: CER={result['aggregate_scores']['average_cer']:.3f}, WER={result['aggregate_scores']['average_wer']:.3f}")
        print(f"🏷️  Categories: {result['category_distribution']}")
        print(f"📋 L0 Content: {result['l0_content_distribution']}")
        print(f"📊 Histograms: Generated in 'output/' directory")
        print(f"💾 JSON Results: {json_filename}")

        print("\n✅ HuggingFace batch pipeline completed successfully!")
        print("📈 Check the 'output/' folder for visualization histograms and JSON results!")tion is working."""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✓ Authenticated as: {user_info.get('name', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\nTo authenticate with HuggingFace:")
        print("1. Get token from: https://huggingface.co/settings/tokens")
        print("2. Run: export HF_TOKEN='your_token_here'")
        print("3. Or run: huggingface-cli login")
        print("4. Request access to dataset: https://huggingface.co/datasets/linhtran92/viet_bud500")
        return False


def load_huggingface_samples(num_samples=5):
    """Load a small subset of samples from HuggingFace dataset using parquet files."""
    try:
        from datasets import load_dataset
        
        print(f"Loading {num_samples} samples from HuggingFace parquet dataset...")
        
        # Set cache directory to persist downloaded data
        cache_dir = "dataset"
        
        # Check if dataset is already cached
        if os.path.exists(cache_dir) and os.listdir(cache_dir):
            print("✓ Found cached dataset, loading from local cache...")
        else:
            print("⬇️  Downloading dataset for first time (will be cached for future use)...")
        
        # Load from parquet file (~4000 samples in a parquet file)
        # Using test parquet file for smaller dataset
        test_url = "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/test-00000-of-00002-531c1d81edb57297.parquet"
        data_files = {"test": test_url}
        
        # Load dataset with cache to avoid re-downloading
        dataset = load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)
        
        # Get test split and take first few samples
        test_data = dataset["test"]
        print("✓ Loaded dataset with samples available")
        
        samples = []
        sample_count = 0
        for sample in test_data:
            if sample_count >= num_samples:
                break
            
            sample_count += 1
            print(f"Processing sample {sample_count}/{num_samples}...")
            
            # Validate sample structure
            if not isinstance(sample, dict):
                print(f"  Warning: Sample {sample_count} is not a dict, skipping")
                continue
                
            # Extract audio according to dataset description
            audio_field = sample.get('audio')
            if audio_field is None:
                print(f"  Warning: Sample {sample_count} missing audio field, skipping")
                continue
            
            # Handle audio decoder object from datasets library
            try:
                # Use get_all_samples method for AudioDecoder
                if hasattr(audio_field, 'get_all_samples'):
                    audio_samples = audio_field.get_all_samples()
                    # AudioSamples object should have data attribute or be array-like
                    if hasattr(audio_samples, 'data'):
                        audio_array = np.array(audio_samples.data)
                    elif hasattr(audio_samples, 'audio'):
                        audio_array = np.array(audio_samples.audio)
                    else:
                        # Try converting directly
                        audio_array = np.array(audio_samples)
                    
                    # Handle multi-channel audio (take first channel if needed)
                    if audio_array.ndim > 1:
                        audio_array = audio_array[0]  # First channel
                    
                    sample_rate = audio_field._desired_sample_rate or 16000
                elif hasattr(audio_field, '__getitem__'):
                    # Try accessing as dict-like
                    audio_array = np.array(audio_field['array'])
                    sample_rate = audio_field.get('sampling_rate', 16000)
                else:
                    print(f"  Warning: Sample {sample_count} unknown audio format, skipping")
                    continue
            except Exception as audio_error:
                print(f"  Warning: Sample {sample_count} audio extraction failed: {audio_error}")
                continue
            
            # Extract transcript according to dataset description
            transcript = sample.get('transcription')
            if not transcript:
                print(f"  Warning: Sample {sample_count} missing transcription, skipping")
                continue
            
            print(f"  ✓ Sample {sample_count}: {len(audio_array)} audio samples, {len(transcript)} chars")
            print(f"    Audio: {len(audio_array)/sample_rate:.1f}s at {sample_rate}Hz")
            print(f"    Text: {transcript[:50]}...")
            
            samples.append({
                'audio_array': audio_array,
                'sample_rate': sample_rate,
                'transcript': transcript,
                'sample_id': sample_count
            })
        
        if not samples:
            raise ValueError("No valid samples found in dataset")
            
        print(f"✓ Successfully loaded {len(samples)} samples")
        return samples
        
    except Exception as e:
        print(f"❌ Failed to load HuggingFace dataset: {e}")
        print("\nPossible issues:")
        print("- Dataset requires authentication (run: huggingface-cli login)")
        print("- Dataset is gated (request access at the dataset page)")
        print("- Missing audio dependencies (install: torchcodec, torchaudio)")
        raise


def process_single_sample(sample_data):
    """Process a single sample through the complete pipeline."""
    audio_array = sample_data['audio_array']
    sample_rate = sample_data['sample_rate']
    transcript = sample_data['transcript']
    sample_id = sample_data['sample_id']
    
    print(f"\n--- Processing Sample {sample_id} ---")
    
    # Step 1: Audio processing
    from src.audio.acoustic_metrics import analyze_audio_metrics
    from src.audio.vad import calculate_speech_ratio
    
    acoustic_metrics = analyze_audio_metrics(audio_array, 0)
    speech_ratio = calculate_speech_ratio(audio_array)
    
    # Step 2: Transcript processing
    from src.transcript.normalization import normalize_transcript
    from src.transcript.metrics import calculate_cer, calculate_wer
    
    normalized_transcript = normalize_transcript(transcript)
    cer = calculate_cer(transcript, normalized_transcript)
    wer = calculate_wer(transcript, normalized_transcript)
    
    # Step 3: Classification
    from src.classification.l0_classifier import L0Classifier
    from src.classification.l1_classifier import L1Classifier
    from src.classification.l2_classifier import L2Classifier
    
    l0_classifier = L0Classifier()
    l1_classifier = L1Classifier()
    l2_classifier = L2Classifier()
    
    l0_result = l0_classifier.classify(normalized_transcript, None)
    l1_result = l1_classifier.classify(normalized_transcript)
    l2_result = l2_classifier.classify(
        {'num_speakers': 1, 'overlap_ratio': 0.1}, 
        {'transcript': normalized_transcript, 'keywords': []}
    )
    
    # Step 4: Suitability scoring
    snr = acoustic_metrics.get('SNR', 0)
    clipping_pct = acoustic_metrics.get('clipping_percentage', 0)
    loudness = acoustic_metrics.get('loudness', -23)

    # Audio score (0-1)
    snr_score = min(1, snr / 20) if snr != float('inf') else 1.0
    clipping_score = 1 - clipping_pct
    speech_score = speech_ratio
    loudness_score = max(0, min(1, (loudness + 60) / 40))
    
    audio_score = (snr_score * 0.3 + clipping_score * 0.3 + speech_score * 0.3 + loudness_score * 0.1)
    
    # Transcript quality
    cer_score = max(0, 1 - cer)
    wer_score = max(0, 1 - wer)
    length_score = min(1, len(transcript) / 50)  # Prefer reasonable length
    
    transcript_score = (1.0 * 0.4 + cer_score * 0.3 + wer_score * 0.2 + length_score * 0.1)
    
    # Final score
    suitability_score = 0.6 * audio_score + 0.4 * transcript_score
    
    # Assessment category
    if suitability_score >= 0.8:
        category = "EXCELLENT ⭐"
    elif suitability_score >= 0.7:
        category = "PASS ✅"
    elif suitability_score >= 0.5:
        category = "REVIEW ⚠️"
    else:
        category = "FAIL ❌"
    
    # Compile results
    result = {
        'sample_id': sample_id,
        'audio_info': {
            'duration_seconds': round(len(audio_array) / sample_rate, 2),
            'sample_rate': sample_rate,
            'num_samples': len(audio_array)
        },
        'suitability_score': round(suitability_score, 3),
        'category': category,
        'classification': {
            'L0_content': l0_result,
            'L1_domain': l1_result,
            'L2_style': l2_result
        },
        'audio_metrics': {
            'speech_ratio': round(speech_ratio, 3),
            'snr': round(float(snr) if snr != float('inf') else 999.0, 3),
            'clipping_pct': round(float(clipping_pct), 3),
            'loudness_lufs': round(float(loudness), 1),
            'audio_score': round(audio_score, 3)
        },
        'transcript_metrics': {
            'original_length': len(transcript),
            'normalized_length': len(normalized_transcript),
            'cer': round(cer, 3),
            'wer': round(wer, 3),
            'transcript_score': round(transcript_score, 3)
        },
        'text_preview': {
            'original': transcript[:80] + "..." if len(transcript) > 80 else transcript,
            'normalized': normalized_transcript[:80] + "..." if len(normalized_transcript) > 80 else normalized_transcript
        }
    }
    
    print(f"Sample {sample_id} Results:")
    print(f"  🎯 {category} (Score: {suitability_score:.3f})")
    print(f"  🎵 Audio: {result['audio_info']['duration_seconds']}s, Speech: {speech_ratio:.2f}")
    print(f"  📝 Text: {len(transcript)} chars, CER: {cer:.3f}, WER: {wer:.3f}")
    print(f"  🏷️  L0: {l0_result}, L1: {l1_result}")
    
    return result
def run_complete_pipeline(num_samples=3):
    """Run the complete pipeline with a small subset of HuggingFace data."""
    
    # Step 1: Ensure authentication
    print("\n1. Checking authentication...")
    if not ensure_authentication():
        raise Exception("Authentication required")
    
    # Step 2: Load small subset of HuggingFace dataset
    print(f"\n2. Loading {num_samples} samples from HuggingFace dataset...")
    samples = load_huggingface_samples(num_samples)
    
    # Step 3: Process each sample through the pipeline
    print(f"\n3. Processing {len(samples)} samples through pipeline...")
    results = []
    
    for sample_data in samples:
        try:
            result = process_single_sample(sample_data)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to process sample {sample_data['sample_id']}: {e}")
            continue
    
    if not results:
        raise Exception("No samples processed successfully")
    
    # Step 4: Aggregate results
    print(f"\n4. Aggregating results from {len(results)} samples...")
    
    # Calculate aggregate statistics
    total_score = sum(r['suitability_score'] for r in results)
    avg_score = total_score / len(results)
    
    total_duration = sum(r['audio_info']['duration_seconds'] for r in results)
    avg_speech_ratio = sum(r['audio_metrics']['speech_ratio'] for r in results) / len(results)
    avg_cer = sum(r['transcript_metrics']['cer'] for r in results) / len(results)
    avg_wer = sum(r['transcript_metrics']['wer'] for r in results) / len(results)
    
    # Count categories
    categories = [r['category'] for r in results]
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Aggregate L0 classifications
    l0_classifications = [r['classification']['L0_content'] for r in results]
    l0_counts = {}
    for l0 in l0_classifications:
        l0_counts[l0] = l0_counts.get(l0, 0) + 1
    
    # Step 5: Generate histograms for dataset analysis
    print("\n5. Generating histograms for dataset visualization...")
    from src.visualization import generate_dataset_histograms
    histogram_stats = generate_dataset_histograms(results, output_dir="output")
    
    aggregate_result = {
        'dataset_info': {
            'source': 'HuggingFace linhtran92/viet_bud500',
            'num_samples_processed': len(results),
            'total_duration_seconds': round(total_duration, 2),
            'avg_duration_per_sample': round(total_duration / len(results), 2)
        },
        'aggregate_scores': {
            'average_suitability': round(avg_score, 3),
            'min_score': round(min(r['suitability_score'] for r in results), 3),
            'max_score': round(max(r['suitability_score'] for r in results), 3),
            'average_speech_ratio': round(avg_speech_ratio, 3),
            'average_cer': round(avg_cer, 3),
            'average_wer': round(avg_wer, 3)
        },
        'category_distribution': category_counts,
        'l0_content_distribution': l0_counts,
        'histogram_stats': histogram_stats,
        'individual_results': results
    }
    
    return aggregate_result


def main(num_samples=3):
    try:
        print("🚀 Starting HuggingFace batch pipeline...")
        result = run_complete_pipeline(num_samples=num_samples)

        print("\n" + "=" * 60)
        print("BATCH HUGGINGFACE DATASET EVALUATION")
        print("=" * 60)

        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Save JSON results to output file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"output/dataset_evaluation_{timestamp}.json"
        
        print(f"\n💾 Saving results to {json_filename}...")
        os.makedirs("output", exist_ok=True)
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to {json_filename}")

        print("\n📊 SUMMARY")
        print(f"🎯 Average Score: {result['aggregate_scores']['average_suitability']}/1.000")
        print(f"📈 Score Range: {result['aggregate_scores']['min_score']} - {result['aggregate_scores']['max_score']}")
        print(f"🎵 Total Audio: {result['dataset_info']['total_duration_seconds']}s ({result['dataset_info']['num_samples_processed']} samples)")
        print(f"📝 Avg Metrics: CER={result['aggregate_scores']['average_cer']:.3f}, WER={result['aggregate_scores']['average_wer']:.3f}")
        print(f"🏷️  Categories: {result['category_distribution']}")
        print(f"� L0 Content: {result['l0_content_distribution']}")

        print("\n✅ HuggingFace batch pipeline completed successfully!")
        return result

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure HuggingFace authentication: huggingface-cli login")
        print("2. Request dataset access: https://huggingface.co/datasets/linhtran92/viet_bud500")
        print("3. Install audio deps: pip install torchcodec torchaudio librosa")
        return None


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Voice Dataset Analyzer - HuggingFace Dataset Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hf_only_demo.py                    # Process 3 samples (default)
  python hf_only_demo.py --samples 10       # Process 10 samples
  python hf_only_demo.py -n 50              # Process 50 samples
  python hf_only_demo.py --samples 100      # Process 100 samples
  python hf_only_demo.py -c --samples 20    # Clear output and process 20 samples
  python hf_only_demo.py --clear-output -n 10  # Clear output and process 10 samples
        """
    )
    
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=3,
        help='Number of samples to process from the dataset (default: 3)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=3750,
        help='Maximum number of samples available in dataset (default: 3750)'
    )
    
    parser.add_argument(
        '--clear-output', '-c',
        action='store_true',
        help='Clear output directory before starting analysis'
    )
    
    args = parser.parse_args()
    
    # Clear output directory if requested
    if args.clear_output:
        import shutil
        output_dir = "output"
        if os.path.exists(output_dir):
            print(f"🧹 Clearing output directory: {output_dir}/")
            shutil.rmtree(output_dir)
            print("✅ Output directory cleared")
        else:
            print(f"ℹ️  Output directory {output_dir}/ does not exist, nothing to clear")
    
    # Validate arguments
    if args.samples <= 0:
        print("❌ Error: Number of samples must be positive")
        sys.exit(1)
    
    if args.samples > args.max_samples:
        print(f"⚠️  Warning: Requested {args.samples} samples, but dataset only has {args.max_samples}")
        print(f"📉 Processing {args.max_samples} samples instead...")
        args.samples = args.max_samples
    
    print(f"🎯 Processing {args.samples} samples from HuggingFace dataset...")
    
    # Run main pipeline with specified number of samples
    main(num_samples=args.samples)
