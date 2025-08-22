#!/usr/bin/env python3
"""
Voice Dataset Analyzer - JSON Audio Evaluation Pipeline

This script performs comprehensive analysis of audio data from local JSON dataset.
It evaluates audio quality, generates visualization reports with histograms and JSON outputs.
Focuses only on audio metrics, skipping transcript-related analysis.

Main Pipeline Steps:
1. JSON Loading - Load audio data from local dataset.json file
2. Audio Processing - Extract features, calculate quality metrics
3. Suitability Scoring - Calculate audio quality score
4. Visualization - Generate histograms for data distribution analysis
5. Export Results - Save detailed JSON reports with timestamps

Usage:
    python json_audio_demo.py --samples 10 --clear-output
    python json_audio_demo.py -n 50 -c
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# Import dataset_reader functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_reader import read_dataset
try:
    from minio_client import MinIOClient
except ImportError:
    MinIOClient = None

MINIO_ACCESS_KEY= os.getenv('HFMINIO_ACCESS_KEY_TOKEN')
MINIO_SECRET_KEY= os.getenv('MINIO_SECRET_KEY')
MINIO_ENDPOINT= os.getenv('MINIO_ENDPOINT')

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

print("=" * 60)
print("VOICE DATASET ANALYZER - JSON AUDIO ONLY")
print("=" * 60)


def load_json_dataset(json_path: str, num_samples: int = None) -> list:
    """
    Load audio data from local JSON dataset file.
    
    Args:
        json_path: Path to the JSON dataset file
        num_samples: Number of samples to load (None for all)
        
    Returns:
        List of audio samples
    """
    try:
        print(f"Loading audio data from: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if not isinstance(dataset, list):
            raise ValueError("JSON file should contain a list of samples")
        
        print(f"✓ Loaded {len(dataset)} samples from JSON")
        
        # Limit samples if specified
        if num_samples and num_samples < len(dataset):
            dataset = dataset[:num_samples]
            print(f"✓ Limited to first {num_samples} samples")
        
        # Validate sample structure
        valid_samples = []
        for i, sample in enumerate(dataset):
            if not isinstance(sample, dict):
                print(f"  Warning: Sample {i} is not a dict, skipping")
                continue
            
            if 'audio' not in sample:
                print(f"  Warning: Sample {i} missing audio field, skipping")
                continue
            
            audio = sample['audio']
            if not isinstance(audio, dict):
                print(f"  Warning: Sample {i} audio field is not a dict, skipping")
                continue
            
            if 'array' not in audio:
                print(f"  Warning: Sample {i} missing audio array, skipping")
                continue
            
            # Convert list to numpy array if necessary
            audio_array = audio['array']
            if isinstance(audio_array, list):
                audio_array = np.array(audio_array, dtype=np.float32)
                sample['audio']['array'] = audio_array
            
            # Validate audio array
            if len(audio_array) == 0:
                print(f"  Warning: Sample {i} has empty audio array, skipping")
                continue
            
            # Add sample ID
            sample['sample_id'] = i + 1
            
            valid_samples.append(sample)
        
        if not valid_samples:
            raise ValueError("No valid audio samples found in dataset")
        
        print(f"✓ Successfully validated {len(valid_samples)} samples")
        return valid_samples
        
    except Exception as e:
        print(f"❌ Failed to load JSON dataset: {e}")
        raise


def load_parquet_dataset(parquet_path: str, num_samples: int = None) -> list:
    """
    Load audio data from local Parquet dataset file.
    
    Args:
        parquet_path: Path to the Parquet dataset file
        num_samples: Number of samples to load (None for all)
        
    Returns:
        List of audio samples in expected format
    """
    try:
        print(f"Loading audio data from: {parquet_path}")
        import pandas as pd

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        print(f"✓ Loaded {len(df)} samples from Parquet")

        # Limit samples if specified
        if num_samples and num_samples < len(df):
            df = df.iloc[:num_samples]
            print(f"✓ Limited to first {num_samples} samples")

        valid_samples = []
        for i, row in df.iterrows():
            audio_array = np.array(row['array'], dtype=np.float32)
            if len(audio_array) == 0:
                print(f"  Warning: Sample {i} has empty audio array, skipping")
                continue
            sample = {
                'audio': {
                    'path': row['path'],
                    'array': audio_array,
                    'sampling_rate': int(row['sampling_rate'])
                },
                'transcript': row.get('transcript', ""),
                'sample_id': i + 1
            }
            if 'error' in df.columns and pd.notnull(row.get('error', None)):
                sample['error'] = row['error']
            valid_samples.append(sample)

        if not valid_samples:
            raise ValueError("No valid audio samples found in dataset")

        print(f"✓ Successfully validated {len(valid_samples)} samples")
        return valid_samples

    except Exception as e:
        print(f"❌ Failed to load Parquet dataset: {e}")
        raise


def load_folder_dataset(folder_path: str, num_samples: int = None, sample_rate: int = 16000) -> list:
    """
    Load audio data from a local folder using dataset_reader.
    """
    print(f"Loading audio data from folder: {folder_path}")
    samples = read_dataset_folder(folder_path, target_sr=sample_rate)
    if num_samples and num_samples < len(samples):
        samples = samples[:num_samples]
        print(f"✓ Limited to first {num_samples} samples")
    # Add sample_id for consistency
    for i, sample in enumerate(samples):
        sample['sample_id'] = i + 1
    print(f"✓ Successfully loaded {len(samples)} samples from folder")
    return samples


def process_single_audio_sample(sample_data):
    """Process a single sample through the complete pipeline (audio + transcript + classification)."""
    audio_array = sample_data['audio']['array']
    sample_rate = sample_data['audio'].get('sampling_rate', 16000)
    sample_id = sample_data['sample_id']
    transcript = sample_data.get('transcript', "")
    print(f"Processing Sample {sample_id} with {len(audio_array)} samples at {sample_rate}Hz")

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

    snr_score = min(1, snr / 20) if snr != float('inf') else 1.0
    clipping_score = 1 - clipping_pct
    speech_score = speech_ratio
    loudness_score = max(0, min(1, (loudness + 60) / 40))

    audio_score = (snr_score * 0.3 + clipping_score * 0.3 + speech_score * 0.3 + loudness_score * 0.1)

    cer_score = max(0, 1 - cer)
    wer_score = max(0, 1 - wer)
    length_score = min(1, len(transcript) / 50)

    transcript_score = (1.0 * 0.4 + cer_score * 0.3 + wer_score * 0.2 + length_score * 0.1)

    suitability_score = 0.6 * audio_score + 0.4 * transcript_score

    if suitability_score >= 0.8:
        category = "EXCELLENT ⭐"
    elif suitability_score >= 0.7:
        category = "PASS ✅"
    elif suitability_score >= 0.5:
        category = "REVIEW ⚠️"
    else:
        category = "FAIL ❌"

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


def run_audio_pipeline(data_path: str, num_samples: int = None, sample_rate: int = 16000):
    """Run the complete audio analysis pipeline and return aggregate_result as in hf_only_demo.py."""
    print(f"\n1. Loading audio data from {data_path}...")
    samples = load_parquet_dataset(data_path, num_samples)

    print(f"\n2. Processing {len(samples)} audio samples through pipeline...")
    results = []
    for sample_data in samples:
        try:
            result = process_single_audio_sample(sample_data)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to process sample {sample_data['sample_id']}: {e}")
            continue

    if not results:
        raise Exception("No samples processed successfully")

    print(f"\n3. Aggregating results from {len(results)} samples...")

    # Aggregate statistics
    total_score = sum(r['suitability_score'] for r in results)
    avg_score = total_score / len(results)
    total_duration = sum(r['audio_info']['duration_seconds'] for r in results)
    avg_speech_ratio = sum(r['audio_metrics']['speech_ratio'] for r in results) / len(results)
    avg_cer = sum(r['transcript_metrics']['cer'] for r in results) / len(results)
    avg_wer = sum(r['transcript_metrics']['wer'] for r in results) / len(results)

    # Category distribution
    categories = [r['category'] for r in results]
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # L0 content distribution
    l0_classifications = [r['classification']['L0_content'] for r in results]
    l0_counts = {}
    for l0 in l0_classifications:
        l0_counts[l0] = l0_counts.get(l0, 0) + 1

    # Histograms (optional, if you have visualization)
    try:
        from src.visualization import generate_dataset_histograms
        histogram_stats = generate_dataset_histograms(results, output_dir="output")
    except Exception as e:
        print(f"  Warning: Histogram generation failed: {e}")
        histogram_stats = {}

    aggregate_result = {
        'dataset_info': {
            'source': f'Local: {data_path}',
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


def main(input_path: str, num_samples: int = None, sample_rate: int = 16000):
    try:
        print("🚀 Starting audio analysis pipeline...")
        parquet_dataset = read_dataset(folder_path=input_path, sample_rate=sample_rate)
        result = run_audio_pipeline(parquet_dataset, num_samples, sample_rate)

        print("\n" + "=" * 60)
        print("AUDIO DATASET EVALUATION")
        print("=" * 60)

        # Use custom encoder for JSON serialization
        print(json.dumps(result, indent=2, ensure_ascii=False, cls=NumpyEncoder))

        # Save JSON results to output file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"output/audio_evaluation_{timestamp}.json"
        
        print(f"\n💾 Saving results to {json_filename}...")
        os.makedirs("output", exist_ok=True)
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"✅ Results saved to {json_filename}")

        print("\n📊 SUMMARY")
        print(f"🎯 Average Score: {result['aggregate_scores']['average_suitability']}/1.000")
        print(f"📈 Score Range: {result['aggregate_scores']['min_score']} - {result['aggregate_scores']['max_score']}")
        print(f"🎵 Total Audio: {result['dataset_info']['total_duration_seconds']}s ({result['dataset_info']['num_samples_processed']} samples)")
        print(f"📊 Avg Metrics: CER={result['aggregate_scores']['average_cer']:.3f}, WER={result['aggregate_scores']['average_wer']:.3f}")
        print(f"🏷️  Categories: {result['category_distribution']}")
        print(f"� L0 Content: {result['l0_content_distribution']}")

        print("\n✅ JSON audio analysis pipeline completed successfully!")
        return result

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure dataset.json file exists and is valid JSON")
        print("2. Check that audio arrays are properly formatted")
        print("3. Install audio dependencies: pip install librosa soundfile")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voice Dataset Analyzer - JSON/Parquet/Folder Audio Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python demo.py --input dataset.json --type json
    python demo.py --input audio.parquet --type parquet
    python demo.py --input ./dataset/vetc_data --type folder
    python demo.py --input ./dataset/vetc_data --type folder --samples 20
        """
    )
    parser.add_argument(
        '--source', 
        choices=['minio', 'local'], 
        required=True, 
        default='local',
        help='Source of data: "minio" to download, "local" to use local folder'
    )
    parser.add_argument(
        '--minio-bucket', 
        help='MinIO bucket name'
    )
    parser.add_argument(
        '--minio-prefix', 
            help='Prefix/folder in MinIO bucket to download'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to dataset file Parquet'
    )

    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=None,
        help='Number of samples to process (default: all samples)'
    )

    parser.add_argument(
        '--sample-rate', '-sr',
        type=int,
        default=16000,
        help='Target sampling rate for folder input (default: 16000)'
    )

    parser.add_argument(
        '--clear-output', '-c',
        action='store_true',
        help='Clear output directory before starting analysis'
    )

    args = parser.parse_args()

    if args.source == 'minio':
        if MinIOClient is None:
            print("MinIOClient not available. Install minio and ensure minio_client.py is present.")
            sys.exit(1)
        required_minio = [args.minio_endpoint, args.minio_access_key, args.minio_secret_key, args.minio_bucket, args.minio_prefix]
        if not all(required_minio):
            print("❌ Missing MinIO arguments for download. Exiting.")
            sys.exit(1)

        minio_client = MinIOClient(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY
        )
        print(f"Downloading files from MinIO bucket '{args.minio_bucket}' with prefix '{args.minio_prefix}' to '{args.input}'...")
        count = minio_client.download_folder(
            bucket_name=args.minio_bucket,
            prefix=args.minio_prefix,
            local_folder=args.input,
            recursive=True
        )
        if count == 0:
            print("No files downloaded from MinIO. Exiting.")
            sys.exit(1)

    if args.samples is not None and args.samples <= 0:
        print("❌ Error: Number of samples must be positive")
        sys.exit(1)

    if args.samples:
        print(f"🎯 Processing first {args.samples} samples from {args.input}...")
    else:
        print(f"🎯 Processing all samples from {args.input}...")

    # Run main pipeline
    main(input_path=args.input, num_samples=args.samples, sample_rate=args.sample_rate)