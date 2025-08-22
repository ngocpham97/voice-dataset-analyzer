#!/usr/bin/env python3
"""
Module to read audio files from data folder and convert them to a standardized format.
This script reads audio files from the vetc_data folder and stores them in the format:
{'audio': {'path': path, 'array': numpy_array, 'sampling_rate': 16000}}
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import librosa
import soundfile as sf
import gc
import pandas as pd


def find_audio_files(folder_path: str, recursive: bool = False) -> List[str]:
    """
    Find all audio files in the specified folder.
    
    Args:
        folder_path: Path to the folder containing audio files
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of audio file paths
    """
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    audio_files = []
    for ext in audio_extensions:
        if recursive:
            audio_files.extend(glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True))
        else:
            audio_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
    
    return sorted(audio_files)


def read_audio_file(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Read audio file and return audio array and sampling rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate (default: 16000)
        
    Returns:
        Tuple of (audio_array, sampling_rate)
    """
    try:
        # Try to read with soundfile first (better for wav, flac)
        if audio_path.lower().endswith(('.wav', '.flac')):
            audio_array, sr = sf.read(audio_path)
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
        else:
            # Use librosa for other formats
            audio_array, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Resample if necessary
        if sr != target_sr:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Ensure audio is float32 and normalized
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        # Normalize audio to [-1, 1] range
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        return audio_array, sr
        
    except Exception as e:
        raise RuntimeError(f"Error reading audio file {audio_path}: {e}")


def create_audio_sample(audio_path: str, target_sr: int = 16000) -> Dict:
    """
    Create an audio sample in the required format, including transcript if available.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate (default: 16000)
        
    Returns:
        Dictionary in the required format
    """
    try:
        # Read audio file
        audio_array, sampling_rate = read_audio_file(audio_path, target_sr)
        
        # Read transcript file (same name as audio, .txt extension)
        base, _ = os.path.splitext(audio_path)
        transcript_path = base + ".txt"
        transcript = None
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
        
        # Create sample structure
        sample = {
            'audio': {
                'path': audio_path,
                'array': audio_array,
                'sampling_rate': sampling_rate
            },
            'transcript': transcript
        }
        
        return sample
        
    except Exception as e:
        raise RuntimeError(f"Error creating audio sample from {audio_path}: {e}")


def read_dataset_folder(folder_path: str, target_sr: int = 16000, batch_size: int = 100) -> int:
    """
    Read all audio files from data folder and save them to Parquet in batches to avoid OOM.
    Returns the number of processed samples.
    """
    print(f"Reading audio folder: {folder_path}")
    audio_files = find_audio_files(folder_path)
    if not audio_files:
        print("No audio files found in the specified folder.")
        return 0

    print(f"Found {len(audio_files)} audio files")
    parent_dir = os.path.dirname(os.path.abspath(folder_path))
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_path = os.path.join(parent_dir, f"{folder_name}.parquet")
    print(f"Saving to: {output_path}")

    batch = []
    successful_conversions = 0
    total_files = len(audio_files)
    first_sample = None

    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"Processing {i}/{total_files}: {os.path.basename(audio_file)}")
            sample = create_audio_sample(audio_file, target_sr)
            batch.append({
                'path': sample['audio']['path'],
                'array': sample['audio']['array'].tolist() if hasattr(sample['audio']['array'], 'tolist') else sample['audio']['array'],
                'sampling_rate': sample['audio']['sampling_rate'],
                'transcript': sample.get('transcript', ""),
                'error': sample.get('error', None)
            })
            successful_conversions += 1
            if first_sample is None:
                first_sample = sample
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            batch.append({
                'path': audio_file,
                'array': [],
                'sampling_rate': target_sr,
                'transcript': "",
                'error': str(e)
            })

        # Save batch to Parquet and clear memory
        if len(batch) >= batch_size or i == total_files:
            df = pd.DataFrame(batch)
            if i == batch_size:
                df.to_parquet(output_path, index=False)
            else:
                df.to_parquet(output_path, index=False, append=True)
            batch.clear()
            gc.collect()

    print(f"Successfully converted {successful_conversions}/{total_files} files")

    # Print first sample info
    if first_sample:
        print("\n" + "="*50)
        print("FIRST SAMPLE EXAMPLE")
        print("="*50)
        print(f"Audio path: {first_sample['audio']['path']}")
        print(f"Sampling rate: {first_sample['audio']['sampling_rate']}")
        audio_array = first_sample['audio']['array']
        array_info = safe_get_array_info(audio_array)
        print(f"Audio array type: {array_info['type']}")
        print(f"Audio array shape: {array_info['shape']}")
        if 'dtype' in array_info:
            print(f"Audio array dtype: {array_info['dtype']}")
        if 'length' in array_info:
            print(f"Audio array length: {array_info['length']}")
        if 'min' in array_info and 'max' in array_info:
            print(f"Audio array range: [{array_info['min']:.6f}, {array_info['max']:.6f}]")
        if 'error' in first_sample:
            print(f"Error: {first_sample['error']}")

    gc.collect()
    return successful_conversions

def get_audio_info(samples: List[Dict]) -> Dict:
    """
    Get basic information about the audio samples.
    
    Args:
        samples: List of audio samples
        
    Returns:
        Dictionary containing audio statistics
    """
    if not samples:
        return {}
    
    total_samples = len(samples)
    total_duration = 0
    error_samples = 0
    total_size_bytes = 0
    
    for sample in samples:
        if 'audio' in sample and 'array' in sample['audio']:
            audio_array = sample['audio']['array']
            # Handle both numpy arrays and lists (from JSON)
            if hasattr(audio_array, 'shape'):  # numpy array
                if len(audio_array) > 0:
                    duration = len(audio_array) / sample['audio']['sampling_rate']
                    total_duration += duration
                    # Estimate size (float32 = 4 bytes per sample)
                    total_size_bytes += len(audio_array) * 4
            elif isinstance(audio_array, list):  # list from JSON
                if len(audio_array) > 0:
                    duration = len(audio_array) / sample['audio']['sampling_rate']
                    total_duration += duration
                    # Estimate size (float32 = 4 bytes per sample)
                    total_size_bytes += len(audio_array) * 4
        
        if 'error' in sample:
            error_samples += 1
    
    info = {
        'total_samples': total_samples,
        'total_duration_seconds': total_duration,
        'total_duration_minutes': total_duration / 60,
        'error_samples': error_samples,
        'successful_samples': total_samples - error_samples,
        'average_duration_seconds': total_duration / total_samples if total_samples > 0 else 0,
        'estimated_size_mb': total_size_bytes / (1024 * 1024)
    }
    
    return info


def safe_get_array_info(audio_array) -> Dict:
    """
    Safely get information about audio array, handling both numpy arrays and lists.
    
    Args:
        audio_array: Audio array (numpy array or list)
        
    Returns:
        Dictionary containing array information
    """
    info = {}
    
    if hasattr(audio_array, 'shape'):  # numpy array
        info['type'] = 'numpy_array'
        info['shape'] = audio_array.shape
        info['dtype'] = str(audio_array.dtype)
        if len(audio_array) > 0:
            info['min'] = float(audio_array.min())
            info['max'] = float(audio_array.max())
            info['length'] = len(audio_array)
    elif isinstance(audio_array, list):  # list from JSON
        info['type'] = 'list'
        info['shape'] = (len(audio_array),)
        info['dtype'] = 'list'
        if len(audio_array) > 0:
            info['min'] = float(min(audio_array))
            info['max'] = float(max(audio_array))
            info['length'] = len(audio_array)
    else:
        info['type'] = 'unknown'
        info['error'] = f'Unexpected type: {type(audio_array)}'
    
    return info


def save_samples_to_json(samples: List[Dict], output_path: str):
    """
    Save audio samples to JSON file. Note: numpy arrays will be converted to lists.
    
    Args:
        samples: List of audio samples
        output_path: Path to output JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_samples = []
    for sample in samples:
        json_sample = sample.copy()
        if 'audio' in json_sample and 'array' in json_sample['audio']:
            json_sample['audio']['array'] = json_sample['audio']['array'].tolist()
        json_samples.append(json_sample)
    
    try:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_samples, f, indent=2, ensure_ascii=False)
        print(f"Audio samples saved to: {output_path}")
    except Exception as e:
        print(f"Error saving samples to {output_path}: {e}")


def save_samples_to_parquet(samples: List[Dict], output_path: str):
    """
    Save audio samples to Parquet file. Numpy arrays will be converted to lists.
    
    Args:
        samples: List of audio samples
        output_path: Path to output Parquet file
    """
    import pandas as pd

    records = []
    for sample in samples:
        record = {
            'path': sample['audio']['path'],
            'array': sample['audio']['array'].tolist() if hasattr(sample['audio']['array'], 'tolist') else sample['audio']['array'],
            'sampling_rate': sample['audio']['sampling_rate'],
            'transcript': sample.get('transcript', ""),
            'error': sample.get('error', None)
        }
        records.append(record)
    df = pd.DataFrame(records)
    try:
        df.to_parquet(output_path, index=False)
        print(f"Audio samples saved to Parquet: {output_path}")
    except Exception as e:
        print(f"Error saving samples to Parquet {output_path}: {e}")


def read_dataset(folder_path: str, sample_rate: int):
    """Optimized usage to avoid OOM."""
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return

    # Process and save in batches
    num_samples = read_dataset_folder(folder_path, target_sr=sample_rate, batch_size=100)
    print(f"Total processed samples: {num_samples}")

    return num_samples

if __name__ == "__main__":
    # Example usage without command line arguments
    if len(os.sys.argv) == 1:
        # Default folder path
        default_folder = "dataset/vetc_data"
        
        if os.path.exists(default_folder):
            print(f"Reading audio folder: {default_folder}")
            samples = read_dataset_folder(default_folder)
            
            if samples:
                info = get_audio_info(samples)
                print(f"\ndataset contains {info['total_samples']} audio samples")
                print(f"Total duration: {info['total_duration_minutes']:.2f} minutes")
                print(f"Estimated size: {info['estimated_size_mb']:.2f} MB")
        else:
            print(f"Default folder '{default_folder}' not found.")
            print("Usage: python dataset_reader.py <data_folder_path> [--output <output_file>]")
            print("Example: python dataset_reader.py ./dataset/vetc_data --output vetc_audio.json")
    else:
        main()