#!/usr/bin/env python3
"""
Dataset visualization module for generating histograms of classification results.
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def create_histogram(data, title, xlabel, ylabel, output_path, figsize=(10, 6)):
    """
    Create and save a histogram from data.

    Args:
        data: List of values to plot
        title: Title for the histogram
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save the histogram image
        figsize: Figure size tuple (width, height)
    """
    plt.figure(figsize=figsize)

    # Count occurrences
    counter = Counter(data)
    categories = list(counter.keys())
    counts = list(counter.values())

    # Create bar plot
    bars = plt.bar(categories, counts, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Rotate x-axis labels if they're long
    if any(len(str(cat)) > 10 for cat in categories):
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Histogram saved: {output_path}")
    return counter


def create_suitability_histogram(scores, output_path, figsize=(10, 6)):
    """
    Create histogram for suitability scores.

    Args:
        scores: List of suitability scores (0-1)
        output_path: Path to save the histogram
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)

    # Create bins for score ranges
    bins = np.arange(0, 1.1, 0.1)
    n, bins, patches = plt.hist(scores, bins=bins, alpha=0.7, edgecolor='black')

    # Color bars based on quality ranges
    for i, (patch, bin_left) in enumerate(zip(patches, bins[:-1])):
        if bin_left >= 0.8:
            patch.set_facecolor('green')  # Excellent
        elif bin_left >= 0.7:
            patch.set_facecolor('lightgreen')  # Pass
        elif bin_left >= 0.5:
            patch.set_facecolor('orange')  # Review
        else:
            patch.set_facecolor('red')  # Fail

    # Add value labels on bars
    for i, count in enumerate(n):
        if count > 0:
            plt.text(bins[i] + 0.05, count + 0.1, f'{int(count)}',
                     ha='center', va='bottom', fontweight='bold')

    plt.title('Suitability Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Suitability Score', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Add legend
    plt.axvspan(0.8, 1.0, alpha=0.2, color='green', label='Excellent (≥0.8)')
    plt.axvspan(0.7, 0.8, alpha=0.2, color='lightgreen', label='Pass (0.7-0.8)')
    plt.axvspan(0.5, 0.7, alpha=0.2, color='orange', label='Review (0.5-0.7)')
    plt.axvspan(0.0, 0.5, alpha=0.2, color='red', label='Fail (<0.5)')
    plt.legend()

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Suitability histogram saved: {output_path}")


def create_audio_metrics_histogram(audio_data, output_path, figsize=(15, 10)):
    """
    Create multi-subplot histogram for audio metrics.

    Args:
        audio_data: Dict with lists of audio metrics
        output_path: Path to save the histogram
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Audio Metrics Distribution', fontsize=16, fontweight='bold')

    # SNR histogram
    if 'snr' in audio_data and audio_data['snr']:
        # Handle infinite SNR values (999.0 represents inf)
        snr_values = []
        inf_count = 0
        for x in audio_data['snr']:
            if x == 999.0 or x == float('inf'):
                inf_count += 1
            elif x < 100:  # Reasonable SNR values
                snr_values.append(x)

        if snr_values:
            axes[0, 0].hist(snr_values, bins=20, alpha=0.7, edgecolor='black', color='skyblue')

        axes[0, 0].set_title(f'Signal-to-Noise Ratio (dB)\n({inf_count} samples with perfect SNR excluded)')
        axes[0, 0].set_xlabel('SNR (dB)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(axis='y', alpha=0.3)

    # Speech ratio histogram
    if 'speech_ratio' in audio_data and audio_data['speech_ratio']:
        axes[0, 1].hist(audio_data['speech_ratio'], bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
        axes[0, 1].set_title('Speech Ratio')
        axes[0, 1].set_xlabel('Speech Ratio (0-1)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(axis='y', alpha=0.3)

    # Clipping percentage histogram
    if 'clipping' in audio_data and audio_data['clipping']:
        axes[1, 0].hist(audio_data['clipping'], bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 0].set_title('Audio Clipping Percentage')
        axes[1, 0].set_xlabel('Clipping (%)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(axis='y', alpha=0.3)

    # Loudness histogram
    if 'loudness' in audio_data and audio_data['loudness']:
        loudness_values = [x for x in audio_data['loudness'] if x != float('-inf')]
        axes[1, 1].hist(loudness_values, bins=20, alpha=0.7, edgecolor='black', color='pink')
        axes[1, 1].set_title('Loudness (LUFS)')
        axes[1, 1].set_xlabel('Loudness (LUFS)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Audio metrics histogram saved: {output_path}")


def generate_dataset_histograms(results, output_dir="output"):
    """
    Generate all histograms for dataset analysis.

    Args:
        results: List of pipeline results from batch processing
        output_dir: Directory to save histogram images
    """
    print("\n📊 Generating dataset histograms...")

    # Extract data for histograms
    l0_categories = []
    l1_categories = []
    l2_categories = []
    suitability_scores = []
    audio_metrics = {
        'snr': [],
        'speech_ratio': [],
        'clipping': [],
        'loudness': []
    }

    for result in results:
        # Classification data
        if 'classification' in result:
            l0_val = result['classification'].get('L0_content', 'Unknown')
            l1_val = result['classification'].get('L1_domain', 'Unknown')
            l2_val = result['classification'].get('L2_style', 'Unknown')

            # Handle case where L2 might return a dict
            if isinstance(l2_val, dict):
                l2_val = str(l2_val)  # Convert dict to string

            l0_categories.append(l0_val)
            l1_categories.append(l1_val)
            l2_categories.append(l2_val)

        # Suitability scores
        if 'suitability_score' in result:
            suitability_scores.append(result['suitability_score'])

        # Audio metrics
        if 'audio_metrics' in result:
            audio_data = result['audio_metrics']
            audio_metrics['snr'].append(audio_data.get('snr', 0))
            audio_metrics['speech_ratio'].append(audio_data.get('speech_ratio', 0))
            audio_metrics['clipping'].append(audio_data.get('clipping_percentage', 0))
            audio_metrics['loudness'].append(audio_data.get('loudness_lufs', -23))

    # Generate histograms
    if l0_categories:
        create_histogram(
            l0_categories,
            'L0 Content Classification Distribution',
            'Content Category',
            'Number of Samples',
            f'{output_dir}/l0_content_distribution.png'
        )

    if l1_categories:
        create_histogram(
            l1_categories,
            'L1 Domain Classification Distribution',
            'Domain Category',
            'Number of Samples',
            f'{output_dir}/l1_domain_distribution.png'
        )

    if l2_categories:
        create_histogram(
            l2_categories,
            'L2 Speech Style Classification Distribution',
            'Speech Style',
            'Number of Samples',
            f'{output_dir}/l2_style_distribution.png'
        )

    if suitability_scores:
        create_suitability_histogram(
            suitability_scores,
            f'{output_dir}/suitability_scores.png'
        )

    if any(audio_metrics.values()):
        create_audio_metrics_histogram(
            audio_metrics,
            f'{output_dir}/audio_metrics.png'
        )

    print(f"✅ All histograms generated in '{output_dir}/' directory")
    return {
        'l0_distribution': Counter(l0_categories),
        'l1_distribution': Counter(l1_categories),
        'l2_distribution': Counter(l2_categories),
        'suitability_stats': {
            'mean': np.mean(suitability_scores) if suitability_scores else 0,
            'std': np.std(suitability_scores) if suitability_scores else 0,
            'min': min(suitability_scores) if suitability_scores else 0,
            'max': max(suitability_scores) if suitability_scores else 0
        }
    }
