# Voice Dataset Analyzer - Vietnamese Audio Dataset Evaluation Framework

## 🎯 Project Overview

The Voice Dataset Analyzer is a comprehensive framework designed to evaluate and analyze Vietnamese voice datasets from HuggingFace. This tool provides automated quality assessment, content classification, and detailed reporting for audio datasets, specifically focusing on the `linhtran92/viet_bud500` dataset.

## 🚀 Key Features

- **🔐 Automated Authentication** - Seamless HuggingFace integration with token management
- **💾 Smart Caching** - Local dataset storage to prevent re-downloading
- **🎵 Audio Quality Analysis** - SNR with intelligent noise estimation, clipping, loudness, and speech activity detection
- **📝 Transcript Evaluation** - Text normalization, CER/WER calculation, and quality scoring
- **🏷️ Multi-Level Classification** - Content categorization across L0, L1, and L2 dimensions
- **📊 Data Visualization** - Automated histogram generation for dataset insights
- **📈 Comprehensive Reporting** - JSON exports with timestamps and detailed metrics

## 🔄 Analysis Pipeline

### Step 1: Authentication & Setup
```bash
# Ensure HuggingFace access is configured
source .env  # Load authentication tokens
```

**What it does:**
- Verifies HuggingFace authentication using stored tokens
- Checks access permissions for gated datasets
- Validates environment configuration

**Technologies & Packages Used:**
- **`huggingface_hub`** - HuggingFace API client for authentication and dataset access
- **`os`** - Environment variable management for token storage
- **Authentication Methods:**
  - Environment variables (`HF_TOKEN`)
  - HuggingFace CLI token storage
  - Token validation via `whoami()` API call

**Output:** Authentication status confirmation

---

### Step 2: Dataset Loading & Caching
```python
# Load samples from parquet files with local caching
dataset = load_dataset("parquet", data_files=data_files, cache_dir="dataset")
```

**What it does:**
- Downloads dataset from HuggingFace parquet URLs (first time only)
- Stores data locally in `dataset/` folder for future use
- Extracts audio arrays and transcriptions from dataset samples
- Handles AudioDecoder objects for proper audio data extraction

**Technologies & Packages Used:**
- **`datasets`** (HuggingFace) - Dataset loading and caching framework
  - Automatic parquet file parsing
  - Built-in caching system with integrity checks
  - Memory-mapped file access for large datasets
- **`numpy`** - Array processing for audio data extraction
- **Audio Processing Pipeline:**
  - Raw audio bytes → numpy arrays (16kHz sampling rate)
  - AudioDecoder object handling for HuggingFace audio format
  - Automatic format detection and conversion

**Output:** 
- Cached dataset files in `dataset/` directory
- Loaded audio arrays (16kHz) and Vietnamese transcriptions

---

### Step 3: Audio Processing & Quality Analysis
```python
# Audio feature extraction and quality metrics
acoustic_metrics = analyze_audio_metrics(audio_array, 0)
speech_ratio = calculate_speech_ratio(audio_array)
```

**What it does:**
- **SNR Calculation** - Intelligent signal-to-noise ratio analysis with automatic noise estimation
- **Clipping Detection** - Identifies audio distortion
- **Loudness Measurement** - LUFS standardized loudness levels
- **Voice Activity Detection** - Determines speech vs silence ratio
- **Audio Quality Scoring** - Combines metrics into 0-1 score

**Technologies & Packages Used:**
- **`numpy`** - Core mathematical operations for signal processing
  - Power calculation: `np.mean(signal ** 2)` for signal power computation
  - Noise floor estimation: `np.percentile(np.abs(signal), 10)` for noise detection
  - RMS calculation: `np.sqrt(np.mean(audio_signal ** 2))` for loudness
  - Statistical analysis: `np.sum()`, `np.abs()` for clipping detection
- **`scipy.io`** - Audio file I/O operations via `wavfile` module
- **Signal Processing Algorithms:**
  - **SNR**: Enhanced calculation using noise floor estimation from signal
    - `estimate_noise_floor()` - Extracts noise from quietest 10% of signal
    - `10 * log10(signal_power / estimated_noise_power)` for realistic SNR values
    - Fallback to -60 dB digital noise floor for very clean signals
  - **EBU R128 Loudness**: `20 * log10(rms) + 94` LUFS standard
  - **Clipping Detection**: Threshold-based analysis at ±1.0 amplitude
  - **Voice Activity Detection**: Energy-based threshold analysis

**Metrics Generated:**
- `SNR`: Signal quality with realistic noise estimation (typical range: 10-40 dB)
- `Clipping %`: Audio distortion level (lower = better)
- `Speech Ratio`: Percentage of speech content (0.5-0.8 ideal)
- `Loudness`: Audio level in LUFS (-23 target for broadcast)

---

### Step 4: Transcript Analysis & Error Calculation
```python
# Text processing and accuracy metrics
normalized_transcript = normalize_transcript(transcript)
cer = calculate_cer(transcript, normalized_transcript)
wer = calculate_wer(transcript, normalized_transcript)
```

**What it does:**
- **Text Normalization** - Standardizes Vietnamese text formatting
- **Character Error Rate (CER)** - Measures character-level accuracy
- **Word Error Rate (WER)** - Evaluates word-level transcription quality
- **Length Analysis** - Assesses transcript completeness
- **Transcript Quality Scoring** - Generates 0-1 quality score

**Technologies & Packages Used:**
- **String Processing (Python built-in):**
  - `str.lower()` - Case normalization
  - `str.split()`, `str.join()` - Whitespace standardization
  - Regular expressions via `re` module for punctuation normalization
- **Edit Distance Algorithm:**
  - **Levenshtein Distance** implementation for CER/WER calculation
  - Dynamic programming approach: O(m×n) time complexity
  - Character-level comparison for CER: `reference.replace(" ", "")`
  - Word-level tokenization for WER: `reference.split()`
- **Vietnamese Text Processing:**
  - Unicode normalization for Vietnamese diacritics
  - Punctuation standardization (`.`, `?`, `!` spacing)
  - Consistent formatting for ASR evaluation

**Metrics Generated:**
- `CER`: Character accuracy (0.0 = perfect)
- `WER`: Word accuracy (0.0 = perfect)
- `Text Length`: Number of characters/words
- `Transcript Score`: Combined quality metric

---

### Step 5: Multi-Level Content Classification

#### L0 Classification - Content Type
```python
# Domain-specific content categorization
l0_result = l0_classifier.classify(normalized_transcript, None)
```

**Technologies & Packages Used:**
- **Keyword-Based Classification System:**
  - `collections.defaultdict` - Efficient score accumulation
  - `re` module - Regular expression pattern matching
  - Custom Vietnamese keyword dictionaries for 14 content categories
- **Scoring Algorithm:**
  - Frequency-based keyword matching
  - Text normalization preprocessing
  - Confidence scoring based on keyword density

**Categories:**
- `hội thoại đời thường` - Daily conversation
- `dịch vụ/CSKH` - Customer service
- `tin tức/thời sự` - News/current affairs
- `giáo dục/giảng giải` - Educational content
- `doanh nghiệp/họp` - Business meetings
- `podcast/tự sự` - Podcast/narrative
- `giải trí/show` - Entertainment
- `hướng dẫn/kỹ thuật` - Technical tutorials
- `tài chính/kinh tế` - Finance/economics
- `y tế` - Healthcare
- `pháp lý` - Legal
- `trẻ em` - Children's content
- `tôn giáo/văn hóa` - Religion/culture
- `thể thao` - Sports

#### L1 Classification - Domain Analysis
```python
# High-level domain classification
l1_result = l1_classifier.classify(normalized_transcript)
```

**Technologies & Packages Used:**
- **Enhanced Keyword Classification:**
  - `collections.defaultdict` - Score aggregation system
  - Extended keyword dictionaries with domain-specific terminology
  - Multi-level scoring based on term frequency and relevance

**Categories:**
- General conversation analysis
- Professional/formal content detection
- Educational material identification

#### L2 Classification - Speech Style Analysis  
```python
# Speech pattern and style evaluation
l2_result = l2_classifier.classify(audio_metadata, transcript_data)
```

**Technologies & Packages Used:**
- **Multi-Modal Analysis:**
  - Audio signal processing for interaction pattern detection
  - Text analysis for style classification
  - Combined audio-text feature extraction
- **Pattern Recognition:**
  - Statistical analysis of speech patterns
  - Temporal audio analysis for dialogue detection
  - Linguistic feature extraction for style categorization

**Categories:**
- `monologue` vs `dialogue` detection
- `serious`, `casual`, `enthusiastic`, `humorous` style classification
- Speaker interaction patterns

---

### Step 6: Suitability Scoring & Quality Assessment
```python
# Combined quality scoring algorithm
suitability_score = 0.6 * audio_score + 0.4 * transcript_score
```

**Technologies & Packages Used:**
- **Weighted Scoring Algorithm:**
  - `numpy` - Mathematical operations for score normalization
  - Custom scoring functions with configurable weights
  - Multi-criteria decision analysis (MCDA) approach
- **Score Aggregation:**
  - Linear combination of normalized metrics
  - Range validation (0.0-1.0) with boundary checking
  - Statistical validation of score distributions

**Scoring Components:**

**Audio Score (60% weight):**
- SNR quality: 30%
- Clipping absence: 30% 
- Speech activity: 30%
- Loudness appropriateness: 10%

**Transcript Score (40% weight):**
- Transcript availability: 40%
- Character accuracy (CER): 30%
- Word accuracy (WER): 20%
- Content length: 10%

**Quality Categories:**
- **EXCELLENT ⭐** (≥0.8) - High-quality samples ready for production
- **PASS ✅** (0.7-0.8) - Good quality, minor issues
- **REVIEW ⚠️** (0.5-0.7) - Requires manual review
- **FAIL ❌** (<0.5) - Poor quality, needs significant improvement

---

### Step 7: Data Visualization & Analysis
```python
# Automated histogram generation
generate_dataset_histograms(results, output_dir="output")
```

**Technologies & Packages Used:**
- **`matplotlib.pyplot`** - Primary plotting and visualization engine
  - Histogram generation with customizable bins and styling
  - Multi-panel subplot layouts (2×2 grids)
  - Color-coded visualizations with transparency settings
- **`numpy`** - Statistical processing for histogram data
  - Data aggregation and binning calculations
  - Array manipulation for plotting optimization
- **`collections.Counter`** - Frequency counting for categorical data
- **File I/O Operations:**
  - `os.makedirs()` - Automatic directory creation
  - PNG export with 300 DPI resolution
  - `plt.savefig()` with optimized settings

**Generated Visualizations:**

1. **L0 Content Distribution** (`l0_content_distribution.png`)
   - Bar chart showing content type frequencies
   - Helps identify dataset composition

2. **L1 Domain Distribution** (`l1_domain_distribution.png`)
   - Domain category breakdown
   - Reveals dataset domain balance

3. **L2 Style Distribution** (`l2_style_distribution.png`)
   - Speech style pattern analysis
   - Shows monologue vs dialogue ratios

4. **Suitability Score Distribution** (`suitability_scores.png`)
   - Quality score histogram with color coding
   - Green: Excellent, Light Green: Pass, Orange: Review, Red: Fail

5. **Audio Metrics Distribution** (`audio_metrics.png`)
   - Multi-panel visualization of:
     - SNR distribution
     - Speech ratio patterns
     - Clipping percentage analysis
     - Loudness level distribution

---

### Step 8: Results Export & Reporting
```python
# Comprehensive JSON report generation
json.dump(result, f, indent=2, ensure_ascii=False)
```

**Technologies & Packages Used:**
- **`json`** (Python standard library) - Structured data serialization
  - UTF-8 encoding with `ensure_ascii=False` for Vietnamese text
  - Pretty-printing with 2-space indentation
  - Nested dictionary structure preservation
- **`datetime`** - Timestamp generation for file naming
- **Data Aggregation:**
  - Statistical calculations (mean, min, max, std)
  - Distribution analysis using `collections.Counter`
  - Nested data structure organization for hierarchical reporting

**Report Contents:**

**Dataset Information:**
- Source dataset identification
- Number of samples processed
- Total audio duration
- Average sample length

**Aggregate Scores:**
- Mean suitability score
- Score range (min/max)
- Average audio metrics
- Mean error rates

**Distribution Analysis:**
- Quality category counts
- L0/L1/L2 classification breakdowns
- Statistical summaries

**Individual Sample Details:**
- Per-sample metrics and scores
- Audio characteristics
- Transcript quality measures
- Classification results
- Text previews

## 📊 Usage Examples

### Basic Analysis (3 samples)
```bash
python hf_only_demo.py
```

### Large Dataset Analysis (100 samples)
```bash
python hf_only_demo.py --samples 100
```

### Clean Analysis (clear previous results)
```bash
python hf_only_demo.py --clear-output --samples 50
```

### Quick Single Sample Test
```bash
python hf_only_demo.py -c -n 1
```

## 📁 Output Structure

```
output/
├── audio_metrics.png                    # Audio quality distributions
├── l0_content_distribution.png          # Content type breakdown
├── l1_domain_distribution.png           # Domain analysis
├── l2_style_distribution.png            # Speech style patterns
├── suitability_scores.png               # Quality score histogram
└── dataset_evaluation_YYYYMMDD_HHMMSS.json  # Detailed results
```

## 🔧 Technical Requirements

**Core Dependencies:**
- **Python 3.11+** - Latest stable Python for optimal performance
- **Audio Processing Stack:**
  - **`numpy`** - Fast numerical operations and array processing
  - **`scipy`** - Scientific computing library for audio I/O (`scipy.io.wavfile`)
  - **`librosa`** - Advanced audio analysis and feature extraction
  - **`pydub`** - Audio file format conversion and manipulation
  - **`webrtcvad`** - Voice Activity Detection (VAD) algorithms
  - **`pyannote.audio`** - Speaker diarization and audio segmentation
- **Machine Learning & NLP:**
  - **`transformers`** - HuggingFace transformer models for text processing
  - **`torch`** - PyTorch backend for neural network operations
- **Data Processing:**
  - **`datasets`** - HuggingFace datasets library for efficient data loading
  - **`huggingface_hub`** - API client for dataset access and authentication
- **Visualization:**
  - **`matplotlib`** - Plotting and histogram generation
- **Development Tools:**
  - **`pytest`** - Unit testing framework
  - **`ffmpeg-python`** - Audio format conversion utilities

**System Requirements:**
- **Memory:** 4GB+ RAM (8GB recommended for large datasets)
- **Storage:** 2GB+ for dataset caching
- **Network:** Stable internet for HuggingFace dataset downloads

## 🎯 Use Cases

1. **Dataset Quality Assessment** - Evaluate audio dataset suitability for ASR training
2. **Content Analysis** - Understand dataset composition and balance
3. **Quality Control** - Identify problematic samples requiring attention
4. **Research Insights** - Generate statistical reports for academic analysis
5. **Production Readiness** - Validate datasets before model training

## 📈 Interpretation Guide

### High-Quality Samples (EXCELLENT ⭐)
- Clear audio with minimal noise (SNR > 20 dB typically)
- Accurate transcriptions
- Appropriate content classification
- Suitable for direct use in training

### Review-Needed Samples (REVIEW ⚠️)
- Audio quality issues or transcription errors
- Unclear content classification
- May require manual correction

### Failed Samples (FAIL ❌)
- Significant audio problems
- Missing or poor transcriptions
- Should be excluded from training sets

This framework provides a comprehensive, automated approach to Vietnamese voice dataset evaluation, enabling researchers and developers to make informed decisions about dataset quality and composition.