# Voice Dataset Evaluator

This project provides a framework for evaluating and analyzing voice datasets based on specified criteria for Automatic Speech Recognition (ASR) suitability. The evaluation process includes audio analysis, transcript evaluation, classification, and scoring.

## Project Structure

```
voice-dataset-evaluator
├── src
│   ├── audio
│   ├── transcript
│   ├── classification
│   ├── scoring
│   ├── pipeline.py
│   └── utils.py
├── tests
├── requirements.txt
├── README.md
└── setup.py
```

## Features

- **Audio Analysis**: Evaluate audio quality using metrics such as Signal-to-Noise Ratio (SNR), clipping percentage, and loudness.
- **Transcript Evaluation**: Normalize transcripts and calculate Character Error Rate (CER) and Word Error Rate (WER).
- **Classification**: Classify transcripts into different levels (L0, L1, L2) based on content and context.
- **Scoring**: Calculate a suitability score for ASR based on audio and transcript metrics.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. **Audio Processing**: Use the functions in `src/audio` to analyze audio files.
2. **Transcript Processing**: Normalize and evaluate transcripts using `src/transcript`.
3. **Classification**: Classify the processed data using the classifiers in `src/classification`.
4. **Scoring**: Calculate the suitability score using the scoring functions in `src/scoring`.
5. **Pipeline**: Use `src/pipeline.py` to run the entire evaluation process in one go.

## Testing

Unit tests are provided in the `tests` directory. To run the tests, use:

```
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.