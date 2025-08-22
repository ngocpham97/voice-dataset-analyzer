import unittest
from src.scoring.suitability import calculate_suitability_score

class TestScoring(unittest.TestCase):

    def test_suitability_score_pass(self):
        audio_metrics = {
            'SNR': 20,
            'clipping_pct': 0.5,
            'speech_ratio': 0.7,
            'overlap_ratio': 0.1
        }
        text_metrics = {
            'has_transcript': True,
            'CER_baseline': 0.2,
            'normalization_consistency': 0.9
        }
        score = calculate_suitability_score(audio_metrics, text_metrics)
        self.assertGreaterEqual(score, 0.7)

    def test_suitability_score_review(self):
        audio_metrics = {
            'SNR': 15,
            'clipping_pct': 0.5,
            'speech_ratio': 0.65,
            'overlap_ratio': 0.2
        }
        text_metrics = {
            'has_transcript': True,
            'CER_baseline': 0.3,
            'normalization_consistency': 0.8
        }
        score = calculate_suitability_score(audio_metrics, text_metrics)
        self.assertGreaterEqual(score, 0.5)
        self.assertLess(score, 0.7)

    def test_suitability_score_fail(self):
        audio_metrics = {
            'SNR': 10,
            'clipping_pct': 1.5,
            'speech_ratio': 0.5,
            'overlap_ratio': 0.3
        }
        text_metrics = {
            'has_transcript': False,
            'CER_baseline': 0.4,
            'normalization_consistency': 0.5
        }
        score = calculate_suitability_score(audio_metrics, text_metrics)
        self.assertLess(score, 0.5)

if __name__ == '__main__':
    unittest.main()