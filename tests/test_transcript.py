import unittest
from src.transcript.normalization import normalize_transcript
from src.transcript.metrics import calculate_cer, calculate_wer

class TestTranscriptNormalization(unittest.TestCase):
    def test_normalize_transcript(self):
        input_transcript = "Đây là một ví dụ về văn bản không chuẩn hóa."
        expected_output = "đây là một ví dụ về văn bản không chuẩn hóa."
        self.assertEqual(normalize_transcript(input_transcript), expected_output)

class TestTranscriptMetrics(unittest.TestCase):
    def test_calculate_cer(self):
        reference = "Đây là một ví dụ."
        hypothesis = "Đây là một ví dụ."
        self.assertEqual(calculate_cer(reference, hypothesis), 0.0)

    def test_calculate_wer(self):
        reference = "Đây là một ví dụ."
        hypothesis = "Đây là ví dụ."
        self.assertEqual(calculate_wer(reference, hypothesis), 0.3333)  # 1 word error

if __name__ == '__main__':
    unittest.main()