import unittest
from src.classification.l0_classifier import L0Classifier
from src.classification.l1_classifier import L1Classifier
from src.classification.l2_classifier import L2Classifier

class TestClassification(unittest.TestCase):

    def setUp(self):
        self.l0_classifier = L0Classifier()
        self.l1_classifier = L1Classifier()
        self.l2_classifier = L2Classifier()

    def test_l0_classification(self):
        transcript = "Hôm nay trời đẹp quá"
        expected_label = "Hội thoại đời thường"
        result = self.l0_classifier.classify(transcript)
        self.assertEqual(result, expected_label)

    def test_l1_classification(self):
        transcript = "Tôi muốn khiếu nại về hóa đơn"
        expected_label = "Khiếu nại"
        result = self.l1_classifier.classify(transcript)
        self.assertEqual(result, expected_label)

    def test_l2_classification_read_speech(self):
        audio_features = {
            'speech_ratio': 0.9,
            'overlap_ratio': 0.0,
            'disfluency_per_min': 1
        }
        expected_label = "Đọc chuẩn"
        result = self.l2_classifier.classify(audio_features)
        self.assertEqual(result, expected_label)

    def test_l2_classification_dialogue(self):
        audio_features = {
            'speech_ratio': 0.7,
            'overlap_ratio': 0.2,
            'disfluency_per_min': 5
        }
        expected_label = "Hội thoại tự nhiên"
        result = self.l2_classifier.classify(audio_features)
        self.assertEqual(result, expected_label)

if __name__ == '__main__':
    unittest.main()