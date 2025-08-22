import unittest
from src.audio.vad import calculate_speech_ratio
from src.audio.acoustic_metrics import calculate_snr, calculate_clipping_percentage, calculate_loudness

class TestAudioProcessing(unittest.TestCase):

    def test_calculate_speech_ratio(self):
        audio_segments = [0.5, 1.0, 0.2, 0.8, 0.0]  # Example segments
        expected_ratio = 0.8  # Expected speech ratio
        result = calculate_speech_ratio(audio_segments)
        self.assertAlmostEqual(result, expected_ratio, places=2)

    def test_calculate_snr(self):
        signal_power = 10  # Example signal power
        noise_power = 1    # Example noise power
        expected_snr = 10   # Expected SNR in dB
        result = calculate_snr(signal_power, noise_power)
        self.assertAlmostEqual(result, expected_snr, places=2)

    def test_calculate_clipping_percentage(self):
        audio_samples = [0.5, 0.7, 1.0, -1.0, 0.9]  # Example audio samples
        expected_clipping = 0.4  # Expected clipping percentage
        result = calculate_clipping_percentage(audio_samples)
        self.assertAlmostEqual(result, expected_clipping, places=2)

    def test_calculate_loudness(self):
        audio_samples = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example audio samples
        expected_loudness = -20  # Expected loudness in LUFS
        result = calculate_loudness(audio_samples)
        self.assertAlmostEqual(result, expected_loudness, places=2)

if __name__ == '__main__':
    unittest.main()