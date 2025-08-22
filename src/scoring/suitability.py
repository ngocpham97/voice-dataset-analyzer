from ..audio.acoustic_metrics import (calculate_clipping_percentage,
                                      calculate_loudness, calculate_snr)
from ..audio.diarization import calculate_overlap_ratio
from ..audio.vad import calculate_speech_ratio
from ..transcript.metrics import calculate_cer, calculate_wer
from ..transcript.normalization import check_normalization_consistency


class SuitabilityEvaluator:
    def __init__(self, audio_metrics, transcript_metrics):
        self.audio_metrics = audio_metrics
        self.transcript_metrics = transcript_metrics

    def evaluate_audio(self):
        snr = self.audio_metrics.get('snr', 0)
        clipping_pct = self.audio_metrics.get('clipping_pct', 0)
        speech_ratio = self.audio_metrics.get('speech_ratio', 0)
        overlap_ratio = self.audio_metrics.get('overlap_ratio', 0)

        q_audio = (
            (min(1, snr / 20) +
             (1 - clipping_pct) +
             speech_ratio +
             (1 - overlap_ratio)) / 4
        )
        return q_audio

    def evaluate_transcript(self):
        has_transcript = 1 if self.transcript_metrics.get('has_transcript') else 0
        cer_baseline = self.transcript_metrics.get('cer_baseline', 1)
        transcript_text = self.transcript_metrics.get('transcript', '')
        normalization_consistency = 1.0 if transcript_text else 0.0

        q_text = (
            (has_transcript +
             (1 - cer_baseline) +
             normalization_consistency) / 3
        )
        return q_text

    def calculate_suitability(self):
        q_audio = self.evaluate_audio()
        q_text = self.evaluate_transcript()

        suitability_score = 0.6 * q_audio + 0.4 * q_text
        return suitability_score

    def get_suitability_category(self, score):
        if score >= 0.7:
            return 'PASS'
        elif 0.5 <= score < 0.7:
            return 'REVIEW'
        else:
            return 'FAIL'


def calculate_suitability_score(audio_metrics, transcript_metrics):
    """Standalone function for calculating suitability score."""
    evaluator = SuitabilityEvaluator(audio_metrics, transcript_metrics)
    return evaluator.calculate_suitability()
    return 'FAIL'
    return 'FAIL'
    return 'FAIL'
    return 'FAIL'
    return 'FAIL'
