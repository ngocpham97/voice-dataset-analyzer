from typing import Dict, List


class L2Classifier:
    def __init__(self):
        self.speech_patterns = {
            "read_speech": self.is_read_speech,
            "natural_dialogue": self.is_natural_dialogue,
            "monologue": self.is_monologue,
            "interview": self.is_interview,
            "presentation": self.is_presentation,
            "storytelling": self.is_storytelling,
        }
        self.styles = {
            "serious": self.is_serious_style,
            "casual": self.is_casual_style,
            "enthusiastic": self.is_enthusiastic_style,
            "humorous": self.is_humorous_style,
        }

    def classify(self, audio_features: Dict, transcript_features: Dict) -> Dict:
        speech_type = self.classify_speech_type(audio_features, transcript_features)
        style = self.classify_style(audio_features, transcript_features)
        return {"speech_type": speech_type, "style": style}

    def classify_speech_type(self, audio_features: Dict, transcript_features: Dict) -> str:
        for speech_type, method in self.speech_patterns.items():
            if method(audio_features, transcript_features):
                return speech_type
        return "unknown"

    def classify_style(self, audio_features: Dict, transcript_features: Dict) -> str:
        for style, method in self.styles.items():
            if method(audio_features, transcript_features):
                return style
        return "unknown"

    def is_read_speech(self, audio_features: Dict, transcript_features: Dict) -> bool:
        overlap_ratio = audio_features.get('overlap_ratio', 0.1)
        disfluency = transcript_features.get('disfluency_per_min', 0)
        return overlap_ratio < 0.05 and disfluency < 1

    def is_natural_dialogue(self, audio_features: Dict, transcript_features: Dict) -> bool:
        num_speakers = audio_features.get('num_speakers', 1)
        overlap_ratio = audio_features.get('overlap_ratio', 0.1)
        return num_speakers >= 2 and overlap_ratio >= 0.15

    def is_monologue(self, audio_features: Dict, transcript_features: Dict) -> bool:
        num_speakers = audio_features.get('num_speakers', 1)
        return num_speakers == 1

    def is_interview(self, audio_features: Dict, transcript_features: Dict) -> bool:
        num_speakers = audio_features.get('num_speakers', 1)
        keywords = transcript_features.get('keywords', [])
        return num_speakers == 2 and 'why' in keywords

    def is_presentation(self, audio_features: Dict, transcript_features: Dict) -> bool:
        num_speakers = audio_features.get('num_speakers', 1)
        structure = transcript_features.get('structure', '')
        return num_speakers == 1 and structure == 'presentation'

    def is_storytelling(self, audio_features: Dict, transcript_features: Dict) -> bool:
        keywords = transcript_features.get('keywords', [])
        return 'story' in keywords

    def is_serious_style(self, audio_features: Dict, transcript_features: Dict) -> bool:
        keywords = transcript_features.get('keywords', [])
        return 'serious' in keywords

    def is_casual_style(self, audio_features: Dict, transcript_features: Dict) -> bool:
        keywords = transcript_features.get('keywords', [])
        return 'hey' in keywords or 'you know' in keywords

    def is_enthusiastic_style(self, audio_features: Dict, transcript_features: Dict) -> bool:
        keywords = transcript_features.get('keywords', [])
        return 'excited' in keywords

    def is_humorous_style(self, audio_features: Dict, transcript_features: Dict) -> bool:
        keywords = transcript_features.get('keywords', [])
        return 'funny' in keywords or 'joke' in keywords
