import numpy as np
import warnings
warnings.filterwarnings('ignore')


def calculate_overlap_ratio(audio_data, sample_rate=16000):
    """
    Calculate overlap ratio based on energy analysis.
    
    Args:
        audio_data: Audio array
        sample_rate: Sampling rate
        
    Returns:
        Overlap ratio (0-1)
    """
    try:
        import librosa
        
        # Calculate energy envelope
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Normalize energy
        rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
        
        # Find high energy regions (potential speech)
        speech_threshold = 0.3
        speech_regions = rms_normalized > speech_threshold
        
        # Calculate overlap based on energy distribution
        if np.sum(speech_regions) > 0:
            # Count frames with very high energy (potential overlap)
            high_energy_threshold = 0.7
            high_energy_frames = np.sum(rms_normalized > high_energy_threshold)
            total_speech_frames = np.sum(speech_regions)
            
            overlap_ratio = high_energy_frames / total_speech_frames if total_speech_frames > 0 else 0.0
        else:
            overlap_ratio = 0.0
        
        return min(overlap_ratio, 1.0)
        
    except Exception as e:
        print(f"Warning: Overlap calculation failed: {e}")
        return 0.1


class Diarization:
    def __init__(self, model='pyannote', auth_token=None):
        """
        Initialize diarization system using pyannote.audio.
        
        Args:
            model: Diarization method ('pyannote', 'fallback')
            auth_token: HuggingFace authentication token for pyannote
        """
        self.model = model
        self.auth_token = auth_token
        self.pipeline = None
        
        if model == 'pyannote':
            self._initialize_pyannote()
    
    def _initialize_pyannote(self):
        """Initialize pyannote.audio pipeline."""
        try:
            from pyannote.audio import Pipeline
            
            if not self.auth_token:
                print("Warning: No auth token provided. Using fallback method.")
                self.model = 'fallback'
                return
            
            # Initialize pyannote pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2022.07",
                use_auth_token=self.auth_token
            )
            
            print("✓ Pyannote.audio pipeline initialized successfully")
            
        except ImportError:
            print("Warning: pyannote.audio not installed. Using fallback method.")
            print("Install with: pip install pyannote.audio")
            self.model = 'fallback'
        except Exception as e:
            print(f"Warning: Pyannote initialization failed: {e}")
            print("Falling back to basic method")
            self.model = 'fallback'
    
    def perform_diarization(self, audio_file):
        """
        Perform speaker diarization using pyannote.audio.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with diarization results
        """
        if self.model == 'pyannote' and self.pipeline:
            return self._pyannote_diarization(audio_file)
        else:
            return self._fallback_diarization(audio_file)
    
    def _pyannote_diarization(self, audio_file):
        """Perform diarization using pyannote.audio."""
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            
            print(f"Running pyannote diarization on: {audio_file}")
            
            # Apply diarization with progress tracking
            with ProgressHook() as hook:
                diarization = self.pipeline(audio_file, hook=hook)
            
            # Extract speaker information
            speakers = set()
            speaker_segments = []
            overlap_ratio = 0.0
            
            # Process diarization results
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                
                # Create speaker segment
                segment = {
                    'start_time': float(turn.start),
                    'end_time': float(turn.end),
                    'duration': float(turn.end - turn.start),
                    'speaker_id': speaker
                }
                speaker_segments.append(segment)
            
            num_speakers = len(speakers)
            
            # Calculate overlap ratio
            if len(speaker_segments) > 1:
                overlap_ratio = self._calculate_pyannote_overlap(diarization)
            
            # Calculate confidence based on diarization quality
            confidence = self._calculate_pyannote_confidence(diarization, num_speakers)
            
            return {
                'num_speakers': int(num_speakers),
                'overlap_ratio': round(float(overlap_ratio), 3),
                'speaker_segments': speaker_segments,
                'confidence': round(float(confidence), 3),
                'method': 'pyannote',
                'total_segments': len(speaker_segments),
                'speaker_ids': list(speakers)
            }
            
        except Exception as e:
            print(f"Warning: Pyannote diarization failed: {e}")
            print("Falling back to basic method")
            return self._fallback_diarization(audio_file)
    
    def _calculate_pyannote_overlap(self, diarization):
        """Calculate overlap ratio from pyannote diarization results."""
        try:
            total_duration = 0
            overlap_duration = 0
            
            # Get timeline of all speech activity
            speech_timeline = diarization.get_timeline()
            
            # Calculate total speech duration
            for segment in speech_timeline:
                total_duration += segment.duration
            
            # Find overlapping regions
            if len(speech_timeline) > 1:
                for i, seg1 in enumerate(speech_timeline):
                    for seg2 in speech_timeline[i+1:]:
                        if seg1.intersects(seg2):
                            overlap = seg1.intersection(seg2)
                            overlap_duration += overlap.duration
            
            overlap_ratio = overlap_duration / total_duration if total_duration > 0 else 0.0
            return min(overlap_ratio, 1.0)
            
        except Exception as e:
            print(f"Warning: Overlap calculation failed: {e}")
            return 0.0
    
    def _calculate_pyannote_confidence(self, diarization, num_speakers):
        """Calculate confidence score for pyannote diarization."""
        try:
            # Base confidence on number of segments
            total_segments = len(list(diarization.itertracks(yield_label=True)))
            segment_confidence = min(1.0, total_segments / (num_speakers * 3))
            
            # Adjust confidence based on number of speakers
            if num_speakers == 1:
                speaker_confidence = 0.95  # Pyannote is very reliable for single speaker
            elif num_speakers == 2:
                speaker_confidence = 0.90
            elif num_speakers == 3:
                speaker_confidence = 0.85
            else:
                speaker_confidence = 0.80
            
            # Combine confidences
            final_confidence = 0.7 * segment_confidence + 0.3 * speaker_confidence
            
            return min(1.0, max(0.0, final_confidence))
            
        except Exception as e:
            return 0.8  # Default high confidence for pyannote
    
    def _fallback_diarization(self, audio_file):
        """Fallback diarization method when pyannote is not available."""
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            if isinstance(audio_file, str):
                audio_array, sample_rate = sf.read(audio_file)
                if len(audio_array.shape) > 1:  # Convert stereo to mono
                    audio_array = np.mean(audio_array, axis=1)
            else:
                audio_array = audio_file
                sample_rate = 16000
            
            # Simple energy-based speaker detection
            num_speakers = self._energy_based_detection(audio_array, sample_rate)
            
            # Calculate overlap ratio
            overlap_ratio = calculate_overlap_ratio(audio_array, sample_rate)
            
            # Create simple speaker segments
            speaker_segments = self._create_simple_segments(audio_array, sample_rate, num_speakers)
            
            return {
                'num_speakers': int(num_speakers),
                'overlap_ratio': round(float(overlap_ratio), 3),
                'speaker_segments': speaker_segments,
                'confidence': 0.6,  # Lower confidence for fallback method
                'method': 'fallback',
                'total_segments': len(speaker_segments),
                'speaker_ids': [f"SPEAKER_{i}" for i in range(num_speakers)]
            }
            
        except Exception as e:
            print(f"Warning: Fallback diarization failed: {e}")
            return {
                'num_speakers': 1,
                'overlap_ratio': 0.1,
                'speaker_segments': [],
                'confidence': 0.0,
                'method': 'fallback',
                'error': str(e)
            }
    
    def _energy_based_detection(self, audio_array, sample_rate):
        """Simple energy-based speaker detection for fallback."""
        try:
            # Calculate energy envelope
            frame_length = int(0.025 * sample_rate)
            hop_length = int(0.010 * sample_rate)
            rms = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Find energy peaks (potential speaker changes)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(rms, height=np.percentile(rms, 70), 
                                distance=int(0.5 * sample_rate / hop_length))
            
            # Estimate speakers based on energy variation
            if len(peaks) <= 2:
                return 1
            elif len(peaks) <= 5:
                return 2
            elif len(peaks) <= 10:
                return 3
            else:
                return min(4, 5)
                
        except Exception as e:
            return 1
    
    def _create_simple_segments(self, audio_array, sample_rate, num_speakers):
        """Create simple speaker segments for fallback method."""
        try:
            segments = []
            segment_duration = 3.0  # 3 seconds per segment
            segment_length = int(segment_duration * sample_rate)
            
            for i in range(0, len(audio_array), segment_length):
                segment = audio_array[i:i + segment_length]
                if len(segment) >= segment_length // 2:
                    segments.append({
                        'start_time': i / sample_rate,
                        'end_time': (i + len(segment)) / sample_rate,
                        'duration': len(segment) / sample_rate,
                        'speaker_id': f"SPEAKER_{i // segment_length % num_speakers}"
                    })
            
            return segments
            
        except Exception as e:
            return []
