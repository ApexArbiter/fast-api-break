from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import io
import logging
from typing import Tuple, Dict, Any, List
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Similarity API", version="1.0.0")

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate
        self.hop_length = 512
        self.n_mfcc = 13
        self.n_fft = 2048
        
    def load_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load audio from bytes - WAV format only"""
        try:
            # Load WAV file directly using soundfile
            import soundfile as sf
            audio_buffer = io.BytesIO(audio_bytes)
            y, sr = sf.read(audio_buffer)
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = librosa.to_mono(y.T)
            
            # Resample if needed
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            
            return y, self.sample_rate
                        
        except Exception as e:
            logger.error(f"Error loading WAV audio: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error loading WAV audio. Please ensure the file is a valid WAV format. Details: {str(e)}")
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Remove silence from beginning and end of audio"""
        try:
            # Trim silence
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
            
            # Ensure we have some audio left
            if len(trimmed_audio) < self.sample_rate * 0.1:  # Less than 0.1 seconds
                logger.warning("Audio too short after trimming")
                return audio  # Return original if trimming removes too much
                
            return trimmed_audio
        except Exception as e:
            logger.error(f"Error trimming silence: {str(e)}")
            return audio
    
    def detect_voice_breaks(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detect breaks/pauses in speech (like aaaaa...aaaaa...aaaa)"""
        try:
            # Calculate RMS energy with a smaller frame size for better break detection
            frame_length = 1024
            hop_length = 512
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Calculate time axis
            times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sample_rate, hop_length=hop_length)
            
            # Detect voice activity using RMS threshold
            # Use adaptive threshold based on audio statistics
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            voice_threshold = max(rms_mean - 0.5 * rms_std, 0.01)
            
            # Create voice activity detection
            voice_activity = rms > voice_threshold
            
            # Find voice segments and breaks
            voice_segments = []
            break_segments = []
            
            # Track state changes
            in_voice = False
            segment_start = 0
            
            for i, is_voice in enumerate(voice_activity):
                if is_voice and not in_voice:
                    # Start of voice segment
                    if in_voice is False and segment_start < i:
                        # End of break segment
                        break_duration = times[i-1] - times[segment_start] if i > 0 else 0
                        if break_duration > 0.1:  # Only count breaks longer than 100ms
                            break_segments.append({
                                'start': times[segment_start],
                                'end': times[i-1] if i > 0 else times[i],
                                'duration': break_duration
                            })
                    segment_start = i
                    in_voice = True
                elif not is_voice and in_voice:
                    # End of voice segment
                    voice_duration = times[i-1] - times[segment_start] if i > 0 else 0
                    if voice_duration > 0.05:  # Only count voice segments longer than 50ms
                        voice_segments.append({
                            'start': times[segment_start],
                            'end': times[i-1] if i > 0 else times[i],
                            'duration': voice_duration
                        })
                    segment_start = i
                    in_voice = False
            
            # Handle final segment
            if in_voice and segment_start < len(voice_activity):
                voice_duration = times[-1] - times[segment_start]
                if voice_duration > 0.05:
                    voice_segments.append({
                        'start': times[segment_start],
                        'end': times[-1],
                        'duration': voice_duration
                    }) 
            elif not in_voice and segment_start < len(voice_activity):
                break_duration = times[-1] - times[segment_start]
                if break_duration > 0.1:
                    break_segments.append({
                        'start': times[segment_start],
                        'end': times[-1],
                        'duration': break_duration
                    })
            
            # Calculate statistics
            total_voice_time = sum(seg['duration'] for seg in voice_segments)
            total_break_time = sum(seg['duration'] for seg in break_segments)
            total_duration = times[-1] if len(times) > 0 else 0
            
            voice_continuity_ratio = total_voice_time / total_duration if total_duration > 0 else 0
            break_frequency = len(break_segments) / total_duration if total_duration > 0 else 0
            
            # Determine speech pattern
            speech_pattern = "continuous"
            if len(break_segments) > 2 and break_frequency > 0.5:
                speech_pattern = "fragmented"
            elif len(break_segments) > 0 and total_break_time > total_voice_time * 0.3:
                speech_pattern = "hesitant"
            
            return {
                'voice_segments': voice_segments,
                'break_segments': break_segments,
                'total_voice_time': float(total_voice_time),
                'total_break_time': float(total_break_time),
                'voice_continuity_ratio': float(voice_continuity_ratio),
                'break_frequency': float(break_frequency),
                'num_breaks': len(break_segments),
                'speech_pattern': speech_pattern,
                'avg_break_duration': float(np.mean([seg['duration'] for seg in break_segments])) if break_segments else 0.0,
                'longest_break': float(max([seg['duration'] for seg in break_segments])) if break_segments else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error detecting voice breaks: {str(e)}")
            return {
                'voice_segments': [],
                'break_segments': [],
                'total_voice_time': 0.0,
                'total_break_time': 0.0,
                'voice_continuity_ratio': 0.0,
                'break_frequency': 0.0,
                'num_breaks': 0,
                'speech_pattern': 'unknown',
                'avg_break_duration': 0.0,
                'longest_break': 0.0
            }
    
    def extract_gender_invariant_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features that are robust to gender differences"""
        try:
            # 1. MFCC features (first 12 coefficients, excluding C0 which is energy-related)
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )[1:13]  # Skip first coefficient
            
            # 2. Delta MFCCs (temporal changes)
            delta_mfccs = librosa.feature.delta(mfccs)
            
            # 3. Spectral features that are gender-invariant
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, 
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, 
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # 4. Chroma features (pitch class profiles)
            chroma = librosa.feature.chroma_stft(
                y=audio, 
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # 5. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio, 
                hop_length=self.hop_length
            )
            
            # 6. Formant-like features using spectral peaks
            # Get magnitude spectrum
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            
            # Find spectral peaks that correspond to formants
            spectral_peaks = []
            for frame in magnitude.T:
                # Find peaks in the spectrum
                peaks = self._find_spectral_peaks(frame)
                spectral_peaks.append(peaks)
            
            spectral_peaks = np.array(spectral_peaks).T
            
            return {
                'mfcc': mfccs,
                'delta_mfcc': delta_mfccs,
                'spectral_centroid': spectral_centroids,
                'spectral_rolloff': spectral_rolloff,
                'chroma': chroma,
                'zcr': zcr,
                'spectral_peaks': spectral_peaks
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")
    
    def _find_spectral_peaks(self, spectrum: np.ndarray, n_peaks: int = 4) -> np.ndarray:
        """Find spectral peaks that may correspond to formants"""
        try:
            # Smooth the spectrum
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(spectrum, sigma=2)
            
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(smoothed, height=np.max(smoothed) * 0.1)
            
            # Get the strongest peaks
            if len(peaks) > n_peaks:
                peak_heights = smoothed[peaks]
                top_peaks_idx = np.argsort(peak_heights)[-n_peaks:]
                peaks = peaks[top_peaks_idx]
            
            # Pad if we don't have enough peaksasd
            if len(peaks) < n_peaks:
                peaks = np.pad(peaks, (0, n_peaks - len(peaks)), 'constant', constant_values=0)
            
            return peaks[:n_peaks]
        except:
            return np.zeros(n_peaks)
    
    def calculate_similarity(self, features1: Dict[str, np.ndarray], features2: Dict[str, np.ndarray]) -> float:
        """Calculate similarity between two feature sets"""
        try:
            similarities = []
            
            # Compare each feature type
            for feature_name in features1.keys():
                if feature_name in features2:
                    feat1 = features1[feature_name]
                    feat2 = features2[feature_name]
                    
                    # Calculate mean and std for each feature
                    feat1_stats = np.concatenate([np.mean(feat1, axis=1), np.std(feat1, axis=1)])
                    feat2_stats = np.concatenate([np.mean(feat2, axis=1), np.std(feat2, axis=1)])
                    
                    # Handle NaN values
                    feat1_stats = np.nan_to_num(feat1_stats)
                    feat2_stats = np.nan_to_num(feat2_stats)
                    
                    # Calculate cosine similarity
                    if np.linalg.norm(feat1_stats) > 0 and np.linalg.norm(feat2_stats) > 0:
                        similarity = 1 - cosine(feat1_stats, feat2_stats)
                        similarities.append(similarity)
            
            if not similarities:
                return 0.0
            
            # Return weighted average (you can adjust weights based on importance)
            weights = {
                'mfcc': 0.25,
                'delta_mfcc': 0.20,
                'spectral_centroid': 0.15,
                'spectral_rolloff': 0.10,
                'chroma': 0.15,
                'zcr': 0.05,
                'spectral_peaks': 0.10
            }
            
            weighted_similarity = 0
            total_weight = 0
            
            for i, feature_name in enumerate(features1.keys()):
                if feature_name in weights and i < len(similarities):
                    weighted_similarity += similarities[i] * weights[feature_name]
                    total_weight += weights[feature_name]
            
            if total_weight > 0:
                return max(0, min(1, weighted_similarity / total_weight))
            else:
                return np.mean(similarities)
                
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

# Initialize processor
processor = AudioProcessor()

@app.post("/compare-audio/")
async def compare_audio(
    reference_audio: UploadFile = File(..., description="Reference audio file (WAV only)"),
    user_audio: UploadFile = File(..., description="User recorded audio file (WAV only, 5 seconds)")
):
    """
    Compare two WAV audio files for vowel sound similarity and detect voice breaks.
    """
    try:
        # Validate file types - only WAV files allowed
        allowed_types = [
            'audio/wav', 'audio/wave', 'audio/x-wav'
        ]
        
        # Check file extensions
        def is_wav_file(filename: str) -> bool:
            if not filename:
                return False
            ext = filename.lower().split('.')[-1]
            return ext == 'wav'
        
        # Validate reference audio
        if (reference_audio.content_type not in allowed_types and 
            not is_wav_file(reference_audio.filename)):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid reference audio format. Only WAV files are supported. Got: {reference_audio.content_type}"
            )
        
        # Validate user audio  
        if (user_audio.content_type not in allowed_types and 
            not is_wav_file(user_audio.filename)):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid user audio format. Only WAV files are supported. Got: {user_audio.content_type}"
            )
        
        # Read audio files
        logger.info(f"Processing reference audio: {reference_audio.filename} ({reference_audio.content_type})")
        logger.info(f"Processing user audio: {user_audio.filename} ({user_audio.content_type})")
        
        ref_audio_bytes = await reference_audio.read()
        user_audio_bytes = await user_audio.read()
        
        logger.info(f"Reference audio size: {len(ref_audio_bytes)} bytes")
        logger.info(f"User audio size: {len(user_audio_bytes)} bytes")
        
        # Load audio data with individual error handling
        try:
            ref_audio, ref_sr = processor.load_audio(ref_audio_bytes)
            logger.info(f"Successfully loaded reference audio")
        except Exception as e:
            logger.error(f"Failed to load reference audio: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to load reference WAV audio: {str(e)}")
        
        try:
            user_audio, user_sr = processor.load_audio(user_audio_bytes)
            logger.info(f"Successfully loaded user audio")
        except Exception as e:
            logger.error(f"Failed to load user audio: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to load user WAV audio: {str(e)}")
        
        # Trim silence from both audio files
        ref_audio_trimmed = processor.trim_silence(ref_audio)
        user_audio_trimmed = processor.trim_silence(user_audio)
        
        # Log audio durations
        logger.info(f"Reference audio duration: {len(ref_audio_trimmed) / processor.sample_rate:.2f}s")
        logger.info(f"User audio duration: {len(user_audio_trimmed) / processor.sample_rate:.2f}s")
        
        # Detect voice breaks in user audio
        user_voice_breaks = processor.detect_voice_breaks(user_audio_trimmed)
        
        # Extract gender-invariant features
        ref_features = processor.extract_gender_invariant_features(ref_audio_trimmed)
        user_features = processor.extract_gender_invariant_features(user_audio_trimmed)
        
        # Calculate similarity
        similarity_score = processor.calculate_similarity(ref_features, user_features)
        
        return JSONResponse(content={
            "similarity_score": float(similarity_score),
            "reference_duration": float(len(ref_audio_trimmed) / processor.sample_rate),
            "user_duration": float(len(user_audio_trimmed) / processor.sample_rate),
            "voice_break_analysis": {
                # "speech_pattern": user_voice_breaks['speech_pattern'],
                "voice_continuity_ratio": user_voice_breaks['voice_continuity_ratio'],
                "total_breaks": user_voice_breaks['num_breaks'],
                "total_voice_time": user_voice_breaks['total_voice_time'],
                "total_break_time": user_voice_breaks['total_break_time'],
                # "break_frequency": user_voice_breaks['break_frequency'],
                # "average_break_duration": user_voice_breaks['avg_break_duration'],
                # "longest_break_duration": user_voice_breaks['longest_break'],
                # "break_details": user_voice_breaks['break_segments']
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Audio Similarity API is running", "version": "1.0.0", "supported_formats": ["WAV"]}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "supported_formats": ["WAV only"],
        "features": [
            "Audio similarity comparison (WAV files)",
            "Voice break detection",
            "Speech pattern analysis",
            "Silence trimming",
            "Gender-invariant feature extraction",
            "Noise-robust analysis"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)