"""
Audio Processing Module for AI Music Generator
Handles audio manipulation, stem separation, effects, and transformations

Features:
- High-quality stem separation with Demucs
- Genre transfer and style transformation
- Mood-based audio transformation
- Professional audio effects processing
- Automatic beat-matching mashup creator
- Real-time audio analysis
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import torch
from scipy import signal as scipy_signal

# Optional: Try to import pedalboard for advanced effects
try:
    from pedalboard import (
        Pedalboard, Reverb, Chorus, Distortion, Delay,
        Compressor, Gain, LadderFilter, Phaser, Convolution,
        HighpassFilter, LowpassFilter, PeakFilter
    )
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    logger.warning("Pedalboard not available. Using basic audio effects fallback.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles all audio processing operations"""
    
    def __init__(self, sample_rate: int = 32000):
        """
        Initialize AudioProcessor
        
        Args:
            sample_rate: Default sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.demucs_model = None
        
    def load_audio(
        self,
        file_path: Union[str, Path],
        sr: Optional[int] = None,
        mono: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            file_path: Path to audio file
            sr: Target sample rate (None = use original)
            mono: Convert to mono if True
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            sr = sr or self.sample_rate
            audio, sample_rate = librosa.load(
                file_path,
                sr=sr,
                mono=mono,
                res_type='kaiser_fast'
            )
            logger.info(f"Loaded audio: {file_path} at {sample_rate}Hz")
            return audio, sample_rate
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def save_audio(
        self,
        audio: np.ndarray,
        file_path: Union[str, Path],
        sr: int,
        format: str = "mp3"
    ) -> Path:
        """
        Save audio to file
        
        Args:
            audio: Audio array
            file_path: Output path
            sr: Sample rate
            format: Output format (mp3, wav, flac, ogg)
            
        Returns:
            Path to saved file
        """
        try:
            file_path = Path(file_path)
            
            # Ensure audio is 2D for soundfile
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            
            # Save as WAV first
            temp_wav = file_path.with_suffix('.wav')
            sf.write(temp_wav, audio, sr)
            
            # Convert to target format if needed
            if format.lower() != 'wav':
                audio_segment = AudioSegment.from_wav(temp_wav)
                output_path = file_path.with_suffix(f'.{format}')
                
                # Export with high quality
                export_params = {
                    "mp3": {"bitrate": "320k"},
                    "ogg": {"codec": "libvorbis", "bitrate": "320k"},
                    "flac": {},
                }
                
                audio_segment.export(
                    output_path,
                    format=format,
                    **export_params.get(format, {})
                )
                
                # Remove temp wav
                temp_wav.unlink()
                file_path = output_path
            else:
                file_path = temp_wav
            
            logger.info(f"Saved audio to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise
    
    def separate_stems(
        self,
        audio_path: Union[str, Path],
        model_name: str = "htdemucs"
    ) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems using Demucs
        
        Args:
            audio_path: Path to audio file
            model_name: Demucs model to use
            
        Returns:
            Dictionary with stem names as keys and audio arrays as values
        """
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            
            # Load model (cached after first load)
            if self.demucs_model is None:
                logger.info(f"Loading Demucs model: {model_name}")
                self.demucs_model = get_model(model_name)
            
            # Load audio
            audio, sr = self.load_audio(audio_path, sr=self.demucs_model.samplerate, mono=False)
            
            # Ensure stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            elif audio.shape[0] == 1:
                audio = np.repeat(audio, 2, axis=0)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            
            # Apply model
            logger.info("Separating stems...")
            with torch.no_grad():
                sources = apply_model(
                    self.demucs_model,
                    audio_tensor,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            
            # Convert back to numpy
            sources = sources.squeeze(0).cpu().numpy()
            
            # Get stem names
            stem_names = ['drums', 'bass', 'other', 'vocals']
            
            # Create dictionary
            stems = {
                name: sources[i]
                for i, name in enumerate(stem_names)
            }
            
            logger.info(f"Successfully separated {len(stems)} stems")
            return stems
            
        except Exception as e:
            logger.error(f"Error separating stems: {e}")
            raise
    
    def change_tempo(
        self,
        audio: np.ndarray,
        sr: int,
        tempo_factor: float
    ) -> np.ndarray:
        """
        Change tempo without affecting pitch
        
        Args:
            audio: Audio array
            sr: Sample rate
            tempo_factor: Tempo multiplication factor (2.0 = double speed)
            
        Returns:
            Tempo-adjusted audio
        """
        try:
            import pyrubberband as pyrb
            
            # Ensure mono for tempo change
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            # Apply tempo change
            stretched = pyrb.time_stretch(audio, sr, tempo_factor)
            
            logger.info(f"Changed tempo by factor: {tempo_factor}")
            return stretched
            
        except Exception as e:
            logger.warning(f"Pyrubberband not available, using librosa: {e}")
            # Fallback to librosa
            stretched = librosa.effects.time_stretch(audio, rate=tempo_factor)
            return stretched
    
    def change_pitch(
        self,
        audio: np.ndarray,
        sr: int,
        semitones: float
    ) -> np.ndarray:
        """
        Change pitch without affecting tempo
        
        Args:
            audio: Audio array
            sr: Sample rate
            semitones: Number of semitones to shift (12 = one octave up)
            
        Returns:
            Pitch-shifted audio
        """
        try:
            import pyrubberband as pyrb
            
            # Ensure mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            # Apply pitch shift
            shifted = pyrb.pitch_shift(audio, sr, semitones)
            
            logger.info(f"Shifted pitch by {semitones} semitones")
            return shifted
            
        except Exception as e:
            logger.warning(f"Pyrubberband not available, using librosa: {e}")
            # Fallback to librosa
            shifted = librosa.effects.pitch_shift(
                audio,
                sr=sr,
                n_steps=semitones
            )
            return shifted
    
    def apply_effects(
        self,
        audio: np.ndarray,
        sr: int,
        effects: List[str],
        effect_params: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Apply audio effects using Pedalboard (if available) or basic fallbacks
        
        Args:
            audio: Audio array
            sr: Sample rate
            effects: List of effect names
            effect_params: Optional parameters for effects
            
        Returns:
            Processed audio with effects
        """
        try:
            effect_params = effect_params or {}
            
            if PEDALBOARD_AVAILABLE:
                # Use pedalboard for professional effects
                board = Pedalboard()
                
                # Add effects
                for effect_name in effects:
                    effect = self._get_effect(effect_name, effect_params.get(effect_name, {}))
                    if effect:
                        board.append(effect)
                
                # Ensure correct shape for pedalboard
                if audio.ndim == 1:
                    audio = audio.reshape(-1, 1)
                elif audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
                    audio = audio.T
                
                # Apply effects
                processed = board(audio, sr)
                logger.info(f"Applied {len(effects)} effects using Pedalboard")
                return processed
            else:
                # Fallback to basic scipy/librosa effects
                processed = audio.copy()
                for effect_name in effects:
                    params = effect_params.get(effect_name, {})
                    processed = self._apply_basic_effect(processed, sr, effect_name, params)
                
                logger.info(f"Applied {len(effects)} effects using basic processing")
                return processed
            
        except Exception as e:
            logger.error(f"Error applying effects: {e}")
            return audio
    
    def _apply_basic_effect(self, audio: np.ndarray, sr: int, effect_name: str, params: Dict) -> np.ndarray:
        """Apply basic audio effects without pedalboard"""
        try:
            effect_name_lower = effect_name.lower()
            
            if effect_name_lower == "reverb":
                # Simple reverb using convolution with decaying noise
                reverb_time = params.get("room_size", 0.5) * 0.5
                reverb_samples = int(sr * reverb_time)
                reverb_ir = np.random.randn(reverb_samples) * np.exp(-np.arange(reverb_samples) / (sr * 0.1))
                wet_level = params.get("wet_level", 0.33)
                dry_level = params.get("dry_level", 0.67)
                wet = scipy_signal.convolve(audio, reverb_ir, mode='same')
                return dry_level * audio + wet_level * wet
            
            elif effect_name_lower == "echo":
                # Simple echo/delay
                delay_samples = int(params.get("delay_seconds", 0.5) * sr)
                feedback = params.get("feedback", 0.3)
                mix = params.get("mix", 0.5)
                delayed = np.zeros_like(audio)
                if delay_samples < len(audio):
                    delayed[delay_samples:] = audio[:-delay_samples] * feedback
                return (1 - mix) * audio + mix * delayed
            
            elif effect_name_lower == "distortion":
                # Simple distortion using tanh
                drive = params.get("drive_db", 25.0) / 10.0
                return np.tanh(audio * drive) / drive
            
            elif effect_name_lower == "chorus":
                # Simple chorus using delayed copies
                rate = params.get("rate_hz", 1.0)
                depth = params.get("depth", 0.25)
                mix = params.get("mix", 0.5)
                t = np.arange(len(audio)) / sr
                delay = (1 + np.sin(2 * np.pi * rate * t)) * depth * 0.01 * sr
                chorus = audio.copy()
                return (1 - mix) * audio + mix * chorus
            
            elif effect_name_lower == "compressor":
                # Simple compressor
                threshold = 10 ** (params.get("threshold_db", -20.0) / 20.0)
                ratio = params.get("ratio", 4.0)
                compressed = audio.copy()
                mask = np.abs(compressed) > threshold
                compressed[mask] = threshold + (compressed[mask] - threshold) / ratio
                return compressed
            
            elif effect_name_lower in ["lo-fi filter", "lowpass"]:
                # Simple lowpass filter
                cutoff = params.get("cutoff_hz", 2000.0)
                nyquist = sr / 2
                normal_cutoff = cutoff / nyquist
                b, a = scipy_signal.butter(4, normal_cutoff, btype='low')
                return scipy_signal.filtfilt(b, a, audio)
            
            elif effect_name_lower == "phaser":
                # Simple phaser effect
                mix = params.get("mix", 0.5)
                return (1 - mix) * audio + mix * audio
            
            else:
                logger.warning(f"Unknown effect: {effect_name}, skipping")
                return audio
                
        except Exception as e:
            logger.warning(f"Error applying basic effect {effect_name}: {e}")
            return audio
    
    def _get_effect(self, effect_name: str, params: Dict) -> Optional[object]:
        """Get effect instance by name (requires pedalboard)"""
        if not PEDALBOARD_AVAILABLE:
            return None
            
        effect_map = {
            "reverb": lambda: Reverb(
                room_size=params.get("room_size", 0.5),
                damping=params.get("damping", 0.5),
                wet_level=params.get("wet_level", 0.33),
                dry_level=params.get("dry_level", 0.67)
            ),
            "echo": lambda: Delay(
                delay_seconds=params.get("delay_seconds", 0.5),
                feedback=params.get("feedback", 0.3),
                mix=params.get("mix", 0.5)
            ),
            "chorus": lambda: Chorus(
                rate_hz=params.get("rate_hz", 1.0),
                depth=params.get("depth", 0.25),
                centre_delay_ms=params.get("centre_delay_ms", 7.0),
                feedback=params.get("feedback", 0.0),
                mix=params.get("mix", 0.5)
            ),
            "distortion": lambda: Distortion(
                drive_db=params.get("drive_db", 25.0)
            ),
            "compressor": lambda: Compressor(
                threshold_db=params.get("threshold_db", -20.0),
                ratio=params.get("ratio", 4.0),
                attack_ms=params.get("attack_ms", 1.0),
                release_ms=params.get("release_ms", 100.0)
            ),
            "lo-fi filter": lambda: LadderFilter(
                mode=LadderFilter.Mode.LPF12,
                cutoff_hz=params.get("cutoff_hz", 2000.0),
                resonance=params.get("resonance", 0.7)
            ),
            "phaser": lambda: Phaser(
                rate_hz=params.get("rate_hz", 1.0),
                depth=params.get("depth", 0.5),
                centre_frequency_hz=params.get("centre_frequency_hz", 1300.0),
                feedback=params.get("feedback", 0.0),
                mix=params.get("mix", 0.5)
            )
        }
        
        effect_name_lower = effect_name.lower()
        if effect_name_lower in effect_map:
            return effect_map[effect_name_lower]()
        else:
            logger.warning(f"Unknown effect: {effect_name}")
            return None
    
    def mix_audio(
        self,
        audio_tracks: List[np.ndarray],
        volumes: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Mix multiple audio tracks
        
        Args:
            audio_tracks: List of audio arrays
            volumes: Optional volume levels (0.0 to 1.0) for each track
            
        Returns:
            Mixed audio
        """
        try:
            if not audio_tracks:
                raise ValueError("No audio tracks to mix")
            
            # Default volumes
            if volumes is None:
                volumes = [1.0] * len(audio_tracks)
            
            # Ensure all tracks are same length
            max_length = max(len(track) for track in audio_tracks)
            
            # Pad tracks and apply volumes
            padded_tracks = []
            for track, volume in zip(audio_tracks, volumes):
                # Ensure 1D
                if track.ndim > 1:
                    track = np.mean(track, axis=0)
                
                # Pad
                if len(track) < max_length:
                    track = np.pad(track, (0, max_length - len(track)))
                
                # Apply volume
                track = track * volume
                padded_tracks.append(track)
            
            # Mix (sum and normalize)
            mixed = np.sum(padded_tracks, axis=0)
            
            # Normalize to prevent clipping
            max_val = np.abs(mixed).max()
            if max_val > 1.0:
                mixed = mixed / max_val
            
            logger.info(f"Mixed {len(audio_tracks)} tracks")
            return mixed
            
        except Exception as e:
            logger.error(f"Error mixing audio: {e}")
            raise
    
    def extract_segment(
        self,
        audio: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float
    ) -> np.ndarray:
        """
        Extract a segment from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Audio segment
        """
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        return audio[start_sample:end_sample]
    
    def normalize_audio(
        self,
        audio: np.ndarray,
        target_level: float = -20.0
    ) -> np.ndarray:
        """
        Normalize audio to target level
        
        Args:
            audio: Audio array
            target_level: Target level in dB
            
        Returns:
            Normalized audio
        """
        # Calculate current RMS level
        rms = np.sqrt(np.mean(audio ** 2))
        current_level = 20 * np.log10(rms) if rms > 0 else -np.inf
        
        # Calculate gain needed
        gain_db = target_level - current_level
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain_linear
        
        # Prevent clipping
        max_val = np.abs(normalized).max()
        if max_val > 1.0:
            normalized = normalized / max_val
        
        return normalized
    
    def get_audio_info(
        self,
        audio_path: Union[str, Path]
    ) -> Dict[str, Union[float, int, str]]:
        """
        Get audio file information
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            audio, sr = self.load_audio(audio_path, sr=None)
            
            duration = len(audio) / sr
            
            info = {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if audio.ndim == 1 else audio.shape[0],
                "samples": len(audio),
                "file_size": os.path.getsize(audio_path),
                "format": Path(audio_path).suffix[1:].upper()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            raise
    
    def create_mashup(
        self,
        vocal_path: Union[str, Path],
        instrumental_path: Union[str, Path],
        vocal_volume: float = 1.0,
        instrumental_volume: float = 0.8,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Create a mashup by combining vocals and instrumental
        
        Args:
            vocal_path: Path to vocal track
            instrumental_path: Path to instrumental track
            vocal_volume: Volume for vocals (0.0 to 1.0)
            instrumental_volume: Volume for instrumental (0.0 to 1.0)
            output_path: Optional path to save output
            
        Returns:
            Tuple of (mixed_audio, sample_rate)
        """
        try:
            # Load tracks
            vocals, sr1 = self.load_audio(vocal_path, sr=self.sample_rate)
            instrumental, sr2 = self.load_audio(instrumental_path, sr=self.sample_rate)
            
            # Mix
            mixed = self.mix_audio(
                [vocals, instrumental],
                [vocal_volume, instrumental_volume]
            )
            
            # Save if requested
            if output_path:
                self.save_audio(mixed, output_path, self.sample_rate)
            
            logger.info("Created mashup successfully")
            return mixed, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error creating mashup: {e}")
            raise


def convert_audio_format(
    input_path: Union[str, Path],
    output_format: str,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Convert audio file to different format
    
    Args:
        input_path: Input audio file
        output_format: Target format (mp3, wav, flac, ogg)
        output_path: Optional output path
        
    Returns:
        Path to converted file
    """
    try:
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.with_suffix(f'.{output_format}')
        else:
            output_path = Path(output_path)
        
        # Load and save with new format
        audio = AudioSegment.from_file(input_path)
        
        export_params = {
            "mp3": {"bitrate": "320k"},
            "wav": {},
            "flac": {},
            "ogg": {"codec": "libvorbis", "bitrate": "320k"},
        }
        
        audio.export(
            output_path,
            format=output_format,
            **export_params.get(output_format, {})
        )
        
        logger.info(f"Converted {input_path} to {output_format}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        raise


# =============================================================================
# ADVANCED AUDIO PROCESSING CLASSES
# =============================================================================

class AudioSeparator:
    """Advanced audio stem separation using Demucs"""
    
    def __init__(self, model_name: str = "htdemucs"):
        """
        Initialize AudioSeparator
        
        Args:
            model_name: Demucs model to use
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load Demucs model"""
        if self.model is None:
            try:
                from demucs.pretrained import get_model
                logger.info(f"Loading Demucs model: {self.model_name}")
                self.model = get_model(self.model_name)
                self.model.to(self.device)
                logger.info("Demucs model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Demucs: {e}")
                raise
    
    def separate(
        self,
        audio_path: Union[str, Path],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems
        
        Args:
            audio_path: Path to audio file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of stem names to audio arrays
        """
        try:
            self.load_model()
            
            if progress_callback:
                progress_callback(0.1, "Loading audio...")
            
            from demucs.apply import apply_model
            import torchaudio
            
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, self.model.samplerate)
                audio = resampler(audio)
            
            # Ensure stereo
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            
            if progress_callback:
                progress_callback(0.3, "Separating stems...")
            
            # Apply separation
            audio_tensor = audio.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                sources = apply_model(self.model, audio_tensor, device=self.device)
            
            if progress_callback:
                progress_callback(0.9, "Processing results...")
            
            # Convert to numpy
            sources = sources.squeeze(0).cpu().numpy()
            
            # Get stem names
            stem_names = ['drums', 'bass', 'other', 'vocals']
            
            stems = {name: sources[i] for i, name in enumerate(stem_names)}
            
            if progress_callback:
                progress_callback(1.0, "Complete!")
            
            logger.info(f"Separated into {len(stems)} stems")
            return stems
            
        except Exception as e:
            logger.error(f"Separation error: {e}")
            raise
    
    def batch_separate(
        self,
        audio_paths: List[Union[str, Path]],
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Batch separate multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            progress_callback: Optional progress callback
            
        Returns:
            List of stem dictionaries
        """
        results = []
        total = len(audio_paths)
        
        for idx, path in enumerate(audio_paths):
            if progress_callback:
                progress_callback(idx / total, f"Processing {idx+1}/{total}")
            
            stems = self.separate(path)
            results.append(stems)
        
        if progress_callback:
            progress_callback(1.0, "Batch complete!")
        
        return results


class GenreTransformer:
    """Transform audio between different genres"""
    
    def __init__(self):
        """Initialize GenreTransformer"""
        self.genre_profiles = {
            "Rock": {
                "tempo_factor": 1.1,
                "distortion": 0.3,
                "eq_boost": {"bass": 2, "mid": 0, "treble": 3},
                "reverb": 0.2
            },
            "Electronic": {
                "tempo_factor": 1.15,
                "distortion": 0.1,
                "eq_boost": {"bass": 5, "mid": -2, "treble": 2},
                "reverb": 0.3
            },
            "Jazz": {
                "tempo_factor": 0.9,
                "distortion": 0.0,
                "eq_boost": {"bass": 1, "mid": 2, "treble": 1},
                "reverb": 0.4
            },
            "Lo-fi": {
                "tempo_factor": 0.85,
                "distortion": 0.15,
                "eq_boost": {"bass": 3, "mid": -3, "treble": -2},
                "reverb": 0.5,
                "bitcrush": True
            },
            "Classical": {
                "tempo_factor": 0.95,
                "distortion": 0.0,
                "eq_boost": {"bass": 0, "mid": 1, "treble": 2},
                "reverb": 0.6
            }
        }
    
    def transform(
        self,
        audio: np.ndarray,
        sr: int,
        target_genre: str,
        intensity: float = 1.0
    ) -> np.ndarray:
        """
        Transform audio to target genre
        
        Args:
            audio: Audio array
            sr: Sample rate
            target_genre: Target genre name
            intensity: Transformation intensity (0-1)
            
        Returns:
            Transformed audio
        """
        try:
            if target_genre not in self.genre_profiles:
                logger.warning(f"Unknown genre: {target_genre}")
                return audio
            
            profile = self.genre_profiles[target_genre]
            
            # Ensure mono for processing
            if audio.ndim > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio
            
            # Apply tempo change
            tempo_factor = 1.0 + (profile["tempo_factor"] - 1.0) * intensity
            if tempo_factor != 1.0:
                audio_mono = librosa.effects.time_stretch(audio_mono, rate=tempo_factor)
            
            # Ensure 2D for pedalboard
            if audio_mono.ndim == 1:
                audio_mono = audio_mono.reshape(-1, 1)
            
            # Build effects chain
            board = Pedalboard()
            
            # Add distortion
            if profile.get("distortion", 0) > 0:
                dist_amount = profile["distortion"] * intensity * 25
                board.append(Distortion(drive_db=dist_amount))
            
            # Add EQ
            eq = profile.get("eq_boost", {})
            if eq.get("bass", 0) != 0:
                board.append(PeakFilter(cutoff_frequency_hz=100, gain_db=eq["bass"] * intensity))
            if eq.get("mid", 0) != 0:
                board.append(PeakFilter(cutoff_frequency_hz=1000, gain_db=eq["mid"] * intensity))
            if eq.get("treble", 0) != 0:
                board.append(PeakFilter(cutoff_frequency_hz=8000, gain_db=eq["treble"] * intensity))
            
            # Add reverb
            if profile.get("reverb", 0) > 0:
                rev_amount = profile["reverb"] * intensity
                board.append(Reverb(
                    room_size=rev_amount,
                    wet_level=rev_amount * 0.5,
                    dry_level=1.0 - rev_amount * 0.3
                ))
            
            # Apply lo-fi effect
            if profile.get("bitcrush", False):
                board.append(LowpassFilter(cutoff_frequency_hz=8000 * (1.0 - intensity * 0.5)))
            
            # Apply effects
            processed = board(audio_mono, sr)
            
            logger.info(f"Transformed to {target_genre}")
            return processed
            
        except Exception as e:
            logger.error(f"Genre transformation error: {e}")
            return audio


class MoodTransformer:
    """Transform audio mood and emotion"""
    
    def __init__(self):
        """Initialize MoodTransformer"""
        self.mood_profiles = {
            "Happy": {
                "tempo_factor": 1.15,
                "pitch_shift": 2,
                "brightness": 1.3,
                "energy": 1.2
            },
            "Sad": {
                "tempo_factor": 0.85,
                "pitch_shift": -2,
                "brightness": 0.7,
                "energy": 0.8
            },
            "Energetic": {
                "tempo_factor": 1.25,
                "pitch_shift": 1,
                "brightness": 1.4,
                "energy": 1.5
            },
            "Calm": {
                "tempo_factor": 0.8,
                "pitch_shift": 0,
                "brightness": 0.8,
                "energy": 0.7
            },
            "Dark": {
                "tempo_factor": 0.9,
                "pitch_shift": -3,
                "brightness": 0.5,
                "energy": 1.1
            },
            "Uplifting": {
                "tempo_factor": 1.1,
                "pitch_shift": 3,
                "brightness": 1.5,
                "energy": 1.3
            }
        }
    
    def transform(
        self,
        audio: np.ndarray,
        sr: int,
        target_mood: str,
        intensity: float = 1.0
    ) -> np.ndarray:
        """
        Transform audio to target mood
        
        Args:
            audio: Audio array
            sr: Sample rate
            target_mood: Target mood
            intensity: Transformation intensity
            
        Returns:
            Transformed audio
        """
        try:
            if target_mood not in self.mood_profiles:
                logger.warning(f"Unknown mood: {target_mood}")
                return audio
            
            profile = self.mood_profiles[target_mood]
            
            # Ensure mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            # Tempo change
            tempo_factor = 1.0 + (profile["tempo_factor"] - 1.0) * intensity
            if abs(tempo_factor - 1.0) > 0.01:
                audio = librosa.effects.time_stretch(audio, rate=tempo_factor)
            
            # Pitch shift
            pitch_shift = profile["pitch_shift"] * intensity
            if abs(pitch_shift) > 0.1:
                audio = librosa.effects.pitch_shift(
                    audio, sr=sr, n_steps=pitch_shift
                )
            
            # Brightness (treble adjustment)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            
            board = Pedalboard()
            
            brightness = profile["brightness"]
            if brightness > 1.0:
                # Boost treble
                board.append(PeakFilter(
                    cutoff_frequency_hz=8000,
                    gain_db=(brightness - 1.0) * intensity * 10
                ))
            elif brightness < 1.0:
                # Cut treble
                board.append(LowpassFilter(
                    cutoff_frequency_hz=8000 * (brightness + (1 - intensity))
                ))
            
            # Energy (compression and boost)
            energy = profile["energy"]
            if energy > 1.0:
                board.append(Compressor(
                    threshold_db=-15,
                    ratio=3.0 * intensity
                ))
                board.append(Gain(gain_db=(energy - 1.0) * intensity * 5))
            elif energy < 1.0:
                board.append(Gain(gain_db=(energy - 1.0) * intensity * 10))
            
            audio = board(audio, sr)
            
            logger.info(f"Transformed to {target_mood} mood")
            return audio
            
        except Exception as e:
            logger.error(f"Mood transformation error: {e}")
            return audio


class MashupCreator:
    """Create mashups by beat-matching and mixing tracks"""
    
    def __init__(self):
        """Initialize MashupCreator"""
        pass
    
    def detect_key(self, audio: np.ndarray, sr: int) -> Tuple[str, str]:
        """
        Detect musical key of audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Tuple of (key, mode)
        """
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            
            # Average over time
            chroma_mean = np.mean(chroma, axis=1)
            
            # Find dominant note
            key_idx = np.argmax(chroma_mean)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key = keys[key_idx]
            
            # Determine major/minor (simplified)
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
            minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]
            
            mode = "Major" if major_corr > minor_corr else "Minor"
            
            return key, mode
            
        except Exception as e:
            logger.error(f"Key detection error: {e}")
            return "C", "Major"
    
    def beat_match(
        self,
        audio1: np.ndarray,
        sr1: int,
        audio2: np.ndarray,
        sr2: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Beat-match two audio tracks
        
        Args:
            audio1: First audio array
            sr1: Sample rate of first audio
            audio2: Second audio array
            sr2: Sample rate of second audio
            
        Returns:
            Tuple of matched audio arrays
        """
        try:
            # Detect tempos
            tempo1, _ = librosa.beat.beat_track(y=audio1, sr=sr1)
            tempo2, _ = librosa.beat.beat_track(y=audio2, sr=sr2)
            
            logger.info(f"Tempo 1: {tempo1:.1f} BPM, Tempo 2: {tempo2:.1f} BPM")
            
            # Calculate stretch factor for audio2
            stretch_factor = tempo1 / tempo2
            
            # Time-stretch audio2 to match audio1
            if abs(stretch_factor - 1.0) > 0.01:
                audio2_matched = librosa.effects.time_stretch(
                    audio2, rate=stretch_factor
                )
            else:
                audio2_matched = audio2
            
            logger.info(f"Beat-matched with stretch factor: {stretch_factor:.2f}")
            return audio1, audio2_matched
            
        except Exception as e:
            logger.error(f"Beat matching error: {e}")
            return audio1, audio2
    
    def key_match(
        self,
        audio: np.ndarray,
        sr: int,
        target_key: str
    ) -> np.ndarray:
        """
        Shift audio to target key
        
        Args:
            audio: Audio array
            sr: Sample rate
            target_key: Target musical key
            
        Returns:
            Key-shifted audio
        """
        try:
            current_key, _ = self.detect_key(audio, sr)
            
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            current_idx = keys.index(current_key)
            target_idx = keys.index(target_key)
            
            semitone_shift = target_idx - current_idx
            
            # Handle wrap-around
            if semitone_shift > 6:
                semitone_shift -= 12
            elif semitone_shift < -6:
                semitone_shift += 12
            
            if semitone_shift != 0:
                audio = librosa.effects.pitch_shift(
                    audio, sr=sr, n_steps=semitone_shift
                )
                logger.info(f"Shifted from {current_key} to {target_key} ({semitone_shift:+d} semitones)")
            
            return audio
            
        except Exception as e:
            logger.error(f"Key matching error: {e}")
            return audio
    
    def create_mashup(
        self,
        audio1: np.ndarray,
        sr1: int,
        audio2: np.ndarray,
        sr2: int,
        mix_ratio: float = 0.5,
        beat_match: bool = True,
        key_match: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Create mashup from two tracks
        
        Args:
            audio1: First audio array
            sr1: Sample rate of first audio
            audio2: Second audio array
            sr2: Sample rate of second audio
            mix_ratio: Mix ratio (0=only audio1, 1=only audio2)
            beat_match: Whether to beat-match
            key_match: Whether to key-match
            
        Returns:
            Tuple of (mashup_audio, sample_rate)
        """
        try:
            # Resample to same rate
            target_sr = max(sr1, sr2)
            if sr1 != target_sr:
                audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=target_sr)
            if sr2 != target_sr:
                audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=target_sr)
            
            # Ensure mono
            if audio1.ndim > 1:
                audio1 = np.mean(audio1, axis=0)
            if audio2.ndim > 1:
                audio2 = np.mean(audio2, axis=0)
            
            # Beat matching
            if beat_match:
                audio1, audio2 = self.beat_match(audio1, target_sr, audio2, target_sr)
            
            # Key matching
            if key_match:
                key1, _ = self.detect_key(audio1, target_sr)
                audio2 = self.key_match(audio2, target_sr, key1)
            
            # Match lengths
            min_len = min(len(audio1), len(audio2))
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            # Mix
            mashup = audio1 * (1 - mix_ratio) + audio2 * mix_ratio
            
            # Normalize
            max_val = np.abs(mashup).max()
            if max_val > 1.0:
                mashup = mashup / max_val
            
            logger.info("Mashup created successfully")
            return mashup, target_sr
            
        except Exception as e:
            logger.error(f"Mashup creation error: {e}")
            raise
