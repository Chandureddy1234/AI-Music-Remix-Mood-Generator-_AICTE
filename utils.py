"""
Utility Functions for AI Music Generator
Includes LLM clients, file handling, audio analysis, and visualization
"""

import os
import time
import tempfile
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime
import shutil

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
import streamlit as st

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# LLM CLIENT IMPLEMENTATIONS
# =============================================================================

class BaseLLMClient:
    """Base class for LLM clients"""
    
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Generate text from prompt"""
        raise NotImplementedError


class GroqClient(BaseLLMClient):
    """Groq API client (FREE - 30 req/min)"""
    
    def __init__(self, api_key: str):
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self.model = config.LLM_MODELS["groq"]["default"]
            logger.info("Groq client initialized")
        except Exception as e:
            logger.error(f"Error initializing Groq: {e}")
            raise
    
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            raise


class TogetherClient(BaseLLMClient):
    """Together AI client (FREE credits)"""
    
    def __init__(self, api_key: str):
        try:
            from together import Together
            self.client = Together(api_key=api_key)
            self.model = config.LLM_MODELS["together"]["default"]
            logger.info("Together AI client initialized")
        except Exception as e:
            logger.error(f"Error initializing Together: {e}")
            raise
    
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Together generation error: {e}")
            raise


class HuggingFaceClient(BaseLLMClient):
    """Hugging Face Inference API client (FREE)"""
    
    def __init__(self, token: str):
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=token)
            self.model = config.LLM_MODELS["huggingface"]["default"]
            logger.info("HuggingFace client initialized")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace: {e}")
            raise
    
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        try:
            prompt = user_prompt
            if system_prompt:
                prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            return response
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client (FREE, unlimited)"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            self.model = config.LLM_MODELS["ollama"]["default"]
            logger.info("Ollama client initialized")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            raise
    
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise


def get_llm_client() -> Optional[BaseLLMClient]:
    """
    Get available LLM client based on configuration
    
    Returns:
        Initialized LLM client or None
    """
    try:
        # Try Groq first (fastest)
        if config.GROQ_API_KEY:
            logger.info("Using Groq API")
            return GroqClient(config.GROQ_API_KEY)
        
        # Try Together AI
        if config.TOGETHER_API_KEY:
            logger.info("Using Together AI")
            return TogetherClient(config.TOGETHER_API_KEY)
        
        # Try Hugging Face
        if config.HUGGINGFACE_TOKEN:
            logger.info("Using Hugging Face")
            return HuggingFaceClient(config.HUGGINGFACE_TOKEN)
        
        # Try Ollama (local)
        if config.USE_OLLAMA:
            logger.info("Using Ollama (local)")
            return OllamaClient(config.OLLAMA_BASE_URL)
        
        logger.warning("No LLM client configured")
        return None
        
    except Exception as e:
        logger.error(f"Error initializing LLM client: {e}")
        return None


# =============================================================================
# AUDIO ANALYSIS
# =============================================================================

class AudioAnalyzer:
    """Analyze audio files for features and characteristics"""
    
    @staticmethod
    def analyze_audio(
        audio_path: Union[str, Path],
        sr: int = 22050
    ) -> Dict[str, Any]:
        """
        Comprehensive audio analysis
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sr)
            
            # Tempo and beat
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Estimate key
            chroma_vals = np.sum(chroma, axis=1)
            key_idx = np.argmax(chroma_vals)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            estimated_key = keys[key_idx]
            
            # Duration
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Energy and loudness
            energy = np.sum(rms ** 2)
            loudness = 20 * np.log10(np.mean(rms) + 1e-10)
            
            analysis = {
                "tempo": float(tempo),
                "key": estimated_key,
                "duration": float(duration),
                "energy": float(energy),
                "loudness": float(loudness),
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "zero_crossing_rate_mean": float(np.mean(zcr)),
                "rms_mean": float(np.mean(rms)),
                "beats": len(beats),
                "sample_rate": sr
            }
            
            # Infer features for music classification
            analysis["danceability"] = AudioAnalyzer._estimate_danceability(tempo, rms)
            analysis["valence"] = AudioAnalyzer._estimate_valence(chroma, tempo)
            analysis["instrumentalness"] = AudioAnalyzer._estimate_instrumentalness(zcr, spectral_centroids)
            
            logger.info(f"Audio analysis complete: {audio_path}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            raise
    
    @staticmethod
    def _estimate_danceability(tempo: float, rms: np.ndarray) -> float:
        """Estimate danceability (0-1)"""
        # Danceability correlates with tempo and energy
        tempo_score = min(1.0, max(0.0, (tempo - 60) / 120))
        energy_score = min(1.0, np.mean(rms) * 10)
        return (tempo_score + energy_score) / 2
    
    @staticmethod
    def _estimate_valence(chroma: np.ndarray, tempo: float) -> float:
        """Estimate valence/positivity (0-1)"""
        # Major chords and faster tempo suggest higher valence
        major_chords = np.sum(chroma[[0, 4, 7], :])  # C, E, G
        minor_chords = np.sum(chroma[[0, 3, 7], :])  # C, Eb, G
        
        chord_score = major_chords / (major_chords + minor_chords + 1e-10)
        tempo_score = min(1.0, tempo / 150)
        
        return (chord_score + tempo_score) / 2
    
    @staticmethod
    def _estimate_instrumentalness(zcr: np.ndarray, spectral_centroids: np.ndarray) -> float:
        """Estimate instrumentalness (0-1)"""
        # Higher for instrumental music
        zcr_var = np.var(zcr)
        spectral_var = np.var(spectral_centroids)
        
        # Voice has more variation
        instrumentalness = 1.0 - min(1.0, (zcr_var + spectral_var / 1000) / 2)
        return max(0.0, instrumentalness)
    
    @staticmethod
    def detect_mood(analysis: Dict[str, Any]) -> str:
        """
        Detect mood from analysis
        
        Args:
            analysis: Audio analysis dictionary
            
        Returns:
            Detected mood string
        """
        valence = analysis.get("valence", 0.5)
        energy = analysis.get("danceability", 0.5)
        tempo = analysis.get("tempo", 120)
        
        # Mood mapping based on valence and energy
        if valence > 0.6 and energy > 0.6:
            return "Happy"
        elif valence > 0.6 and energy < 0.4:
            return "Peaceful"
        elif valence < 0.4 and energy > 0.6:
            return "Energetic"
        elif valence < 0.4 and energy < 0.4:
            return "Sad"
        elif tempo > 140:
            return "Energetic"
        elif tempo < 80:
            return "Calm"
        else:
            return "Neutral"
    
    @staticmethod
    def detect_genre(analysis: Dict[str, Any]) -> str:
        """
        Detect genre from analysis (simplified)
        
        Args:
            analysis: Audio analysis dictionary
            
        Returns:
            Detected genre string
        """
        tempo = analysis.get("tempo", 120)
        instrumentalness = analysis.get("instrumentalness", 0.5)
        energy = analysis.get("energy", 0.5)
        
        # Simple genre classification
        if tempo > 140 and energy > 0.7:
            return "Electronic"
        elif tempo > 120 and tempo < 140 and energy > 0.6:
            return "Pop"
        elif tempo < 90 and instrumentalness > 0.7:
            return "Classical"
        elif tempo > 90 and tempo < 110:
            return "Hip-Hop"
        elif instrumentalness > 0.8:
            return "Ambient"
        else:
            return "General"


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_waveform(
    audio: np.ndarray,
    sr: int,
    title: str = "Waveform"
) -> go.Figure:
    """
    Create interactive waveform plot
    
    Args:
        audio: Audio array
        sr: Sample rate
        title: Plot title
        
    Returns:
        Plotly figure
    """
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio if audio.ndim == 1 else audio[0],
        mode='lines',
        name='Amplitude',
        line=dict(color='#00ff88', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def plot_spectrogram(
    audio: np.ndarray,
    sr: int,
    title: str = "Spectrogram"
) -> go.Figure:
    """
    Create interactive spectrogram
    
    Args:
        audio: Audio array
        sr: Sample rate
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=np.max
    )
    
    # Time and frequency axes
    times = librosa.times_like(D, sr=sr)
    freqs = librosa.fft_frequencies(sr=sr)
    
    fig = go.Figure(data=go.Heatmap(
        z=D,
        x=times,
        y=freqs,
        colorscale='Viridis',
        colorbar=dict(title="dB")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_yaxis(type="log", range=[np.log10(20), np.log10(sr/2)])
    
    return fig


def plot_audio_features(analysis: Dict[str, Any]) -> go.Figure:
    """
    Create radar chart of audio features
    
    Args:
        analysis: Audio analysis dictionary
        
    Returns:
        Plotly figure
    """
    features = ['Danceability', 'Energy', 'Valence', 'Instrumentalness']
    values = [
        analysis.get('danceability', 0.5),
        min(1.0, analysis.get('energy', 0.5) / 100),
        analysis.get('valence', 0.5),
        analysis.get('instrumentalness', 0.5)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],  # Close the loop
        theta=features + [features[0]],
        fill='toself',
        line=dict(color='#00ff88')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        template="plotly_dark",
        height=400,
        title="Audio Features"
    )
    
    return fig


# =============================================================================
# FILE HANDLING
# =============================================================================

def save_uploaded_file(uploaded_file) -> Path:
    """
    Save uploaded Streamlit file to temp directory
    
    Args:
        uploaded_file: Streamlit UploadedFile
        
    Returns:
        Path to saved file
    """
    try:
        # Create temp directory
        temp_dir = Path(tempfile.gettempdir()) / "ai_music_generator"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = Path(uploaded_file.name).suffix
        temp_path = temp_dir / f"{timestamp}_{uploaded_file.name}"
        
        # Save file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Saved uploaded file: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise


def cleanup_old_files(directory: Path, age_seconds: int = 3600):
    """
    Clean up old temporary files
    
    Args:
        directory: Directory to clean
        age_seconds: Files older than this will be deleted
    """
    try:
        if not directory.exists():
            return
        
        current_time = time.time()
        deleted_count = 0
        
        for file_path in directory.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > age_seconds:
                    file_path.unlink()
                    deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old files from {directory}")
            
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")


def get_file_hash(file_path: Path) -> str:
    """
    Get MD5 hash of file for caching
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state variables"""
    defaults = {
        "generation_history": [],
        "favorites": [],
        "current_project": None,
        "llm_client": None,
        "music_generator": None,
        "audio_processor": None,
        "first_visit": True
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_to_history(item: Dict[str, Any]):
    """Add item to generation history"""
    if "generation_history" not in st.session_state:
        st.session_state.generation_history = []
    
    st.session_state.generation_history.insert(0, item)
    
    # Keep only last 50
    st.session_state.generation_history = st.session_state.generation_history[:50]


def add_to_favorites(item: Dict[str, Any]):
    """Add item to favorites"""
    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    
    st.session_state.favorites.append(item)


# =============================================================================
# CACHING UTILITIES
# =============================================================================

@st.cache_resource
def get_cached_llm_client():
    """Get cached LLM client"""
    return get_llm_client()


@st.cache_resource
def get_cached_audio_analyzer():
    """Get cached audio analyzer"""
    return AudioAnalyzer()


# =============================================================================
# MISC UTILITIES
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in seconds to MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def estimate_generation_time(duration: float, model_size: str) -> float:
    """
    Estimate music generation time
    
    Args:
        duration: Audio duration in seconds
        model_size: Model size (small, medium, large)
        
    Returns:
        Estimated time in seconds
    """
    # Rough estimates (vary by hardware)
    time_multipliers = {
        "small": 0.5,
        "medium": 1.0,
        "large": 2.0,
        "melody": 1.5
    }
    
    base_time = duration * time_multipliers.get(model_size, 1.0)
    
    # Adjust for CPU vs GPU
    if not torch.cuda.is_available():
        base_time *= 3  # CPU is ~3x slower
    
    return base_time


def validate_audio_file(file_path: Path) -> bool:
    """
    Validate audio file
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check file exists
        if not file_path.exists():
            return False
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > config.MAX_UPLOAD_SIZE:
            return False
        
        # Check format
        ext = file_path.suffix[1:].lower()
        if ext not in config.ALLOWED_AUDIO_FORMATS:
            return False
        
        # Try to load
        librosa.load(file_path, sr=None, duration=1.0)
        
        return True
        
    except Exception as e:
        logger.error(f"Audio validation failed: {e}")
        return False


import torch
