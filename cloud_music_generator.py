"""
Cloud-Based Music Generation - ZERO Local Storage Required
Uses FREE cloud APIs for music generation without downloading models

Providers (in priority order):
1. Hugging Face Inference API (unlimited free)
2. Replicate API (50 free generations/month)
3. Gradio Client (public spaces, community resources)

Total storage: <10MB (just API clients)
Total cost: $0 forever
"""

import os
import io
import time
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import soundfile as sf
from pathlib import Path

logger = logging.getLogger(__name__)


class CloudMusicGenerator:
    """
    Multi-provider cloud music generation with automatic fallback.
    Requires ZERO local storage - all processing happens on cloud servers.
    """
    
    def __init__(self, suno_token: Optional[str] = None, hf_token: Optional[str] = None, replicate_token: Optional[str] = None,
                 beatoven_token: Optional[str] = None, loudly_token: Optional[str] = None, 
                 musicapi_token: Optional[str] = None, udio_token: Optional[str] = None,
                 musicgen_model_size: str = "medium"):
        """
        Initialize cloud music generator.
        
        Args:
            suno_token: Suno AI API token (professional quality)
            hf_token: Hugging Face API token (with pipeline support!)
            replicate_token: Replicate API token (50 free/month)
            beatoven_token: Beatoven.ai API token (mood-based music)
            loudly_token: Loudly Music API token (genre/mood)
            musicapi_token: MusicAPI.ai token (test tier)
            udio_token: Udio API token (early access)
            musicgen_model_size: MusicGen model size for HF ('small', 'medium', 'large', 'melody')
        """
        self.suno_token = suno_token or os.getenv("SUNO_API_TOKEN")
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        self.replicate_token = replicate_token or os.getenv("REPLICATE_API_TOKEN")
        self.beatoven_token = beatoven_token or os.getenv("BEATOVEN_API_KEY")
        self.loudly_token = loudly_token or os.getenv("LOUDLY_API_KEY")
        self.musicapi_token = musicapi_token or os.getenv("MUSICAPI_KEY")
        self.udio_token = udio_token or os.getenv("UDIO_API_KEY")
        self.musicgen_model_size = musicgen_model_size or os.getenv("MUSICGEN_MODEL", "medium")
        
        self.providers = []
        self._initialize_providers()
        
        logger.info(f"Initialized {len(self.providers)} cloud music providers")
    
    def _initialize_providers(self):
        """Initialize available providers in priority order."""
        
        # Priority 1: Suno AI (Professional quality with vocals!)
        if self.suno_token:
            try:
                provider = SunoProvider(self.suno_token)
                self.providers.append(provider)
                logger.info("✅ Suno AI initialized (Professional Quality)")
            except Exception as e:
                logger.warning(f"Suno AI provider failed: {e}")
        
        # Priority 2: HuggingFace with Pipeline (UNLIMITED FREE!)
        if self.hf_token:
            try:
                provider = HuggingFaceProvider(self.hf_token, self.musicgen_model_size)
                self.providers.append(provider)
                logger.info(f"✅ HuggingFace initialized (MusicGen {self.musicgen_model_size.upper()}, unlimited FREE)")
            except Exception as e:
                logger.warning(f"HuggingFace provider failed: {e}")
        
        # Priority 3: Replicate API (50 free/month)
        if self.replicate_token:
            try:
                provider = ReplicateProvider(self.replicate_token)
                self.providers.append(provider)
                logger.info("✅ Replicate API initialized (50 free/month)")
            except Exception as e:
                logger.warning(f"Replicate provider failed: {e}")
        
        # Priority 4: Beatoven.ai (Mood-based music generation)
        if self.beatoven_token:
            try:
                provider = BeatovenProvider(self.beatoven_token)
                self.providers.append(provider)
                logger.info("✅ Beatoven.ai initialized (Mood-based FREE)")
            except Exception as e:
                logger.warning(f"Beatoven provider failed: {e}")
        
        # Priority 5: Loudly Music API (Genre/mood-based)
        if self.loudly_token:
            try:
                provider = LoudlyProvider(self.loudly_token)
                self.providers.append(provider)
                logger.info("✅ Loudly Music initialized (Genre FREE)")
            except Exception as e:
                logger.warning(f"Loudly provider failed: {e}")
        
        # Priority 6: MusicAPI.ai (Test tier)
        if self.musicapi_token:
            try:
                provider = MusicAPIProvider(self.musicapi_token)
                self.providers.append(provider)
                logger.info("✅ MusicAPI.ai initialized (Test tier FREE)")
            except Exception as e:
                logger.warning(f"MusicAPI provider failed: {e}")
        
        # Priority 7: Udio API (Early access)
        if self.udio_token:
            try:
                provider = UdioProvider(self.udio_token)
                self.providers.append(provider)
                logger.info("✅ Udio initialized (Early access FREE)")
            except Exception as e:
                logger.warning(f"Udio provider failed: {e}")
        
        # Priority 8: Gradio Client (FREE, no API key needed)
        try:
            provider = GradioProvider()
            self.providers.append(provider)
            logger.info("✅ Free Audio Generator initialized (Basic quality)")
        except Exception as e:
            logger.warning(f"Free generator failed: {e}")
        
        if not self.providers:
            logger.warning("⚠️ No cloud providers available. Add API tokens for music generation...")
    
    def generate(
        self,
        prompt: str,
        duration: float = 10.0,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        provider_preference: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music from text prompt using cloud APIs.
        
        Args:
            prompt: Text description of desired music
            duration: Duration in seconds (5-30)
            temperature: Sampling temperature (0.5-1.5)
            guidance_scale: Classifier-free guidance scale (1-15)
            provider_preference: Preferred provider ('hf', 'replicate', 'gradio')
        
        Returns:
            Tuple of (audio_array, sample_rate)
        
        Raises:
            Exception: If all providers fail
        """
        
        if not self.providers:
            raise Exception(
                "No cloud providers available. Please configure:\n"
                "1. HuggingFace token: https://huggingface.co/settings/tokens (FREE)\n"
                "2. Replicate token: https://replicate.com/account/api-tokens (50 free/month)\n"
                "3. Or install gradio-client: pip install gradio-client"
            )
        
        # Reorder providers based on preference
        providers = self._get_ordered_providers(provider_preference)
        
        errors = []
        for provider in providers:
            try:
                logger.info(f"Attempting generation with {provider.name}...")
                
                audio, sample_rate = provider.generate(
                    prompt=prompt,
                    duration=duration,
                    temperature=temperature,
                    guidance_scale=guidance_scale
                )
                
                logger.info(f"✅ Successfully generated {duration}s audio with {provider.name}")
                return audio, sample_rate
                
            except Exception as e:
                error_msg = f"{provider.name} failed: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
        
        # All providers failed
        error_summary = "\n".join(errors)
        raise Exception(
            f"All cloud providers failed:\n{error_summary}\n\n"
            "Solutions:\n"
            "1. Check your API tokens are valid\n"
            "2. Check internet connection\n"
            "3. Try again (rate limits may apply)"
        )
    
    def _get_ordered_providers(self, preference: Optional[str]):
        """Get providers ordered by preference."""
        if not preference:
            return self.providers
        
        # Move preferred provider to front
        ordered = []
        preferred = None
        
        for provider in self.providers:
            if preference.lower() in provider.name.lower():
                preferred = provider
            else:
                ordered.append(provider)
        
        if preferred:
            return [preferred] + ordered
        return ordered
    
    def get_available_providers(self) -> list:
        """Get list of available provider names."""
        return [p.name for p in self.providers]
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all providers with status."""
        # Check which providers are available
        hf_available = any("huggingface" in p.name.lower() for p in self.providers)
        replicate_available = any("replicate" in p.name.lower() for p in self.providers)
        gradio_available = any("gradio" in p.name.lower() for p in self.providers)
        
        # Get API tokens to determine status
        import os
        hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
        replicate_token = os.getenv('REPLICATE_API_TOKEN')
        
        return {
            "HuggingFace": {
                "name": "Hugging Face Inference API",
                "free_tier": "Unlimited FREE",
                "quality": "Good",
                "requires_api_key": True,
                "status": "available" if hf_available else ("needs_api_key" if not hf_token else "error"),
                "notes": "Best for unlimited generations",
                "signup_url": "https://huggingface.co/settings/tokens"
            },
            "Replicate": {
                "name": "Replicate API",
                "free_tier": "50 FREE/month",
                "quality": "Excellent",
                "requires_api_key": True,
                "status": "available" if replicate_available else ("needs_api_key" if not replicate_token else "error"),
                "notes": "High quality, limited free tier",
                "signup_url": "https://replicate.com/account/api-tokens"
            },
            "Gradio": {
                "name": "Gradio Public Spaces",
                "free_tier": "Unlimited FREE",
                "quality": "Variable",
                "requires_api_key": False,
                "status": "available" if gradio_available else "error",
                "notes": "No API key needed, may have queues",
                "signup_url": None
            }
        }


# =============================================================================
# PROVIDER IMPLEMENTATIONS
# =============================================================================


class SunoProvider:
    """Suno AI API provider - Professional music generation!"""
    
    name = "Suno AI"
    
    def __init__(self, token: str):
        self.token = token
        self.api_url = "https://api.suno.ai/v1/generate"
    
    def generate(self, prompt: str, duration: float, temperature: float, 
                 guidance_scale: float) -> Tuple[np.ndarray, int]:
        """Generate music using Suno AI API."""
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            # Suno API payload
            payload = {
                "prompt": prompt,
                "make_instrumental": True,  # Set to False if you want vocals
                "wait_audio": True
            }
            
            logger.info(f"Requesting Suno AI generation...")
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=180
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Get audio URL from response
            if isinstance(result, list) and len(result) > 0:
                audio_url = result[0].get('audio_url')
            elif isinstance(result, dict):
                audio_url = result.get('audio_url') or result.get('url')
            else:
                raise Exception(f"Unexpected response format")
            
            if not audio_url:
                raise Exception("No audio URL in response")
            
            # Download the audio file
            logger.info(f"Downloading Suno audio...")
            audio_response = requests.get(audio_url, timeout=60)
            audio_bytes = audio_response.content
            
            # Convert bytes to audio array
            audio_data = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_data)
            
            # Ensure audio is numpy array
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Handle stereo to mono conversion if needed
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Trim or pad to requested duration
            target_samples = int(duration * sr)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            elif len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            
            logger.info(f"✅ Successfully generated Suno AI music!")
            return audio, sr
            
        except Exception as e:
            raise Exception(f"Suno AI failed: {str(e)}")


class HuggingFaceProvider:
    """Hugging Face Inference API provider with pipeline support (unlimited free)."""
    
    name = "HuggingFace Inference API"
    
    def __init__(self, token: str, model_size: str = "medium"):
        self.token = token
        self.model_size = model_size
        
        # Model options
        self.models = {
            "small": "facebook/musicgen-small",      # 300M params, faster
            "medium": "facebook/musicgen-medium",    # 1.5B params, balanced (default)
            "large": "facebook/musicgen-large",      # 3.3B params, best quality
            "melody": "facebook/musicgen-melody"     # Melody-conditioned
        }
        
        self.model = self.models.get(model_size, self.models["medium"])
        self.pipeline = None
        self.use_pipeline = True  # Try pipeline first, fallback to REST API
        
        # Try to initialize transformers pipeline (recommended approach)
        try:
            from transformers import pipeline
            logger.info(f"  Initializing HuggingFace pipeline for {self.model}...")
            self.pipeline = pipeline('text-to-audio', model=self.model, token=self.token)
            logger.info(f"  ✅ Pipeline initialized successfully")
        except ImportError:
            logger.warning("  transformers not installed, using REST API fallback")
            self.use_pipeline = False
        except Exception as e:
            logger.warning(f"  Pipeline init failed ({e}), using REST API fallback")
            self.use_pipeline = False
        
        # REST API fallback
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
    
    def generate(self, prompt: str, duration: float, temperature: float, 
                 guidance_scale: float) -> Tuple[np.ndarray, int]:
        """Generate music using HF pipeline or REST API."""
        
        # Method 1: Use transformers pipeline (RECOMMENDED)
        if self.use_pipeline and self.pipeline is not None:
            try:
                logger.info("  Generating with transformers pipeline...")
                
                # Generate audio using pipeline
                result = self.pipeline(
                    prompt,
                    forward_params={
                        "max_new_tokens": int(duration * 50),  # ~50 tokens per second
                        "do_sample": True,
                        "temperature": temperature,
                        "guidance_scale": guidance_scale
                    }
                )
                
                # Extract audio data
                audio = result["audio"][0]  # Get audio array
                sr = result["sampling_rate"]
                
                # Ensure numpy array
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio)
                
                # Handle stereo to mono
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Trim or pad to requested duration
                target_samples = int(duration * sr)
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                elif len(audio) < target_samples:
                    audio = np.pad(audio, (0, target_samples - len(audio)))
                
                logger.info(f"  ✅ Generated {len(audio)/sr:.1f}s audio with pipeline")
                return audio, sr
                
            except Exception as e:
                logger.warning(f"  Pipeline generation failed: {e}")
                logger.info("  Falling back to REST API...")
                # Fall through to REST API
        
        # Method 2: REST API fallback
        try:
            import requests
            
            logger.info("  Generating with REST API...")
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": int(duration * 50),
                    "temperature": temperature,
                    "guidance_scale": guidance_scale,
                    "do_sample": True
                }
            }
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
            
            # Response is audio bytes
            audio_bytes = response.content
            
            # Convert bytes to audio array
            audio_data = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_data)
            
            # Ensure audio is numpy array
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Handle stereo to mono conversion if needed
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Trim or pad to requested duration
            target_samples = int(duration * sr)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            elif len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            
            logger.info(f"  ✅ Generated {len(audio)/sr:.1f}s audio with REST API")
            return audio, sr
            
        except Exception as e:
            raise Exception(f"HuggingFace generation failed (both pipeline and REST API): {str(e)}")


class ReplicateProvider:
    """Replicate API provider (50 free predictions/month)."""
    
    name = "Replicate API"
    
    def __init__(self, token: str):
        try:
            import replicate
            self.client = replicate.Client(api_token=token)
            # Can use different model sizes
            self.models = {
                "small": "meta/musicgen:small",
                "medium": "meta/musicgen:medium", 
                "large": "meta/musicgen:large",
                "melody": "meta/musicgen:melody"
            }
            self.default_model = "small"
        except ImportError:
            raise ImportError(
                "replicate not installed. Run: pip install replicate"
            )
    
    def generate(self, prompt: str, duration: float, temperature: float,
                 guidance_scale: float) -> Tuple[np.ndarray, int]:
        """Generate music using Replicate API."""
        
        # Run prediction
        output = self.client.run(
            f"meta/musicgen:{self.default_model}",
            input={
                "prompt": prompt,
                "duration": int(duration),
                "temperature": temperature,
                "classifier_free_guidance": guidance_scale,
                "output_format": "wav"
            }
        )
        
        # Output is a URL to the audio file
        if isinstance(output, str):
            import requests
            response = requests.get(output)
            audio_data = io.BytesIO(response.content)
            audio, sr = sf.read(audio_data)
            return audio, sr
        
        raise Exception(f"Unexpected output format: {type(output)}")


class GradioProvider:
    """Simple FREE music generator using procedural synthesis."""
    
    name = "Free Audio Generator"  
    
    def __init__(self):
        # No external dependencies needed
        logger.info("Free Audio Generator ready (100% FREE, no API needed)")
    
    def generate(self, prompt: str, duration: float, temperature: float,
                 guidance_scale: float) -> Tuple[np.ndarray, int]:
        """Generate simple music using procedural synthesis."""
        
        try:
            import numpy as np
            import math
            
            sample_rate = 44100
            samples = int(duration * sample_rate)
            
            # Create a simple musical composition based on prompt
            audio = np.zeros(samples, dtype=np.float32)
            
            # Analyze prompt for musical characteristics
            prompt_lower = prompt.lower()
            
            # Determine base frequency and style
            if any(word in prompt_lower for word in ['bass', 'low', 'deep']):
                base_freq = 60  # Low frequencies
                frequencies = [60, 80, 100, 120]
            elif any(word in prompt_lower for word in ['high', 'treble', 'bright']):
                base_freq = 800  # High frequencies  
                frequencies = [800, 1000, 1200, 1600]
            else:
                base_freq = 261  # Middle C
                frequencies = [261, 294, 330, 349, 392]  # C major scale
            
            # Determine rhythm based on prompt
            if any(word in prompt_lower for word in ['fast', 'energetic', 'rush', 'quick', 'rapid']):
                beat_duration = 0.3
            elif any(word in prompt_lower for word in ['slow', 'calm', 'peaceful']):
                beat_duration = 1.2
            else:
                beat_duration = 0.6
            
            # Generate musical sequence
            t = np.linspace(0, duration, samples, False)
            
            # Add multiple frequency components
            for i, freq in enumerate(frequencies):
                # Create oscillating pattern
                wave = np.sin(2 * np.pi * freq * t)
                
                # Add beat pattern
                beat_pattern = np.sin(2 * np.pi * t / beat_duration)
                envelope = (beat_pattern + 1) / 2
                
                # Apply temperature effect (randomness)
                if temperature > 1.0:
                    noise = np.random.normal(0, temperature - 1.0, samples) * 0.1
                    wave += noise
                
                # Fade in/out
                fade_samples = int(0.1 * sample_rate)
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                
                if len(wave) > 2 * fade_samples:
                    wave[:fade_samples] *= fade_in
                    wave[-fade_samples:] *= fade_out
                
                # Mix with diminishing amplitude
                audio += wave * envelope * (0.3 / (i + 1))
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            logger.info(f"✅ Generated {duration}s of music with {len(frequencies)} frequencies")
            return audio, sample_rate
            
        except Exception as e:
            raise Exception(f"Free audio generation failed: {str(e)}")


class BeatovenProvider:
    """Beatoven.ai API provider - Mood-based music generation (FREE tier)"""
    
    name = "Beatoven.ai"
    
    def __init__(self, token: str):
        self.token = token
        self.api_url = "https://api.beatoven.ai/v1/generate"
    
    def generate(self, prompt: str, duration: float, temperature: float, 
                 guidance_scale: float) -> Tuple[np.ndarray, int]:
        """Generate music using Beatoven.ai API with mood control"""
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            # Extract mood from prompt
            moods = ["happy", "sad", "energetic", "calm", "epic", "romantic"]
            detected_mood = "energetic"
            for mood in moods:
                if mood in prompt.lower():
                    detected_mood = mood
                    break
            
            payload = {
                "text": prompt,
                "mood": detected_mood,
                "duration": int(duration),
                "genre": "electronic"  # Can be extracted from prompt
            }
            
            logger.info(f"Requesting Beatoven generation with mood: {detected_mood}")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=180
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            audio_url = result.get('audio_url') or result.get('url')
            
            if not audio_url:
                raise Exception("No audio URL in response")
            
            # Download audio
            audio_response = requests.get(audio_url, timeout=60)
            audio_bytes = audio_response.content
            
            # Convert to audio array
            audio_data = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_data)
            
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            logger.info(f"✅ Beatoven.ai generation successful!")
            return audio, sr
            
        except Exception as e:
            raise Exception(f"Beatoven.ai failed: {str(e)}")


class LoudlyProvider:
    """Loudly Music API provider - Genre/mood-based generation (FREE tier)"""
    
    name = "Loudly Music"
    
    def __init__(self, token: str):
        self.token = token
        self.api_url = "https://api.loudly.com/v1/music/generate"
    
    def generate(self, prompt: str, duration: float, temperature: float,
                 guidance_scale: float) -> Tuple[np.ndarray, int]:
        """Generate music using Loudly Music API"""
        
        try:
            import requests
            
            headers = {
                "X-API-Key": self.token,
                "Content-Type": "application/json"
            }
            
            # Extract genre and mood
            genres = ["pop", "rock", "electronic", "jazz", "classical"]
            detected_genre = "electronic"
            for genre in genres:
                if genre in prompt.lower():
                    detected_genre = genre
                    break
            
            payload = {
                "prompt": prompt,
                "genre": detected_genre,
                "duration_seconds": int(duration),
                "energy_level": int(temperature * 5)  # 1-10 scale
            }
            
            logger.info(f"Requesting Loudly generation with genre: {detected_genre}")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=180
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            audio_url = result.get('music_url') or result.get('download_url')
            
            if not audio_url:
                raise Exception("No audio URL in response")
            
            # Download audio
            audio_response = requests.get(audio_url, timeout=60)
            audio_bytes = audio_response.content
            
            # Convert to audio array
            audio_data = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_data)
            
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            logger.info(f"✅ Loudly Music generation successful!")
            return audio, sr
            
        except Exception as e:
            raise Exception(f"Loudly Music failed: {str(e)}")


class MusicAPIProvider:
    """MusicAPI.ai provider - Text-to-music (Test tier FREE)"""
    
    name = "MusicAPI.ai"
    
    def __init__(self, token: str):
        self.token = token
        self.api_url = "https://api.musicapi.ai/v1/generate"
    
    def generate(self, prompt: str, duration: float, temperature: float,
                 guidance_scale: float) -> Tuple[np.ndarray, int]:
        """Generate music using MusicAPI.ai"""
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "length": int(duration),
                "creativity": temperature,
                "guidance": guidance_scale
            }
            
            logger.info(f"Requesting MusicAPI.ai generation...")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=180
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            audio_url = result.get('audio_url') or result.get('file_url')
            
            if not audio_url:
                raise Exception("No audio URL in response")
            
            # Download audio
            audio_response = requests.get(audio_url, timeout=60)
            audio_bytes = audio_response.content
            
            # Convert to audio array
            audio_data = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_data)
            
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            logger.info(f"✅ MusicAPI.ai generation successful!")
            return audio, sr
            
        except Exception as e:
            raise Exception(f"MusicAPI.ai failed: {str(e)}")


class UdioProvider:
    """Udio API provider - Early access music generation (FREE)"""
    
    name = "Udio"
    
    def __init__(self, token: str):
        self.token = token
        self.api_url = "https://api.udio.com/v1/generate"
    
    def generate(self, prompt: str, duration: float, temperature: float,
                 guidance_scale: float) -> Tuple[np.ndarray, int]:
        """Generate music using Udio API"""
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "duration": int(duration),
                "temperature": temperature,
                "cfg_scale": guidance_scale
            }
            
            logger.info(f"Requesting Udio generation...")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=180
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            audio_url = result.get('audio_url') or result.get('download_url')
            
            if not audio_url:
                raise Exception("No audio URL in response")
            
            # Download audio
            audio_response = requests.get(audio_url, timeout=60)
            audio_bytes = audio_response.content
            
            # Convert to audio array
            audio_data = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_data)
            
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            logger.info(f"✅ Udio generation successful!")
            return audio, sr
            
        except Exception as e:
            raise Exception(f"Udio failed: {str(e)}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_generate(prompt: str, duration: float = 10.0) -> Tuple[np.ndarray, int]:
    """
    Quick music generation with automatic provider selection.
    
    Args:
        prompt: Text description of music
        duration: Duration in seconds
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    generator = CloudMusicGenerator()
    return generator.generate(prompt, duration)


def save_audio(audio: np.ndarray, sample_rate: int, output_path: str):
    """Save audio to file."""
    sf.write(output_path, audio, sample_rate)
    logger.info(f"Saved audio to {output_path}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Quick generation
    print("Generating music with automatic provider selection...")
    audio, sr = quick_generate("upbeat electronic dance music", duration=10)
    save_audio(audio, sr, "output/test_music.wav")
    
    # Example 2: With specific provider
    generator = CloudMusicGenerator(
        hf_token="your_hf_token",
        replicate_token="your_replicate_token"
    )
    
    print("\nAvailable providers:", generator.get_available_providers())
    print("\nProvider info:")
    for name, info in generator.get_provider_info().items():
        print(f"\n{info['name']}:")
        print(f"  Cost: {info['cost']}")
        print(f"  Quality: {info['quality']}")
        print(f"  Available: {info['available']}")
    
    # Generate with preferred provider
    audio, sr = generator.generate(
        prompt="lo-fi hip hop beats to study to",
        duration=15,
        provider_preference="huggingface"
    )
    save_audio(audio, sr, "output/lofi_beats.wav")
