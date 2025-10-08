"""
AI Music Generation Pipeline with MusicGen & Groq LLM
====================================================

Production-ready music generation system using Meta's MusicGen
with intelligent prompt enhancement via Groq LLM (Llama 3.1 70B).

Author: AI Music Generator Team
Version: 2.0.0
"""

import os
import logging
import torch
import torchaudio
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write
    AUDIOCRAFT_AVAILABLE = True
except ImportError:
    logger.warning("AudioCraft not installed. Falling back to cloud providers...")
    AUDIOCRAFT_AVAILABLE = False

# Try importing cloud music generator (zero storage alternative)
try:
    from cloud_music_generator import CloudMusicGenerator
    CLOUD_GENERATOR_AVAILABLE = True
    logger.info("✅ Cloud music generation available (zero storage)")
except ImportError:
    CLOUD_GENERATOR_AVAILABLE = False
    logger.warning("Cloud generator not available")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("Librosa not installed. Install with: pip install librosa")
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    logger.warning("SoundFile not installed. Install with: pip install soundfile")
    SOUNDFILE_AVAILABLE = False

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("SciPy not installed. Install with: pip install scipy")
    SCIPY_AVAILABLE = False

# Constants
DEFAULT_SAMPLE_RATE = 32000
DEFAULT_DURATION = 30.0
MAX_DURATION = 300.0

MUSICGEN_MODELS = {
    'small': 'facebook/musicgen-small',
    'medium': 'facebook/musicgen-medium',
    'large': 'facebook/musicgen-large',
    'melody': 'facebook/musicgen-melody'
}

# Genre templates for prompt enhancement
GENRE_TEMPLATES = {
    'pop': {
        'description': 'catchy melody, major key, steady beat, polished production',
        'instruments': 'synths, electric guitar, drums, bass'
    },
    'rock': {
        'description': 'electric guitars, powerful drums, energetic, driving rhythm',
        'instruments': 'electric guitar, bass guitar, drums'
    },
    'electronic': {
        'description': 'synthesized sounds, electronic beats, digital production',
        'instruments': 'synthesizers, drum machines, electronic bass'
    },
    'jazz': {
        'description': 'improvisation, swing rhythm, complex harmonies',
        'instruments': 'saxophone, piano, double bass, drums'
    },
    'classical': {
        'description': 'orchestral instruments, structured composition, elegant',
        'instruments': 'strings, woodwinds, brass, percussion'
    }
}

MOOD_DESCRIPTORS = {
    'happy': 'upbeat, cheerful, bright, optimistic, major key, energetic',
    'sad': 'melancholic, somber, minor key, slow tempo, emotional',
    'energetic': 'fast-paced, dynamic, powerful, driving, intense',
    'calm': 'peaceful, serene, gentle, relaxing, soft, tranquil',
    'dark': 'ominous, brooding, mysterious, heavy, intense'
}


class MusicGenPipeline:
    """
    Production-ready pipeline for AI music generation.
    Supports both local (AudioCraft) and cloud-based generation.
    
    Example:
        >>> pipeline = MusicGenPipeline(model_size='medium')
        >>> audio, sr = pipeline.generate_music("upbeat jazz piano", duration=30)
        >>> pipeline.save_audio(audio, "output.wav", sr)
    """
    
    def __init__(
        self,
        model_size: str = 'medium',
        device: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        use_cloud: bool = None
    ):
        """
        Initialize MusicGen pipeline.
        
        Args:
            model_size: Model size ('small', 'medium', 'large', 'melody')
            device: Device to use ('cuda', 'cpu', or None for auto)
            groq_api_key: Groq API key for prompt enhancement
            use_cloud: Force cloud generation (None = auto-detect)
        """
        self.model_size = model_size
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        
        # Determine if we should use cloud
        if use_cloud is None:
            # Auto-detect: use cloud if AudioCraft not available
            self.use_cloud = not AUDIOCRAFT_AVAILABLE
        else:
            self.use_cloud = use_cloud
        
        # Determine device (only for local)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize appropriate backend
        self.model = None
        self.cloud_generator = None
        self.sample_rate = DEFAULT_SAMPLE_RATE
        
        if self.use_cloud:
            if not CLOUD_GENERATOR_AVAILABLE:
                raise ImportError(
                    "Cloud generator not available. Please ensure cloud_music_generator.py is present."
                )
            logger.info("🌥️  Using CLOUD music generation (zero storage)")
            logger.info("  Providers: HuggingFace, Replicate, Gradio")
            self._init_cloud_generator()
        else:
            if not AUDIOCRAFT_AVAILABLE:
                raise ImportError(
                    "AudioCraft not available. Install with: pip install audiocraft\n"
                    "Or set use_cloud=True for cloud-based generation."
                )
            logger.info(f"💾 Using LOCAL music generation")
            logger.info(f"  Model: {model_size}, Device: {self.device}")
        
        # Initialize Groq client if available
        self.groq_client = None
        if self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("  Groq LLM: enabled")
            except ImportError:
                logger.warning("  Groq package not installed")
            except Exception as e:
                logger.warning(f"  Groq initialization failed: {e}")
    
    def _init_cloud_generator(self):
        """Initialize cloud music generator."""
        suno_token = os.getenv("SUNO_API_TOKEN")
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        beatoven_token = os.getenv("BEATOVEN_API_KEY")
        loudly_token = os.getenv("LOUDLY_API_KEY")
        musicapi_token = os.getenv("MUSICAPI_KEY")
        udio_token = os.getenv("UDIO_API_KEY")
        
        self.cloud_generator = CloudMusicGenerator(
            suno_token=suno_token,
            hf_token=hf_token,
            replicate_token=replicate_token,
            beatoven_token=beatoven_token,
            loudly_token=loudly_token,
            musicapi_token=musicapi_token,
            udio_token=udio_token
        )
        
        providers = self.cloud_generator.get_available_providers()
        if providers:
            logger.info(f"  Available providers: {', '.join(providers)}")
        else:
            logger.warning("  ⚠️  No API tokens configured. Add tokens for best results:")
            logger.warning("     Suno AI: https://suno.com (Professional quality!)")
            logger.warning("     Replicate: https://replicate.com/account/api-tokens")
            logger.warning("     Beatoven: https://www.beatoven.ai/api (Mood-based)")
            logger.warning("     Loudly: https://www.loudly.com/music-api (Genre-based)")
            logger.warning("     MusicAPI: https://musicapi.ai (Text-to-music)")
            logger.warning("     Udio: https://www.udio.com (Early access)")
    
    def _load_model(self):
        """Load MusicGen model (lazy loading) - for local generation only."""
        if self.use_cloud:
            return  # No model loading needed for cloud
        
        if self.model is None:
            logger.info(f"Loading MusicGen {self.model_size} model...")
            self.model = MusicGen.get_pretrained(self.model_size, device=self.device)
            logger.info("Model loaded successfully")
    
    def generate_music(
        self,
        prompt: str,
        duration: float = 30.0,
        temperature: float = 1.0,
        cfg_coef: float = 3.0,
        enhance_prompt: bool = False,
        genre: Optional[str] = None,
        mood: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music from text prompt.
        
        Args:
            prompt: Text description
            duration: Duration in seconds (max 300)
            temperature: Sampling temperature (0.5-1.5)
            cfg_coef: Guidance coefficient (1.0-10.0)
            enhance_prompt: Use prompt enhancement
            genre: Genre for enhancement
            mood: Mood for enhancement
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Validate duration
        duration = max(1.0, min(duration, MAX_DURATION))
        
        # Enhance prompt if requested
        if enhance_prompt:
            original = prompt
            prompt = self._enhance_prompt(prompt, genre, mood)
            logger.info(f"Prompt enhanced: '{original}' -> '{prompt}'")
        
        logger.info(f"Generating {duration}s of music using {'cloud' if self.use_cloud else 'local'} backend...")
        
        # Route to cloud or local generation
        if self.use_cloud:
            return self._generate_cloud(
                prompt=prompt,
                duration=duration,
                temperature=temperature,
                guidance_scale=cfg_coef
            )
        else:
            return self._generate_local(
                prompt=prompt,
                duration=duration,
                temperature=temperature,
                cfg_coef=cfg_coef
            )
    
    def _generate_local(
        self,
        prompt: str,
        duration: float,
        temperature: float,
        cfg_coef: float
    ) -> Tuple[np.ndarray, int]:
        """Generate music using local AudioCraft model."""
        self._load_model()
        
        # Set generation parameters
        self.model.set_generation_params(
            duration=duration,
            temperature=temperature,
            cfg_coef=cfg_coef
        )
        
        # Generate
        with torch.no_grad():
            wav = self.model.generate([prompt])
        
        audio = wav[0].cpu().numpy()
        logger.info(f"Generated local audio: shape={audio.shape}")
        
        return audio, self.sample_rate
    
    def _generate_cloud(
        self,
        prompt: str,
        duration: float,
        temperature: float,
        guidance_scale: float
    ) -> Tuple[np.ndarray, int]:
        """Generate music using cloud providers."""
        try:
            result = self.cloud_generator.generate(
                prompt=prompt,
                duration=duration,
                temperature=temperature,
                guidance_scale=guidance_scale
            )
            
            # Handle different return formats from cloud generators
            if isinstance(result, tuple) and len(result) == 2:
                audio_array, sample_rate = result
            else:
                audio_array = result
                sample_rate = 32000
            
            logger.info(f"Generated cloud audio: shape={audio_array.shape}, rate={sample_rate}")
            
            return audio_array, sample_rate
            
        except Exception as e:
            logger.error(f"Cloud generation failed: {e}")
            raise RuntimeError(f"Failed to generate music using cloud providers: {e}")
    
    def _enhance_prompt(
        self,
        user_input: str,
        genre: Optional[str] = None,
        mood: Optional[str] = None
    ) -> str:
        """Enhance prompt using templates or LLM."""
        parts = []
        
        # Add mood descriptors
        if mood and mood.lower() in MOOD_DESCRIPTORS:
            parts.append(MOOD_DESCRIPTORS[mood.lower()])
        
        # Add genre information
        if genre and genre.lower() in GENRE_TEMPLATES:
            template = GENRE_TEMPLATES[genre.lower()]
            parts.append(f"{genre} style")
            parts.append(template['description'])
        
        # Add user input
        parts.append(user_input)
        
        enhanced = ", ".join(parts)
        return enhanced
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: Optional[int] = None
    ) -> str:
        """
        Save audio to file.
        
        Args:
            audio: Audio array
            output_path: Output file path
            sample_rate: Sample rate (None = use default)
            
        Returns:
            Path to saved file
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure 2D array
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # Save using soundfile
        if SOUNDFILE_AVAILABLE:
            # Transpose for soundfile (expects samples, channels)
            audio_t = audio.T if audio.ndim > 1 else audio
            sf.write(str(output_path), audio_t, sample_rate)
            logger.info(f"Saved audio: {output_path}")
        else:
            logger.error("SoundFile not available, cannot save audio")
            raise ImportError("Install soundfile: pip install soundfile")
        
        return str(output_path)


# Helper functions
def get_available_genres() -> List[str]:
    """Get list of available genres."""
    return sorted(list(GENRE_TEMPLATES.keys()))


def get_available_moods() -> List[str]:
    """Get list of available moods."""
    return sorted(list(MOOD_DESCRIPTORS.keys()))


def get_genre_info(genre: str) -> Dict[str, str]:
    """Get information about a genre."""
    return GENRE_TEMPLATES.get(genre.lower(), {})


def get_mood_descriptors(mood: str) -> str:
    """Get descriptors for a mood."""
    return MOOD_DESCRIPTORS.get(mood.lower(), "")


# Legacy compatibility
MusicGenerator = MusicGenPipeline


# =============================================================================
# PROMPT ENHANCER CLASS
# =============================================================================

class PromptEnhancer:
    """
    Intelligent prompt enhancement using FREE LLM providers.
    
    Features:
    - Multi-provider support (Groq, Together AI, HuggingFace, Ollama)
    - Automatic fallback chain
    - Rule-based fallback when no APIs available
    - Caching for repeated prompts
    - Music-specific knowledge base
    
    Usage:
        enhancer = PromptEnhancer()
        enhanced = enhancer.enhance_prompt("chill beat", genre="lo-fi", mood="calm")
    """
    
    def __init__(self):
        """Initialize with available LLM providers."""
        self.cache = {}
        self.providers = []
        
        # Try to import and setup Groq (FASTEST, 30 req/min free)
        try:
            from groq import Groq
            groq_key = os.getenv('GROQ_API_KEY', '')
            if groq_key:
                self.groq_client = Groq(api_key=groq_key)
                self.providers.append('groq')
                logger.info("✓ Groq LLM initialized")
        except Exception as e:
            logger.debug(f"Groq not available: {e}")
            self.groq_client = None
        
        # Try OpenRouter (FREE Llama 3.1 & many models!)
        try:
            import requests
            openrouter_key = os.getenv('OPENROUTER_API_KEY', '')
            if openrouter_key:
                self.openrouter_key = openrouter_key
                self.providers.append('openrouter')
                logger.info("✓ OpenRouter initialized (FREE Llama models)")
        except Exception as e:
            logger.debug(f"OpenRouter not available: {e}")
            self.openrouter_key = None
        
        # Try HuggingFace (Completely FREE)
        try:
            from huggingface_hub import InferenceClient
            hf_token = os.getenv('HUGGINGFACE_TOKEN', '')
            if hf_token:
                self.hf_client = InferenceClient(token=hf_token)
                self.providers.append('huggingface')
                logger.info("✓ HuggingFace Inference initialized")
        except Exception:
            self.hf_client = None
        
        # Log status
        if self.providers:
            logger.info(f"Prompt enhancement enabled with: {', '.join(self.providers)}")
        else:
            logger.info("No LLM providers configured, using rule-based enhancement")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        return self.providers.copy()
    
    def enhance_prompt(
        self,
        user_input: str,
        genre: str = None,
        mood: str = None,
        duration: int = 30,
        instruments: List[str] = None,
        bpm: int = None,
        key: str = None
    ) -> str:
        """
        Enhance user prompt with detailed musical description.
        
        Args:
            user_input: Basic description from user
            genre: Musical genre
            mood: Desired mood
            duration: Length in seconds
            instruments: List of instruments
            bpm: Beats per minute
            key: Musical key (e.g., "C Major", "A Minor")
        
        Returns:
            Enhanced, detailed prompt for MusicGen
        """
        # Check cache first
        cache_key = f"{user_input}_{genre}_{mood}_{duration}_{instruments}_{bpm}_{key}"
        if cache_key in self.cache:
            logger.debug("Using cached enhanced prompt")
            return self.cache[cache_key]
        
        # Build context
        context = self._build_context(genre, mood, instruments, duration, bpm, key)
        
        # Try LLM providers in order
        enhanced = None
        
        if 'groq' in self.providers and self.groq_client:
            enhanced = self._try_groq(user_input, context)
        
        if not enhanced and 'openrouter' in self.providers and self.openrouter_key:
            enhanced = self._try_openrouter(user_input, context)
        
        if not enhanced and 'huggingface' in self.providers and self.hf_client:
            enhanced = self._try_huggingface(user_input, context)
        
        # Fallback to rule-based enhancement
        if not enhanced:
            enhanced = self._fallback_enhancement(user_input, genre, mood, instruments, bpm, key)
            logger.debug("Using rule-based prompt enhancement")
        
        # Cache result
        self.cache[cache_key] = enhanced
        
        return enhanced
    
    def _build_context(
        self,
        genre: Optional[str],
        mood: Optional[str],
        instruments: Optional[List[str]],
        duration: int,
        bpm: Optional[int] = None,
        key: Optional[str] = None
    ) -> str:
        """Build context string from parameters."""
        parts = []
        
        if genre:
            genre_info = GENRE_TEMPLATES.get(genre.lower(), {})
            if genre_info:
                parts.append(f"Genre: {genre} ({genre_info.get('description', '')})")
        
        if mood:
            mood_desc = MOOD_DESCRIPTORS.get(mood.lower(), '')
            if mood_desc:
                parts.append(f"Mood: {mood} ({mood_desc})")
        
        if instruments:
            parts.append(f"Instruments: {', '.join(instruments)}")
        
        if bpm:
            parts.append(f"BPM: {bpm}")
        
        if key:
            parts.append(f"Key: {key}")
        
        parts.append(f"Duration: {duration} seconds")
        
        return " | ".join(parts)
    
    def _try_groq(self, user_input: str, context: str) -> Optional[str]:
        """Try Groq API (Llama 3.1 70B)."""
        try:
            system_prompt = """You are a music production expert. Convert simple music descriptions into detailed, technical prompts for AI music generation. Focus on:
- Specific instruments and their characteristics
- Tempo, rhythm, and time signature
- Musical key and harmony
- Production style and effects
- Mood and emotional qualities

Keep responses under 100 words, be specific and technical."""
            
            user_prompt = f"User wants: {user_input}\nContext: {context}\n\nCreate a detailed music generation prompt:"
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200,
                timeout=10,
            )
            
            enhanced = response.choices[0].message.content.strip()
            logger.info("✓ Prompt enhanced with Groq")
            return enhanced
            
        except Exception as e:
            logger.debug(f"Groq API failed: {e}")
            return None
    
    def _try_openrouter(self, user_input: str, context: str) -> Optional[str]:
        """Try OpenRouter API (FREE Llama 3.1 8B and other models)."""
        try:
            import requests
            
            system_prompt = """You are a music production expert. Convert simple music descriptions into detailed, technical prompts for AI music generation. Focus on:
- Specific instruments and their characteristics
- Tempo, rhythm, and time signature
- Musical key and harmony
- Production style and effects
- Mood and emotional qualities

Keep responses under 100 words, be specific and technical."""
            
            user_prompt = f"User wants: {user_input}\nContext: {context}\n\nCreate a detailed music generation prompt:"
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-3.1-8b-instruct:free",  # FREE model!
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                enhanced = data['choices'][0]['message']['content'].strip()
                logger.info("✓ Prompt enhanced with OpenRouter (FREE Llama 3.1)")
                return enhanced
            else:
                logger.debug(f"OpenRouter API failed with status {response.status_code}")
                return None
            
        except Exception as e:
            logger.debug(f"OpenRouter API failed: {e}")
            return None
    
    def _try_huggingface(self, user_input: str, context: str) -> Optional[str]:
        """Try HuggingFace Inference API."""
        try:
            prompt = f"Create a detailed music generation prompt for: {user_input}\nContext: {context}\nDetailed prompt:"
            
            response = self.hf_client.text_generation(
                prompt,
                model="meta-llama/Meta-Llama-3-70B-Instruct",
                max_new_tokens=200,
                temperature=0.7,
            )
            
            enhanced = response.strip()
            logger.info("✓ Prompt enhanced with HuggingFace")
            return enhanced
            
        except Exception as e:
            logger.debug(f"HuggingFace failed: {e}")
            return None
    
    def _fallback_enhancement(
        self,
        user_input: str,
        genre: Optional[str],
        mood: Optional[str],
        instruments: Optional[List[str]],
        bpm: Optional[int] = None,
        key: Optional[str] = None
    ) -> str:
        """
        Rule-based prompt enhancement (NO LLM needed).
        Always works even without API keys.
        """
        parts = [user_input] if user_input else []
        
        # Add genre characteristics
        if genre:
            genre_info = GENRE_TEMPLATES.get(genre.lower(), {})
            if genre_info:
                parts.append(genre_info.get('description', ''))
                parts.append(f"featuring {genre_info.get('instruments', '')}")
        
        # Add mood descriptors
        if mood:
            mood_desc = MOOD_DESCRIPTORS.get(mood.lower(), '')
            if mood_desc:
                parts.append(f"with a {mood_desc} feeling")
        
        # Add instruments
        if instruments:
            parts.append(f"using {', '.join(instruments)}")
        
        # Add BPM
        if bpm:
            parts.append(f"at {bpm} BPM")
        
        # Add key
        if key:
            parts.append(f"in {key}")
        
        # Combine into coherent prompt
        enhanced = ', '.join(filter(None, parts))
        
        # Ensure it's not empty
        if not enhanced:
            enhanced = "instrumental music with rich melodic content"
        
        return enhanced


__version__ = "2.0.0"
__all__ = [
    'MusicGenPipeline',
    'MusicGenerator',
    'PromptEnhancer',
    'get_available_genres',
    'get_available_moods',
    'get_genre_info',
    'get_mood_descriptors'
]

logger.info(f"MusicGen module v{__version__} initialized")
