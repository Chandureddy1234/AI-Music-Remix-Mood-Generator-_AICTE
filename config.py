"""
Configuration Module for AI Music Generator
Manages all application settings, API keys, and model configurations

Features:
- Multi-provider LLM support with automatic fallback
- Secure Hybrid Key Management (Streamlit Secrets + Session Storage)
- Cloud-compatible (works on Streamlit Cloud)
- Per-user session isolation
- Rate limiting and quota tracking
- Model configuration for MusicGen
- Audio processing parameters
- UI customization settings

SECURITY ARCHITECTURE:
Priority 1: User session keys (per-user, session-only)
Priority 2: Streamlit Secrets (admin defaults, encrypted)
Priority 3: Environment variables (local dev only)
Priority 4: None (Free Generator fallback)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
import time
from functools import wraps

# Optional: PyTorch (not needed for cloud-only deployment)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Load environment variables from .env file (for local development)
load_dotenv()

# ============================================================================
# SECURE HYBRID KEY MANAGER (Streamlit Cloud Compatible)
# ============================================================================

class SecureKeyManager:
    """
    Secure API key management for Streamlit Cloud deployment.
    
    SECURITY FEATURES:
    ✅ Session-only user keys (no persistence)
    ✅ Streamlit Secrets for admin defaults
    ✅ Per-user isolation
    ✅ Cloud-compatible (no local file storage)
    ✅ GDPR compliant (no user data stored)
    
    PRIORITY FALLBACK:
    1. User session key (if user provided)
    2. Streamlit secrets (admin default)
    3. Environment variables (local dev)
    4. None (use Free Generator)
    """
    
    @staticmethod
    def get_key(provider: str) -> Optional[str]:
        """
        Get API key with secure priority fallback.
        
        Args:
            provider: Provider name (e.g., 'GROQ_API_KEY', 'REPLICATE_API_TOKEN')
            
        Returns:
            API key or None
        """
        try:
            import streamlit as st
            
            # Priority 1: User's session key (highest priority)
            if 'api_keys' in st.session_state:
                if provider in st.session_state.api_keys:
                    return st.session_state.api_keys[provider]
            
            # Priority 2: Streamlit secrets (admin defaults)
            if hasattr(st, 'secrets'):
                # Try api_keys section first
                if 'api_keys' in st.secrets:
                    if provider in st.secrets.api_keys:
                        return st.secrets.api_keys[provider]
                # Try direct key access
                if provider in st.secrets:
                    return st.secrets[provider]
            
        except ImportError:
            # Streamlit not available (running outside Streamlit)
            pass
        except Exception as e:
            # Secrets not configured (expected for local dev)
            pass
        
        # Priority 3: Environment variables (local development)
        env_value = os.getenv(provider)
        if env_value:
            return env_value
        
        # Priority 4: None (will use Free Generator fallback)
        return None
    
    @staticmethod
    def save_key(provider: str, api_key: str) -> bool:
        """
        Save API key to session state ONLY (no persistence).
        
        SECURITY: Keys are stored in RAM only and lost when session ends.
        This is MORE secure than persistent storage!
        
        Args:
            provider: Provider key name
            api_key: API key value
            
        Returns:
            Success status
        """
        try:
            import streamlit as st
            
            # Initialize session state if needed
            if 'api_keys' not in st.session_state:
                st.session_state.api_keys = {}
            
            # Save to session ONLY (no persistence)
            st.session_state.api_keys[provider] = api_key
            
            return True
        except ImportError:
            # Streamlit not available
            return False
        except Exception as e:
            print(f"Error saving key: {e}")
            return False
    
    @staticmethod
    def clear_key(provider: str) -> bool:
        """Remove API key from session"""
        try:
            import streamlit as st
            if 'api_keys' in st.session_state:
                if provider in st.session_state.api_keys:
                    del st.session_state.api_keys[provider]
            return True
        except:
            return False
    
    @staticmethod
    def clear_all_keys() -> bool:
        """Clear all user session keys"""
        try:
            import streamlit as st
            if 'api_keys' in st.session_state:
                st.session_state.api_keys = {}
            return True
        except:
            return False
    
    @staticmethod
    def has_key(provider: str) -> bool:
        """Check if API key is available (any source)"""
        return SecureKeyManager.get_key(provider) is not None
    
    @staticmethod
    def get_key_source(provider: str) -> str:
        """
        Get the source of the API key (for debugging).
        
        Returns:
            'session', 'secrets', 'env', or 'none'
        """
        try:
            import streamlit as st
            
            # Check session
            if 'api_keys' in st.session_state:
                if provider in st.session_state.api_keys:
                    return 'session'
            
            # Check secrets
            if hasattr(st, 'secrets'):
                if 'api_keys' in st.secrets:
                    if provider in st.secrets.api_keys:
                        return 'secrets'
                if provider in st.secrets:
                    return 'secrets'
        except:
            pass
        
        # Check environment
        if os.getenv(provider):
            return 'env'
        
        return 'none'
    
    @staticmethod
    def validate_api_key(provider: str, api_key: str) -> bool:
        """
        Validate API key format (basic check)
        
        Args:
            provider: API provider name
            api_key: API key to validate
            
        Returns:
            bool: Whether key format is valid
        """
        if not api_key or len(api_key) < 10:
            return False
        
        # Provider-specific validation
        provider_lower = provider.lower()
        
        # LLM Providers
        if provider_lower == "groq":
            return api_key.startswith("gsk_")
        elif provider_lower == "huggingface":
            return api_key.startswith("hf_")
        elif provider_lower == "together":
            return len(api_key) > 30  # Together keys are long strings
        elif provider_lower == "openrouter":
            return api_key.startswith("sk-or-")
        
        # Music Generation Providers
        elif provider_lower == "replicate":
            return api_key.startswith("r8_")
        elif provider_lower == "beatoven":
            return len(api_key) >= 32  # Beatoven uses standard UUID format
        elif provider_lower == "loudly":
            return len(api_key) >= 20  # Loudly API keys are alphanumeric
        elif provider_lower == "musicapi":
            return len(api_key) >= 16  # MusicAPI test keys
        elif provider_lower == "udio":
            return len(api_key) >= 24  # Udio early access keys
        elif provider_lower == "suno":
            return len(api_key) >= 20  # Suno API tokens
        
        # Audio Analysis Providers
        elif provider_lower == "hume":
            return len(api_key) >= 32  # Hume AI keys are long
        elif provider_lower == "eden":
            return api_key.startswith("Bearer ") or len(api_key) >= 20
        elif provider_lower == "audd":
            return len(api_key) >= 16  # Audd.io API tokens
        
        return True
    
    @staticmethod
    def get_provider_info() -> Dict[str, Dict[str, str]]:
        """
        Get information about API providers
        
        Returns:
            dict: Provider information including signup URLs
        """
        return {
            # LLM Providers (for lyrics generation)
            "groq": {
                "name": "Groq",
                "signup_url": "https://console.groq.com/keys",
                "description": "⚡ Fastest inference (30 req/min free)",
                "docs": "https://console.groq.com/docs"
            },
            "together": {
                "name": "Together AI",
                "signup_url": "https://api.together.xyz/settings/api-keys",
                "description": "🚀 $25 free credits",
                "docs": "https://docs.together.ai/"
            },
            "huggingface": {
                "name": "Hugging Face",
                "signup_url": "https://huggingface.co/settings/tokens",
                "description": "🤗 Free inference API",
                "docs": "https://huggingface.co/docs/api-inference/"
            },
            "openrouter": {
                "name": "OpenRouter",
                "signup_url": "https://openrouter.ai/keys",
                "description": "🔀 Multiple models, pay-as-you-go",
                "docs": "https://openrouter.ai/docs"
            },
            
            # Music Generation Providers
            "suno": {
                "name": "Suno AI",
                "signup_url": "https://suno.ai/",
                "description": "🎵 Text-to-music (50 free generations)",
                "docs": "https://suno.ai/docs"
            },
            "replicate": {
                "name": "Replicate",
                "signup_url": "https://replicate.com/account/api-tokens",
                "description": "🔁 MusicGen model (free tier available)",
                "docs": "https://replicate.com/docs"
            },
            "beatoven": {
                "name": "Beatoven.ai",
                "signup_url": "https://www.beatoven.ai/",
                "description": "🎹 AI music composition (free trial)",
                "docs": "https://www.beatoven.ai/docs"
            },
            "loudly": {
                "name": "Loudly",
                "signup_url": "https://www.loudly.com/",
                "description": "🔊 Royalty-free music (free tier)",
                "docs": "https://www.loudly.com/docs"
            },
            "udio": {
                "name": "Udio",
                "signup_url": "https://udio.ai/",
                "description": "🎼 AI music generator (early access)",
                "docs": "https://udio.ai/docs"
            },
            
            # Audio Analysis Providers
            "hume": {
                "name": "Hume AI",
                "signup_url": "https://www.hume.ai/",
                "description": "🧠 Emotion recognition (free tier)",
                "docs": "https://docs.hume.ai/"
            },
            "eden": {
                "name": "Eden AI",
                "signup_url": "https://www.edenai.co/",
                "description": "🌍 Multi-provider audio analysis (free credits)",
                "docs": "https://docs.edenai.co/"
            },
            "audd": {
                "name": "AudD.io",
                "signup_url": "https://audd.io/",
                "description": "🎧 Music recognition ($10 free credits)",
                "docs": "https://docs.audd.io/"
            }
        }

# ============================================================================
# RATE LIMITING (Prevent API Abuse)
# ============================================================================

class RateLimiter:
    """Rate limiting for API calls (per user session)"""
    
    @staticmethod
    def rate_limit(max_calls: int = 10, period: int = 60):
        """
        Decorator to rate limit function calls.
        
        Args:
            max_calls: Maximum calls allowed
            period: Time period in seconds
            
        Example:
            @RateLimiter.rate_limit(max_calls=5, period=60)
            def generate_music(prompt):
                # This can only be called 5 times per minute
                pass
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    import streamlit as st
                    
                    # Initialize rate limit tracking
                    if 'rate_limits' not in st.session_state:
                        st.session_state.rate_limits = {}
                    
                    # Get user identifier (session ID)
                    user_key = st.session_state.get('session_id', 'anonymous')
                    func_key = f"{func.__name__}_{user_key}"
                    now = time.time()
                    
                    # Initialize for this function
                    if func_key not in st.session_state.rate_limits:
                        st.session_state.rate_limits[func_key] = []
                    
                    # Remove old timestamps outside the period
                    st.session_state.rate_limits[func_key] = [
                        timestamp for timestamp in st.session_state.rate_limits[func_key]
                        if now - timestamp < period
                    ]
                    
                    # Check if limit exceeded
                    if len(st.session_state.rate_limits[func_key]) >= max_calls:
                        raise Exception(
                            f"⚠️ Rate limit exceeded! Maximum {max_calls} calls "
                            f"per {period} seconds. Please wait."
                        )
                    
                    # Add current timestamp
                    st.session_state.rate_limits[func_key].append(now)
                    
                except ImportError:
                    # Streamlit not available, skip rate limiting
                    pass
                except Exception as e:
                    # Re-raise rate limit errors
                    if "Rate limit exceeded" in str(e):
                        raise
                
                # Execute function
                return func(*args, **kwargs)
            
            return wrapper
        return decorator

# ============================================================================
# ORIGINAL CONFIGURATION (Updated to use SecureKeyManager)
# ============================================================================

# Helper function to get secrets from Streamlit Cloud or .env
def get_secret(key: str, default: str = "") -> str:
    """
    UPDATED: Now uses SecureKeyManager for secure key retrieval.
    
    Priority:
    1. User session keys (per-user, session-only)
    2. Streamlit Cloud secrets (admin defaults, encrypted)
    3. Environment variables from .env (local development)
    4. Default value
    
    Args:
        key: Secret key name
        default: Default value if not found
    
    Returns:
        Secret value or default
    """
    result = SecureKeyManager.get_key(key)
    return result if result is not None else default
    # Fall back to environment variable (for local development)
    return os.getenv(key, default)


class Config:
    """Central configuration class for the application"""
    
    # =============================================================================
    # DIRECTORIES
    # =============================================================================
    BASE_DIR = Path(__file__).parent
    CACHE_DIR = BASE_DIR / "cache"
    TEMP_DIR = BASE_DIR / "temp"
    OUTPUT_DIR = BASE_DIR / "output"
    ASSETS_DIR = BASE_DIR / "assets"
    UPLOAD_DIR = BASE_DIR / "temp"  # For file uploads
    
    # Create directories if they don't exist
    for directory in [CACHE_DIR, TEMP_DIR, OUTPUT_DIR, ASSETS_DIR]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # =============================================================================
    # FILE HANDLING
    # =============================================================================
    
    # Allowed audio file extensions
    ALLOWED_AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"]
    
    # Maximum file upload size (in MB)
    MAX_UPLOAD_SIZE_MB = 100
    
    # =============================================================================
    # API KEYS & CREDENTIALS (ALL FREE OPTIONS)
    # =============================================================================
    
    # LLM Provider API Keys (supports both .env and Streamlit Cloud secrets)
    GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
    TOGETHER_API_KEY = get_secret("TOGETHER_API_KEY", "")
    HUGGINGFACE_TOKEN = get_secret("HUGGINGFACE_TOKEN", "") or get_secret("HF_TOKEN", "")
    OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY", "")
    
    # Cloud Music Generation API Keys
    SUNO_API_TOKEN = get_secret("SUNO_API_TOKEN", "")  # Professional music with vocals!
    REPLICATE_API_TOKEN = get_secret("REPLICATE_API_TOKEN", "")
    
    # Additional FREE Music Generation APIs
    BEATOVEN_API_KEY = get_secret("BEATOVEN_API_KEY", "")  # Text-to-music with mood control
    LOUDLY_API_KEY = get_secret("LOUDLY_API_KEY", "")  # Genre/mood-based generation
    MUSICAPI_KEY = get_secret("MUSICAPI_KEY", "")  # Test tier music generation
    UDIO_API_KEY = get_secret("UDIO_API_KEY", "")  # Early access free tier
    
    # Audio Analysis & Emotion Detection APIs (Cloud-based)
    HUME_API_KEY = get_secret("HUME_API_KEY", "")  # FREE emotion detection
    EDEN_API_KEY = get_secret("EDEN_API_KEY", "")  # Multi-model audio analysis
    AUDD_API_KEY = get_secret("AUDD_API_KEY", "")  # Mood/genre/tempo detection
    
    # Ollama settings (local LLM)
    USE_OLLAMA = get_secret("USE_OLLAMA", "false").lower() == "true"
    OLLAMA_BASE_URL = get_secret("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # =============================================================================
    # LLM CONFIGURATIONS (FREE MODELS PRIORITIZED)
    # =============================================================================
    
    # LLM Provider priority order (fastest to slowest)
    LLM_PRIORITY = [
        "groq",        # Fastest, 30/min free
        "together",    # Fast, $25 free credits
        "huggingface", # Free but slower
        "ollama",      # Local, unlimited
        "openrouter"   # Backup option
    ]
    
    # Model configurations for each provider
    LLM_MODELS = {
        "groq": {
            "default": "llama-3.1-70b-versatile",
            "fast": "llama-3.1-8b-instant",
            "models": [
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma-7b-it"
            ]
        },
        "together": {
            "default": "meta-llama/Llama-3-70b-chat-hf",
            "models": [
                "meta-llama/Llama-3-70b-chat-hf",
                "meta-llama/Llama-3-8b-chat-hf",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-72B-Instruct"
            ]
        },
        "huggingface": {
            "default": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "models": [
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "mistralai/Mixtral-8x7B-Instruct-v0.1"
            ]
        },
        "ollama": {
            "default": "llama3.2",
            "models": [
                "llama3.2",
                "llama3.1",
                "mistral",
                "mixtral",
                "phi3"
            ]
        }
    }
    
    # =============================================================================
    # MUSIC GENERATION SETTINGS
    # =============================================================================
    
    # MusicGen model options (all free, run locally)
    MUSICGEN_MODELS = {
        "small": "facebook/musicgen-small",      # 300M params, faster
        "medium": "facebook/musicgen-medium",    # 1.5B params, balanced
        "large": "facebook/musicgen-large",      # 3.3B params, best quality
        "melody": "facebook/musicgen-melody"     # Conditioned on melody
    }
    
    # Default model
    MUSICGEN_MODEL = os.getenv("MUSICGEN_MODEL", "medium")
    
    # Audio settings
    DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "32000"))
    OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "mp3")
    MAX_DURATION = 120  # seconds
    MIN_DURATION = 15   # seconds
    
    # =============================================================================
    # MUSIC GENRES & MOODS
    # =============================================================================
    
    GENRES = [
        "Pop", "Rock", "Jazz", "Electronic", "Classical",
        "Hip-Hop", "Lo-fi", "Ambient", "R&B", "Country",
        "Reggae", "Blues", "Metal", "Folk", "House"
    ]
    
    MOODS = [
        "Happy", "Sad", "Energetic", "Calm", "Mysterious",
        "Epic", "Romantic", "Melancholic", "Uplifting", "Dark",
        "Playful", "Dramatic", "Peaceful", "Aggressive", "Dreamy"
    ]
    
    INSTRUMENTS = [
        "Piano", "Guitar", "Drums", "Bass", "Synth",
        "Strings", "Brass", "Woodwinds", "Vocals", "Percussion"
    ]
    
    # =============================================================================
    # REMIX & EFFECTS
    # =============================================================================
    
    EFFECTS = [
        "Reverb", "Echo", "Chorus", "Distortion", "Lo-fi Filter",
        "Compressor", "EQ", "Delay", "Flanger", "Phaser"
    ]
    
    # Demucs stem separation models
    DEMUCS_MODELS = {
        "htdemucs": "htdemucs",           # High-quality, 4 stems
        "htdemucs_ft": "htdemucs_ft",     # Fine-tuned
        "htdemucs_6s": "htdemucs_6s",     # 6 stems
        "mdx_extra": "mdx_extra"          # Extra quality
    }
    
    STEM_TYPES = ["vocals", "drums", "bass", "other"]
    
    # =============================================================================
    # PRESET TEMPLATES
    # =============================================================================
    
    PRESETS = {
        "Chill Study Music": {
            "genre": "Lo-fi",
            "mood": "Calm",
            "bpm": 80,
            "duration": 60,
            "instruments": ["Piano", "Synth"]
        },
        "Workout Energy": {
            "genre": "Electronic",
            "mood": "Energetic",
            "bpm": 128,
            "duration": 90,
            "instruments": ["Drums", "Bass", "Synth"]
        },
        "Sleep Ambient": {
            "genre": "Ambient",
            "mood": "Peaceful",
            "bpm": 60,
            "duration": 120,
            "instruments": ["Synth", "Strings"]
        },
        "Creative Flow": {
            "genre": "Jazz",
            "mood": "Uplifting",
            "bpm": 100,
            "duration": 75,
            "instruments": ["Piano", "Bass", "Drums"]
        },
        "Party Vibes": {
            "genre": "Pop",
            "mood": "Happy",
            "bpm": 120,
            "duration": 60,
            "instruments": ["Synth", "Drums", "Bass"]
        }
    }
    
    # =============================================================================
    # UI SETTINGS
    # =============================================================================
    
    APP_TITLE = "🎵 AI Music Remix & Mood Generator"
    APP_ICON = "🎵"
    THEME = "dark"
    
    # Page configuration
    PAGE_CONFIG = {
        "page_title": "AI Music Generator",
        "page_icon": "🎵",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # =============================================================================
    # FILE HANDLING
    # =============================================================================
    
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "100")) * 1024 * 1024  # MB to bytes
    ALLOWED_AUDIO_FORMATS = ["mp3", "wav", "ogg", "flac", "m4a"]
    TEMP_FILE_CLEANUP = int(os.getenv("TEMP_FILE_CLEANUP", "3600"))  # seconds
    
    # =============================================================================
    # CACHE SETTINGS
    # =============================================================================
    
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL = 3600  # seconds
    
    # =============================================================================
    # AUDIO ANALYSIS
    # =============================================================================
    
    ANALYSIS_FEATURES = [
        "tempo", "key", "energy", "danceability",
        "valence", "loudness", "speechiness", "instrumentalness"
    ]
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    @classmethod
    def get_available_llm_provider(cls) -> Optional[str]:
        """Get the first available LLM provider based on priority"""
        for provider in cls.LLM_PRIORITY:
            if provider == "groq" and cls.GROQ_API_KEY:
                return "groq"
            elif provider == "together" and cls.TOGETHER_API_KEY:
                return "together"
            elif provider == "huggingface" and cls.HUGGINGFACE_TOKEN:
                return "huggingface"
            elif provider == "ollama" and cls.USE_OLLAMA:
                return "ollama"
            elif provider == "openrouter" and cls.OPENROUTER_API_KEY:
                return "openrouter"
        return None
    
    @classmethod
    def get_llm_model(cls, provider: str) -> str:
        """Get default model for a provider"""
        return cls.LLM_MODELS.get(provider, {}).get("default", "")
    
    @classmethod
    def is_configured(cls) -> Dict[str, bool]:
        """Check which services are configured (includes session state keys)"""
        # Import here to avoid circular dependency
        try:
            from config import SecureKeyManager
            
            # Helper function to check if key exists (env var OR session state)
            def has_key(provider: str, env_value: str) -> bool:
                return bool(env_value) or SecureKeyManager.has_key(provider)
            
            return {
                # LLM Providers
                "groq": has_key("groq", cls.GROQ_API_KEY),
                "together": has_key("together", cls.TOGETHER_API_KEY),
                "huggingface": has_key("huggingface", cls.HUGGINGFACE_TOKEN),
                "ollama": cls.USE_OLLAMA,
                "openrouter": has_key("openrouter", cls.OPENROUTER_API_KEY),
                # Music Generation Providers
                "suno": has_key("suno", cls.SUNO_API_TOKEN),
                "replicate": has_key("replicate", cls.REPLICATE_API_TOKEN),
                "beatoven": has_key("beatoven", cls.BEATOVEN_API_KEY),
                "loudly": has_key("loudly", cls.LOUDLY_API_KEY),
                "musicapi": has_key("musicapi", cls.MUSICAPI_KEY),
                "udio": has_key("udio", cls.UDIO_API_KEY),
                # Audio Analysis Providers
                "hume": has_key("hume", cls.HUME_API_KEY),
                "eden": has_key("eden", cls.EDEN_API_KEY),
                "audd": has_key("audd", cls.AUDD_API_KEY),
                # Free provider (always available)
                "free": True
            }
        except Exception as e:
            # Fallback to env vars only if SecureKeyManager not available
            return {
                # LLM Providers
                "groq": bool(cls.GROQ_API_KEY),
                "together": bool(cls.TOGETHER_API_KEY),
                "huggingface": bool(cls.HUGGINGFACE_TOKEN),
                "ollama": cls.USE_OLLAMA,
                "openrouter": bool(cls.OPENROUTER_API_KEY),
                # Music Generation Providers
                "suno": bool(cls.SUNO_API_TOKEN),
                "replicate": bool(cls.REPLICATE_API_TOKEN),
                "beatoven": bool(cls.BEATOVEN_API_KEY),
                "loudly": bool(cls.LOUDLY_API_KEY),
                "musicapi": bool(cls.MUSICAPI_KEY),
                "udio": bool(cls.UDIO_API_KEY),
                # Audio Analysis Providers
                "hume": bool(cls.HUME_API_KEY),
                "eden": bool(cls.EDEN_API_KEY),
                "audd": bool(cls.AUDD_API_KEY),
                # Free provider (always available)
                "free": True
            }
    
    @classmethod
    def get_musicgen_model_path(cls, model_size: str = None) -> str:
        """Get MusicGen model path"""
        if model_size is None:
            model_size = cls.MUSICGEN_MODEL
        return cls.MUSICGEN_MODELS.get(model_size, cls.MUSICGEN_MODELS["medium"])
    
    @classmethod
    def get_device(cls) -> str:
        """Get compute device (cuda/cpu)"""
        if TORCH_AVAILABLE:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    
    @classmethod
    def get_system_info(cls) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "device": cls.get_device(),
            "python_version": os.sys.version
        }
        if TORCH_AVAILABLE:
            info.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "pytorch_version": torch.__version__,
            })
        return info


# =============================================================================
# RATE LIMITING & QUOTA TRACKING
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration for API providers"""
    requests_per_minute: int
    requests_per_day: int
    tokens_per_minute: Optional[int] = None
    tokens_per_day: Optional[int] = None


class RateLimiter:
    """Track API usage and enforce rate limits"""
    
    def __init__(self):
        """Initialize rate limiter"""
        self.usage_log = {}
        self.limits = {
            "groq": RateLimitConfig(
                requests_per_minute=30,
                requests_per_day=14400,  # 30/min * 60 * 24
                tokens_per_minute=6000,
                tokens_per_day=None
            ),
            "together": RateLimitConfig(
                requests_per_minute=60,
                requests_per_day=10000,
                tokens_per_minute=None,
                tokens_per_day=None
            ),
            "huggingface": RateLimitConfig(
                requests_per_minute=10,
                requests_per_day=1000,
                tokens_per_minute=None,
                tokens_per_day=None
            ),
            "ollama": RateLimitConfig(
                requests_per_minute=9999,
                requests_per_day=9999,
                tokens_per_minute=None,
                tokens_per_day=None
            )
        }
    
    def can_make_request(self, provider: str) -> bool:
        """
        Check if request can be made
        
        Args:
            provider: LLM provider name
            
        Returns:
            True if request can be made
        """
        if provider not in self.limits:
            return True
        
        if provider not in self.usage_log:
            self.usage_log[provider] = []
        
        now = datetime.now()
        
        # Clean old entries
        self.usage_log[provider] = [
            ts for ts in self.usage_log[provider]
            if now - ts < timedelta(days=1)
        ]
        
        limit = self.limits[provider]
        
        # Check per-minute limit
        recent = [ts for ts in self.usage_log[provider] if now - ts < timedelta(minutes=1)]
        if len(recent) >= limit.requests_per_minute:
            return False
        
        # Check per-day limit
        if len(self.usage_log[provider]) >= limit.requests_per_day:
            return False
        
        return True
    
    def record_request(self, provider: str):
        """Record a request"""
        if provider not in self.usage_log:
            self.usage_log[provider] = []
        self.usage_log[provider].append(datetime.now())
    
    def get_usage_stats(self, provider: str) -> Dict[str, int]:
        """Get usage statistics"""
        if provider not in self.usage_log:
            return {"last_minute": 0, "last_hour": 0, "last_day": 0}
        
        now = datetime.now()
        
        return {
            "last_minute": len([ts for ts in self.usage_log[provider] if now - ts < timedelta(minutes=1)]),
            "last_hour": len([ts for ts in self.usage_log[provider] if now - ts < timedelta(hours=1)]),
            "last_day": len([ts for ts in self.usage_log[provider] if now - ts < timedelta(days=1)])
        }


# =============================================================================
# LLM PROVIDER MANAGER WITH FALLBACK
# =============================================================================

class LLMProviderManager:
    """Manage multiple LLM providers with automatic fallback"""
    
    def __init__(self):
        """Initialize provider manager"""
        self.config = Config()
        self.rate_limiter = RateLimiter()
        self.provider_health = {}
        self.current_provider = None
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        available = []
        
        for provider in self.config.LLM_PRIORITY:
            if provider == "groq" and self.config.GROQ_API_KEY:
                available.append("groq")
            elif provider == "together" and self.config.TOGETHER_API_KEY:
                available.append("together")
            elif provider == "huggingface" and self.config.HUGGINGFACE_TOKEN:
                available.append("huggingface")
            elif provider == "ollama" and self.config.USE_OLLAMA:
                available.append("ollama")
            elif provider == "openrouter" and self.config.OPENROUTER_API_KEY:
                available.append("openrouter")
        
        return available
    
    def get_next_provider(self, exclude: Optional[List[str]] = None) -> Optional[str]:
        """
        Get next available provider with fallback
        
        Args:
            exclude: Providers to exclude
            
        Returns:
            Provider name or None
        """
        exclude = exclude or []
        available = self.get_available_providers()
        
        for provider in available:
            if provider in exclude:
                continue
            
            # Check rate limits
            if not self.rate_limiter.can_make_request(provider):
                continue
            
            # Check health
            if self.provider_health.get(provider, {}).get("failed", 0) > 3:
                continue
            
            return provider
        
        return None
    
    def record_success(self, provider: str):
        """Record successful request"""
        if provider not in self.provider_health:
            self.provider_health[provider] = {"failed": 0, "success": 0}
        
        self.provider_health[provider]["success"] += 1
        self.provider_health[provider]["failed"] = max(0, self.provider_health[provider]["failed"] - 1)
        self.rate_limiter.record_request(provider)
    
    def record_failure(self, provider: str):
        """Record failed request"""
        if provider not in self.provider_health:
            self.provider_health[provider] = {"failed": 0, "success": 0}
        
        self.provider_health[provider]["failed"] += 1
    
    def get_provider_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all providers"""
        stats = {}
        
        for provider in self.get_available_providers():
            stats[provider] = {
                "health": self.provider_health.get(provider, {"failed": 0, "success": 0}),
                "usage": self.rate_limiter.get_usage_stats(provider),
                "can_request": self.rate_limiter.can_make_request(provider)
            }
        
        return stats


# =============================================================================
# AUDIO PROCESSING PRESETS
# =============================================================================

@dataclass
class AudioPreset:
    """Audio processing preset"""
    name: str
    effects: List[str]
    parameters: Dict[str, Any]
    description: str


class AudioPresets:
    """Collection of audio processing presets"""
    
    PRESETS = {
        "Studio Quality": AudioPreset(
            name="Studio Quality",
            effects=["Compressor", "EQ", "Reverb"],
            parameters={
                "compressor": {"threshold_db": -15, "ratio": 3.0},
                "eq": {"bass": 2, "mid": 0, "treble": 1},
                "reverb": {"room_size": 0.3, "wet_level": 0.2}
            },
            description="Professional studio sound"
        ),
        "Lo-fi Chill": AudioPreset(
            name="Lo-fi Chill",
            effects=["Lo-fi Filter", "Reverb", "Chorus"],
            parameters={
                "lo-fi filter": {"cutoff_hz": 3000, "resonance": 0.7},
                "reverb": {"room_size": 0.6, "wet_level": 0.4},
                "chorus": {"rate_hz": 0.5, "depth": 0.3}
            },
            description="Warm, nostalgic lo-fi sound"
        ),
        "Heavy Rock": AudioPreset(
            name="Heavy Rock",
            effects=["Distortion", "Compressor", "EQ"],
            parameters={
                "distortion": {"drive_db": 30},
                "compressor": {"threshold_db": -10, "ratio": 4.0},
                "eq": {"bass": 4, "mid": -1, "treble": 3}
            },
            description="Aggressive rock tone"
        ),
        "Ambient Soundscape": AudioPreset(
            name="Ambient Soundscape",
            effects=["Reverb", "Delay", "Chorus"],
            parameters={
                "reverb": {"room_size": 0.9, "wet_level": 0.6},
                "delay": {"delay_seconds": 1.0, "feedback": 0.5},
                "chorus": {"rate_hz": 0.3, "depth": 0.5}
            },
            description="Spacious, atmospheric sound"
        ),
        "Radio Voice": AudioPreset(
            name="Radio Voice",
            effects=["Compressor", "EQ"],
            parameters={
                "compressor": {"threshold_db": -20, "ratio": 6.0},
                "eq": {"bass": -3, "mid": 5, "treble": 2}
            },
            description="Clear, broadcast-ready voice"
        )
    }
    
    
    @classmethod
    def get_preset(cls, name: str) -> Optional[AudioPreset]:
        """Get preset by name"""
        return cls.PRESETS.get(name)
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """List all preset names"""
        return list(cls.PRESETS.keys())


# =============================================================================
# API KEY MANAGEMENT (SECURE CLOUD-COMPATIBLE SYSTEM)
# =============================================================================
# 
# ⚠️ IMPORTANT: The old APIKeyManager with file-based storage has been REMOVED
# 
# Why? The old system used local file encryption (~/.ai_music_generator/keys.enc)
# which does NOT work on Streamlit Cloud (ephemeral containers) and created
# security risks in multi-user environments (shared filesystem).
#
# New System: SecureKeyManager (defined at top of file)
# - Session-only storage (more secure, no persistence)
# - Cloud-compatible (works on Streamlit Cloud)
# - Per-user isolation (each session separate)
# - Priority fallback: session → secrets → env → none
# - GDPR compliant (no permanent user data storage)
#
# For backward compatibility, APIKeyManager is aliased to SecureKeyManager below.
# All old code using APIKeyManager will automatically use the new secure system.
# 
# =============================================================================


# Export singleton instances

# ============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# ============================================================================
# Wrapper class to maintain compatibility with old APIKeyManager method names

class _APIKeyManagerCompat:
    """
    Backward compatibility wrapper for old APIKeyManager.
    Maps old method names to new SecureKeyManager methods.
    """
    
    @staticmethod
    def get_api_key(provider: str) -> str:
        """Old method name - forwards to SecureKeyManager.get_key()"""
        key = SecureKeyManager.get_key(provider)
        return key if key else ""
    
    @staticmethod
    def save_api_key(provider: str, api_key: str) -> bool:
        """Old method name - forwards to SecureKeyManager.save_key()"""
        return SecureKeyManager.save_key(provider, api_key)
    
    @staticmethod
    def delete_api_key(provider: str) -> bool:
        """Old method name - forwards to SecureKeyManager.clear_key()"""
        return SecureKeyManager.clear_key(provider)
    
    @staticmethod
    def has_key(provider: str) -> bool:
        """Check if key exists - forwards to SecureKeyManager.has_key()"""
        return SecureKeyManager.has_key(provider)
    
    @staticmethod
    def get_key_source(provider: str) -> str:
        """Get key source - forwards to SecureKeyManager.get_key_source()"""
        source = SecureKeyManager.get_key_source(provider)
        return source if source else ""
    
    @staticmethod
    def validate_api_key(provider: str, api_key: str) -> bool:
        """Forwards to SecureKeyManager.validate_api_key()"""
        return SecureKeyManager.validate_api_key(provider, api_key)
    
    @staticmethod
    def get_provider_info():
        """Forwards to SecureKeyManager.get_provider_info()"""
        return SecureKeyManager.get_provider_info()


# Export singleton instances
config = Config()
rate_limiter = RateLimiter()
llm_provider_manager = LLMProviderManager()
audio_presets = AudioPresets()

# Backward compatibility: Old APIKeyManager replaced with SecureKeyManager
# Use the compatibility wrapper to support old method names
APIKeyManager = _APIKeyManagerCompat
api_key_manager = _APIKeyManagerCompat
