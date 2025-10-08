"""
Cloud-Based Audio Analysis - Emotion & Mood Detection APIs
Uses FREE cloud APIs for audio analysis without local processing

Providers:
1. Hume AI - FREE emotion detection API
2. Eden AI - Multi-model audio analysis (FREE tier)
3. Audd.io - Mood/genre/tempo detection (FREE trial)

Total storage: <5MB (just API clients)
Total cost: $0 with free tiers
"""

import os
import io
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import base64

logger = logging.getLogger(__name__)


class CloudAudioAnalyzer:
    """
    Multi-provider cloud audio analysis with automatic fallback.
    Analyzes audio for emotions, mood, genre, and tempo using cloud APIs.
    """
    
    def __init__(self, hume_token: Optional[str] = None, eden_token: Optional[str] = None, 
                 audd_token: Optional[str] = None):
        """
        Initialize cloud audio analyzer.
        
        Args:
            hume_token: Hume AI API token (emotion detection)
            eden_token: Eden AI API token (multi-model analysis)
            audd_token: Audd.io API token (mood/genre/tempo)
        """
        self.hume_token = hume_token or os.getenv("HUME_API_KEY")
        self.eden_token = eden_token or os.getenv("EDEN_API_KEY")
        self.audd_token = audd_token or os.getenv("AUDD_API_KEY")
        
        self.providers = []
        self._initialize_providers()
        
        logger.info(f"Initialized {len(self.providers)} cloud audio analysis providers")
    
    def _initialize_providers(self):
        """Initialize available providers in priority order."""
        
        # Priority 1: Hume AI (Most accurate emotion detection)
        if self.hume_token:
            try:
                provider = HumeAIProvider(self.hume_token)
                self.providers.append(provider)
                logger.info("✅ Hume AI initialized (Emotion detection)")
            except Exception as e:
                logger.warning(f"Hume AI provider failed: {e}")
        
        # Priority 2: Eden AI (Multi-model analysis)
        if self.eden_token:
            try:
                provider = EdenAIProvider(self.eden_token)
                self.providers.append(provider)
                logger.info("✅ Eden AI initialized (Multi-model)")
            except Exception as e:
                logger.warning(f"Eden AI provider failed: {e}")
        
        # Priority 3: Audd.io (Mood/genre/tempo)
        if self.audd_token:
            try:
                provider = AuddProvider(self.audd_token)
                self.providers.append(provider)
                logger.info("✅ Audd.io initialized (Mood/genre)")
            except Exception as e:
                logger.warning(f"Audd provider failed: {e}")
        
        if not self.providers:
            logger.warning("⚠️ No cloud analysis providers available. Analysis will use local processing.")
    
    def analyze_emotion(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze emotions in audio using cloud APIs.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with emotion scores and detected mood
        """
        
        if not self.providers:
            return {
                "emotions": {},
                "dominant_emotion": "neutral",
                "mood": "neutral",
                "confidence": 0.0,
                "provider": "none"
            }
        
        errors = []
        for provider in self.providers:
            try:
                logger.info(f"Analyzing with {provider.name}...")
                
                result = provider.analyze_emotion(audio_path)
                
                logger.info(f"✅ Successfully analyzed with {provider.name}")
                return result
                
            except Exception as e:
                error_msg = f"{provider.name} failed: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
        
        # All providers failed
        logger.warning("All cloud analysis providers failed, returning default")
        return {
            "emotions": {},
            "dominant_emotion": "neutral",
            "mood": "neutral",
            "confidence": 0.0,
            "provider": "none",
            "errors": errors
        }
    
    def analyze_music_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze music features (genre, tempo, key) using cloud APIs.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with music features
        """
        
        if not self.providers:
            return {
                "genre": "unknown",
                "tempo": 0,
                "key": "unknown",
                "energy": 0.0,
                "provider": "none"
            }
        
        errors = []
        for provider in self.providers:
            try:
                logger.info(f"Analyzing features with {provider.name}...")
                
                result = provider.analyze_music_features(audio_path)
                
                logger.info(f"✅ Successfully analyzed features with {provider.name}")
                return result
                
            except Exception as e:
                error_msg = f"{provider.name} failed: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
        
        # All providers failed
        logger.warning("All cloud analysis providers failed, returning default")
        return {
            "genre": "unknown",
            "tempo": 0,
            "key": "unknown",
            "energy": 0.0,
            "provider": "none",
            "errors": errors
        }
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [p.name for p in self.providers]


# =============================================================================
# PROVIDER IMPLEMENTATIONS
# =============================================================================


class HumeAIProvider:
    """Hume AI API provider - Advanced emotion detection (FREE)"""
    
    name = "Hume AI"
    
    def __init__(self, token: str):
        self.token = token
        self.api_url = "https://api.hume.ai/v0/batch/jobs"
    
    def analyze_emotion(self, audio_path: str) -> Dict[str, Any]:
        """Analyze emotions using Hume AI"""
        
        try:
            import requests
            
            headers = {
                "X-Hume-Api-Key": self.token
            }
            
            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Encode as base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Create job
            payload = {
                "models": {
                    "prosody": {}
                },
                "urls": [],
                "text": [],
                "files": [{
                    "filename": Path(audio_path).name,
                    "content": audio_base64
                }]
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code not in [200, 201]:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            job_id = result.get('job_id')
            
            # Poll for results
            import time
            max_attempts = 30
            for _ in range(max_attempts):
                status_response = requests.get(
                    f"{self.api_url}/{job_id}/predictions",
                    headers=headers,
                    timeout=30
                )
                
                if status_response.status_code == 200:
                    predictions = status_response.json()
                    
                    if predictions and len(predictions) > 0:
                        # Extract emotions
                        emotions = predictions[0].get('results', {}).get('predictions', [])
                        
                        if emotions:
                            emotion_dict = emotions[0].get('emotions', {})
                            
                            # Get dominant emotion
                            if emotion_dict:
                                dominant = max(emotion_dict.items(), key=lambda x: x[1])
                                
                                return {
                                    "emotions": emotion_dict,
                                    "dominant_emotion": dominant[0],
                                    "mood": self._map_emotion_to_mood(dominant[0]),
                                    "confidence": dominant[1],
                                    "provider": self.name
                                }
                
                time.sleep(2)
            
            raise Exception("Timeout waiting for results")
            
        except Exception as e:
            raise Exception(f"Hume AI failed: {str(e)}")
    
    def analyze_music_features(self, audio_path: str) -> Dict[str, Any]:
        """Hume AI doesn't provide music feature analysis"""
        return {
            "genre": "unknown",
            "tempo": 0,
            "key": "unknown",
            "energy": 0.0,
            "provider": self.name
        }
    
    @staticmethod
    def _map_emotion_to_mood(emotion: str) -> str:
        """Map Hume AI emotions to music moods"""
        mood_map = {
            "joy": "happy",
            "excitement": "energetic",
            "contentment": "calm",
            "sadness": "sad",
            "anger": "aggressive",
            "fear": "dark",
            "surprise": "mysterious",
            "love": "romantic"
        }
        return mood_map.get(emotion.lower(), "neutral")


class EdenAIProvider:
    """Eden AI API provider - Multi-model audio analysis (FREE tier)"""
    
    name = "Eden AI"
    
    def __init__(self, token: str):
        self.token = token
        self.api_url = "https://api.edenai.run/v2/audio/audio_analysis"
    
    def analyze_emotion(self, audio_path: str) -> Dict[str, Any]:
        """Analyze emotions using Eden AI"""
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.token}"
            }
            
            # Read audio file
            with open(audio_path, 'rb') as f:
                files = {"file": f}
                data = {"providers": "amazon,google"}
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Extract emotions from first available provider
            for provider_key in ["amazon", "google"]:
                if provider_key in result:
                    provider_result = result[provider_key]
                    emotions = provider_result.get("emotions", [])
                    
                    if emotions:
                        # Convert to dict
                        emotion_dict = {e["name"]: e["score"] for e in emotions}
                        dominant = max(emotion_dict.items(), key=lambda x: x[1])
                        
                        return {
                            "emotions": emotion_dict,
                            "dominant_emotion": dominant[0],
                            "mood": self._map_emotion_to_mood(dominant[0]),
                            "confidence": dominant[1],
                            "provider": f"{self.name} ({provider_key})"
                        }
            
            raise Exception("No emotion data in response")
            
        except Exception as e:
            raise Exception(f"Eden AI failed: {str(e)}")
    
    def analyze_music_features(self, audio_path: str) -> Dict[str, Any]:
        """Analyze music features using Eden AI"""
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.token}"
            }
            
            with open(audio_path, 'rb') as f:
                files = {"file": f}
                data = {"providers": "amazon"}
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}")
            
            result = response.json()
            
            return {
                "genre": result.get("genre", "unknown"),
                "tempo": result.get("tempo", 0),
                "key": result.get("key", "unknown"),
                "energy": result.get("energy", 0.0),
                "provider": self.name
            }
            
        except Exception as e:
            raise Exception(f"Eden AI features failed: {str(e)}")
    
    @staticmethod
    def _map_emotion_to_mood(emotion: str) -> str:
        """Map Eden AI emotions to music moods"""
        mood_map = {
            "HAPPY": "happy",
            "EXCITED": "energetic",
            "CALM": "calm",
            "SAD": "sad",
            "ANGRY": "aggressive",
            "FEARFUL": "dark"
        }
        return mood_map.get(emotion.upper(), "neutral")


class AuddProvider:
    """Audd.io API provider - Mood/genre/tempo detection (FREE trial)"""
    
    name = "Audd.io"
    
    def __init__(self, token: str):
        self.token = token
        self.api_url = "https://api.audd.io/"
    
    def analyze_emotion(self, audio_path: str) -> Dict[str, Any]:
        """Audd.io focuses on music features, limited emotion analysis"""
        
        # Get music features and infer mood from genre/tempo
        features = self.analyze_music_features(audio_path)
        
        # Infer mood from tempo and genre
        tempo = features.get("tempo", 120)
        genre = features.get("genre", "unknown").lower()
        
        if tempo > 140:
            mood = "energetic"
        elif tempo < 80:
            mood = "calm"
        elif "sad" in genre or "blues" in genre:
            mood = "sad"
        elif "happy" in genre or "pop" in genre:
            mood = "happy"
        else:
            mood = "neutral"
        
        return {
            "emotions": {},
            "dominant_emotion": mood,
            "mood": mood,
            "confidence": 0.6,
            "provider": self.name
        }
    
    def analyze_music_features(self, audio_path: str) -> Dict[str, Any]:
        """Analyze music features using Audd.io"""
        
        try:
            import requests
            
            with open(audio_path, 'rb') as f:
                files = {"file": f}
                data = {"api_token": self.token, "return": "timecode,apple_music,spotify"}
                
                response = requests.post(
                    self.api_url,
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}")
            
            result = response.json()
            
            if result.get("status") == "success":
                song_data = result.get("result", {})
                
                return {
                    "genre": song_data.get("genre", "unknown"),
                    "tempo": song_data.get("tempo", 0),
                    "key": song_data.get("key", "unknown"),
                    "energy": song_data.get("energy", 0.5),
                    "provider": self.name,
                    "title": song_data.get("title"),
                    "artist": song_data.get("artist")
                }
            
            raise Exception("No recognition data")
            
        except Exception as e:
            raise Exception(f"Audd.io failed: {str(e)}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def analyze_audio_cloud(audio_path: str, 
                         hume_token: Optional[str] = None,
                         eden_token: Optional[str] = None,
                         audd_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze audio with cloud APIs.
    
    Args:
        audio_path: Path to audio file
        hume_token: Hume AI token (optional)
        eden_token: Eden AI token (optional)
        audd_token: Audd.io token (optional)
    
    Returns:
        Combined analysis results
    """
    analyzer = CloudAudioAnalyzer(hume_token, eden_token, audd_token)
    
    emotion_result = analyzer.analyze_emotion(audio_path)
    feature_result = analyzer.analyze_music_features(audio_path)
    
    return {
        **emotion_result,
        **feature_result,
        "cloud_analysis": True
    }


if __name__ == "__main__":
    # Example usage
    print("Cloud Audio Analysis Module")
    print("=" * 50)
    print("Available providers:")
    print("1. Hume AI - https://dev.hume.ai/")
    print("2. Eden AI - https://www.edenai.co/")
    print("3. Audd.io - https://audd.io/")
