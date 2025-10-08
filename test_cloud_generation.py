"""
Quick test script for cloud music generation.
Tests all three providers and fallback logic.
"""

import sys
import logging
from cloud_music_generator import CloudMusicGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_cloud_generator():
    """Test cloud music generation with all providers"""
    
    print("=" * 80)
    print("CLOUD MUSIC GENERATION TEST")
    print("=" * 80)
    print()
    
    # Initialize cloud generator
    logger.info("Initializing CloudMusicGenerator...")
    cloud_gen = CloudMusicGenerator()
    
    # Get provider info
    print("PROVIDER STATUS:")
    print("-" * 80)
    provider_info = cloud_gen.get_provider_info()
    
    for name, info in provider_info.items():
        status_icon = "✅" if info['status'] == 'available' else "🔑" if info['status'] == 'needs_api_key' else "❌"
        print(f"{status_icon} {name:20} | Status: {info['status']:20} | Free Tier: {info['free_tier']}")
        if info['notes']:
            print(f"   Note: {info['notes']}")
    
    print()
    print("-" * 80)
    print()
    
    # Get available providers
    available = [name for name, info in provider_info.items() if info['status'] == 'available']
    
    if not available:
        print("❌ NO PROVIDERS AVAILABLE")
        print()
        print("To enable cloud music generation, you need at least one API key:")
        print("1. HuggingFace: https://huggingface.co/settings/tokens")
        print("2. Replicate: https://replicate.com/signin")
        print("3. Or use Gradio (community spaces, no key needed - but may be slow/unavailable)")
        print()
        print("Add keys to your .env file:")
        print("  HUGGINGFACE_TOKEN=your_token_here")
        print("  REPLICATE_API_TOKEN=your_token_here")
        return False
    
    print(f"✅ {len(available)} provider(s) available: {', '.join(available)}")
    print()
    
    # Test generation
    print("TESTING MUSIC GENERATION:")
    print("-" * 80)
    
    test_prompt = "upbeat electronic dance music with drums and bass"
    test_duration = 5  # Short for testing
    
    print(f"Prompt: '{test_prompt}'")
    print(f"Duration: {test_duration}s")
    print()
    
    try:
        logger.info("Starting generation...")
        audio = cloud_gen.generate(
            prompt=test_prompt,
            duration=test_duration,
            temperature=1.0,
            guidance_scale=3.0
        )
        
        print()
        print("✅ GENERATION SUCCESSFUL!")
        print(f"   Audio shape: {audio.shape}")
        print(f"   Sample rate: 32000 Hz (standard for MusicGen)")
        print(f"   Duration: {audio.shape[-1] / 32000:.1f} seconds")
        print()
        
        # Try to save
        try:
            import soundfile as sf
            output_path = "test_cloud_output.wav"
            sf.write(output_path, audio.T, 32000)
            print(f"✅ Audio saved to: {output_path}")
            print("   You can play this file to verify the generation!")
        except ImportError:
            print("ℹ️  Install soundfile to save audio: pip install soundfile")
        
        print()
        print("=" * 80)
        print("SUCCESS! Cloud music generation is working! 🎉")
        print("=" * 80)
        return True
        
    except Exception as e:
        print()
        print(f"❌ GENERATION FAILED: {e}")
        print()
        logger.exception("Generation error:")
        return False


if __name__ == "__main__":
    success = test_cloud_generator()
    sys.exit(0 if success else 1)
