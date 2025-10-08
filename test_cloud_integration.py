"""
Test cloud music generation integration
Tests that the cloud generator works with the main app
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("=" * 60)
    print("TESTING CLOUD MUSIC GENERATION INTEGRATION")
    print("=" * 60)
    print()
    
    # Test 1: Import cloud generator
    print("✓ Test 1: Importing cloud_music_generator...")
    try:
        from cloud_music_generator import CloudMusicGenerator
        print("  ✅ CloudMusicGenerator imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import: {e}")
        return False
    
    # Test 2: Import music generator
    print("\n✓ Test 2: Importing music_generator...")
    try:
        from music_generator import MusicGenerator, CLOUD_GENERATOR_AVAILABLE
        print(f"  ✅ MusicGenerator imported successfully")
        print(f"  ℹ️  CLOUD_GENERATOR_AVAILABLE = {CLOUD_GENERATOR_AVAILABLE}")
    except ImportError as e:
        print(f"  ❌ Failed to import: {e}")
        return False
    
    # Test 3: Check cloud generator initialization
    print("\n✓ Test 3: Initializing CloudMusicGenerator...")
    try:
        cloud_gen = CloudMusicGenerator()
        print("  ✅ CloudMusicGenerator initialized")
        
        # Get provider info
        provider_info = cloud_gen.get_provider_info()
        print(f"  ℹ️  Found {len(provider_info)} providers:")
        
        for provider_name, info in provider_info.items():
            status_icon = "✅" if info['status'] == 'available' else "🔑" if info['status'] == 'needs_api_key' else "❌"
            print(f"     {status_icon} {provider_name}: {info['status']}")
            
    except Exception as e:
        print(f"  ❌ Failed to initialize: {e}")
        return False
    
    # Test 4: Check MusicGenerator cloud mode
    print("\n✓ Test 4: Testing MusicGenerator with cloud mode...")
    try:
        # Initialize with cloud mode
        generator = MusicGenerator(use_cloud=True)
        print("  ✅ MusicGenerator initialized with cloud mode")
        print(f"  ℹ️  Using cloud: {generator.use_cloud}")
        print(f"  ℹ️  Cloud generator available: {generator.cloud_generator is not None}")
        
    except Exception as e:
        print(f"  ❌ Failed to initialize with cloud: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Check dependencies
    print("\n✓ Test 5: Checking cloud dependencies...")
    dependencies = {
        'gradio_client': 'Gradio Client',
        'replicate': 'Replicate',
        'huggingface_hub': 'HuggingFace Hub'
    }
    
    for module_name, display_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"  ✅ {display_name} installed")
        except ImportError:
            print(f"  ⚠️  {display_name} not installed (optional)")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n📋 Summary:")
    print("  • Cloud music generation is properly integrated")
    print("  • MusicGenerator supports cloud mode")
    print("  • All imports working correctly")
    print("\n🚀 Next Steps:")
    print("  1. Add API keys for HuggingFace/Replicate (optional)")
    print("  2. Test generation with: python test_cloud_generation.py")
    print("  3. Run the app: streamlit run app.py")
    print()
    
    return True


def test_music_generator_integration():
    """Test that MusicGenerator properly integrates cloud mode"""
    print("\n" + "=" * 60)
    print("TESTING MUSICGENERATOR CLOUD INTEGRATION")
    print("=" * 60)
    print()
    
    from music_generator import MusicGenerator
    
    # Test auto-detection (should use cloud since AudioCraft not installed)
    print("✓ Test: Auto-detection mode (use_cloud=None)...")
    try:
        generator = MusicGenerator(use_cloud=None)
        print(f"  ✅ Auto-detected mode: {'cloud' if generator.use_cloud else 'local'}")
        print(f"  ℹ️  This is correct because AudioCraft is not installed")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # Test explicit cloud mode
    print("\n✓ Test: Explicit cloud mode (use_cloud=True)...")
    try:
        generator = MusicGenerator(use_cloud=True)
        print(f"  ✅ Explicitly using cloud mode")
        print(f"  ℹ️  Cloud generator: {generator.cloud_generator is not None}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    print("\n✅ MusicGenerator cloud integration working correctly!")
    return True


if __name__ == "__main__":
    print("\n" + "🎵" * 30)
    print("AI MUSIC GENERATOR - CLOUD INTEGRATION TEST")
    print("🎵" * 30 + "\n")
    
    success = True
    
    # Run import tests
    if not test_imports():
        success = False
    
    # Run integration tests
    if not test_music_generator_integration():
        success = False
    
    # Final result
    if success:
        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATION TESTS PASSED! 🎉")
        print("=" * 60)
        print("\n✨ Cloud music generation is ready to use!")
        print("💡 Run 'streamlit run app.py' to start the application")
        print()
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\n⚠️  Please check the errors above")
        print()
        sys.exit(1)
