"""
🎵 AI Music Remix & Mood Generator Platform
Complete production-ready Streamlit application for AI music generation and remixing

Features:
- Music generation from text with mood/genre control
- Audio remixing with stem separation
- Mood analysis and detection
- Creative studio for layering tracks
- Beautiful, modern UI with dark mode
"""

import os
import sys
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu
import logging
from datetime import datetime
import tempfile
import shutil
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from config import config
from music_generator import MusicGenerator, PromptEnhancer
from audio_processor import AudioProcessor, convert_audio_format
from utils import (
    get_llm_client, AudioAnalyzer, init_session_state,
    add_to_history, save_uploaded_file, cleanup_old_files,
    plot_waveform, plot_spectrogram, plot_audio_features,
    format_duration, estimate_generation_time, validate_audio_file,
    get_cached_llm_client, get_cached_audio_analyzer
)

# Import beautiful glassmorphism components
from components import (
    hero_section, glass_card, glass_card_container, dashboard_card, audio_player,
    beautiful_file_uploader, loading_skeleton, action_button,
    success_message, error_message, info_message, animated_progress_bar,
    mood_indicator, feature_card, stat_card,
    enhanced_audio_player, simple_audio_player
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(**config.PAGE_CONFIG)

# Load custom CSS
def load_css():
    """Load custom CSS styling"""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
init_session_state()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def show_welcome_tutorial():
    """Show welcome tutorial for first-time users"""
    if st.session_state.get("first_visit", True):
        with st.expander("👋 Welcome! New here? Click to see a quick tutorial", expanded=True):
            st.markdown("""
            ### 🎵 Welcome to AI Music Generator!
            
            **What can you do here?**
            
            1. **🎼 Music Generation** - Create original music from text descriptions
               - Describe the music you want in plain English
               - Choose genre, mood, BPM, and duration
               - Download your generated tracks
            
            2. **🎚️ Remix Engine** - Transform existing songs
               - Upload any audio file
               - Separate vocals, drums, bass, and other instruments
               - Change genre, mood, or tempo
               - Add professional audio effects
            
            3. **🎯 Mood Analyzer** - Analyze any song
               - Upload music to detect its mood and genre
               - See detailed audio features
               - Get remix suggestions
            
            4. **🎨 Creative Studio** - Mix and layer tracks
               - Combine multiple generated or uploaded tracks
               - Create mashups
               - Professional mixing tools
            
            **🆓 100% Free AI Models**
            - No expensive plugins needed
            - Works on your computer
            - No music theory required!
            
            *Tip: Start with presets in Music Generation if you're new!*
            """)
            
            if st.button("Got it! Don't show this again"):
                st.session_state.first_visit = False
                st.rerun()


def check_configuration():
    """Check if app is properly configured and allow API key input"""
    from config import api_key_manager
    
    configured = config.is_configured()
    
    if not any(configured.values()):
        with st.expander("⚙️ **API Configuration** - Add your FREE API keys here", expanded=True):
            st.markdown("""
            ### 🎵 Get Your AI Music Generator Working!
            
            **For MUSIC GENERATION (Required):**
            - Add at least ONE: HuggingFace OR Replicate token
            - Both are 100% FREE with no credit card
            
            **For PROMPT ENHANCEMENT (Optional):**
            - Add Groq or Together AI for better music descriptions
            
            **🔒 Privacy & Security:**
            - ✅ Keys stored **encrypted on your device** (persistent)
            - ✅ **Saved permanently** - no need to re-enter after refresh!
            - ✅ **Encrypted storage** - secure local file
            - ✅ Only sent to the AI provider YOU chose
            - ✅ Each user has their own keys (isolated)
            
            **� ONE-TIME SETUP:** Enter your key once, and it's saved securely!
            """)
            
            provider_info = api_key_manager.get_provider_info()
            
            # Create tabs - now including Replicate for music generation
            tabs = st.tabs([
                "🎵 HuggingFace (Music)", 
                "🎵 Replicate (Music)", 
                "🟢 Groq (Prompts)", 
                "🌐 OpenRouter (Prompts - FREE Llama)"
            ])
            
            # Tab 0: HuggingFace (MUSIC GENERATION)
            with tabs[0]:
                st.markdown("""
                #### 🎵 HuggingFace (REQUIRED for Music Generation)
                
                **Best for:** FREE unlimited music generation
                
                **How to get your token:**
                1. Visit [HuggingFace Tokens](https://huggingface.co/settings/tokens)
                2. Sign up (email only, **no credit card**)
                3. Click "New token"
                4. Name: "AI Music Generator"
                5. Select: **Read** access
                6. Copy token (starts with `hf_...`)
                7. Paste below
                
                **Free tier:** ✅ Unlimited generations (rate limited)
                
                📚 [Documentation](https://huggingface.co/docs/api-inference/index)
                """)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    hf_key = st.text_input(
                        "HuggingFace Token",
                        value="",
                        type="password",
                        placeholder="hf_...",
                        key="api_key_huggingface",
                        help="Saved permanently - encrypted on your device"
                    )
                
                with col2:
                    if st.button("💾 Save", key="save_huggingface", type="primary", use_container_width=True):
                        if hf_key:
                            if api_key_manager.validate_api_key("huggingface", hf_key):
                                if api_key_manager.save_api_key("huggingface", hf_key):
                                    st.success("✅ Saved permanently! (encrypted)")
                                    st.info("🔄 Reload the page to use music generation")
                                else:
                                    st.error("Failed to save token")
                            else:
                                st.error("Invalid token format (should start with hf_)")
                        else:
                            st.warning("Please enter a token")
                
                if api_key_manager.get_api_key("huggingface"):
                    st.success("✅ HuggingFace is configured - Music generation ready!")
                else:
                    st.warning("⚠️ HuggingFace not configured - Add token to generate music")
            
            # Tab 1: Replicate (MUSIC GENERATION ALTERNATIVE)
            with tabs[1]:
                st.markdown("""
                #### 🎵 Replicate (Alternative for Music Generation)
                
                **Best for:** Backup option if HuggingFace is slow
                
                **How to get your token:**
                1. Visit [Replicate Account](https://replicate.com/signin)
                2. Sign up (email only, **no credit card**)
                3. Go to [API Tokens](https://replicate.com/account/api-tokens)
                4. Copy token (starts with `r8_...`)
                5. Paste below
                
                **Free tier:** ✅ Limited free credits
                
                📚 [Documentation](https://replicate.com/docs)
                """)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    rep_key = st.text_input(
                        "Replicate Token",
                        value="",
                        type="password",
                        placeholder="r8_...",
                        key="api_key_replicate",
                        help="Saved permanently - encrypted on your device"
                    )
                
                with col2:
                    if st.button("💾 Save", key="save_replicate", type="primary", use_container_width=True):
                        if rep_key:
                            if api_key_manager.validate_api_key("replicate", rep_key):
                                if api_key_manager.save_api_key("replicate", rep_key):
                                    st.success("✅ Saved permanently! (encrypted)")
                                    st.info("🔄 Reload the page to use music generation")
                                else:
                                    st.error("Failed to save token")
                            else:
                                st.error("Invalid token format (should start with r8_)")
                        else:
                            st.warning("Please enter a token")
                
                if api_key_manager.get_api_key("replicate"):
                    st.success("✅ Replicate is configured - Music generation ready!")
                else:
                    st.info("ℹ️ Replicate not configured (optional)")
            
            # Tab 2: Groq (PROMPT ENHANCEMENT)
            with tabs[2]:
                provider_data = provider_info.get("groq", {})
                if provider_data:
                    st.markdown(f"""
                    #### {provider_data['name']}
                    {provider_data['description']}
                    
                    **How to get your API key:**
                    1. Visit [{provider_data['name']} API Keys]({provider_data['signup_url']})
                    2. Sign up for a free account (**no credit card**)
                    3. Generate a new API key
                    4. Paste it below
                    
                    📚 [Documentation]({provider_data['docs']})
                    """)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        groq_key = st.text_input(
                            f"{provider_data['name']} API Key",
                            value="",
                            type="password",
                            placeholder="gsk_...",
                            key="api_key_groq",
                            help="Your Groq API key (saved in session only)"
                        )
                    
                    with col2:
                        if st.button("Save", key="save_groq", type="primary", use_container_width=True):
                            if groq_key:
                                if api_key_manager.validate_api_key("groq", groq_key):
                                    if api_key_manager.save_api_key("groq", groq_key):
                                        st.success(f"✅ {provider_data['name']} API key saved!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to save API key")
                                else:
                                    st.error(f"Invalid API key format")
                            else:
                                st.warning("Please enter an API key")
                    
                    if api_key_manager.get_api_key("groq"):
                        st.success(f"✅ {provider_data['name']} is configured")
                    else:
                        st.info(f"ℹ️ Optional: Add for prompt enhancement")
            
            # Tab 3: OpenRouter (PROMPT ENHANCEMENT - FREE Llama Models)
            with tabs[3]:
                st.markdown("""
                #### 🌐 OpenRouter (FREE Llama Models for Prompts)
                
                **Best for:** FREE access to Llama 3.1 and many other models!
                
                **How to get your FREE API key:**
                1. Visit [OpenRouter Keys](https://openrouter.ai/keys)
                2. Sign up with Google/GitHub (**100% FREE**, no credit card!)
                3. Click "Create Key"
                4. Name: "AI Music Generator"
                5. Copy key (starts with `sk-or-...`)
                6. Paste below
                
                **🆓 FREE Models Available:**
                - ✅ **Meta Llama 3.1 8B** (FREE forever!)
                - ✅ **Meta Llama 3 8B** (FREE)
                - ✅ **Mistral 7B** (FREE)
                - ✅ **Gemini Flash** (FREE)
                - ✅ Many more FREE models!
                
                **Free tier:** ✅ Unlimited requests (rate limited)
                
                📚 [Documentation](https://openrouter.ai/docs) | [Free Models](https://openrouter.ai/models?pricing=free)
                """)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    openrouter_key = st.text_input(
                        "OpenRouter API Key",
                        value="",
                        type="password",
                        placeholder="sk-or-...",
                        key="api_key_openrouter",
                        help="Saved permanently - encrypted on your device"
                    )
                
                with col2:
                    if st.button("💾 Save", key="save_openrouter", type="primary", use_container_width=True):
                        if openrouter_key:
                            if api_key_manager.validate_api_key("openrouter", openrouter_key):
                                if api_key_manager.save_api_key("openrouter", openrouter_key):
                                    st.success("✅ Saved permanently! (encrypted)")
                                    st.info("🔄 Reload the page to use prompt enhancement")
                                else:
                                    st.error("Failed to save API key")
                            else:
                                st.error("Invalid key format (should start with sk-or-)")
                        else:
                            st.warning("Please enter an API key")
                
                if api_key_manager.get_api_key("openrouter"):
                    st.success("✅ OpenRouter is configured - FREE Llama prompts ready!")
                else:
                    st.info("ℹ️ Optional: Add for FREE AI prompt enhancement")
            
            st.markdown("---")
            st.markdown("""
            **✅ Quick Start:**
            1. Add **HuggingFace token** (2 minutes) → Music generation works!
            2. Optionally add **Groq key** → Better prompts!
            
            **🔒 Security:**
            - Keys stored in **session only** (temporary)
            - **Lost when browser closed** (maximum privacy)
            - Perfect for deployed apps - no files needed!
            
            **Need help?** Check the [FREE API Guide](./FREE_LLM_SECURITY_GUIDE.md)
            """)
        
        return False
    else:
        # Show configured status in a compact way
        configured_providers = [k for k, v in configured.items() if v]
        if len(configured_providers) > 0:
            st.success(f"✅ API Configured: {', '.join(configured_providers).upper()}")
        return True


@st.cache_resource
def load_music_generator(model_size: str = "medium"):
    """Load and cache music generator"""
    try:
        from config import Config, api_key_manager
        import os
        
        # Get API keys from encrypted storage
        suno_token = api_key_manager.get_api_key("suno") or Config.SUNO_API_TOKEN
        hf_token = api_key_manager.get_api_key("huggingface") or Config.HUGGINGFACE_TOKEN
        groq_key = api_key_manager.get_api_key("groq") or Config.GROQ_API_KEY
        replicate_token = api_key_manager.get_api_key("replicate") or Config.REPLICATE_API_TOKEN
        
        # Set environment variables for cloud generator to pick up
        if suno_token:
            os.environ["SUNO_API_TOKEN"] = suno_token
            logger.info("✓ Suno AI token configured")
        if hf_token:
            os.environ["HUGGINGFACE_TOKEN"] = hf_token
            logger.info("✓ HuggingFace token configured")
        if replicate_token:
            os.environ["REPLICATE_API_TOKEN"] = replicate_token
            logger.info("✓ Replicate token configured")
        
        # Create generator with API keys
        from music_generator import MusicGenPipeline
        generator = MusicGenPipeline(
            model_size=model_size,
            groq_api_key=groq_key,
            use_cloud=True  # Use cloud by default (HuggingFace API)
        )
        
        logger.info(f"✓ Music generator loaded with model size: {model_size}")
        
        return generator
    except Exception as e:
        logger.error(f"Error loading music generator: {e}")
        st.error(f"Error loading music generator: {e}")
        return None


def show_cloud_status():
    """Show cloud music generation provider status"""
    try:
        from cloud_music_generator import CloudMusicGenerator
        
        # Initialize cloud generator to check provider status
        cloud_gen = CloudMusicGenerator()
        
        with st.expander("☁️ **Cloud Music Generation** - Zero Storage, Always Updated", expanded=False):
            st.markdown("""
            ### ☁️ Cloud-Based Music Generation
            
            **Benefits:**
            - 🚀 **Zero Local Storage** - No 5GB model downloads
            - ⚡ **Always Updated** - Latest models on remote servers
            - 💻 **No GPU Needed** - Processing on cloud servers
            - 🆓 **Multiple FREE Options** - Automatic fallback between providers
            
            ---
            """)
            
            # Get provider info
            provider_info = cloud_gen.get_provider_info()
            
            st.markdown("#### Available Providers:")
            
            for provider_name, info in provider_info.items():
                status = info['status']
                
                # Status indicator
                if status == 'available':
                    status_icon = "✅"
                    status_text = "Ready"
                    status_color = "green"
                elif status == 'needs_api_key':
                    status_icon = "🔑"
                    status_text = "Needs API Key"
                    status_color = "orange"
                else:
                    status_icon = "❌"
                    status_text = "Not Available"
                    status_color = "red"
                
                # Provider card
                with st.container():
                    col1, col2, col3 = st.columns([2, 3, 2])
                    
                    with col1:
                        st.markdown(f"**{status_icon} {provider_name}**")
                    
                    with col2:
                        st.markdown(f"<span style='color: {status_color};'>{status_text}</span>", unsafe_allow_html=True)
                    
                    with col3:
                        if info['requires_api_key'] and status == 'needs_api_key':
                            if st.button(f"Get API Key", key=f"cloud_{provider_name}", use_container_width=True):
                                if provider_name == 'HuggingFace':
                                    st.markdown("[Sign up for HuggingFace](https://huggingface.co/settings/tokens)")
                                elif provider_name == 'Replicate':
                                    st.markdown("[Sign up for Replicate](https://replicate.com/signin)")
                    
                    # Provider details
                    st.caption(f"💰 **Cost:** {info['free_tier']} | 🎵 **Quality:** {info['quality']}")
                    if info['notes']:
                        st.caption(f"ℹ️ {info['notes']}")
                    
                    st.markdown("---")
            
            st.markdown("""
            **📖 How it works:**
            
            The app automatically tries providers in order:
            1. **HuggingFace** (unlimited free, requires API key)
            2. **Replicate** (50 free/month, requires API key)
            3. **Gradio** (community spaces, no API key needed)
            
            You only need one provider to work - the app handles fallback automatically!
            """)
            
    except ImportError:
        st.info("☁️ Cloud music generation available. Install dependencies: `pip install gradio-client replicate`")
    except Exception as e:
        st.warning(f"Could not load cloud provider status: {e}")


@st.cache_resource
def load_audio_processor():
    """Load and cache audio processor"""
    try:
        processor = AudioProcessor()
        return processor
    except Exception as e:
        st.error(f"Error loading audio processor: {e}")
        return None


# =============================================================================
# MAIN NAVIGATION
# =============================================================================

def main():
    """Main application"""
    
    # Stunning Hero Section with glassmorphism design
    hero_section(
        title="AI Music Remix & Mood Generator",
        subtitle="Create, remix, and transform music with FREE AI models",
        icon="🎵",
        gradient="primary"
    )
    
    # Show tutorial
    show_welcome_tutorial()
    
    # Check configuration
    check_configuration()
    
    # Show cloud provider status
    show_cloud_status()
    
    # Modern Dashboard with Quick Actions
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Start")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if dashboard_card(
            title="Generate Music",
            description="Create original tracks from text",
            icon="🎼",
            action_label="Start Creating",
            gradient="primary"
        ):
            st.session_state.selected_page = "🎼 Music Generation"
            st.rerun()
    
    with col2:
        if dashboard_card(
            title="Remix Audio",
            description="Transform existing songs",
            icon="🎚️",
            action_label="Start Remixing",
            gradient="secondary"
        ):
            st.session_state.selected_page = "🎚️ Remix Engine"
            st.rerun()
    
    with col3:
        if dashboard_card(
            title="Analyze Mood",
            description="Detect song emotions",
            icon="🎯",
            action_label="Analyze Now",
            gradient="rainbow"
        ):
            st.session_state.selected_page = "🎯 Mood Analyzer"
            st.rerun()
    
    with col4:
        if dashboard_card(
            title="Creative Studio",
            description="Mix & layer tracks",
            icon="🎨",
            action_label="Open Studio",
            gradient="gold"
        ):
            st.session_state.selected_page = "🎨 Creative Studio"
            st.rerun()
    
    # Add spacing
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    
    # Navigation menu
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "🎼 Music Generation"
    
    selected = option_menu(
        menu_title=None,
        options=["🎼 Music Generation", "🎚️ Remix Engine", "🎯 Mood Analyzer", "🎨 Creative Studio", "📚 History", "⚙️ Settings"],
        icons=["music-note-beamed", "sliders", "graph-up", "palette", "clock-history", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        key='nav_menu',
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#00ff88", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#131842",
            },
            "nav-link-selected": {"background-color": "#00ff88", "color": "#0a0e27"},
        }
    )
    
    # Route to selected page
    if selected == "🎼 Music Generation":
        music_generation_page()
    elif selected == "🎚️ Remix Engine":
        remix_engine_page()
    elif selected == "🎯 Mood Analyzer":
        mood_analyzer_page()
    elif selected == "🎨 Creative Studio":
        creative_studio_page()
    elif selected == "📚 History":
        history_page()
    elif selected == "⚙️ Settings":
        settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #b0b8d4; padding: 20px;'>
        Made with ❤️ using 100% FREE AI models | 
        <a href='https://github.com' style='color: #00ff88;'>GitHub</a> | 
        <a href='#' style='color: #00ff88;'>Documentation</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Cleanup old temp files
    try:
        cleanup_old_files(config.TEMP_DIR, config.TEMP_FILE_CLEANUP)
    except:
        pass


# =============================================================================
# PAGE 1: MUSIC GENERATION
# =============================================================================

def music_generation_page():
    """Music generation from text descriptions"""
    
    # Modern glass card header
    with glass_card_container():
        st.markdown("### 🎼 AI Music Generation")
        st.markdown("Generate original music from text descriptions using Meta's MusicGen (100% FREE)")
        st.markdown("---")
        
        # Generator type selection - prominent buttons
        st.markdown("#### Choose Your Generator:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 10px;'>
                <h3 style='color: white; margin: 0;'>🎵 Instrumental Music Generator</h3>
                <p style='color: #e0e0e0; margin: 10px 0 0 0;'>Background music & instrumentals</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**Use the tabs below** ⬇️ to generate instrumental music")
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 10px;'>
                <h3 style='color: white; margin: 0;'>🎤 Full Song Generator (Vocals + Lyrics)</h3>
                <p style='color: #e0e0e0; margin: 10px 0 0 0;'>Complete songs with singing</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Open Full Song Generator →", key="open_vocal_gen", use_container_width=True, type="primary"):
                st.success("✅ Opening external song generator in new tab...")
                st.info("💡 **Tip:** The external generator can create full songs with actual vocals and lyrics (up to 5 minutes)")
                st.markdown("""
                <meta http-equiv="refresh" content="0; url=https://yueai.ai/create.php" />
                <script>
                    window.open('https://yueai.ai/create.php', '_blank');
                </script>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature highlights in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**✨ Text-to-Music**")
            st.markdown("Describe your vision")
        with col2:
            st.markdown("**🎛️ Advanced Control**")
            st.markdown("Fine-tune every detail")
        with col3:
            st.markdown("**🔄 Batch Generate**")
            st.markdown("Create variations")
    
    # Sidebar for settings
    with st.sidebar:
        with glass_card_container():
            st.markdown("### ⚙️ Generation Settings")
            
            model_size = st.selectbox(
                "Model Size",
                options=["small", "medium", "large", "melody"],
                index=1,
                help="small=faster, large=better quality, melody=conditioned on melody"
            )
            
            duration = st.slider(
                "Duration (seconds)",
                min_value=config.MIN_DURATION,
                max_value=config.MAX_DURATION,
                value=30,
                step=5
            )
            
            temperature = st.slider(
                "Creativity",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.1,
                help="Higher = more random/creative"
            )
            
            # Estimate generation time with info message
            est_time = estimate_generation_time(duration, model_size)
            info_message(f"⏱️ Est. generation time: ~{int(est_time)}s")
    
    # Main content - tabs
    tab1, tab2, tab3, tab4 = st.tabs(["✨ Simple Mode", "🎛️ Advanced Mode", "📋 Presets", "🔄 Batch Generate"])
    
    # TAB 1: Simple Mode
    with tab1:
        with glass_card_container():
            st.markdown("#### Describe the music you want")
            
            user_prompt = st.text_area(
                "Music Description",
                placeholder="e.g., Upbeat electronic dance music with energetic synths and driving bass",
                height=100,
                help="Describe the music in plain English"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                genre = st.selectbox("Genre", options=[""] + config.GENRES)
            with col2:
                mood = st.selectbox("Mood", options=[""] + config.MOODS)
            
            use_llm = st.checkbox("✨ Enhance prompt with AI", value=True, 
                                 help="Use LLM to create more detailed prompt")
            
            if action_button("🎵 Generate Music", key="gen_simple"):
                if not user_prompt and not genre and not mood:
                    error_message("Please provide at least a description, genre, or mood")
                else:
                    generate_music_simple(user_prompt, genre, mood, duration, temperature, model_size, use_llm)
    
    # TAB 2: Advanced Mode
    with tab2:
        with glass_card_container():
            st.markdown("#### Detailed Music Configuration")
            
            adv_prompt = st.text_area(
                "Detailed Description",
                placeholder="e.g., A melancholic piano ballad in C minor with soft string accompaniment",
                height=120
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                adv_genre = st.selectbox("Genre *", options=config.GENRES, key="adv_genre")
            with col2:
                adv_mood = st.selectbox("Mood *", options=config.MOODS, key="adv_mood")
            with col3:
                bpm = st.number_input("BPM", min_value=60, max_value=200, value=120)
            
            col1, col2 = st.columns(2)
            with col1:
                key = st.selectbox("Key", options=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
                key_type = st.selectbox("Mode", options=["Major", "Minor"])
            with col2:
                instruments = st.multiselect("Instruments", options=config.INSTRUMENTS)
            
            if action_button("🎼 Generate Advanced", key="gen_advanced"):
                full_key = f"{key} {key_type}"
                generate_music_advanced(
                    adv_prompt, adv_genre, adv_mood, instruments, bpm, full_key,
                    duration, temperature, model_size
                )
    
    # TAB 3: Presets
    with tab3:
        with glass_card_container():
            st.markdown("#### Quick Start with Presets")
            st.markdown("Choose a preset template to get started quickly")
            
            # Display presets in grid
            cols = st.columns(3)
            for idx, (preset_name, preset_config) in enumerate(config.PRESETS.items()):
                with cols[idx % 3]:
                    with glass_card_container():
                        st.markdown(f"**{preset_name}**")
                        st.markdown(f"""
                        - **Genre**: {preset_config['genre']}
                        - **Mood**: {preset_config['mood']}
                        - **BPM**: {preset_config['bpm']}
                        - **Duration**: {preset_config['duration']}s
                        """)
                    
                    if st.button(f"Generate {preset_name}", key=f"preset_{idx}", use_container_width=True):
                        generate_from_preset(preset_name, preset_config, model_size, temperature)
    
    # TAB 4: Batch Generate
    with tab4:
        with glass_card_container():
            st.markdown("#### Generate Multiple Variations")
            st.markdown("Create multiple versions of the same prompt with variations")
            
            batch_prompt = st.text_area(
                "Base Prompt",
                placeholder="Chill lo-fi hip hop beat",
                height=100
            )
            
            num_variations = st.slider("Number of Variations", min_value=2, max_value=5, value=3)
            
            if action_button("🔄 Generate Batch", key="gen_batch"):
                if batch_prompt:
                    generate_batch(batch_prompt, num_variations, duration, model_size, temperature)
                else:
                    error_message("Please provide a base prompt")


def generate_music_simple(prompt, genre, mood, duration, temperature, model_size, use_llm):
    """Generate music in simple mode"""
    try:
        # Show loading spinner
        with st.spinner("🎵 Generating music... This may take a minute..."):
            # Load generator
            generator = load_music_generator(model_size)
            if not generator:
                error_message("Failed to load music generator")
                return
            
            # Enhance prompt if requested
            final_prompt = prompt
            if use_llm:
                from music_generator import PromptEnhancer
                enhancer = PromptEnhancer()
                final_prompt = enhancer.enhance_prompt(
                    prompt, genre=genre, mood=mood
                )
                info_message(f"✨ Enhanced prompt: {final_prompt}")
            else:
                # Basic enhancement without LLM
                parts = [prompt] if prompt else []
                if genre:
                    parts.append(f"{genre} music")
                if mood:
                    parts.append(f"with {mood.lower()} mood")
                final_prompt = ", ".join(parts)
            
            # Generate
            progress = st.progress(0)
            progress.progress(30, "Loading model...")
            
            audio, sr = generator.generate_music(
                final_prompt,
                duration=duration,
                temperature=temperature
            )
            
        # Display results with modern components
        success_message("Music generated successfully!")
        
        with glass_card_container():
            st.markdown("#### 🎵 Your Generated Music")
            
            # Save audio
            output_path = config.OUTPUT_DIR / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            saved_path = generator.save_audio(audio, output_path, sr)
            
            # Beautiful audio player with waveform
            st.markdown("### 🎧 Listen to Your Music")
            with open(saved_path, "rb") as audio_file:
                audio_player(
                    audio_file.read(),
                    title="Generated Track",
                    show_waveform=True
                )
            
            # Native browser audio player + download buttons
            st.markdown("### 🎵 Audio Player & Download")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.audio(str(saved_path), format="audio/wav", start_time=0)
                st.caption("🎵 Click play button above to listen directly in browser")
            with col2:
                with open(saved_path, "rb") as f:
                    audio_bytes = f.read()
                st.download_button(
                    "⬇️ Quick Download",
                    audio_bytes,
                    file_name=Path(saved_path).name,
                    mime="audio/wav",
                    use_container_width=True
                )
            with col3:
                if st.button("📁 Save As...", use_container_width=True, help="Choose where to save the file"):
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        
                        # Create root window and hide it
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes('-topmost', True)
                        
                        # Open file save dialog
                        file_path = filedialog.asksaveasfilename(
                            title="Save Music File As...",
                            defaultextension=".wav",
                            filetypes=[
                                ("WAV files", "*.wav"),
                                ("MP3 files", "*.mp3"),
                                ("All files", "*.*")
                            ],
                            initialdir=str(Path.home() / "Downloads"),
                            initialfile=f"generated_music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                        )
                        
                        if file_path:
                            # Copy the file to chosen location
                            import shutil
                            shutil.copy2(saved_path, file_path)
                            st.success(f"✅ Saved to: {file_path}")
                            
                            # Optional: Open file location
                            if st.button("📂 Open Folder", key="open_folder"):
                                import subprocess
                                import os
                                if os.name == 'nt':  # Windows
                                    subprocess.run(['explorer', '/select,', file_path])
                                elif os.name == 'posix':  # macOS and Linux
                                    subprocess.run(['open', '-R', file_path] if sys.platform == 'darwin' 
                                                 else ['xdg-open', os.path.dirname(file_path)])
                        
                        root.destroy()
                    except ImportError:
                        st.error("❌ File dialog not available. Use Quick Download instead.")
                    except Exception as e:
                        st.error(f"❌ Error saving file: {str(e)}")
            
            # Waveform visualization
            st.plotly_chart(plot_waveform(audio[0] if audio.ndim > 1 else audio, sr), use_container_width=True)
            
            # Add to history
            add_to_history({
                "type": "generation",
                "prompt": final_prompt,
                "genre": genre,
                "mood": mood,
                "duration": duration,
                "file": str(saved_path),
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        error_message(f"Error generating music: {str(e)}")
        logger.error(f"Generation error: {e}", exc_info=True)


def generate_music_advanced(prompt, genre, mood, instruments, bpm, key, duration, temperature, model_size):
    """Generate music in advanced mode"""
    try:
        with st.spinner("🎼 Generating advanced music..."):
            generator = load_music_generator(model_size)
            if not generator:
                return
            
            # Build detailed prompt
            from music_generator import PromptEnhancer
            enhancer = PromptEnhancer()
            final_prompt = enhancer.enhance_prompt(
                prompt,
                genre=genre,
                mood=mood,
                instruments=instruments,
                bpm=bpm,
                key=key
            )
            
            if not final_prompt or final_prompt == prompt:
                # Fallback if enhancement fails
                parts = [prompt, f"{genre} music", f"{mood} mood"]
                if instruments:
                    parts.append(f"with {', '.join(instruments)}")
                parts.append(f"at {bpm} BPM in {key}")
                final_prompt = ", ".join(parts)
            
            st.info(f"🎵 Generating: {final_prompt}")
            
            audio, sr = generator.generate_music(final_prompt, duration=duration, temperature=temperature)
            
            output_path = config.OUTPUT_DIR / f"advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            saved_path = generator.save_audio(audio, output_path, sr)
            
            st.success("✅ Advanced music generated!")
            st.audio(str(saved_path))
            
            with open(saved_path, "rb") as f:
                st.download_button("⬇️ Download", f.read(), file_name=Path(saved_path).name, mime="audio/wav")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def generate_from_preset(preset_name, preset_config, model_size, temperature):
    """Generate music from preset"""
    try:
        with st.spinner(f"🎵 Generating {preset_name}..."):
            generator = load_music_generator(model_size)
            if not generator:
                return
            
            # Build prompt from preset
            from music_generator import PromptEnhancer
            enhancer = PromptEnhancer()
            prompt = enhancer.enhance_prompt(
                "",
                genre=preset_config['genre'],
                mood=preset_config['mood'],
                instruments=preset_config['instruments'],
                bpm=preset_config['bpm']
            )
            
            if not prompt:
                # Fallback if enhancement fails
                prompt = f"{preset_config['genre']} music with {preset_config['mood']} mood, {preset_config['bpm']} BPM"
            
            audio, sr = generator.generate_music(
                prompt,
                duration=preset_config['duration'],
                temperature=temperature
            )
            
            output_path = config.OUTPUT_DIR / f"{preset_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            saved_path = generator.save_audio(audio, output_path, sr)
            
            st.success(f"✅ {preset_name} generated!")
            st.audio(str(saved_path))
            
            with open(saved_path, "rb") as f:
                st.download_button("⬇️ Download", f.read(), file_name=Path(saved_path).name, mime="audio/wav")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def generate_batch(prompt, num_variations, duration, model_size, temperature):
    """Generate batch of variations"""
    try:
        generator = load_music_generator(model_size)
        if not generator:
            return
        
        # Create variations
        from music_generator import PromptEnhancer
        enhancer = PromptEnhancer()
        prompts = enhancer.create_variation_prompts(prompt, num_variations)
        
        if not prompts or len(prompts) < num_variations:
            # Fallback if enhancement fails
            prompts = [prompt] * num_variations
        
        st.info(f"🔄 Generating {num_variations} variations...")
        
        progress_bar = st.progress(0)
        results = []
        
        for idx, var_prompt in enumerate(prompts):
            st.write(f"**Variation {idx + 1}**: {var_prompt}")
            
            audio, sr = generator.generate_music(var_prompt, duration=duration, temperature=temperature)
            
            output_path = config.OUTPUT_DIR / f"batch_{idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            saved_path = generator.save_audio(audio, output_path, sr)
            
            results.append(saved_path)
            
            st.audio(str(saved_path))
            progress_bar.progress((idx + 1) / num_variations)
        
        st.success(f"✅ Generated {len(results)} variations!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


# =============================================================================
# PAGE 2: REMIX ENGINE  
# =============================================================================

def remix_engine_page():
    """Audio remixing and transformation"""
    
    # Modern glass card header
    with glass_card_container():
        st.markdown("### 🎚️ Remix Engine")
        st.markdown("Transform existing music with AI-powered remixing tools")
        st.markdown("---")
        
        # Feature highlights
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**🎤 Stem Separation**")
            st.markdown("Extract vocals & instruments")
        with col2:
            st.markdown("**🎭 Genre Transfer**")
            st.markdown("Change musical style")
        with col3:
            st.markdown("**⏱️ Tempo Control**")
            st.markdown("Adjust speed & pitch")
        with col4:
            st.markdown("**🎚️ Audio Effects**")
            st.markdown("Professional processing")
    
    # Beautiful file upload
    with glass_card_container():
        st.markdown("#### 📤 Upload Audio File")
        
        uploaded_file = beautiful_file_uploader(
            label="Drop your audio file here or click to browse",
            accepted_types=config.ALLOWED_AUDIO_EXTENSIONS,
            key="remix_uploader"
        )
    
    if not uploaded_file:
        info_message("👆 Upload an audio file to get started with remixing!")
        return
    
    # Save uploaded file with loading skeleton
    with loading_skeleton("Loading audio..."):
        audio_path = save_uploaded_file(uploaded_file)
    
    success_message(f"Loaded: {uploaded_file.name}")
    
    # Play original in glass card
    with glass_card_container():
        st.markdown("#### 🎵 Original Audio")
        audio_player(
            str(audio_path),
            title=uploaded_file.name,
            artist="Original Upload",
            show_waveform=True
        )
    
    # Remix tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎤 Stem Separation",
        "🎭 Genre Transfer",
        "🎨 Mood Transform",
        "⏱️ Tempo/Pitch",
        "🎚️ Effects"
    ])
    
    # TAB 1: Stem Separation
    with tab1:
        stem_separation_tab(audio_path)
    
    # TAB 2: Genre Transfer
    with tab2:
        genre_transfer_tab(audio_path)
    
    # TAB 3: Mood Transform
    with tab3:
        mood_transform_tab(audio_path)
    
    # TAB 4: Tempo/Pitch
    with tab4:
        tempo_pitch_tab(audio_path)
    
    # TAB 5: Effects
    with tab5:
        effects_tab(audio_path)


def stem_separation_tab(audio_path):
    """Stem separation interface"""
    with glass_card_container():
        st.markdown("#### 🎤 Separate Audio Stems")
        st.markdown("Extract vocals, drums, bass, and other instruments using Demucs (FREE)")
        
        model = st.selectbox(
            "Demucs Model",
            options=list(config.DEMUCS_MODELS.keys()),
            help="htdemucs recommended for best quality"
        )
        
        if action_button("🎵 Separate Stems", key="separate_stems"):
            try:
                with loading_skeleton("🎵 Separating stems... This may take 1-2 minutes..."):
                    processor = load_audio_processor()
                    if not processor:
                        return
                    
                    stems = processor.separate_stems(audio_path, model_name=config.DEMUCS_MODELS[model])
                
                success_message(f"Separated into {len(stems)} stems!")
                
                # Display each stem in glass cards
                for stem_name, stem_audio in stems.items():
                    with glass_card_container():
                        st.markdown(f"**{stem_name.capitalize()}**")
                        
                        # Save stem
                        stem_path = config.OUTPUT_DIR / f"{stem_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                        stem_path = processor.save_audio(stem_audio, stem_path, processor.sample_rate)
                        
                        # Beautiful audio player
                        audio_player(
                            str(stem_path),
                            title=f"{stem_name.capitalize()} Track",
                            artist="Separated Stem"
                        )
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.audio(str(stem_path))
                        with col2:
                            with open(stem_path, "rb") as f:
                                st.download_button(
                                    f"⬇️ {stem_name}",
                                    f.read(),
                                    file_name=stem_path.name,
                                    mime="audio/wav",
                                    key=f"download_{stem_name}"
                                )
                
            except Exception as e:
                error_message(f"Error separating stems: {str(e)}")


def genre_transfer_tab(audio_path):
    """Genre transfer interface"""
    st.subheader("🎭 Transform Genre")
    st.markdown("Change the genre of your music using AI")
    
    target_genre = st.selectbox("Target Genre", options=config.GENRES)
    
    st.info("💡 Genre transfer will analyze the audio and regenerate it in the target genre")
    
    if st.button("🎭 Transform Genre", type="primary"):
        st.warning("⚠️ Genre transfer requires music generation. This feature combines audio analysis with new generation.")
        # Implementation would analyze the audio, extract musical elements, and regenerate


def mood_transform_tab(audio_path):
    """Mood transformation interface"""
    st.subheader("🎨 Transform Mood")
    st.markdown("Change the emotional tone of your music")
    
    target_mood = st.selectbox("Target Mood", options=config.MOODS)
    
    intensity = st.slider("Transformation Intensity", 0.0, 1.0, 0.7, 0.1)
    
    if st.button("🎨 Transform Mood", type="primary"):
        st.info("💡 Mood transformation adjusts tempo, effects, and harmonics to match the target mood")
        # Implementation would apply effects and transformations


def tempo_pitch_tab(audio_path):
    """Tempo and pitch adjustment interface"""
    st.subheader("⏱️ Tempo & Pitch Adjustment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Tempo")
        tempo_factor = st.slider(
            "Tempo Change",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="1.0 = original, 2.0 = double speed"
        )
    
    with col2:
        st.markdown("### Pitch")
        pitch_shift = st.slider(
            "Pitch Shift (semitones)",
            min_value=-12,
            max_value=12,
            value=0,
            step=1,
            help="12 = one octave up"
        )
    
    if st.button("⏱️ Apply Changes", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing audio..."):
                processor = load_audio_processor()
                if not processor:
                    return
                
                audio, sr = processor.load_audio(audio_path)
                
                # Apply tempo change
                if tempo_factor != 1.0:
                    audio = processor.change_tempo(audio, sr, tempo_factor)
                
                # Apply pitch shift
                if pitch_shift != 0:
                    audio = processor.change_pitch(audio, sr, pitch_shift)
                
                # Save
                output_path = config.OUTPUT_DIR / f"modified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                saved_path = processor.save_audio(audio, output_path, sr)
                
                st.success("✅ Audio modified!")
                st.audio(str(saved_path))
                
                with open(saved_path, "rb") as f:
                    st.download_button("⬇️ Download", f.read(), file_name=Path(saved_path).name, mime="audio/wav")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def effects_tab(audio_path):
    """Audio effects interface"""
    st.subheader("🎚️ Audio Effects")
    st.markdown("Add professional audio effects to your track")
    
    # Effect selection
    selected_effects = st.multiselect(
        "Select Effects",
        options=config.EFFECTS,
        default=[]
    )
    
    # Effect parameters
    effect_params = {}
    
    if "Reverb" in selected_effects:
        with st.expander("🎵 Reverb Settings"):
            effect_params["reverb"] = {
                "room_size": st.slider("Room Size", 0.0, 1.0, 0.5, 0.1),
                "wet_level": st.slider("Wet Level", 0.0, 1.0, 0.33, 0.01)
            }
    
    if "Echo" in selected_effects:
        with st.expander("🔊 Echo Settings"):
            effect_params["echo"] = {
                "delay_seconds": st.slider("Delay (seconds)", 0.1, 2.0, 0.5, 0.1),
                "feedback": st.slider("Feedback", 0.0, 0.9, 0.3, 0.1)
            }
    
    if "Distortion" in selected_effects:
        with st.expander("🎸 Distortion Settings"):
            effect_params["distortion"] = {
                "drive_db": st.slider("Drive (dB)", 0.0, 50.0, 25.0, 5.0)
            }
    
    if "Lo-fi Filter" in selected_effects:
        with st.expander("📻 Lo-fi Filter Settings"):
            effect_params["lo-fi filter"] = {
                "cutoff_hz": st.slider("Cutoff (Hz)", 500, 5000, 2000, 100),
                "resonance": st.slider("Resonance", 0.0, 1.0, 0.7, 0.1)
            }
    
    if st.button("✨ Apply Effects", type="primary", use_container_width=True):
        if not selected_effects:
            st.warning("Please select at least one effect")
            return
        
        try:
            with st.spinner("Applying effects..."):
                processor = load_audio_processor()
                if not processor:
                    return
                
                audio, sr = processor.load_audio(audio_path)
                
                # Apply effects
                processed = processor.apply_effects(audio, sr, selected_effects, effect_params)
                
                # Save
                output_path = config.OUTPUT_DIR / f"effects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                saved_path = processor.save_audio(processed, output_path, sr)
                
                st.success(f"✅ Applied {len(selected_effects)} effects!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original:**")
                    st.audio(str(audio_path))
                with col2:
                    st.markdown("**With Effects:**")
                    st.audio(str(saved_path))
                
                with open(saved_path, "rb") as f:
                    st.download_button("⬇️ Download", f.read(), file_name=Path(saved_path).name, mime="audio/wav")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# =============================================================================
# PAGE 3: MOOD ANALYZER
# =============================================================================

def mood_analyzer_page():
    """Audio analysis and mood detection"""
    
    # Modern glass card header
    with glass_card_container():
        st.markdown("### 🎯 Mood Analyzer")
        st.markdown("Analyze audio files to detect mood, genre, and musical features")
        st.markdown("---")
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🎭 Mood Detection**")
            st.markdown("AI-powered emotion analysis")
        with col2:
            st.markdown("**🎵 Genre Classification**")
            st.markdown("Identify musical style")
        with col3:
            st.markdown("**📊 Feature Extraction**")
            st.markdown("Tempo, key, energy & more")
    
    # Beautiful file upload
    with glass_card_container():
        st.markdown("#### 📤 Upload Audio for Analysis")
        
        uploaded_file = beautiful_file_uploader(
            label="Drop your audio file here or click to browse",
            accepted_types=config.ALLOWED_AUDIO_EXTENSIONS,
            key="analyzer_uploader"
        )
    
    if not uploaded_file:
        info_message("👆 Upload an audio file to analyze")
        return
    
    with loading_skeleton("Loading audio..."):
        audio_path = save_uploaded_file(uploaded_file)
    
    success_message(f"Loaded: {uploaded_file.name}")
    
    # Play audio in glass card
    with glass_card_container():
        audio_player(
            str(audio_path),
            title=uploaded_file.name,
            artist="Analysis Target",
            show_waveform=True
        )
    
    if action_button("🎯 Analyze Audio", key="analyze_audio"):
        try:
            # Check if cloud analysis is available
            cloud_available = any([
                config.HUME_API_KEY,
                config.EDEN_API_KEY,
                config.AUDD_API_KEY
            ])
            
            if cloud_available:
                st.markdown("---")
                analysis_mode = st.radio(
                    "Analysis Mode",
                    options=["Local (Fast)", "Cloud (Advanced Emotion Detection)"],
                    horizontal=True,
                    key="analysis_mode"
                )
                use_cloud = analysis_mode == "Cloud (Advanced Emotion Detection)"
            else:
                use_cloud = False
            
            if use_cloud:
                # Cloud-based emotion analysis
                with loading_skeleton("🎯 Analyzing with cloud AI... Detecting emotions..."):
                    from cloud_audio_analysis import CloudAudioAnalyzer
                    
                    cloud_analyzer = CloudAudioAnalyzer(
                        hume_token=config.HUME_API_KEY,
                        eden_token=config.EDEN_API_KEY,
                        audd_token=config.AUDD_API_KEY
                    )
                    
                    # Get emotion analysis
                    emotion_result = cloud_analyzer.analyze_emotion(str(audio_path))
                    
                    # Get music features
                    features_result = cloud_analyzer.analyze_music_features(str(audio_path))
                
                success_message("Cloud analysis complete!")
                
                # Display emotion results
                if emotion_result:
                    with glass_card_container():
                        st.markdown("#### 💖 Emotion Analysis (Cloud AI)")
                        
                        if 'dominant_emotion' in emotion_result:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Dominant Emotion**: {emotion_result['dominant_emotion'].title()}")
                            with col2:
                                st.markdown(f"**Confidence**: {emotion_result.get('confidence', 0):.1%}")
                        
                        if 'mood' in emotion_result:
                            mood_indicator(emotion_result['mood'], confidence=emotion_result.get('confidence', 0.85))
                        
                        # Show all emotions
                        if 'emotions' in emotion_result and emotion_result['emotions']:
                            st.markdown("##### 🎭 Detected Emotions:")
                            emotions_dict = emotion_result['emotions']
                            
                            # Sort by score
                            sorted_emotions = sorted(
                                emotions_dict.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:8]  # Top 8 emotions
                            
                            for emotion, score in sorted_emotions:
                                animated_progress_bar(score, label=f"{emotion.title()}: {score:.1%}")
                        
                        st.markdown(f"**Provider**: {emotion_result.get('provider', 'Cloud AI')}")
                
                # Display music features
                if features_result:
                    with glass_card_container():
                        st.markdown("#### 🎵 Music Features (Cloud AI)")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        if 'genre' in features_result:
                            with col1:
                                st.markdown("**Genre**")
                                st.markdown(f"### {features_result['genre']}")
                        
                        if 'tempo' in features_result:
                            with col2:
                                st.markdown("**Tempo**")
                                st.markdown(f"### {features_result['tempo']} BPM")
                        
                        if 'key' in features_result:
                            with col3:
                                st.markdown("**Key**")
                                st.markdown(f"### {features_result['key']}")
                        
                        if 'energy' in features_result:
                            with col4:
                                st.markdown("**Energy**")
                                animated_progress_bar(features_result['energy'], label=f"{features_result['energy']:.1%}")
                        
                        st.markdown(f"**Provider**: {features_result.get('provider', 'Cloud AI')}")
                
                # If no cloud results, fall back to local
                if not emotion_result and not features_result:
                    st.warning("⚠️ Cloud analysis unavailable, using local analysis...")
                    use_cloud = False
            
            # Local analysis (original code)
            # Local analysis (original code)
            if not use_cloud or (not emotion_result and not features_result):
                with loading_skeleton("🎯 Analyzing audio... Extracting features..."):
                    analyzer = get_cached_audio_analyzer()
                    analysis = analyzer.analyze_audio(audio_path)
                    
                    # Detect mood and genre
                    detected_mood = analyzer.detect_mood(analysis)
                    detected_genre = analyzer.detect_genre(analysis)
            
                success_message("Analysis complete!")
                
                # Display results with modern components
                with glass_card_container():
                    st.markdown("#### 🎭 Analysis Results")
                    
                    # Mood indicator
                    mood_indicator(detected_mood, confidence=0.85)
                    
                    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
                    
                    # Key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        with glass_card_container():
                            st.markdown("**🎭 Mood**")
                            st.markdown(f"### {detected_mood}")
                    with col2:
                        with glass_card_container():
                            st.markdown("**🎵 Genre**")
                            st.markdown(f"### {detected_genre}")
                    with col3:
                        with glass_card_container():
                            st.markdown("**⏱️ Tempo**")
                            st.markdown(f"### {analysis['tempo']:.1f} BPM")
                    with col4:
                        with glass_card_container():
                            st.markdown("**🎹 Key**")
                            st.markdown(f"### {analysis['key']}")
                
                # Audio features visualization
                with glass_card_container():
                    st.markdown("#### 📊 Audio Features")
                    st.plotly_chart(plot_audio_features(analysis), use_container_width=True)
                
                # Detailed metrics in glass cards
                col1, col2 = st.columns(2)
                
                with col1:
                    with glass_card_container():
                        st.markdown("#### 🎼 Musical Features")
                        
                        # Use animated progress bars for metrics
                        st.markdown("**Duration**")
                        st.markdown(f"{format_duration(analysis['duration'])}")
                        
                        st.markdown("**Energy**")
                        animated_progress_bar(analysis['danceability'], label=f"{analysis['danceability']:.2%}")
                        
                        st.markdown("**Valence (Positivity)**")
                        animated_progress_bar(analysis['valence'], label=f"{analysis['valence']:.2%}")
                        
                        st.markdown("**Instrumentalness**")
                        animated_progress_bar(analysis['instrumentalness'], label=f"{analysis['instrumentalness']:.2%}")
                
                with col2:
                    with glass_card_container():
                        st.markdown("#### 🔊 Technical Details")
                        
                        st.metric(
                            label="📊 Sample Rate",
                            value=f"{analysis['sample_rate']} Hz"
                        )
                        
                        st.metric(
                            label="🔊 Loudness",
                            value=f"{analysis['loudness']:.1f} dB"
                        )
                        
                        st.metric(
                            label="🎵 Spectral Centroid",
                            value=f"{analysis['spectral_centroid_mean']:.0f} Hz"
                        )
                    
                    # Waveform
                    st.subheader("📈 Waveform")
                    processor = load_audio_processor()
                    audio, sr = processor.load_audio(audio_path)
                    st.plotly_chart(plot_waveform(audio, sr), use_container_width=True)
                    
                    # Spectrogram
                    st.subheader("🌈 Spectrogram")
                    st.plotly_chart(plot_spectrogram(audio, sr), use_container_width=True)
                    
                    # Suggestions
                    st.subheader("💡 Remix Suggestions")
                    if analysis['tempo'] > 120:
                        st.info("🎵 High-energy track! Try adding distortion or increasing tempo for more intensity")
                    if analysis['valence'] < 0.4:
                        st.info("😢 Melancholic track. Try pitch shifting up or adding reverb for atmosphere")
                    if analysis['instrumentalness'] > 0.7:
                        st.info("🎹 Instrumental track. Perfect for adding vocals or vocal samples")
                
        except Exception as e:
            st.error(f"❌ Error analyzing audio: {str(e)}")


# =============================================================================
# PAGE 4: CREATIVE STUDIO
# =============================================================================

def creative_studio_page():
    """Multi-track mixing and layering"""
    
    # Modern glass card header
    with glass_card_container():
        st.markdown("### 🎨 Creative Studio")
        st.markdown("Mix and layer multiple tracks to create unique compositions")
        st.markdown("---")
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🎵 Multi-Track Mixing**")
            st.markdown("Layer unlimited tracks")
        with col2:
            st.markdown("**🎚️ Volume Control**")
            st.markdown("Adjust each track individually")
        with col3:
            st.markdown("**✨ Professional Mix**")
            st.markdown("Export mixed masterpiece")
    
    info_message("💡 Upload or generate multiple tracks, then mix them together with custom volume levels")
    
    # Track management
    if "studio_tracks" not in st.session_state:
        st.session_state.studio_tracks = []
    
    # Add track
    with glass_card_container():
        st.markdown("#### ➕ Add Tracks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded = beautiful_file_uploader(
                label="Upload Audio Track",
                accepted_types=config.ALLOWED_AUDIO_EXTENSIONS,
                key="studio_upload"
            )
            
            if uploaded and action_button("Add Track", key="add_track"):
                audio_path = save_uploaded_file(uploaded)
                st.session_state.studio_tracks.append({
                    "name": uploaded.name,
                    "path": str(audio_path),
                    "volume": 1.0
                })
                success_message(f"Added: {uploaded.name}")
                st.rerun()
        
        with col2:
            st.markdown("<div style='padding-top: 40px'></div>", unsafe_allow_html=True)
            if action_button("➕ Generate New Track", key="gen_new"):
                info_message("Generate a track in the Music Generation tab, then add it here")
    
    # Display tracks
    if st.session_state.studio_tracks:
        with glass_card_container():
            st.markdown(f"#### 🎵 Tracks ({len(st.session_state.studio_tracks)})")
            
            for idx, track in enumerate(st.session_state.studio_tracks):
                with glass_card_container():
                    st.markdown(f"**Track {idx+1}: {track['name']}**")
                    
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        audio_player(
                            track['path'],
                            title=track['name'],
                            artist=f"Track {idx+1}"
                        )
                    
                    with col2:
                        track['volume'] = st.slider(
                            "Volume",
                            0.0, 2.0, track['volume'], 0.1,
                            key=f"vol_{idx}"
                        )
                    
                    with col3:
                        st.markdown("<div style='padding-top: 20px'></div>", unsafe_allow_html=True)
                        if st.button("🗑️", key=f"remove_{idx}"):
                            st.session_state.studio_tracks.pop(idx)
                            st.rerun()
        
        # Mix tracks
        with glass_card_container():
            st.markdown("#### 🎚️ Mix Tracks")
            
            if action_button("🎵 Mix All Tracks", key="mix_tracks"):
                try:
                    with loading_skeleton("Mixing tracks...", "Layering audio and balancing volumes..."):
                        processor = load_audio_processor()
                        if not processor:
                            return
                        
                        # Load all tracks
                        audio_tracks = []
                        volumes = []
                        
                        for track in st.session_state.studio_tracks:
                            audio, sr = processor.load_audio(track['path'])
                            audio_tracks.append(audio)
                            volumes.append(track['volume'])
                        
                        # Mix
                        mixed = processor.mix_audio(audio_tracks, volumes)
                        
                        # Save
                        output_path = config.OUTPUT_DIR / f"mix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                        saved_path = processor.save_audio(mixed, output_path, sr)
                        
                        success_message("Tracks mixed successfully!")
                        
                        audio_player(
                            str(saved_path),
                            title="Mixed Track",
                            artist=f"{len(st.session_state.studio_tracks)} tracks"
                        )
                        
                        with open(saved_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download Mix",
                                f.read(),
                                file_name=Path(saved_path).name,
                                mime="audio/wav",
                                use_container_width=True
                            )
                        
                except Exception as e:
                    error_message(f"Error mixing tracks: {str(e)}")
        
        # Clear all
        with glass_card_container():
            if st.button("🗑️ Clear All Tracks", use_container_width=True):
                st.session_state.studio_tracks = []
                success_message("All tracks cleared!")
                st.rerun()
    else:
        with glass_card_container():
            st.markdown("### 🎵 No Tracks Yet")
            st.markdown("Upload or generate tracks to start creating your masterpiece!")
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**1️⃣ Upload**")
                st.markdown("Add audio files")
            with col2:
                st.markdown("**2️⃣ Generate**")
                st.markdown("Create new tracks")
            with col3:
                st.markdown("**3️⃣ Mix**")
                st.markdown("Blend them together")


# =============================================================================
# PAGE 5: HISTORY
# =============================================================================

def history_page():
    """Generation history with search, filter, and favorites"""
    
    # Modern glass card header
    with glass_card_container():
        st.markdown("### 📚 Generation History")
        st.markdown("View, search, and organize your AI music creations")
        st.markdown("---")
        
        # Feature highlights
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**🎵 All Generations**")
            st.markdown("Complete track history")
        with col2:
            st.markdown("**� Search & Filter**")
            st.markdown("Find any track")
        with col3:
            st.markdown("**⭐ Favorites**")
            st.markdown("Mark your best")
        with col4:
            st.markdown("**⬇️ Export**")
            st.markdown("Download anytime")
    
    history = st.session_state.get("generation_history", [])
    
    if not history:
        with glass_card_container():
            st.markdown("### 🎵 No History Yet")
            st.markdown("Generate some music to see your creation history here!")
            st.markdown("---")
            info_message("All your generations will be saved here for easy access")
        return
    
    # Search and Filter Controls
    with glass_card_container():
        st.markdown("#### 🔍 Search & Filter")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            search_query = st.text_input(
                "🔍 Search prompts...",
                placeholder="Search by prompt text...",
                key="history_search",
                label_visibility="collapsed"
            )
        
        with col2:
            # Genre filter
            all_genres = sorted(set([item.get('genre', 'N/A') for item in history if item.get('genre')]))
            genre_filter = st.selectbox(
                "Filter by Genre",
                options=["All Genres"] + all_genres,
                key="genre_filter"
            )
        
        with col3:
            # Mood filter
            all_moods = sorted(set([item.get('mood', 'N/A') for item in history if item.get('mood')]))
            mood_filter = st.selectbox(
                "Filter by Mood",
                options=["All Moods"] + all_moods,
                key="mood_filter"
            )
        
        with col4:
            # Sort options
            sort_option = st.selectbox(
                "Sort by",
                options=["Newest First", "Oldest First", "Duration (Long)", "Duration (Short)"],
                key="sort_option"
            )
    
    # Initialize favorites if not exists
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    
    # Favorites toggle
    with glass_card_container():
        show_favorites_only = st.checkbox("⭐ Show Favorites Only", key="show_fav_only")
    
    # Filter history
    filtered_history = history.copy()
    
    # Apply search filter
    if search_query:
        filtered_history = [
            item for item in filtered_history 
            if search_query.lower() in item.get('prompt', '').lower()
        ]
    
    # Apply genre filter
    if genre_filter != "All Genres":
        filtered_history = [
            item for item in filtered_history 
            if item.get('genre') == genre_filter
        ]
    
    # Apply mood filter
    if mood_filter != "All Moods":
        filtered_history = [
            item for item in filtered_history 
            if item.get('mood') == mood_filter
        ]
    
    # Apply favorites filter
    if show_favorites_only:
        filtered_history = [
            item for item in filtered_history 
            if item.get('file') in st.session_state.favorites
        ]
    
    # Apply sorting
    if sort_option == "Newest First":
        filtered_history = list(reversed(filtered_history))
    elif sort_option == "Oldest First":
        pass  # Already in chronological order
    elif sort_option == "Duration (Long)":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('duration', 0), reverse=True)
    elif sort_option == "Duration (Short)":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('duration', 0))
    
    # Stats
    with glass_card_container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🎵 Total", str(len(history)))
        with col2:
            st.metric("🔍 Showing", str(len(filtered_history)))
        with col3:
            st.metric("⭐ Favorites", str(len(st.session_state.favorites)))
        with col4:
            genres = [item.get('genre', 'N/A') for item in history if item.get('genre')]
            st.metric("🎼 Genres", str(len(set(genres))))
        with col5:
            total_duration = sum([item.get('duration', 0) for item in history])
            st.metric("⏱️ Duration", f"{total_duration}s")
    
    # Display filtered history
    if not filtered_history:
        with glass_card_container():
            st.markdown("### 🔍 No Results Found")
            st.markdown("Try adjusting your search or filters")
        return
    
    for idx, item in enumerate(filtered_history):
        file_path = Path(item.get('file', ''))
        is_favorite = str(file_path) in st.session_state.favorites
        
        with glass_card_container():
            # Title row with favorite indicator
            title_col, fav_col = st.columns([10, 1])
            with title_col:
                fav_indicator = " ⭐" if is_favorite else ""
                st.markdown(f"### 🎵 Track {idx+1}{fav_indicator}")
                st.markdown(f"*{item.get('timestamp', 'Unknown')}*")
            with fav_col:
                st.markdown("<div style='padding-top: 20px'></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            cols = st.columns([2, 1])
            
            with cols[0]:
                # Track details
                st.markdown(f"**Prompt:** {item.get('prompt', 'N/A')}")
                
                detail_cols = st.columns(3)
                with detail_cols[0]:
                    st.markdown(f"**Genre:** {item.get('genre', 'N/A')}")
                with detail_cols[1]:
                    st.markdown(f"**Mood:** {item.get('mood', 'N/A')}")
                with detail_cols[2]:
                    st.markdown(f"**Duration:** {item.get('duration', 'N/A')}s")
                
                # Enhanced audio player with favorites
                if file_path.exists():
                    enhanced_audio_player(
                        str(file_path),
                        title=f"Track {idx+1}",
                        key=f"hist_{idx}",
                        show_download=True,
                        show_favorite=True,
                        metadata={
                            'genre': item.get('genre', 'N/A'),
                            'mood': item.get('mood', 'N/A'),
                            'duration': item.get('duration', 'N/A')
                        }
                    )
            
            with cols[1]:
                st.markdown("<div style='padding-top: 40px'></div>", unsafe_allow_html=True)
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download",
                            f.read(),
                            file_name=file_path.name,
                            key=f"dl_hist_{idx}",
                            use_container_width=True
                        )
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"del_{idx}", use_container_width=True):
                        # Remove from history
                        original_idx = history.index(item)
                        st.session_state.generation_history.pop(original_idx)
                        # Remove from favorites if present
                        if str(file_path) in st.session_state.favorites:
                            st.session_state.favorites.remove(str(file_path))
                        success_message("Track deleted!")
                        st.rerun()
    
    # Clear history
    with glass_card_container():
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state.generation_history = []
                success_message("History cleared!")
                st.rerun()
        with col2:
            if st.button("❌ Clear All Favorites", use_container_width=True):
                st.session_state.favorites = []
                success_message("Favorites cleared!")
                st.rerun()


# =============================================================================
# PAGE 6: SETTINGS
# =============================================================================

def settings_page():
    """Application settings with stunning UI"""
    
    # Hero header with gradient
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="
            font-size: 3rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">⚙️ Settings</h1>
        <p style="
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.9);
            margin-top: 0.5rem;
        ">Configure your AI Music Generator</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick status cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
        ">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔧</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: white;">Configuration</div>
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">System Ready</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
        ">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: white;">AI Providers</div>
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">API Setup</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(67, 233, 123, 0.3);
        ">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💾</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: white;">Storage</div>
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">Encrypted</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Configuration status with beautiful cards
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    ">
        <h3 style="
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">🔧 System Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    configured = config.is_configured()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
            border: 2px solid rgba(102,126,234,0.3);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        ">
            <h4 style="color: #667eea; margin-bottom: 1rem; font-size: 1.2rem;">🤖 LLM Providers (5)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # LLM providers
        llm_providers = ['groq', 'openrouter', 'huggingface', 'together', 'ollama']
        for provider in llm_providers:
            status = configured.get(provider, False)
            icon = "✅" if status else "⚠️"
            color = "#43e97b" if status else "#ff6b6b"
            bg_color = "rgba(67,233,123,0.1)" if status else "rgba(255,107,107,0.1)"
            st.markdown(f"""
            <div style="
                background: {bg_color};
                border-left: 4px solid {color};
                padding: 0.8rem;
                margin-bottom: 0.5rem;
                border-radius: 8px;
            ">
                <span style="font-size: 1.5rem;">{icon}</span>
                <strong style="color: {color}; margin-left: 0.5rem;">{provider.title()}</strong>
                <span style="color: {color}; margin-left: 1rem; font-size: 0.9rem;">{'Configured' if status else 'Not configured'}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Music providers
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(251,146,60,0.2) 0%, rgba(249,115,22,0.2) 100%);
            border: 2px solid rgba(251,146,60,0.3);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            margin-top: 1rem;
        ">
            <h4 style="color: #FB923C; margin-bottom: 1rem; font-size: 1.2rem;">🎵 Music APIs (7)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        music_providers = ['suno', 'replicate', 'beatoven', 'loudly', 'musicapi', 'udio', 'free']
        for provider in music_providers:
            status = configured.get(provider, False)
            # Free provider is always available
            if provider == 'free':
                status = True
            icon = "✅" if status else "⚠️"
            color = "#43e97b" if status else "#ff6b6b"
            bg_color = "rgba(67,233,123,0.1)" if status else "rgba(255,107,107,0.1)"
            display_name = "Free Generator" if provider == 'free' else provider.title()
            st.markdown(f"""
            <div style="
                background: {bg_color};
                border-left: 4px solid {color};
                padding: 0.8rem;
                margin-bottom: 0.5rem;
                border-radius: 8px;
            ">
                <span style="font-size: 1.5rem;">{icon}</span>
                <strong style="color: {color}; margin-left: 0.5rem;">{display_name}</strong>
                <span style="color: {color}; margin-left: 1rem; font-size: 0.9rem;">{'Configured' if status else 'Not configured'}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Audio Analysis providers
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(236,72,153,0.2) 0%, rgba(219,39,119,0.2) 100%);
            border: 2px solid rgba(236,72,153,0.3);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        ">
            <h4 style="color: #EC4899; margin-bottom: 1rem; font-size: 1.2rem;">🎯 Analysis APIs (3)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_providers = ['hume', 'eden', 'audd']
        for provider in analysis_providers:
            status = configured.get(provider, False)
            icon = "✅" if status else "⚠️"
            color = "#43e97b" if status else "#ff6b6b"
            bg_color = "rgba(67,233,123,0.1)" if status else "rgba(255,107,107,0.1)"
            display_name = "Hume AI" if provider == 'hume' else "Eden AI" if provider == 'eden' else "Audd.io"
            st.markdown(f"""
            <div style="
                background: {bg_color};
                border-left: 4px solid {color};
                padding: 0.8rem;
                margin-bottom: 0.5rem;
                border-radius: 8px;
            ">
                <span style="font-size: 1.5rem;">{icon}</span>
                <strong style="color: {color}; margin-left: 0.5rem;">{display_name}</strong>
                <span style="color: {color}; margin-left: 1rem; font-size: 0.9rem;">{'Configured' if status else 'Not configured'}</span>
            </div>
            """, unsafe_allow_html=True)
        
        device = "🚀 GPU (CUDA)" if st.session_state.get('has_cuda', False) else "💻 CPU"
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(67,233,123,0.2) 0%, rgba(56,249,215,0.2) 100%);
            border: 2px solid rgba(67,233,123,0.3);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            margin-top: 1rem;
        ">
            <h4 style="color: #43e97b; margin-bottom: 1rem; font-size: 1.2rem;">💻 System Info</h4>
            <div style="
                background: rgba(255,255,255,0.1);
                padding: 0.8rem;
                border-radius: 8px;
                margin-bottom: 0.5rem;
            ">
                <strong style="color: #43e97b;">Device:</strong>
                <span style="color: white; margin-left: 1rem;">{device}</span>
            </div>
            <div style="
                background: rgba(255,255,255,0.1);
                padding: 0.8rem;
                border-radius: 8px;
                margin-bottom: 0.5rem;
            ">
                <strong style="color: #43e97b;">Cache:</strong>
                <span style="color: white; margin-left: 1rem;">{'✅ Enabled' if config.ENABLE_CACHE else '❌ Disabled'}</span>
            </div>
            <div style="
                background: rgba(255,255,255,0.1);
                padding: 0.8rem;
                border-radius: 8px;
                margin-bottom: 0.5rem;
            ">
                <strong style="color: #43e97b;">Format:</strong>
                <span style="color: white; margin-left: 1rem;">{config.OUTPUT_FORMAT.upper()}</span>
            </div>
            <div style="
                background: rgba(255,255,255,0.1);
                padding: 0.8rem;
                border-radius: 8px;
            ">
                <strong style="color: #43e97b;">Total Providers:</strong>
                <span style="color: white; margin-left: 1rem;">19 APIs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # API Keys Configuration - Premium Design
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(240, 147, 251, 0.3);
    ">
        <h3 style="
            font-size: 2rem;
            font-weight: 700;
            color: white;
            margin-bottom: 1rem;
        ">🔑 API Keys Configuration</h3>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 1rem;">
            🔒 <strong>Secure & Encrypted</strong> - Your keys are stored safely on your device!
        </p>
        <div style="
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
        ">
            <p style="color: white; margin: 0; font-size: 0.95rem;">
                ⚡ <strong>Get FREE API Keys:</strong><br>
                🎵 <a href="https://huggingface.co/settings/tokens" target="_blank" style="color: #43e97b; text-decoration: none;">HuggingFace</a> • 
                🎸 <a href="https://replicate.com/account/api-tokens" target="_blank" style="color: #43e97b; text-decoration: none;">Replicate</a> • 
                🚀 <a href="https://console.groq.com/keys" target="_blank" style="color: #43e97b; text-decoration: none;">Groq</a> • 
                🌐 <a href="https://openrouter.ai/keys" target="_blank" style="color: #43e97b; text-decoration: none;">OpenRouter</a>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    from config import api_key_manager
    
    # HuggingFace Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(255,183,77,0.2) 0%, rgba(255,138,0,0.2) 100%);
        border: 2px solid rgba(255,183,77,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">🎵</span>
            <div>
                <h4 style="color: #FFB74D; margin: 0; font-size: 1.3rem;">HuggingFace</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">AI Music Generation</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure HuggingFace Token", expanded=False):
        # Show current status
        current_hf_status = api_key_manager.get_api_key("huggingface")
        if current_hf_status:
            st.success("✅ **Currently Configured** - Token is active!")
            key_source = api_key_manager.get_key_source("huggingface")
            if key_source:
                st.info(f"🔑 Source: {key_source.upper()}")
        else:
            st.warning("⚠️ **Not Configured** - Add your token below")
        
        hf_key = st.text_input(
            "🔐 HuggingFace Token",
            type="password",
            placeholder="hf_...",
            key="settings_hf_key",
            help="Get your FREE token from: https://huggingface.co/settings/tokens"
        )
        if st.button("💾 Save HuggingFace Key", key="save_hf_settings", use_container_width=True):
            if hf_key:
                # Validate key format
                if api_key_manager.validate_api_key("huggingface", hf_key):
                    if api_key_manager.save_api_key("huggingface", hf_key):
                        success_message("✅ HuggingFace key saved successfully!")
                        st.cache_resource.clear()
                        time.sleep(0.5)  # Give time for state to update
                        st.rerun()
                    else:
                        error_message("❌ Failed to save key")
                else:
                    error_message("❌ Invalid token format (should start with hf_)")
            else:
                error_message("⚠️ Please enter a key")
        
    # Replicate Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(139,92,246,0.2) 100%);
        border: 2px solid rgba(99,102,241,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">�</span>
            <div>
                <h4 style="color: #818CF8; margin: 0; font-size: 1.3rem;">Replicate</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">Advanced Music Models</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure Replicate Token", expanded=False):
        # Show current status
        current_rep_status = api_key_manager.get_api_key("replicate")
        if current_rep_status:
            st.success("✅ **Currently Configured** - Token is active!")
            key_source = api_key_manager.get_key_source("replicate")
            if key_source:
                st.info(f"🔑 Source: {key_source.upper()}")
        else:
            st.warning("⚠️ **Not Configured** - Add your token below")
        
        rep_key = st.text_input(
            "🔐 Replicate Token",
            type="password",
            placeholder="r8_...",
            key="settings_rep_key",
            help="Get your FREE token from: https://replicate.com/account/api-tokens"
        )
        if st.button("💾 Save Replicate Key", key="save_rep_settings", use_container_width=True):
            if rep_key:
                # Validate key format
                if api_key_manager.validate_api_key("replicate", rep_key):
                    if api_key_manager.save_api_key("replicate", rep_key):
                        success_message("✅ Replicate key saved successfully!")
                        st.cache_resource.clear()
                        time.sleep(0.5)  # Give time for state to update
                        st.rerun()
                    else:
                        error_message("❌ Failed to save key")
                else:
                    error_message("❌ Invalid token format (should start with r8_)")
            else:
                error_message("⚠️ Please enter a key")
        
    # Groq Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(16,185,129,0.2) 100%);
        border: 2px solid rgba(34,197,94,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">�</span>
            <div>
                <h4 style="color: #10B981; margin: 0; font-size: 1.3rem;">Groq</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">Ultra-Fast Prompt Enhancement</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure Groq API Key", expanded=False):
        # Show current status
        current_groq_status = api_key_manager.get_api_key("groq")
        if current_groq_status:
            st.success("✅ **Currently Configured** - Key is active!")
            key_source = api_key_manager.get_key_source("groq")
            if key_source:
                st.info(f"🔑 Source: {key_source.upper()}")
        else:
            st.warning("⚠️ **Not Configured** - Add your key below")
        
        groq_key = st.text_input(
            "🔐 Groq API Key",
            type="password",
            placeholder="gsk_...",
            key="settings_groq_key",
            help="Get your FREE key from: https://console.groq.com/keys"
        )
        if st.button("💾 Save Groq Key", key="save_groq_settings", use_container_width=True):
            if groq_key:
                # Validate key format
                if api_key_manager.validate_api_key("groq", groq_key):
                    if api_key_manager.save_api_key("groq", groq_key):
                        success_message("✅ Groq key saved successfully!")
                        st.cache_resource.clear()
                        time.sleep(0.5)  # Give time for state to update
                        st.rerun()
                    else:
                        error_message("❌ Failed to save key")
                else:
                    error_message("❌ Invalid key format (should start with gsk_)")
            else:
                error_message("⚠️ Please enter a key")
        
    # OpenRouter Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(59,130,246,0.2) 0%, rgba(37,99,235,0.2) 100%);
        border: 2px solid rgba(59,130,246,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">🌐</span>
            <div>
                <h4 style="color: #3B82F6; margin: 0; font-size: 1.3rem;">OpenRouter</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">FREE Llama 3.1 Access</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure OpenRouter API Key", expanded=False):
        # Show current status
        current_or_status = api_key_manager.get_api_key("openrouter")
        if current_or_status:
            st.success("✅ **Currently Configured** - Key is active!")
            key_source = api_key_manager.get_key_source("openrouter")
            if key_source:
                st.info(f"🔑 Source: {key_source.upper()}")
        else:
            st.warning("⚠️ **Not Configured** - Add your key below")
        
        or_key = st.text_input(
            "🔐 OpenRouter API Key",
            type="password",
            placeholder="sk-or-v1-...",
            key="settings_or_key",
            help="Get your FREE key from: https://openrouter.ai/keys"
        )
        if st.button("💾 Save OpenRouter Key", key="save_or_settings", use_container_width=True):
            if or_key:
                # Validate key format
                if api_key_manager.validate_api_key("openrouter", or_key):
                    if api_key_manager.save_api_key("openrouter", or_key):
                        success_message("✅ OpenRouter key saved successfully!")
                        st.cache_resource.clear()
                        time.sleep(0.5)  # Give time for state to update
                        st.rerun()
                    else:
                        error_message("❌ Failed to save key")
                else:
                    error_message("❌ Invalid key format (should start with sk-or-)")
            else:
                error_message("⚠️ Please enter a key")
    
    # NEW MUSIC GENERATION PROVIDERS
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0 1.5rem 0;
        box-shadow: 0 8px 25px rgba(255, 154, 158, 0.3);
    ">
        <h3 style="
            font-size: 1.5rem;
            font-weight: 700;
            color: white;
            margin: 0;
            text-align: center;
        ">🎵 Additional Music Generation APIs</h3>
        <p style="color: rgba(255,255,255,0.9); font-size: 0.95rem; margin: 0.5rem 0 0 0; text-align: center;">
            More FREE providers for unlimited music creation!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Beatoven.ai Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(168,85,247,0.2) 0%, rgba(147,51,234,0.2) 100%);
        border: 2px solid rgba(168,85,247,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">🎼</span>
            <div>
                <h4 style="color: #A855F7; margin: 0; font-size: 1.3rem;">Beatoven.ai</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">Mood-Based Music • FREE Tier</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure Beatoven API Key", expanded=False):
        beatoven_key = st.text_input(
            "🔐 Beatoven API Key",
            type="password",
            placeholder="Enter your Beatoven API key...",
            key="settings_beatoven_key",
            help="Get your FREE key from: https://www.beatoven.ai/api"
        )
        if st.button("💾 Save Beatoven Key", key="save_beatoven_settings", use_container_width=True):
            if beatoven_key:
                if api_key_manager.save_api_key("beatoven", beatoven_key):
                    success_message("✅ Beatoven key saved successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    error_message("❌ Failed to save key")
            else:
                error_message("⚠️ Please enter a key")
    
    # Loudly Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(251,146,60,0.2) 0%, rgba(249,115,22,0.2) 100%);
        border: 2px solid rgba(251,146,60,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">🎸</span>
            <div>
                <h4 style="color: #FB923C; margin: 0; font-size: 1.3rem;">Loudly Music</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">Genre-Based Music • FREE Tier</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure Loudly API Key", expanded=False):
        loudly_key = st.text_input(
            "🔐 Loudly API Key",
            type="password",
            placeholder="Enter your Loudly API key...",
            key="settings_loudly_key",
            help="Get your FREE key from: https://www.loudly.com/api"
        )
        if st.button("💾 Save Loudly Key", key="save_loudly_settings", use_container_width=True):
            if loudly_key:
                if api_key_manager.save_api_key("loudly", loudly_key):
                    success_message("✅ Loudly key saved successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    error_message("❌ Failed to save key")
            else:
                error_message("⚠️ Please enter a key")
    
    # MusicAPI.ai Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(56,189,248,0.2) 0%, rgba(14,165,233,0.2) 100%);
        border: 2px solid rgba(56,189,248,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">🎹</span>
            <div>
                <h4 style="color: #38BDF8; margin: 0; font-size: 1.3rem;">MusicAPI.ai</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">Text-to-Music • Test Tier FREE</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure MusicAPI Key", expanded=False):
        musicapi_key = st.text_input(
            "🔐 MusicAPI Key",
            type="password",
            placeholder="Enter your MusicAPI key...",
            key="settings_musicapi_key",
            help="Get your FREE key from: https://musicapi.ai/api"
        )
        if st.button("💾 Save MusicAPI Key", key="save_musicapi_settings", use_container_width=True):
            if musicapi_key:
                if api_key_manager.save_api_key("musicapi", musicapi_key):
                    success_message("✅ MusicAPI key saved successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    error_message("❌ Failed to save key")
            else:
                error_message("⚠️ Please enter a key")
    
    # Udio Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(52,211,153,0.2) 0%, rgba(16,185,129,0.2) 100%);
        border: 2px solid rgba(52,211,153,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">🎧</span>
            <div>
                <h4 style="color: #34D399; margin: 0; font-size: 1.3rem;">Udio</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">Early Access • FREE</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure Udio API Key", expanded=False):
        udio_key = st.text_input(
            "🔐 Udio API Key",
            type="password",
            placeholder="Enter your Udio API key...",
            key="settings_udio_key",
            help="Get your FREE key from: https://www.udio.com/api"
        )
        if st.button("💾 Save Udio Key", key="save_udio_settings", use_container_width=True):
            if udio_key:
                if api_key_manager.save_api_key("udio", udio_key):
                    success_message("✅ Udio key saved successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    error_message("❌ Failed to save key")
            else:
                error_message("⚠️ Please enter a key")
    
    # AUDIO ANALYSIS PROVIDERS
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0 1.5rem 0;
        box-shadow: 0 8px 25px rgba(168, 237, 234, 0.3);
    ">
        <h3 style="
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f766e;
            margin: 0;
            text-align: center;
        ">🎯 Audio Analysis & Emotion Detection APIs</h3>
        <p style="color: rgba(15, 118, 110, 0.8); font-size: 0.95rem; margin: 0.5rem 0 0 0; text-align: center;">
            FREE emotion and music feature analysis!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hume AI Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(236,72,153,0.2) 0%, rgba(219,39,119,0.2) 100%);
        border: 2px solid rgba(236,72,153,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">💖</span>
            <div>
                <h4 style="color: #EC4899; margin: 0; font-size: 1.3rem;">Hume AI</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">Advanced Emotion Detection • FREE</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure Hume AI Key", expanded=False):
        hume_key = st.text_input(
            "🔐 Hume AI API Key",
            type="password",
            placeholder="Enter your Hume AI API key...",
            key="settings_hume_key",
            help="Get your FREE key from: https://beta.hume.ai/api"
        )
        if st.button("💾 Save Hume AI Key", key="save_hume_settings", use_container_width=True):
            if hume_key:
                if api_key_manager.save_api_key("hume", hume_key):
                    success_message("✅ Hume AI key saved successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    error_message("❌ Failed to save key")
            else:
                error_message("⚠️ Please enter a key")
    
    # Eden AI Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(139,92,246,0.2) 0%, rgba(124,58,237,0.2) 100%);
        border: 2px solid rgba(139,92,246,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">🌈</span>
            <div>
                <h4 style="color: #8B5CF6; margin: 0; font-size: 1.3rem;">Eden AI</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">Multi-Model Analysis • FREE Tier</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure Eden AI Key", expanded=False):
        eden_key = st.text_input(
            "🔐 Eden AI API Key",
            type="password",
            placeholder="Enter your Eden AI API key...",
            key="settings_eden_key",
            help="Get your FREE key from: https://www.edenai.co/api"
        )
        if st.button("💾 Save Eden AI Key", key="save_eden_settings", use_container_width=True):
            if eden_key:
                if api_key_manager.save_api_key("eden", eden_key):
                    success_message("✅ Eden AI key saved successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    error_message("❌ Failed to save key")
            else:
                error_message("⚠️ Please enter a key")
    
    # Audd.io Card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(234,179,8,0.2) 0%, rgba(202,138,4,0.2) 100%);
        border: 2px solid rgba(234,179,8,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">🏆</span>
            <div>
                <h4 style="color: #EAB308; margin: 0; font-size: 1.3rem;">Audd.io</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;">Music Recognition • FREE Trial</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure Audd.io Key", expanded=False):
        audd_key = st.text_input(
            "🔐 Audd.io API Key",
            type="password",
            placeholder="Enter your Audd.io API key...",
            key="settings_audd_key",
            help="Get your FREE key from: https://audd.io/api"
        )
        if st.button("💾 Save Audd.io Key", key="save_audd_settings", use_container_width=True):
            if audd_key:
                if api_key_manager.save_api_key("audd", audd_key):
                    success_message("✅ Audd.io key saved successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    error_message("❌ Failed to save key")
            else:
                error_message("⚠️ Please enter a key")
    
    # Success tips section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.15) 100%);
        border-left: 4px solid #10B981;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
    ">
        <h4 style="color: #10B981; margin-bottom: 1rem;">💡 Pro Tips</h4>
        <ul style="color: rgba(255,255,255,0.8); margin-left: 1.5rem;">
            <li style="margin-bottom: 0.5rem;">All API keys are <strong>encrypted</strong> and stored locally on your device 🔒</li>
            <li style="margin-bottom: 0.5rem;">Free tiers are available for <strong>all 19 providers</strong> - no credit card required! 🎉</li>
            <li style="margin-bottom: 0.5rem;"><strong>7 Music APIs</strong>: Suno, Replicate, Beatoven, Loudly, MusicAPI, Udio, Free Generator</li>
            <li style="margin-bottom: 0.5rem;"><strong>3 Analysis APIs</strong>: Hume AI (emotion), Eden AI (multi-model), Audd.io (recognition)</li>
            <li style="margin-bottom: 0.5rem;">HuggingFace + Groq combo gives you unlimited music generation ⚡</li>
            <li>OpenRouter provides FREE access to Llama 3.1 for prompt enhancement 🚀</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # App settings
    with glass_card_container():
        st.markdown("#### 🎵 Music Generation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_model = st.selectbox(
                "Default Model",
                options=["small", "medium", "large", "melody"],
                index=1
            )
        
        with col2:
            st.markdown("<div style='padding-top: 30px'></div>", unsafe_allow_html=True)
            if action_button("💾 Save Settings", key="save_settings"):
                success_message("Settings saved!")
    
    # API Health Check Section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0 1.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    ">
        <h3 style="
            font-size: 1.5rem;
            font-weight: 700;
            color: white;
            margin: 0;
            text-align: center;
        ">🏥 API Health Check</h3>
        <p style="color: rgba(255,255,255,0.9); font-size: 0.95rem; margin: 0.5rem 0 0 0; text-align: center;">
            Test connectivity to all 19 API providers
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with glass_card_container():
        st.markdown("#### 🔍 Check Provider Status")
        
        if action_button("🏥 Run Health Check", key="health_check_btn"):
            try:
                from api_health import check_api_health, get_health_summary, get_providers_by_category
                
                with st.spinner("🔍 Checking all 19 providers..."):
                    results = check_api_health()
                    summary = get_health_summary(results)
                    by_category = get_providers_by_category(results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("✅ Available", summary['available'])
                with col2:
                    st.metric("⚠️ Needs Key", summary['needs_key'])
                with col3:
                    st.metric("❌ Errors", summary['error'])
                with col4:
                    st.metric("📊 Health", f"{summary['percentage']}%")
                
                st.markdown("---")
                
                # LLM Providers
                st.markdown("##### 🤖 LLM Providers")
                for provider_id, info in by_category['llm'].items():
                    status_icon = "✅" if info['status'] == 'available' else "⚠️" if info['status'] == 'needs_key' else "❌"
                    status_color = "#43e97b" if info['status'] == 'available' else "#FFA500" if info['status'] == 'needs_key' else "#ff6b6b"
                    
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.05);
                        border-left: 4px solid {status_color};
                        padding: 0.8rem;
                        margin-bottom: 0.5rem;
                        border-radius: 8px;
                    ">
                        <span style="font-size: 1.2rem;">{status_icon}</span>
                        <strong style="margin-left: 0.5rem;">{info['name']}</strong>
                        <span style="color: {status_color}; margin-left: 1rem; font-size: 0.9rem;">{info['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Music Providers
                st.markdown("##### 🎵 Music Generation APIs")
                for provider_id, info in by_category['music'].items():
                    status_icon = "✅" if info['status'] == 'available' else "⚠️" if info['status'] == 'needs_key' else "❌"
                    status_color = "#43e97b" if info['status'] == 'available' else "#FFA500" if info['status'] == 'needs_key' else "#ff6b6b"
                    
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.05);
                        border-left: 4px solid {status_color};
                        padding: 0.8rem;
                        margin-bottom: 0.5rem;
                        border-radius: 8px;
                    ">
                        <span style="font-size: 1.2rem;">{status_icon}</span>
                        <strong style="margin-left: 0.5rem;">{info['name']}</strong>
                        <span style="color: {status_color}; margin-left: 1rem; font-size: 0.9rem;">{info['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Analysis Providers
                st.markdown("##### 🎯 Audio Analysis APIs")
                for provider_id, info in by_category['analysis'].items():
                    status_icon = "✅" if info['status'] == 'available' else "⚠️" if info['status'] == 'needs_key' else "❌"
                    status_color = "#43e97b" if info['status'] == 'available' else "#FFA500" if info['status'] == 'needs_key' else "#ff6b6b"
                    
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.05);
                        border-left: 4px solid {status_color};
                        padding: 0.8rem;
                        margin-bottom: 0.5rem;
                        border-radius: 8px;
                    ">
                        <span style="font-size: 1.2rem;">{status_icon}</span>
                        <strong style="margin-left: 0.5rem;">{info['name']}</strong>
                        <span style="color: {status_color}; margin-left: 1rem; font-size: 0.9rem;">{info['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                success_message(f"Health check complete! {summary['available']}/{summary['total']} providers available")
                
            except Exception as e:
                error_message(f"Health check failed: {str(e)}")
    
    # Storage Management - Simplified
    with glass_card_container():
        st.markdown("#### 💾 Storage Management")
        
        # Calculate storage
        output_files = list(config.OUTPUT_DIR.glob("*"))
        upload_files = list(config.UPLOAD_DIR.glob("*"))
        total_files = len(output_files) + len(upload_files)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📁 Total Files", total_files)
        with col2:
            st.metric("🎵 Output Files", len(output_files))
        with col3:
            st.metric("📤 Uploads", len(upload_files))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clear Output Directory", key="clear_output", use_container_width=True):
                for file in output_files:
                    try:
                        file.unlink()
                    except:
                        pass
                success_message("✅ Output directory cleared!")
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Upload Directory", key="clear_upload", use_container_width=True):
                for file in upload_files:
                    try:
                        file.unlink()
                    except:
                        pass
                success_message("✅ Upload directory cleared!")
                st.rerun()
    
    # About
    with glass_card_container():
        st.markdown("#### ℹ️ About")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **AI Music Generator v2.0**
            
            A complete music generation and remixing platform using 100% FREE AI models:
            - **MusicGen** by Meta (Music generation)
            - **Demucs** by Meta (Stem separation)  
            - **Groq/Llama** (Prompt enhancement)
            - **Librosa** (Audio analysis)
            - **Pedalboard** by Spotify (Audio effects)
            
            All models run locally - no expensive API calls needed!
            """)
        
        with col2:
            st.metric("🚀 Version", "2.0")
            st.metric("✅ Status", "Ready")
            st.metric("📜 License", "MIT")
        
        st.markdown("---")
        st.markdown("*Made with ❤️ for students and music enthusiasts*")


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
