"""
AI Music Generator - Custom Reusable UI Components
Modern, glassmorphic components inspired by Spotify, SoundCloud, and professional DAWs
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Optional, List, Dict, Any, Callable
import base64
from io import BytesIO
from contextlib import contextmanager


# =============================================================================
# GLASSMORPHIC CARD COMPONENTS
# =============================================================================

def glass_card(content: str = "", title: Optional[str] = None, icon: str = "🎵", gradient: str = "primary") -> None:
    """
    Create a beautiful glassmorphic card with optional title and icon.
    
    Args:
        content: HTML or markdown content to display in the card
        title: Optional title for the card header
        icon: Emoji icon to display with the title
        gradient: Gradient style - 'primary', 'secondary', 'rainbow', or 'gold'
    
    Example:
        >>> glass_card("Your content here", title="Feature Name", icon="🎸", gradient="primary")
    """
    gradient_map = {
        "primary": "var(--gradient-primary)",
        "secondary": "var(--gradient-secondary)",
        "rainbow": "var(--gradient-rainbow)",
        "gold": "var(--gradient-gold)"
    }
    
    gradient_style = gradient_map.get(gradient, gradient_map["primary"])
    
    card_html = f"""
    <div class="feature-card" style="animation: fadeInScale 0.5s cubic-bezier(0.16, 1, 0.3, 1);">
        {f'<h3 style="margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">{icon} {title}</h3>' if title else ''}
        <div style="color: var(--text-secondary);">{content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


@contextmanager
def glass_card_container(title: Optional[str] = None, icon: str = "🎵", gradient: str = "primary"):
    """
    Context manager version of glass_card for wrapping Streamlit content.
    Use this with 'with' statement to wrap streamlit components.
    
    Args:
        title: Optional title for the card header
        icon: Emoji icon to display with the title
        gradient: Gradient style - 'primary', 'secondary', 'rainbow', or 'gold'
    
    Example:
        >>> with glass_card_container(title="Settings", icon="⚙️"):
        >>>     st.write("Card content here")
        >>>     st.button("Click me")
    """
    gradient_map = {
        "primary": "rgba(102, 126, 234, 0.1)",
        "secondary": "rgba(240, 147, 251, 0.1)",
        "rainbow": "rgba(79, 172, 254, 0.1)",
        "gold": "rgba(245, 158, 11, 0.1)"
    }
    
    bg_color = gradient_map.get(gradient, gradient_map["primary"])
    
    # Start the card HTML
    if title:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {bg_color} 0%, rgba(255, 255, 255, 0.05) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            animation: fadeInScale 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        ">
            <h3 style="margin: 0 0 1.5rem 0; display: flex; align-items: center; gap: 0.5rem; color: #fff;">
                {icon} {title}
            </h3>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {bg_color} 0%, rgba(255, 255, 255, 0.05) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            animation: fadeInScale 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        ">
        """, unsafe_allow_html=True)
    
    # Yield control to the with block
    try:
        yield
    finally:
        # Close the card HTML
        st.markdown("</div>", unsafe_allow_html=True)


def dashboard_card(
    title: str, 
    value: Optional[str] = None,
    description: Optional[str] = None,
    delta: Optional[str] = None, 
    icon: str = "📊", 
    color: str = "primary",
    gradient: str = "primary",
    action_label: Optional[str] = None
) -> bool:
    """
    Create a dashboard metric card with glassmorphism effect.
    Can be used as a display card (with value) or interactive card (with description + action_label).
    
    Args:
        title: Card title/label
        value: Main value to display (for stats cards)
        description: Description text (for interactive cards)
        delta: Optional change indicator
        icon: Emoji icon
        color: Color theme - 'primary', 'secondary', 'accent'
        gradient: Gradient style for interactive cards
        action_label: Label for action button (makes card clickable)
    
    Returns:
        bool: True if card was clicked (interactive mode), False otherwise
    
    Example:
        >>> dashboard_card("Total Tracks", value="42", delta="+5", icon="🎵", color="primary")
        >>> if dashboard_card("Generate", description="Create music", action_label="Start", icon="🎼"):
        >>>     print("Card clicked!")
    """
    color_map = {
        "primary": "#1DB954",
        "secondary": "#6366F1",
        "accent": "#F59E0B"
    }
    
    gradient_map = {
        "primary": "linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%)",
        "secondary": "linear-gradient(135deg, rgba(240, 147, 251, 0.2) 0%, rgba(245, 87, 108, 0.2) 100%)",
        "success": "linear-gradient(135deg, rgba(79, 172, 254, 0.2) 0%, rgba(0, 242, 254, 0.2) 100%)"
    }
    
    card_color = color_map.get(color, color_map["primary"])
    
    # Interactive card mode (with action button)
    if action_label and description:
        gradient_style = gradient_map.get(gradient, gradient_map["primary"])
        key = f"card_{title.replace(' ', '_').lower()}"
        
        with st.container():
            st.markdown(f"""
                <div style="
                    background: {gradient_style};
                    backdrop-filter: blur(10px);
                    border: 2px solid rgba(255, 255, 255, 0.2);
                    border-radius: 16px;
                    padding: 1.5rem;
                    text-align: center;
                    transition: all 0.3s ease;
                    cursor: pointer;
                    height: 100%;
                    animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
                ">
                    <div style="font-size: 3rem; margin-bottom: 0.75rem;">{icon}</div>
                    <h3 style="color: #fff; margin: 0 0 0.5rem 0; font-size: 1.1rem;">{title}</h3>
                    <p style="color: rgba(255, 255, 255, 0.7); margin: 0 0 1rem 0; font-size: 0.9rem;">
                        {description}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Return button click state
            return st.button(action_label, key=key, use_container_width=True, type="primary")
    
    # Stats display mode (original functionality)
    else:
        display_value = value if value else description if description else "N/A"
        card_html = f"""
        <div class="dashboard-card" style="text-align: center; animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
                {title}
            </div>
            <div style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, {card_color}, {card_color}dd); 
                        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.25rem;">
                {display_value}
            </div>
            {f'<div style="color: {card_color}; font-weight: 600; font-size: 0.9rem;">{delta}</div>' if delta else ''}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        return False


def info_box(message: str, type: str = "info", icon: Optional[str] = None) -> None:
    """
    Display an information box with custom styling.
    
    Args:
        message: Message to display
        type: Box type - 'info', 'success', 'warning', 'error'
        icon: Optional custom icon (defaults based on type)
    
    Example:
        >>> info_box("Audio file loaded successfully!", type="success")
    """
    type_config = {
        "info": {"icon": "ℹ️", "color": "#6366F1"},
        "success": {"icon": "✅", "color": "#1DB954"},
        "warning": {"icon": "⚠️", "color": "#F59E0B"},
        "error": {"icon": "❌", "color": "#EC4899"}
    }
    
    config = type_config.get(type, type_config["info"])
    display_icon = icon or config["icon"]
    
    box_html = f"""
    <div style="background: {config['color']}15; backdrop-filter: blur(16px); border-left: 4px solid {config['color']}; 
                border-radius: 12px; padding: 1rem 1.5rem; margin: 1rem 0; animation: slideInLeft 0.4s cubic-bezier(0.16, 1, 0.3, 1);
                box-shadow: 0 0 20px {config['color']}20;">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 1.5rem;">{display_icon}</span>
            <span style="color: var(--text-primary); font-weight: 500;">{message}</span>
        </div>
    </div>
    """
    st.markdown(box_html, unsafe_allow_html=True)


# =============================================================================
# AUDIO PLAYER COMPONENTS
# =============================================================================

def audio_player(audio_data: bytes, title: str = "Audio Track", show_waveform: bool = True) -> None:
    """
    Create a professional audio player with optional waveform visualization.
    
    Args:
        audio_data: Audio file bytes
        title: Display title for the audio
        show_waveform: Whether to show waveform visualization
    
    Example:
        >>> with open("track.wav", "rb") as f:
        >>>     audio_player(f.read(), title="My Track", show_waveform=True)
    """
    # Create two columns for player and info
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: var(--glass-bg); backdrop-filter: blur(16px); border: 1px solid var(--glass-border); 
                    border-radius: 16px; padding: 1.5rem; box-shadow: var(--shadow-md);">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <div class="music-bar"></div>
                <div class="music-bar"></div>
                <div class="music-bar"></div>
                <div class="music-bar"></div>
                <h3 style="margin: 0; flex: 1;">{title}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.audio(audio_data, format="audio/wav")
    
    with col2:
        st.markdown("""
        <div style="background: var(--glass-bg); backdrop-filter: blur(16px); border: 1px solid var(--glass-border); 
                    border-radius: 16px; padding: 1.5rem; text-align: center; height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎵</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary);">Ready to Play</div>
        </div>
        """, unsafe_allow_html=True)


def mini_audio_player(audio_data: bytes, label: str = "Preview") -> None:
    """
    Create a compact audio player for quick previews.
    
    Args:
        audio_data: Audio file bytes
        label: Label for the player
    
    Example:
        >>> mini_audio_player(audio_bytes, label="Quick Preview")
    """
    st.markdown(f"""
    <div style="background: var(--glass-bg); backdrop-filter: blur(16px); border: 1px solid var(--glass-border); 
                border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
        <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem; font-weight: 600;">
            🎧 {label}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.audio(audio_data, format="audio/wav")


# =============================================================================
# FILE UPLOADER COMPONENTS
# =============================================================================

def beautiful_file_uploader(
    label: str = "Drop your audio file here",
    accepted_types: List[str] = ["wav", "mp3", "flac"],
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> Optional[Any]:
    """
    Create a beautiful file uploader with drag-and-drop support and custom styling.
    
    Args:
        label: Uploader label text
        accepted_types: List of accepted file extensions
        help_text: Optional help text to display
        key: Optional unique key for the uploader
    
    Returns:
        Uploaded file object or None
    
    Example:
        >>> file = beautiful_file_uploader("Upload Audio", accepted_types=["wav", "mp3"])
    """
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">📁</div>
        <h3 style="color: var(--primary); margin-bottom: 0.5rem;">{label}</h3>
        {f'<p style="color: var(--text-secondary); font-size: 0.9rem;">{help_text}</p>' if help_text else ''}
        <p style="color: #888; font-size: 0.85rem;">
            Supported formats: {', '.join(ext.upper() if ext.startswith('.') else f'.{ext.upper()}' for ext in accepted_types)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "",
        type=accepted_types,
        key=key,
        label_visibility="collapsed"
    )
    
    return uploaded_file


# =============================================================================
# LOADING & PROGRESS COMPONENTS
# =============================================================================

from contextlib import contextmanager

@contextmanager
def loading_skeleton(message: str = "Loading...", *args):
    """
    Context manager for displaying loading state using Streamlit's spinner.
    
    Args:
        message: Loading message to display
    
    Example:
        >>> with loading_skeleton("Processing..."):
        ...     # Do work here
        ...     pass
    """
    with st.spinner(message):
        yield


def loading_spinner(message: str = "Processing...") -> None:
    """
    Display a custom loading spinner with message.
    
    Args:
        message: Loading message to display
    
    Example:
        >>> loading_spinner("Generating music...")
    """
    spinner_html = f"""
    <div style="text-align: center; padding: 3rem 0;">
        <div style="display: inline-block; width: 50px; height: 50px; border: 5px solid rgba(29, 185, 84, 0.2); 
                    border-top-color: #1DB954; border-radius: 50%; animation: spin 1s linear infinite;"></div>
        <p style="margin-top: 1rem; color: var(--text-secondary); font-weight: 600; font-size: 1.1rem;">
            {message}
        </p>
    </div>
    """
    st.markdown(spinner_html, unsafe_allow_html=True)


def progress_bar(value: float, label: str = "", show_percentage: bool = True) -> None:
    """
    Display a custom styled progress bar.
    
    Args:
        value: Progress value between 0 and 1
        label: Optional label for the progress bar
        show_percentage: Whether to show percentage value
    
    Example:
        >>> progress_bar(0.65, label="Processing", show_percentage=True)
    """
    percentage = int(value * 100)
    
    bar_html = f"""
    <div style="margin: 1rem 0;">
        {f'<div style="margin-bottom: 0.5rem; color: var(--text-secondary); font-weight: 600;">{label}</div>' if label else ''}
        <div style="background: rgba(255, 255, 255, 0.1); border-radius: 9999px; height: 12px; overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);">
            <div style="background: var(--gradient-primary); height: 100%; width: {percentage}%; 
                        transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1); border-radius: 9999px;
                        box-shadow: 0 0 20px rgba(29, 185, 84, 0.5);"></div>
        </div>
        {f'<div style="text-align: right; margin-top: 0.25rem; color: var(--primary); font-weight: 700; font-size: 0.9rem;">{percentage}%</div>' if show_percentage else ''}
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


# =============================================================================
# BUTTON COMPONENTS
# =============================================================================

def action_button(label: str, icon: str = "▶️", gradient: str = "primary", key: Optional[str] = None) -> bool:
    """
    Create a stylish action button with icon.
    
    Args:
        label: Button text
        icon: Emoji icon
        gradient: Gradient style - 'primary', 'secondary', 'gold'
        key: Optional unique key
    
    Returns:
        True if button was clicked
    
    Example:
        >>> if action_button("Generate Music", icon="🎵", gradient="primary"):
        >>>     # Do something
    """
    button_html = f"""
    <style>
        .action-btn-{key or 'default'} {{
            background: var(--gradient-{gradient});
            color: var(--text-inverse);
            border: none;
            border-radius: 9999px;
            padding: 1rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow-md), var(--shadow-glow);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .action-btn-{key or 'default'}:hover {{
            transform: translateY(-3px) scale(1.02);
            box-shadow: var(--shadow-lg), 0 0 40px rgba(29, 185, 84, 0.6);
        }}
    </style>
    """
    st.markdown(button_html, unsafe_allow_html=True)
    
    return st.button(f"{icon} {label}", key=key, use_container_width=False)


def preset_button(label: str, description: str, icon: str = "🎵", key: Optional[str] = None) -> bool:
    """
    Create a preset selection button with description.
    
    Args:
        label: Preset name
        description: Preset description
        icon: Emoji icon
        key: Optional unique key
    
    Returns:
        True if button was clicked
    
    Example:
        >>> if preset_button("Jazz Fusion", "Smooth and sophisticated", icon="🎷"):
        >>>     # Load jazz preset
    """
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown(f"""
        <div class="preset-btn" style="text-align: left; animation: fadeInScale 0.4s cubic-bezier(0.16, 1, 0.3, 1);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <h4 style="margin-bottom: 0.25rem; color: var(--text-primary);">{label}</h4>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        return st.button("Select", key=key)


# =============================================================================
# STATUS & BADGE COMPONENTS
# =============================================================================

def status_indicator(status: str, label: str = "") -> None:
    """
    Display a status indicator with optional label.
    
    Args:
        status: Status type - 'success', 'warning', 'error', 'info'
        label: Optional status label
    
    Example:
        >>> status_indicator("success", "Recording Active")
    """
    status_colors = {
        "success": "#1DB954",
        "warning": "#F59E0B",
        "error": "#EC4899",
        "info": "#6366F1"
    }
    
    color = status_colors.get(status, status_colors["info"])
    
    indicator_html = f"""
    <div style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; 
                background: {color}20; border-radius: 9999px; border: 1px solid {color}50;">
        <div class="status-indicator status-{status}" style="width: 10px; height: 10px; background: {color}; 
                    border-radius: 50%; box-shadow: 0 0 12px {color}; animation: pulse 2s infinite;"></div>
        <span style="color: {color}; font-weight: 600; font-size: 0.9rem;">{label or status.upper()}</span>
    </div>
    """
    st.markdown(indicator_html, unsafe_allow_html=True)


def badge(text: str, type: str = "primary") -> None:
    """
    Display a badge with text.
    
    Args:
        text: Badge text
        type: Badge type - 'primary', 'secondary', 'accent'
    
    Example:
        >>> badge("New Feature", type="accent")
    """
    type_gradients = {
        "primary": "var(--gradient-primary)",
        "secondary": "var(--gradient-secondary)",
        "accent": "var(--gradient-gold)"
    }
    
    gradient = type_gradients.get(type, type_gradients["primary"])
    
    badge_html = f"""
    <span class="badge" style="background: {gradient}; display: inline-block; padding: 0.35rem 0.75rem; 
                border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: var(--text-inverse); 
                text-transform: uppercase; letter-spacing: 0.05em; box-shadow: var(--shadow-sm); margin: 0.25rem;">
        {text}
    </span>
    """
    st.markdown(badge_html, unsafe_allow_html=True)


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

def waveform_viz(audio_array: np.ndarray, sample_rate: int = 44100, title: str = "Waveform") -> None:
    """
    Create an interactive waveform visualization.
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate of the audio
        title: Chart title
    
    Example:
        >>> waveform_viz(audio_data, sample_rate=44100, title="Track Waveform")
    """
    # Generate time axis
    duration = len(audio_array) / sample_rate
    time = np.linspace(0, duration, len(audio_array))
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=audio_array,
        mode='lines',
        line=dict(color='#1DB954', width=1),
        fill='tozeroy',
        fillcolor='rgba(29, 185, 84, 0.3)',
        name='Waveform'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1DB954')),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color='#A0AEC0'),
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)


def spectrum_viz(frequencies: np.ndarray, magnitudes: np.ndarray, title: str = "Frequency Spectrum") -> None:
    """
    Create a frequency spectrum visualization.
    
    Args:
        frequencies: Frequency values
        magnitudes: Magnitude values
        title: Chart title
    
    Example:
        >>> spectrum_viz(freqs, mags, title="Audio Spectrum")
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=magnitudes,
        mode='lines',
        line=dict(color='#6366F1', width=2),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.3)',
        name='Spectrum'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#6366F1')),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color='#A0AEC0'),
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_xaxes(type="log", showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# LAYOUT COMPONENTS
# =============================================================================

def hero_section(title: str, subtitle: str, icon: str = "🎵", gradient: str = "primary") -> None:
    """
    Create a hero section for the landing page.
    
    Args:
        title: Main hero title
        subtitle: Hero subtitle/description
        icon: Large hero icon
        gradient: Gradient style ('primary', 'secondary', 'success', 'danger')
    
    Example:
        >>> hero_section("AI Music Generator", "Create stunning music with AI", icon="🎵", gradient="primary")
    """
    gradient_map = {
        "primary": "var(--gradient-primary)",
        "secondary": "var(--gradient-secondary)",
        "success": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "danger": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
    }
    
    gradient_style = gradient_map.get(gradient, gradient_map["primary"])
    
    hero_html = f"""
    <div class="hero-section" style="text-align: center; padding: 4rem 0; position: relative;">
        <div style="font-size: 6rem; margin-bottom: 1rem; animation: fadeInScale 0.8s cubic-bezier(0.16, 1, 0.3, 1);">
            {icon}
        </div>
        <h1 style="font-size: clamp(2.5rem, 5vw, 4.5rem); margin-bottom: 1rem; 
                   background: {gradient_style}; -webkit-background-clip: text; background-clip: text;
                   -webkit-text-fill-color: transparent; animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.2s both;">
            {title}
        </h1>
        <p style="font-size: 1.25rem; color: var(--text-secondary); max-width: 600px; margin: 0 auto; line-height: 1.6;
                  animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.4s both;">
            {subtitle}
        </p>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)


def section_header(title: str, subtitle: Optional[str] = None, icon: str = "🎵") -> None:
    """
    Create a section header with optional subtitle.
    
    Args:
        title: Section title
        subtitle: Optional subtitle
        icon: Section icon
    
    Example:
        >>> section_header("Generate Music", "Create your masterpiece", icon="🎸")
    """
    header_html = f"""
    <div style="margin: 3rem 0 2rem 0; animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);">
        <h2 style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
            <span style="font-size: 2rem;">{icon}</span>
            <span>{title}</span>
        </h2>
        {f'<p style="color: var(--text-secondary); font-size: 1.1rem; margin-left: 3rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def divider(text: Optional[str] = None, gradient: bool = True) -> None:
    """
    Create a stylish divider with optional text.
    
    Args:
        text: Optional text to display in the divider
        gradient: Whether to use gradient colors
    
    Example:
        >>> divider("OR", gradient=True)
    """
    if text:
        divider_html = f"""
        <div style="display: flex; align-items: center; margin: 2rem 0;">
            <div style="flex: 1; height: 2px; background: {'var(--gradient-primary)' if gradient else 'rgba(255,255,255,0.1)'};"></div>
            <span style="padding: 0 1rem; color: var(--text-secondary); font-weight: 600; font-size: 0.9rem;">{text}</span>
            <div style="flex: 1; height: 2px; background: {'var(--gradient-primary)' if gradient else 'rgba(255,255,255,0.1)'};"></div>
        </div>
        """
    else:
        divider_html = f"""
        <div style="height: 2px; background: {'var(--gradient-rainbow)' if gradient else 'rgba(255,255,255,0.1)'}; 
                    margin: 2rem 0; border-radius: 9999px;"></div>
        """
    
    st.markdown(divider_html, unsafe_allow_html=True)


# =============================================================================
# EMPTY STATE COMPONENTS
# =============================================================================

def empty_state(
    icon: str = "📁",
    title: str = "No files yet",
    description: str = "Upload a file to get started",
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None
) -> None:
    """
    Display an empty state placeholder.
    
    Args:
        icon: Large icon for empty state
        title: Empty state title
        description: Empty state description
        action_label: Optional action button label
        action_callback: Optional callback function for action button
    
    Example:
        >>> empty_state(icon="🎵", title="No tracks", description="Generate your first track")
    """
    empty_html = f"""
    <div style="text-align: center; padding: 4rem 2rem; background: var(--glass-bg); backdrop-filter: blur(16px);
                border: 2px dashed var(--glass-border); border-radius: 16px; margin: 2rem 0;">
        <div style="font-size: 5rem; margin-bottom: 1rem; opacity: 0.5;">{icon}</div>
        <h3 style="color: var(--text-primary); margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">{description}</p>
    </div>
    """
    st.markdown(empty_html, unsafe_allow_html=True)
    
    if action_label and action_callback:
        if st.button(action_label, key="empty_state_action"):
            action_callback()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def add_vertical_space(height: int = 2) -> None:
    """
    Add vertical spacing between components.
    
    Args:
        height: Number of rem units for spacing
    
    Example:
        >>> add_vertical_space(3)
    """
    st.markdown(f'<div style="height: {height}rem;"></div>', unsafe_allow_html=True)


def centered_container(content_func: Callable) -> None:
    """
    Wrap content in a centered container with max width.
    
    Args:
        content_func: Function that renders the content
    
    Example:
        >>> def my_content():
        >>>     st.write("Centered content")
        >>> centered_container(my_content)
    """
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        content_func()


def success_message(message: str, icon: str = "✅") -> None:
    """
    Display a beautiful success message with glassmorphism styling.
    
    Args:
        message: Success message text
        icon: Icon to display
    """
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 16px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15);
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <span style="color: #10b981; font-weight: 500; font-size: 0.95rem;">{message}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def error_message(message: str, icon: str = "❌") -> None:
    """
    Display a beautiful error message with glassmorphism styling.
    
    Args:
        message: Error message text
        icon: Icon to display
    """
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 16px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(239, 68, 68, 0.15);
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <span style="color: #ef4444; font-weight: 500; font-size: 0.95rem;">{message}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def info_message(message: str, icon: str = "ℹ️") -> None:
    """
    Display a beautiful info message with glassmorphism styling.
    
    Args:
        message: Info message text
        icon: Icon to display
    """
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.15) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 16px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
        ">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <span style="color: #3b82f6; font-weight: 500; font-size: 0.95rem;">{message}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def animated_progress_bar(value: float, label: str = "", gradient: str = "primary") -> None:
    """
    Display an animated progress bar with gradient styling.
    
    Args:
        value: Progress value (0-1)
        label: Label text
        gradient: Gradient style ('primary', 'secondary', 'success')
    """
    gradients = {
        "primary": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
        "secondary": "linear-gradient(90deg, #f093fb 0%, #f5576c 100%)",
        "success": "linear-gradient(90deg, #4facfe 0%, #00f2fe 100%)"
    }
    
    gradient_style = gradients.get(gradient, gradients["primary"])
    percentage = int(value * 100)
    
    st.markdown(f"""
        <div style="margin: 1rem 0;">
            {f'<div style="color: #fff; margin-bottom: 0.5rem; font-size: 0.9rem;">{label}</div>' if label else ''}
            <div style="
                background: rgba(255, 255, 255, 0.1);
                border-radius: 50px;
                height: 8px;
                overflow: hidden;
                backdrop-filter: blur(10px);
            ">
                <div style="
                    width: {percentage}%;
                    height: 100%;
                    background: {gradient_style};
                    border-radius: 50px;
                    transition: width 0.3s ease;
                    box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
                "></div>
            </div>
            <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.85rem; margin-top: 0.25rem; text-align: right;">
                {percentage}%
            </div>
        </div>
    """, unsafe_allow_html=True)


def mood_indicator(mood: str, confidence: float = 1.0) -> None:
    """
    Display a mood indicator with confidence level.
    
    Args:
        mood: Mood text (e.g., 'Happy', 'Sad', 'Energetic')
        confidence: Confidence level (0-1)
    """
    mood_colors = {
        "happy": "#fbbf24",
        "sad": "#60a5fa",
        "energetic": "#f87171",
        "calm": "#34d399",
        "romantic": "#f472b6",
        "dark": "#9333ea"
    }
    
    mood_lower = mood.lower()
    color = mood_colors.get(mood_lower, "#a78bfa")
    
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            display: inline-block;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background: {color};
                    box-shadow: 0 0 10px {color};
                "></div>
                <span style="color: #fff; font-weight: 500;">{mood}</span>
                <span style="color: rgba(255, 255, 255, 0.6); font-size: 0.85rem;">({int(confidence * 100)}%)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def feature_card(title: str, description: str, icon: str = "✨", active: bool = False) -> None:
    """
    Display a feature card with icon and description.
    
    Args:
        title: Feature title
        description: Feature description
        icon: Icon emoji
        active: Whether the feature is active/selected
    """
    border_color = "rgba(102, 126, 234, 0.5)" if active else "rgba(255, 255, 255, 0.2)"
    bg_gradient = "linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%)" if active else "linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%)"
    
    st.markdown(f"""
        <div style="
            background: {bg_gradient};
            backdrop-filter: blur(10px);
            border: 2px solid {border_color};
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
            cursor: pointer;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.75rem;">{icon}</div>
            <h3 style="color: #fff; margin: 0 0 0.5rem 0; font-size: 1.1rem;">{title}</h3>
            <p style="color: rgba(255, 255, 255, 0.7); margin: 0; font-size: 0.9rem; line-height: 1.5;">
                {description}
            </p>
        </div>
    """, unsafe_allow_html=True)


def stat_card(value: str, label: str, icon: str = "📊", trend: Optional[str] = None) -> None:
    """
    Display a statistics card with value, label, and optional trend.
    
    Args:
        value: Main statistic value
        label: Label text
        icon: Icon emoji
        trend: Trend indicator (e.g., '+12%', '-5%')
    """
    trend_color = "#10b981" if trend and trend.startswith("+") else "#ef4444" if trend and trend.startswith("-") else "#6b7280"
    
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                <span style="font-size: 1.75rem;">{icon}</span>
                {f'<span style="color: {trend_color}; font-size: 0.85rem; font-weight: 500;">{trend}</span>' if trend else ''}
            </div>
            <div style="color: #fff; font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;">
                {value}
            </div>
            <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">
                {label}
            </div>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ENHANCED AUDIO PLAYER COMPONENT
# =============================================================================

def enhanced_audio_player(
    file_path: str,
    title: str = "Audio Player",
    key: str = "player",
    show_download: bool = True,
    show_favorite: bool = True,
    on_favorite: Optional[Callable] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced audio player with controls, download, and favorite buttons.
    
    Args:
        file_path: Path to audio file
        title: Display title
        key: Unique key for component
        show_download: Show download button
        show_favorite: Show favorite button  
        on_favorite: Callback when favorite clicked
        metadata: Optional metadata (genre, mood, duration, etc.)
    
    Returns:
        Dict with user actions: {'download': bool, 'favorite': bool}
    """
    actions = {'download': False, 'favorite': False}
    
    # Check if already favorited
    is_favorited = False
    if 'favorites' in st.session_state:
        is_favorited = file_path in st.session_state.favorites
    
    # Player container with glassmorphic style
    with glass_card_container():
        # Title row
        col_title, col_buttons = st.columns([3, 1])
        with col_title:
            st.markdown(f"### 🎵 {title}")
            if metadata:
                meta_parts = []
                if 'genre' in metadata and metadata['genre']:
                    meta_parts.append(f"**Genre:** {metadata['genre']}")
                if 'mood' in metadata and metadata['mood']:
                    meta_parts.append(f"**Mood:** {metadata['mood']}")
                if 'duration' in metadata and metadata['duration']:
                    meta_parts.append(f"**Duration:** {metadata['duration']}s")
                if meta_parts:
                    st.markdown(" • ".join(meta_parts))
        
        with col_buttons:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if show_download:
                    if st.button("⬇️", key=f"dl_{key}", help="Download audio"):
                        actions['download'] = True
            with btn_col2:
                if show_favorite:
                    fav_icon = "⭐" if is_favorited else "☆"
                    if st.button(fav_icon, key=f"fav_{key}", help="Add to favorites"):
                        actions['favorite'] = True
                        if on_favorite:
                            on_favorite(file_path)
                        else:
                            # Default favorite handling
                            if 'favorites' not in st.session_state:
                                st.session_state.favorites = []
                            if file_path in st.session_state.favorites:
                                st.session_state.favorites.remove(file_path)
                                st.toast("Removed from favorites", icon="💔")
                            else:
                                st.session_state.favorites.append(file_path)
                                st.toast("Added to favorites!", icon="⭐")
        
        # Audio player
        st.audio(file_path, format='audio/wav')
    
    return actions


def simple_audio_player(file_path: str, title: str = None, key: str = "player"):
    """
    Simple audio player for quick preview.
    
    Args:
        file_path: Path to audio file
        title: Optional title
        key: Unique key
    """
    if title:
        st.markdown(f"**🎵 {title}**")
    st.audio(file_path, format='audio/wav')


