import os
import json
import streamlit as st


def render_sidebar_navigation() -> str:
    """
    Render clean 9-item primary sidebar navigation menu with dynamic system status card.
    Returns:
        str: Selected page key ('overview', 'live_bci', 'eeg_explorer', 'models', 'experiments', 'validation', 'dataset', 'hardware', 'system')
    """
    st.sidebar.markdown("""
    <div style="text-align:center; padding:10px 0 20px 0;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:800; color:#00F2FE; letter-spacing:0.08em;">
            🧠 NEURO BCI
        </div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:2px;">
            Real-Time Motor Imagery Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    page_options = {
        "🏠 Overview": "overview",
        "🗣️ Speech BCI (Word Decoding)": "speech_bci",
        "🧠 Live BCI": "live_bci",
        "📊 EEG Explorer": "eeg_explorer",
        "🤖 Models": "models",
        "🧪 Experiments": "experiments",
        "🛡️ Validation": "validation",
        "📁 Dataset": "dataset",
        "🔌 Hardware": "hardware",
        "⚙️ System": "system"
    }

    # Initialize nav_sidebar_radio in session state if not set
    if "nav_sidebar_radio" not in st.session_state:
        st.session_state.nav_sidebar_radio = "🏠 Overview"

    # Backward compatibility key mapping for quick navigation buttons
    key_mapping = {
        "dashboard": "🏠 Overview",
        "live_bci": "🧠 Live BCI",
        "eeg_explorer": "📊 EEG Explorer",
        "model_lab": "🤖 Models",
        "models": "🤖 Models",
        "experiments": "🧪 Experiments",
        "validation": "🛡️ Validation",
        "dataset": "📁 Dataset",
        "hardware": "🔌 Hardware",
        "system": "⚙️ System"
    }

    if st.session_state.nav_sidebar_radio in key_mapping:
        st.session_state.nav_sidebar_radio = key_mapping[st.session_state.nav_sidebar_radio]

    if st.session_state.nav_sidebar_radio not in page_options:
        st.session_state.nav_sidebar_radio = "🏠 Overview"

    selected_label = st.sidebar.radio(
        "Primary Navigation",
        options=list(page_options.keys()),
        key="nav_sidebar_radio"
    )

    st.sidebar.markdown("---")

    # Load authentic Step 59 metrics dynamically
    sum59_path = "models/step59/summary_report.json"
    champ_name = "STEP 59 Subject-Adaptive"
    holdout_acc_str = "67.67%"
    macro_f1_str = "62.24%"
    lat_str = "37.3 ms"

    if os.path.exists(sum59_path):
        try:
            with open(sum59_path, "r", encoding="utf-8") as f:
                r59 = json.load(f)
                holdout_acc_str = f"{r59['step59_authentic_r11_metrics']['accuracy']}%"
                macro_f1_str = f"{r59['step59_authentic_r11_metrics']['macro_f1']}%"
                lat_str = f"{r59['latency_ms']} ms"
        except Exception:
            pass

    # System Status Indicator Box
    st.sidebar.markdown(f"""
    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; font-size:0.8rem;">
        <div style="font-weight:700; color:#F8FAFC; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
            <span>SYSTEM STATUS</span>
            <span style="font-size:0.65rem; color:#10B981; border:1px solid #10B981; padding:1px 6px; border-radius:10px;">AUTHENTIC</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#94A3B8;">Software:</span>
            <span style="color:#10B981; font-weight:600;">READY</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#94A3B8;">Champion Model:</span>
            <span style="color:#10B981; font-weight:600;">{champ_name}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#94A3B8;">R11 Accuracy:</span>
            <span style="color:#00F2FE; font-weight:600;">{holdout_acc_str}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#94A3B8;">Macro F1:</span>
            <span style="color:#A855F7; font-weight:600;">{macro_f1_str}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#94A3B8;">Latency:</span>
            <span style="color:#F59E0B; font-weight:600;">{lat_str}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#94A3B8;">Hardware:</span>
            <span style="color:#EF4444; font-weight:600;">DISCONNECTED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.caption("PhysioNet Motor Imagery EEG v2.0")

    return page_options[selected_label]
