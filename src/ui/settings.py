import streamlit as st
from src.ui.components import render_scientific_disclosure_footer


def render_settings_page():
    """
    Render Application Settings Page.
    Theme options, graph resolution, animation quality, and debug mode toggle.
    """
    st.markdown("## APPLICATION SETTINGS")
    st.caption("Customize UI theme, visual rendering quality, and developer diagnostic controls.")

    st.markdown("### Visual & Interface Preferences")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("UI Color Theme:", options=["Cyberpunk Dark (Default)", "OLED Black", "Midnight Blue"], index=0, key="settings_theme_sel")
        st.selectbox("Graph Plotly Template:", options=["plotly_dark (Default)", "plotly_white"], index=0, key="settings_plot_template")
    with c2:
        st.slider("3D Avatar Canvas Height (px):", min_value=300, max_value=600, value=380, step=20, key="settings_avatar_height")
        st.select_slider("Animation Quality:", options=["Low", "Medium", "High (Default)", "Ultra"], value="High (Default)", key="settings_anim_quality")

    st.markdown("---")
    st.markdown("### Developer & Diagnostic Controls")
    st.checkbox("⚡ Developer Debug Mode (Expose raw arrays, signal hashes, and tracebacks)", value=False, key="settings_debug_mode_chk")

    if st.session_state.get("settings_debug_mode_chk", False):
        st.info("Developer Debug Mode is ACTIVE. Extended diagnostic panels will appear in Live BCI and Signal Lab.")

    render_scientific_disclosure_footer()
