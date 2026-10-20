import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal

from src.feature_extraction import calculate_fft_psd, calculate_band_powers
from src.step17b_pipeline import FILTER_BANKS
from src.ui.components import render_scientific_disclosure_footer


def render_eeg_explorer_page(raw, info, df_events):
    """
    Render EEG Signal & Feature Explorer Page (Requirement 7).
    Lazy-loaded tabs: RAW EEG, FREQUENCY SPECTRUM, BAND POWERS, FILTER BANK, CSP FEATURES, EVENT TIMELINE.
    """
    st.markdown("## 📊 EEG SIGNAL & FEATURE EXPLORER")
    st.caption("Interactive multi-channel signal inspection, spectral estimation, filter bank sub-band decompositions, CSP spatial patterns, and event timeline for PhysioNet motor imagery recordings.")

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🌊 RAW EEG",
        "📈 FREQUENCY SPECTRUM",
        "⚡ BAND POWERS",
        "🎛️ FILTER BANK",
        "🧩 CSP FEATURES",
        "⏱️ EVENT TIMELINE"
    ])

    # Extract 3-second segment for visualization
    sfreq = float(info.get("sfreq", 160.0))
    n_samples_3s = int(sfreq * 3.0)
    data_2d = raw.get_data()[:, :n_samples_3s] * 1e6 # microvolts
    ch_names = info.get("ch_names", [f"Ch{i+1}" for i in range(data_2d.shape[0])])

    # ------------------------------------------------------------------
    # TAB 1: RAW EEG
    # ------------------------------------------------------------------
    with t1:
        st.markdown("### Raw Multi-Channel EEG Signals")
        selected_chans = st.multiselect("Select Channels to Display:", options=ch_names, default=ch_names[:5], key="explorer_raw_chans")

        if selected_chans:
            t_arr = np.arange(data_2d.shape[1]) / sfreq
            fig_raw = go.Figure()
            for ch in selected_chans:
                idx = ch_names.index(ch)
                fig_raw.add_trace(go.Scatter(x=t_arr, y=data_2d[idx], mode='lines', name=ch))
            fig_raw.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                height=400,
                xaxis=dict(title="Time (sec)", gridcolor="#334155"),
                yaxis=dict(title="Amplitude (µV)", gridcolor="#334155")
            )
            st.plotly_chart(fig_raw, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2: FREQUENCY SPECTRUM
    # ------------------------------------------------------------------
    with t2:
        st.markdown("### Power Spectral Density (FFT PSD)")
        ch_target = st.selectbox("Select Target Channel:", options=ch_names, index=0, key="explorer_psd_ch")
        idx_psd = ch_names.index(ch_target)
        freqs, psd = calculate_fft_psd(data_2d[idx_psd], sfreq=sfreq)

        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', line=dict(color='#00F2FE', width=2), name=ch_target))
        fig_psd.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            height=380,
            xaxis=dict(title="Frequency (Hz)", range=[0, 50], gridcolor="#334155"),
            yaxis=dict(title="PSD (µV²/Hz)", gridcolor="#334155")
        )
        st.plotly_chart(fig_psd, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 3: BAND POWERS
    # ------------------------------------------------------------------
    with t3:
        st.markdown("### Canonical EEG Band Power Distribution")
        bp_dict = calculate_band_powers(data_2d[0], sfreq=sfreq)
        df_bp = pd.DataFrame(list(bp_dict.items()), columns=["Band", "Power (µV²)"])
        fig_bp = go.Figure(go.Bar(x=df_bp["Band"], y=df_bp["Power (µV²)"], marker_color='#10B981'))
        fig_bp.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            height=350,
            xaxis=dict(title="Canonical Band"),
            yaxis=dict(title="Absolute Power (µV²)")
        )
        st.plotly_chart(fig_bp, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 4: FILTER BANK
    # ------------------------------------------------------------------
    with t4:
        st.markdown("### Step 17B FBCSP 7 Sub-band Decompositions")
        df_fb = pd.DataFrame(FILTER_BANKS, columns=["Sub-band Name", "Low Cut (Hz)", "High Cut (Hz)"])
        st.dataframe(df_fb, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # TAB 5: CSP FEATURES
    # ------------------------------------------------------------------
    with t5:
        st.markdown("### Multiclass Spatial Patterns (CSP)")
        st.info("Filter Bank Common Spatial Pattern (FBCSP) projects 15-channel sensor signals into 4 spatial components per sub-band (28 total log-variance features).")

    # ------------------------------------------------------------------
    # TAB 6: EVENT TIMELINE
    # ------------------------------------------------------------------
    with t6:
        st.markdown("### Recorded Annotations & Event Markers")
        if df_events is not None and not df_events.empty:
            st.dataframe(df_events, use_container_width=True)
        else:
            st.info("No event markers present in active recording segment.")

    render_scientific_disclosure_footer()
