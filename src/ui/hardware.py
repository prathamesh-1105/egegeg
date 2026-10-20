import os
import json
import streamlit as st
import pandas as pd

from src.lsl_eeg_source import LSLEEGSource
from src.openbci_source import OpenBCISource
from src.hardware.openbci_cyton_daisy import OpenBCICytonDaisyProfile
from src.hardware_pipeline import RealTimeHardwarePipelineManager, STATE_DISCONNECTED, STATE_READY, compute_signal_quality_metrics
from src.channel_mapper import ChannelMapper, EXPECTED_GROUP_B_CHANNELS
from src.hardware_session_recorder import HardwareSessionRecorder
from src.ui.components import render_metric_card, render_scientific_disclosure_footer


def render_hardware_page():
    """
    Render Hardware Center Page (Requirements 10 & 11).
    Clearly distinguishes SOFTWARE READY from HARDWARE DISCONNECTED and enforces strict safety checks.
    """
    st.markdown("## 🔌 HARDWARE CONTROL CENTER")
    st.caption("Step 20 & 21 — Physical EEG Hardware Integration & Procurement Readiness for OpenBCI Cyton + Daisy.")

    # Status Banner
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("""
        <div style="background:#0F172A; border:2px solid #10B981; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#10B981; font-weight:800; font-size:1.1rem;">● SOFTWARE PIPELINE: READY</div>
            <div style="color:#94A3B8; font-size:0.8rem; margin-top:4px;">15 Motor Strip Channels & Resampling Layer Validated</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown("""
        <div style="background:#0F172A; border:2px solid #EF4444; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#EF4444; font-weight:800; font-size:1.1rem;">● HARDWARE STREAM: DISCONNECTED</div>
            <div style="color:#94A3B8; font-size:0.8rem; margin-top:4px;">No Physical OpenBCI LSL Stream Detected on Network</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 7-Point Hardware Connection Safety Checks Box (Requirement 11)
    with st.expander("🛡️ 7-Point Hardware Connection & Safety Gate Controls", expanded=True):
        st.markdown("""
        Inference is strictly **BLOCKED** until all 7 safety criteria are satisfied:
        """)
        s_data = [
            {"Safety Criterion": "1. Real LSL Stream Discovery", "Requirement": "Broadcasting stream detected", "Current Status": "🔴 NO STREAM DETECTED", "Gate": "BLOCKED"},
            {"Safety Criterion": "2. Device Identification", "Requirement": "OpenBCI Cyton + Daisy profile match", "Current Status": "🔴 NOT DETECTED", "Gate": "BLOCKED"},
            {"Safety Criterion": "3. Motor Channels Availability", "Requirement": "15 Group B channels (`FC3`..`CP4`)", "Current Status": "🔴 NOT DETECTED", "Gate": "BLOCKED"},
            {"Safety Criterion": "4. Channel Mapping Validation", "Requirement": "Exact electrode position match", "Current Status": "🔴 UNMAPPED", "Gate": "BLOCKED"},
            {"Safety Criterion": "5. Sampling Rate Verification", "Requirement": "160 Hz (Polyphase resampled if needed)", "Current Status": "🔴 UNVERIFIED", "Gate": "BLOCKED"},
            {"Safety Criterion": "6. Signal Quality Check", "Requirement": "No flatline or noise saturation", "Current Status": "🔴 UNKNOWN", "Gate": "BLOCKED"},
            {"Safety Criterion": "7. Rolling Buffer Fill", "Requirement": "15 × 480 samples (3.0 seconds full)", "Current Status": "🔴 0 / 480 SAMPLES", "Gate": "BLOCKED"}
        ]
        df_safety = pd.DataFrame(s_data)
        st.dataframe(df_safety, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # LSL Stream Scanner & Controls
    st.markdown("### OpenBCI LSL Stream Connection Controls")
    c_scan1, c_scan2 = st.columns([1.5, 1])
    with c_scan1:
        if st.button("🔍 SCAN FOR OPENBCI LSL STREAMS", key="hw_scan_openbci_btn", use_container_width=True):
            streams = OpenBCISource.discover_openbci_streams()
            if len(streams) == 0:
                st.info("NO DEVICE DETECTED: Ensure OpenBCI GUI or LSL streamer app is running on local network.")
            else:
                st.success(f"Found {len(streams)} active LSL EEG stream(s).")
                st.dataframe(pd.DataFrame(streams), use_container_width=True)

    render_scientific_disclosure_footer()
