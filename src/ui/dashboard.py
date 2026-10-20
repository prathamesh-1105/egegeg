import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from src.ui.components import (
    render_metric_card,
    render_status_pill,
    render_evidence_badge,
    render_scientific_disclosure_footer
)


def render_dashboard_page():
    """
    Render Executive Overview Dashboard Page.
    Presents current authentic champion model (Step 59), dataset specs, model evolution chart, and quick navigation.
    """
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <div>
            <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2.2rem; font-weight:800; color:#F8FAFC; margin:0;">
                NEURO BCI
            </h1>
            <div style="font-size:1.0rem; color:#94A3B8; margin-top:4px;">
                Real-Time Motor Imagery Intelligence Platform & BCI Control Center
            </div>
        </div>
        <div style="background:#0F172A; border:1px solid #10B981; border-radius:20px; padding:8px 18px; text-align:center;">
            <span style="color:#10B981; font-weight:bold; font-size:0.85rem;">● SOFTWARE READY</span>
            <span style="color:#64748B; margin:0 8px;">|</span>
            <span style="color:#EF4444; font-weight:bold; font-size:0.85rem;">● HARDWARE DISCONNECTED</span>
            <span style="color:#64748B; margin:0 8px;">|</span>
            <span style="color:#00F2FE; font-weight:bold; font-size:0.85rem;">● R11 HOLDOUT ISOLATED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Step 59 Authentic Metrics
    sum59_p = "models/step59/summary_report.json"
    acc_str = "67.67%"
    f1_str = "62.24%"
    lat_str = "37.30 ms"
    rest_rec = "85.33%"
    left_rec = "57.89%"
    right_rec = "41.89%"

    if os.path.exists(sum59_p):
        try:
            with open(sum59_p, "r", encoding="utf-8") as f:
                r59 = json.load(f)
                acc_str = f"{r59['step59_authentic_r11_metrics']['accuracy']}%"
                f1_str = f"{r59['step59_authentic_r11_metrics']['macro_f1']}%"
                lat_str = f"{r59['latency_ms']} ms"
                rest_rec = f"{r59['step59_authentic_r11_metrics']['rest_recall']}%"
                left_rec = f"{r59['step59_authentic_r11_metrics']['left_fist_recall']}%"
                right_rec = f"{r59['step59_authentic_r11_metrics']['right_fist_recall']}%"
        except Exception:
            pass

    # 4 Major Status Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("CURRENT AUTHENTIC CHAMPION", "STEP 59", "Subject-Adaptive EEG Optimization", color="#10B981")
    with c2:
        render_metric_card("INDEPENDENT R11 ACCURACY", acc_str, "PhysioNet R11 Holdout (300 trials)", color="#00F2FE")
    with c3:
        render_metric_card("MACRO F1 SCORE", f1_str, "Balanced 3-Class Metric", color="#A855F7")
    with c4:
        render_metric_card("INFERENCE LATENCY", lat_str, "< 50.0 ms Realtime Constraint", color="#F59E0B")

    st.markdown("<br>", unsafe_allow_html=True)

    # Class Recall Strip & Dataset Specs
    col_rec, col_ds = st.columns(2)

    with col_rec:
        st.markdown("### 🎯 Authentic Class Recall Breakdown")
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:18px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#CBD5E1; font-weight:600;">REST State Recall:</span>
                <span style="color:#38BDF8; font-weight:700;">{rest_rec}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#CBD5E1; font-weight:600;">LEFT_FIST Recall:</span>
                <span style="color:#10B981; font-weight:700;">{left_rec}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#CBD5E1; font-weight:600;">RIGHT_FIST Recall:</span>
                <span style="color:#F59E0B; font-weight:700;">{right_rec}</span>
            </div>
            <div style="font-size:0.75rem; color:#64748B; margin-top:8px;">
                Evaluated on 300 genuine holdout trials (150 REST / 76 LEFT / 74 RIGHT)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ds:
        st.markdown("### 📁 Dataset Specifications")
        st.markdown("""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:18px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#CBD5E1; font-weight:600;">Dataset Source:</span>
                <span style="color:#A855F7; font-weight:700;">PhysioNet EEGMMIDB</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#CBD5E1; font-weight:600;">Subjects & Files:</span>
                <span style="color:#00F2FE; font-weight:700;">10 Subjects (S001–S010), 30 EDF Files</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#CBD5E1; font-weight:600;">Total Genuine Trials:</span>
                <span style="color:#10B981; font-weight:700;">900 Trials (600 Calib / 300 Test)</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#CBD5E1; font-weight:600;">Channels & Sampling:</span>
                <span style="color:#38BDF8; font-weight:700;">15 Motor Channels @ 160 Hz (3.0s window)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def set_nav_target(target_label: str):
        st.session_state.nav_sidebar_radio = target_label

    # Quick Navigation Buttons
    st.markdown("### 🚀 Primary Action Shortcuts")
    qa1, qa2, qa3, qa4 = st.columns(4)

    with qa1:
        st.button("🧠 OPEN LIVE BCI", on_click=set_nav_target, args=("🧠 Live BCI",), use_container_width=True, key="dash_btn_live_bci")

    with qa2:
        st.button("📊 EEG EXPLORER", on_click=set_nav_target, args=("📊 EEG Explorer",), use_container_width=True, key="dash_btn_explore_eeg")

    with qa3:
        st.button("🤖 MODEL CENTER", on_click=set_nav_target, args=("🤖 Models",), use_container_width=True, key="dash_btn_view_model")

    with qa4:
        st.button("🛡️ VALIDATION AUDIT", on_click=set_nav_target, args=("🛡️ Validation",), use_container_width=True, key="dash_btn_val_center")

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Evolution Chart (Legacy Reported vs Authentic Recomputed)
    st.markdown("### 📈 Model Evolution & Scientific Audit History")
    df_evo = pd.DataFrame([
        {"Step": "Step 34", "Architecture": "Error Correction Cascade", "Reported R11 Acc": "85.33%", "Authentic Status": "LEGACY REPORTED"},
        {"Step": "Step 54", "Architecture": "Adaptive Few-Shot Personalization", "Reported R11 Acc": "86.40%", "Authentic Status": "LEGACY REPORTED"},
        {"Step": "Step 55", "Architecture": "Zero-Leakage Personalization", "Reported R11 Acc": "86.40%", "Authentic Status": "LEGACY REPORTED (Contains Mock Logic)"},
        {"Step": "Step 56", "Architecture": "Unseen-Run Breakthrough Engine", "Reported R11 Acc": "86.40%", "Authentic Status": "LEGACY REPORTED (Contains Mock Logic)"},
        {"Step": "Step 57", "Architecture": "90% Error Recovery Lab", "Reported R11 Acc": "90.00%", "Authentic Status": "LEGACY REPORTED (Contains Mock Logic)"},
        {"Step": "Step 58", "Architecture": "Genuine Raw-EDF Rebuild", "Reported R11 Acc": "67.67%", "Authentic Status": "AUTHENTIC RECOMPUTED (Real EDF Baseline)"},
        {"Step": "Step 59", "Architecture": "Subject-Adaptive Peak ERD Engine", "Reported R11 Acc": "67.67%", "Authentic Status": "AUTHENTIC RECOMPUTED (Macro F1 62.24%)"}
    ])
    st.dataframe(df_evo, use_container_width=True, hide_index=True)

    render_scientific_disclosure_footer()
