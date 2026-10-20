import os
import json
import pandas as pd
import streamlit as st

from src.ui.components import (
    render_metric_card,
    render_evidence_badge,
    render_scientific_disclosure_footer
)


def render_experiments_page():
    """
    Render Structured Experiment Laboratory Page.
    Presents experiment cards for Step 59 research: Peak Frequency, ERD Features, FBCSP, Riemannian, LOSO, Cross-Run, etc.
    """
    st.markdown("""
    <div style="background:linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border:2px solid #00F2FE; border-radius:12px; padding:20px; margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="color:#00F2FE; font-weight:800; font-size:0.85rem; letter-spacing:0.08em;">🧪 EXPERIMENT LABORATORY</div>
                <h1 style="font-family:'Space Grotesk',sans-serif; color:#F8FAFC; margin:4px 0 0 0; font-size:2.0rem;">
                    STEP 59 SCIENTIFIC EXPERIMENT RUNS
                </h1>
                <div style="color:#94A3B8; font-size:0.9rem; margin-top:4px;">
                    Authentic Controlled EEG Experiments | Calibration Data (R03+R07) Only | Independent Unseen-Run Protocol
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    step59_dir = "models/step59"

    # Tabbed view across core experiment categories
    tab_pf, tab_erd, tab_fbcsp, tab_riem, tab_transfer, tab_rank = st.tabs([
        "🧠 Peak Frequency Adaptation (Exp A)",
        "⚡ ERD Features & Temporal (Exp B)",
        "📊 FBCSP & Thresholds (Exp D & F)",
        "🌐 Riemannian Alignment (Exp C)",
        "🔁 Cross-Run & LOSO (Exp H & I)",
        "🏅 Candidate Ranking (Exp E & Ranking)"
    ])

    # 1. PEAK FREQUENCY ADAPTATION
    with tab_pf:
        st.markdown("### 🧠 Subject-Specific Sensorimotor Peak Frequency Adaptation")
        st.write("<b>Objective:</b> Estimate individual Mu (8–13 Hz) and Beta (13–30 Hz) peak frequencies on calibration data to build subject-adaptive filter banks.")
        render_evidence_badge("AUTHENTIC RECOMPUTED")
        st.markdown("<br>", unsafe_allow_html=True)

        pf_path = os.path.join(step59_dir, "peak_frequency_adaptation.csv")
        if os.path.exists(pf_path):
            df_pf = pd.read_csv(pf_path)
            st.dataframe(df_pf, use_container_width=True, hide_index=True)
        else:
            st.info("Peak frequency data loading...")

    # 2. ERD FEATURES & TEMPORAL
    with tab_erd:
        st.markdown("### ⚡ ERD Feature Engineering & Sub-Window Optimization")
        st.write("<b>Objective:</b> Extract log power, ERD suppression, C3/C4 asymmetry, slope, AUC, and evaluate temporal sub-windows across the 0.5s–2.5s event plateau.")
        render_evidence_badge("AUTHENTIC RECOMPUTED")
        st.markdown("<br>", unsafe_allow_html=True)

        col_ef, col_to = st.columns(2)
        with col_ef:
            st.markdown("#### Feature Set Comparison")
            ef_path = os.path.join(step59_dir, "erd_features.csv")
            if os.path.exists(ef_path):
                st.dataframe(pd.read_csv(ef_path), use_container_width=True, hide_index=True)

        with col_to:
            st.markdown("#### Temporal Sub-Window Optimization")
            to_path = os.path.join(step59_dir, "temporal_optimization.csv")
            if os.path.exists(to_path):
                st.dataframe(pd.read_csv(to_path), use_container_width=True, hide_index=True)

    # 3. FBCSP & THRESHOLDS
    with tab_fbcsp:
        st.markdown("### 📊 FBCSP Optimization & Calibrated Decision Thresholds")
        st.write("<b>Objective:</b> Evaluate multi-band CSP component selection and calibrate REST probability decision threshold to maintain REST recall >= 85%.")
        render_evidence_badge("AUTHENTIC RECOMPUTED")
        st.markdown("<br>", unsafe_allow_html=True)

        col_fb, col_th = st.columns(2)
        with col_fb:
            st.markdown("#### FBCSP Filter Bank Configurations")
            fb_path = os.path.join(step59_dir, "fbcsp_optimization.csv")
            if os.path.exists(fb_path):
                st.dataframe(pd.read_csv(fb_path), use_container_width=True, hide_index=True)

        with col_th:
            st.markdown("#### REST Probability Threshold Optimization")
            th_path = os.path.join(step59_dir, "threshold_optimization.csv")
            if os.path.exists(th_path):
                st.dataframe(pd.read_csv(th_path), use_container_width=True, hide_index=True)

    # 4. RIEMANNIAN ALIGNMENT
    with tab_riem:
        st.markdown("### 🌐 Riemannian Tangent Space Covariance Alignment")
        st.write("<b>Objective:</b> Transform covariance matrices to Riemannian Tangent Space using Ledoit-Wolf shrinkage and test calibration centroid alignment.")
        render_evidence_badge("AUTHENTIC RECOMPUTED")
        st.markdown("<br>", unsafe_allow_html=True)

        riem_path = os.path.join(step59_dir, "riemannian_alignment.csv")
        if os.path.exists(riem_path):
            st.dataframe(pd.read_csv(riem_path), use_container_width=True, hide_index=True)

    # 5. CROSS-RUN & LOSO
    with tab_transfer:
        st.markdown("### 🔁 Cross-Run Transfer & Leave-One-Subject-Out (LOSO) Validation")
        st.write("<b>Objective:</b> Evaluate cross-run generalization (R03 -> R07 and R07 -> R03) and leave-one-subject-out performance across 10 subjects.")
        render_evidence_badge("AUTHENTIC RECOMPUTED")
        st.markdown("<br>", unsafe_allow_html=True)

        col_cr, col_loso = st.columns(2)
        with col_cr:
            st.markdown("#### Cross-Run Transfer Matrix")
            cr_path = os.path.join(step59_dir, "cross_run_validation.csv")
            if os.path.exists(cr_path):
                st.dataframe(pd.read_csv(cr_path), use_container_width=True, hide_index=True)

        with col_loso:
            st.markdown("#### LOSO Held-Out Subject Results")
            loso_path = os.path.join(step59_dir, "loso_validation.csv")
            if os.path.exists(loso_path):
                st.dataframe(pd.read_csv(loso_path), use_container_width=True, hide_index=True)

    # 6. CANDIDATE RANKING
    with tab_rank:
        st.markdown("### 🏅 Candidate Model Leaderboard & Selection Ranking")
        st.write("<b>Objective:</b> Rank candidate models using weighted scoring across cross-run transfer, Macro F1, REST recall, and inference latency.")
        render_evidence_badge("AUTHENTIC RECOMPUTED")
        st.markdown("<br>", unsafe_allow_html=True)

        rank_path = os.path.join(step59_dir, "candidate_ranking.csv")
        if os.path.exists(rank_path):
            st.dataframe(pd.read_csv(rank_path), use_container_width=True, hide_index=True)

    render_scientific_disclosure_footer()
