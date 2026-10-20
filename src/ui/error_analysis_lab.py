import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.ui.components import render_scientific_disclosure_footer


def render_error_analysis_lab_page():
    """
    Render 🔬 STEP 29 — ERROR ANALYSIS LAB
    Sections:
    1. Executive Summary
    2. Confusion Matrix
    3. Per-Subject Errors
    4. LEFT vs RIGHT Analysis
    5. Probability Analysis
    6. Frequency Band Analysis
    7. CSP Component Analysis
    8. Channel Analysis
    9. Specialist Classifier
    10. Signal Quality
    11. Confidence Analysis
    12. Personalization Analysis
    13. Command-Level Impact
    14. Final Candidate Comparison
    """
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
        <div>
            <h2 style="font-family:'Space Grotesk',sans-serif; margin:0; color:#F8FAFC;">🔬 STEP 29 — ERROR ANALYSIS LAB</h2>
            <div style="font-size:0.85rem; color:#94A3B8;">Deep Error Analysis & Left/Right Motor-Class Optimization Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    step29_dir = os.path.abspath("models/step29")
    has_report = os.path.exists(os.path.join(step29_dir, "summary_report.json"))

    if not has_report:
        st.warning("⚠️ Step 29 summary report missing. Run step29_pipeline.py to populate results.")
        return

    with open(os.path.join(step29_dir, "summary_report.json"), "r") as f:
        rep = json.load(f)

    # ------------------------------------------------------------------
    # 1. EXECUTIVE SUMMARY
    # ------------------------------------------------------------------
    st.markdown("### 1. Executive Summary & Champion Status")
    col_e1, col_e2, col_e3, col_e4 = st.columns([1, 1, 1, 1])

    with col_e1:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #10B981; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">DECISION</div>
            <div style="font-size:1.2rem; font-weight:bold; color:#10B981; margin:4px 0;">{rep.get("final_decision", "KEEP STEP 24")}</div>
            <div style="font-size:0.75rem; color:#F8FAFC;">Champion: {rep.get("best_candidate", "STEP 24 FBCSP")}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_e2:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">HOLDOUT ACCURACY</div>
            <div style="font-size:1.3rem; font-weight:bold; color:#00F2FE; margin:4px 0;">{rep.get("candidate_accuracy", 63.0)}%</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Baseline: {rep.get("baseline_accuracy", 74.67)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_e3:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">MACRO F1 SCORE</div>
            <div style="font-size:1.3rem; font-weight:bold; color:#00F2FE; margin:4px 0;">{rep.get("candidate_macro_f1", 55.8)}%</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Baseline F1: {rep.get("baseline_macro_f1", 70.18)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_e4:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">COMPUTATION LATENCY</div>
            <div style="font-size:1.3rem; font-weight:bold; color:#10B981; margin:4px 0;">{rep.get("latency_ms", 0.54)} ms</div>
            <div style="font-size:0.75rem; color:#94A3B8;">P95: {rep.get("p95_latency_ms", 0.63)} ms</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:14px 0; border-color:#334155;'>", unsafe_allow_html=True)

    # Tabs for 14 Sections
    tab_conf, tab_motor, tab_prob, tab_band, tab_chan, tab_spec, tab_cmd = st.tabs([
        "📊 Confusion & Errors",
        "🖐️ LEFT vs RIGHT Analysis",
        "📈 Probability & Bands",
        "🧠 CSP & Channels",
        "🤖 Specialist Search",
        "🛡️ Quality & Confidence",
        "🏆 Candidate Benchmark"
    ])

    # ------------------------------------------------------------------
    # TAB 1: 2. CONFUSION MATRIX & 3. PER-SUBJECT ERRORS
    # ------------------------------------------------------------------
    with tab_conf:
        col_c1, col_c2 = st.columns([1.1, 1.3])
        with col_c1:
            st.markdown("### 2. Validation Confusion Analysis")
            if os.path.exists(os.path.join(step29_dir, "confusion_analysis.csv")):
                df_cm = pd.read_csv(os.path.join(step29_dir, "confusion_analysis.csv"), index_col=0)
                fig_cm = px.imshow(
                    df_cm.values,
                    x=["Pred_REST", "Pred_LEFT", "Pred_RIGHT"],
                    y=["True_REST", "True_LEFT", "True_RIGHT"],
                    color_continuous_scale="Blues",
                    text_auto=True
                )
                fig_cm.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)

        with col_c2:
            st.markdown("### 3. Class-Level Precision & Recall Breakdown")
            if os.path.exists(os.path.join(step29_dir, "error_analysis.csv")):
                df_err = pd.read_csv(os.path.join(step29_dir, "error_analysis.csv"))
                st.dataframe(df_err, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2: 4. LEFT VS RIGHT ANALYSIS
    # ------------------------------------------------------------------
    with tab_motor:
        st.markdown("### 4. LEFT vs RIGHT Motor-Class Separation Analysis")
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; margin-bottom:12px;">
            <div style="color:#F8FAFC; font-weight:bold;">Motor Recall Breakdown:</div>
            <div>REST Recall: <strong style="color:#38BDF8;">{rep.get('rest_recall', 87.33)}%</strong></div>
            <div>LEFT_FIST Recall: <strong style="color:#10B981;">{rep.get('left_recall', 32.89)}%</strong></div>
            <div>RIGHT_FIST Recall: <strong style="color:#F59E0B;">{rep.get('right_recall', 44.59)}%</strong></div>
            <div>False LEFT Rate: <span>{rep.get('false_left_rate', 12.67)}%</span> | False RIGHT Rate: <span>{rep.get('false_right_rate', 12.67)}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # TAB 3: 5. PROBABILITY & 6. FREQUENCY BAND ANALYSIS
    # ------------------------------------------------------------------
    with tab_prob:
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.markdown("### 5. Probability Analysis")
            if os.path.exists(os.path.join(step29_dir, "probability_analysis.csv")):
                df_prob = pd.read_csv(os.path.join(step29_dir, "probability_analysis.csv"))
                st.dataframe(df_prob, use_container_width=True)

        with col_p2:
            st.markdown("### 6. Frequency Band Contribution")
            if os.path.exists(os.path.join(step29_dir, "band_analysis.csv")):
                df_band = pd.read_csv(os.path.join(step29_dir, "band_analysis.csv"))
                fig_band = px.bar(df_band, x="frequency_band", y="f_statistic", color="contribution_level", title="Band Feature Importance (F-Stat)")
                fig_band.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_band, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 4: 7. CSP COMPONENT & 8. CHANNEL ANALYSIS
    # ------------------------------------------------------------------
    with tab_chan:
        col_cs1, col_cs2 = st.columns([1, 1])
        with col_cs1:
            st.markdown("### 7. CSP Component Feature Importance")
            if os.path.exists(os.path.join(step29_dir, "csp_component_analysis.csv")):
                df_csp = pd.read_csv(os.path.join(step29_dir, "csp_component_analysis.csv"))
                st.dataframe(df_csp, use_container_width=True)

        with col_cs2:
            st.markdown("### 8. 15 Motor Channel Sensitivity")
            if os.path.exists(os.path.join(step29_dir, "channel_analysis.csv")):
                df_chan = pd.read_csv(os.path.join(step29_dir, "channel_analysis.csv"))
                fig_ch = px.bar(df_chan, x="channel_name", y="left_right_power_ratio", color="importance", title="Channel L/R Power Variance Ratio")
                fig_ch.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_ch, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 5: 9. SPECIALIST CLASSIFIER SEARCH
    # ------------------------------------------------------------------
    with tab_spec:
        st.markdown("### 9. 2-Stage Hierarchical Specialist Classifier Matrix")
        if os.path.exists(os.path.join(step29_dir, "specialist_results.csv")):
            df_spec = pd.read_csv(os.path.join(step29_dir, "specialist_results.csv"))
            st.dataframe(df_spec, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 6: 10. SIGNAL QUALITY & 11. CONFIDENCE & 12. PERSONALIZATION
    # ------------------------------------------------------------------
    with tab_cmd:
        st.markdown("### 10 & 11. Signal Quality & Step 26 Confidence Analysis")
        col_q1, col_q2 = st.columns([1, 1])
        with col_q1:
            if os.path.exists(os.path.join(step29_dir, "validation_results.csv")):
                st.markdown("##### Candidate Validation Matrix")
                df_val = pd.read_csv(os.path.join(step29_dir, "validation_results.csv"))
                st.dataframe(df_val, use_container_width=True)
        with col_q2:
            if os.path.exists(os.path.join(step29_dir, "final_benchmark.csv")):
                st.markdown("##### Final Holdout R11 Benchmark")
                df_bm = pd.read_csv(os.path.join(step29_dir, "final_benchmark.csv"))
                st.dataframe(df_bm, use_container_width=True)

    render_scientific_disclosure_footer()
