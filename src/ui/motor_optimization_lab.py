import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.ui.components import render_scientific_disclosure_footer


def render_motor_optimization_lab_page():
    """
    Render 🧠 STEP 32 — MOTOR OPTIMIZATION LAB
    Sections:
    1. Official Baseline
    2. Error Analysis
    3. Subject Difficulty
    4. Frequency Optimization
    5. CSP Optimization
    6. Motor Features
    7. Temporal Optimization
    8. Classifier Search
    9. Personalization
    10. Ensemble
    11. Validation Leaderboard
    12. Final Holdout Comparison
    """
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
        <div>
            <h2 style="font-family:'Space Grotesk',sans-serif; margin:0; color:#F8FAFC;">🧠 STEP 32 — MOTOR OPTIMIZATION LAB</h2>
            <div style="font-size:0.85rem; color:#94A3B8;">Subject-Specific Motor-Imagery Decoding & Frequency/Asymmetry Optimization</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    step32_dir = os.path.abspath("models/step32")
    has_report = os.path.exists(os.path.join(step32_dir, "summary_report.json"))

    if not has_report:
        st.warning("⚠️ Step 32 summary report missing. Run step32_pipeline.py to populate results.")
        return

    with open(os.path.join(step32_dir, "summary_report.json"), "r", encoding="utf-8") as f:
        rep = json.load(f)

    # ------------------------------------------------------------------
    # 1. OFFICIAL BASELINE & 12. FINAL HOLDOUT COMPARISON
    # ------------------------------------------------------------------
    st.markdown("### 1 & 12. Official Baseline & Final Holdout Decision")
    col_m1, col_m2, col_m3, col_m4 = st.columns([1, 1, 1, 1])

    with col_m1:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #10B981; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">FINAL DECISION</div>
            <div style="font-size:1.2rem; font-weight:bold; color:#10B981; margin:4px 0;">{rep.get("final_decision", "KEEP STEP 24")}</div>
            <div style="font-size:0.75rem; color:#F8FAFC;">Baseline Retained</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">HOLDOUT R11 ACCURACY</div>
            <div style="font-size:1.3rem; font-weight:bold; color:#00F2FE; margin:4px 0;">{rep.get("step32_candidate_r11", {}).get("accuracy", 70.33)}%</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Baseline: {rep.get("official_step24_baseline", {}).get("accuracy", 70.67)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">HOLDOUT MACRO F1</div>
            <div style="font-size:1.3rem; font-weight:bold; color:#00F2FE; margin:4px 0;">{rep.get("step32_candidate_r11", {}).get("macro_f1", 63.70)}%</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Baseline F1: {rep.get("official_step24_baseline", {}).get("macro_f1", 64.33)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">80% / 85% TARGETS</div>
            <div style="font-size:0.95rem; font-weight:bold; color:#F59E0B; margin:4px 0;">NOT REACHED</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Honest Evaluation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:14px 0; border-color:#334155;'>", unsafe_allow_html=True)

    tab_err, tab_freq, tab_csp, tab_temp, tab_clf, tab_lb = st.tabs([
        "🔍 Error Analysis & Transitions",
        "🌊 Frequency Search",
        "📐 CSP & Motor Features",
        "⏱️ Temporal Windows",
        "🤖 Classifiers & Personalization",
        "🏆 Validation Leaderboard"
    ])

    # ------------------------------------------------------------------
    # TAB 1: 2. ERROR ANALYSIS, 3. SUBJECT DIFFICULTY
    # ------------------------------------------------------------------
    with tab_err:
        col_e1, col_e2 = st.columns([1.2, 1.0])
        with col_e1:
            st.markdown("### 2 & 3. Per-Subject Accuracy & Difficulty Tier")
            if os.path.exists(os.path.join(step32_dir, "subject_error_analysis.csv")):
                df_err = pd.read_csv(os.path.join(step32_dir, "subject_error_analysis.csv"))
                fig_err = px.bar(
                    df_err,
                    x="subject",
                    y="accuracy",
                    color="difficulty_tier",
                    title="Per-Subject Accuracy (%) & Difficulty Tier",
                    labels={"accuracy": "Accuracy (%)", "subject": "Subject"}
                )
                fig_err.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_err, use_container_width=True)

        with col_e2:
            st.markdown("### Subject Error Transitions & Confusion Matrix")
            if os.path.exists(os.path.join(step32_dir, "confusion_matrix.csv")):
                df_cm = pd.read_csv(os.path.join(step32_dir, "confusion_matrix.csv"), index_col=0)
                st.dataframe(df_cm, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2: 4. FREQUENCY OPTIMIZATION
    # ------------------------------------------------------------------
    with tab_freq:
        st.markdown("### 4. Subject-Specific Frequency Range Search")
        if os.path.exists(os.path.join(step32_dir, "frequency_search.csv")):
            df_freq = pd.read_csv(os.path.join(step32_dir, "frequency_search.csv"))
            fig_freq = px.bar(
                df_freq,
                x="frequency_range",
                y="val_macro_f1",
                title="Validation Macro F1 (%) by Frequency Sub-Band"
            )
            fig_freq.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_freq, use_container_width=True)
            st.dataframe(df_freq, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 3: 5. CSP OPTIMIZATION, 6. MOTOR FEATURES
    # ------------------------------------------------------------------
    with tab_csp:
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("### 5. CSP Component Count Search")
            if os.path.exists(os.path.join(step32_dir, "csp_search.csv")):
                df_csp = pd.read_csv(os.path.join(step32_dir, "csp_search.csv"))
                st.dataframe(df_csp, use_container_width=True)

        with col_c2:
            st.markdown("### 6. Contralateral Sensorimotor Asymmetry Features")
            if os.path.exists(os.path.join(step32_dir, "motor_feature_search.csv")):
                df_motor = pd.read_csv(os.path.join(step32_dir, "motor_feature_search.csv"))
                st.dataframe(df_motor, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 4: 7. TEMPORAL OPTIMIZATION
    # ------------------------------------------------------------------
    with tab_temp:
        st.markdown("### 7. Event-Aligned Temporal Window & Onset Search")
        if os.path.exists(os.path.join(step32_dir, "temporal_search.csv")):
            df_temp = pd.read_csv(os.path.join(step32_dir, "temporal_search.csv"))
            fig_temp = px.bar(
                df_temp,
                x="temporal_config",
                y="val_macro_f1",
                title="Validation Macro F1 (%) by Temporal Window Configuration"
            )
            fig_temp.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_temp, use_container_width=True)
            st.dataframe(df_temp, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 5: 8. CLASSIFIERS, 9. PERSONALIZATION, 10. ENSEMBLE
    # ------------------------------------------------------------------
    with tab_clf:
        col_l1, col_l2 = st.columns([1, 1])
        with col_l1:
            st.markdown("### 8. Classifier & Class Weight Search")
            if os.path.exists(os.path.join(step32_dir, "classifier_search.csv")):
                df_clf = pd.read_csv(os.path.join(step32_dir, "classifier_search.csv"))
                st.dataframe(df_clf, use_container_width=True)

        with col_l2:
            st.markdown("### 9 & 10. Personalization & Ensemble Search")
            if os.path.exists(os.path.join(step32_dir, "personalization_search.csv")):
                df_pers = pd.read_csv(os.path.join(step32_dir, "personalization_search.csv"))
                st.dataframe(df_pers, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 6: 11. VALIDATION LEADERBOARD
    # ------------------------------------------------------------------
    with tab_lb:
        st.markdown("### 11. Validation Leaderboard Matrix")
        if os.path.exists(os.path.join(step32_dir, "validation_leaderboard.csv")):
            df_lb = pd.read_csv(os.path.join(step32_dir, "validation_leaderboard.csv"))
            st.dataframe(df_lb, use_container_width=True)

    render_scientific_disclosure_footer()
