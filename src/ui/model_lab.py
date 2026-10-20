import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.ui.components import (
    render_metric_card,
    render_evidence_badge,
    render_scientific_disclosure_footer
)


def render_model_lab_page():
    """
    Render Clean Consolidated Model Center Page.
    Organized into 4 structured sub-tabs:
    1. 🏆 Current Champion (Step 59)
    2. 📜 Model History (Steps 17B-59)
    3. 🥊 Model Comparison
    4. 🔍 Checkpoints & Artifacts
    """
    st.markdown("""
    <div style="background:linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border:2px solid #10B981; border-radius:12px; padding:20px; margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="color:#10B981; font-weight:800; font-size:0.85rem; letter-spacing:0.08em;">🤖 MODEL CENTER</div>
                <h1 style="font-family:'Space Grotesk',sans-serif; color:#F8FAFC; margin:4px 0 0 0; font-size:2.0rem;">
                    MOTOR IMAGERY DECODER SUITE
                </h1>
                <div style="color:#94A3B8; font-size:0.9rem; margin-top:4px;">
                    Current Authentic Champion (Step 59) | Historical Model Registry | Comparative Benchmark
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_champ, tab_hist, tab_comp, tab_arts = st.tabs([
        "🏆 CURRENT CHAMPION (STEP 59)",
        "📜 MODEL HISTORY (STEPS 17B–59)",
        "🥊 MODEL COMPARISON",
        "🔍 CHECKPOINTS & ARTIFACTS"
    ])

    # ------------------------------------------------------------------
    # TAB 1: CURRENT CHAMPION (STEP 59)
    # ------------------------------------------------------------------
    with tab_champ:
        sum59_path = "models/step59/summary_report.json"
        if os.path.exists(sum59_path):
            with open(sum59_path, "r", encoding="utf-8") as f:
                r59 = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("AUTHENTIC R11 ACCURACY", f"{r59['step59_authentic_r11_metrics']['accuracy']}%", "300 Genuine Holdout Trials", color="#10B981")
            with c2:
                render_metric_card("MACRO F1 SCORE", f"{r59['step59_authentic_r11_metrics']['macro_f1']}%", "Balanced 3-Class Metric", color="#00F2FE")
            with c3:
                render_metric_card("REST STATE RECALL", f"{r59['step59_authentic_r11_metrics']['rest_recall']}%", "150 REST Windows", color="#38BDF8")
            with c4:
                render_metric_card("INFERENCE LATENCY", f"{r59['latency_ms']} ms", "< 50.0 ms Realtime Constraint", color="#F59E0B")

            st.markdown("<br>", unsafe_allow_html=True)

            col_arch, col_recs = st.columns(2)
            with col_arch:
                st.markdown("### 🧩 Champion Pipeline Architecture")
                st.markdown("""
                <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:16px;">
                    <div style="font-weight:700; color:#10B981; margin-bottom:8px;">STEP 59 — Subject-Adaptive EEG Optimization</div>
                    <div style="color:#CBD5E1; font-size:0.85rem; margin-bottom:6px;"><b>Spatial Filter:</b> Subject-Adaptive 6-Band FBCSP + Riemannian Centroid Alignment</div>
                    <div style="color:#CBD5E1; font-size:0.85rem; margin-bottom:6px;"><b>Feature Fusion:</b> FBCSP Log-Var + C3/C4 Asymmetry + ERD Slope & AUC</div>
                    <div style="color:#CBD5E1; font-size:0.85rem; margin-bottom:6px;"><b>Classifier:</b> Regularized LDA (solver='lsqr', shrinkage='auto')</div>
                    <div style="color:#CBD5E1; font-size:0.85rem;"><b>Decision Gating:</b> Calibrated REST Probability Threshold (0.62)</div>
                </div>
                """, unsafe_allow_html=True)

            with col_recs:
                st.markdown("### 🎯 Class Recall Readouts")
                st.markdown(f"""
                <div style="background:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:16px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span style="color:#CBD5E1;">REST (Baseline):</span>
                        <span style="color:#38BDF8; font-weight:700;">{r59['step59_authentic_r11_metrics']['rest_recall']}%</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span style="color:#CBD5E1;">LEFT_FIST Motor Imagery:</span>
                        <span style="color:#10B981; font-weight:700;">{r59['step59_authentic_r11_metrics']['left_fist_recall']}% (+10.5% gain)</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#CBD5E1;">RIGHT_FIST Motor Imagery:</span>
                        <span style="color:#F59E0B; font-weight:700;">{r59['step59_authentic_r11_metrics']['right_fist_recall']}% (Primary Bottleneck)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Step 59 Champion model metrics loading...")

    # ------------------------------------------------------------------
    # TAB 2: MODEL HISTORY (STEPS 17B–59)
    # ------------------------------------------------------------------
    with tab_hist:
        st.markdown("### 📜 Comprehensive Model Evolution History")
        st.caption("Explicitly distinguishes Legacy Reported territory vs Authentic Recomputed territory established in Step 58.")

        history_data = [
            {"Step": "Step 17B", "Pipeline Architecture": "Filter Bank CSP (FBCSP)", "Reported Acc": "71.33%", "Macro F1": "66.02%", "Evidence Status": "LEGACY REPORTED"},
            {"Step": "Step 24", "Pipeline Architecture": "GroupKFold FBCSP Baseline", "Reported Acc": "71.33%", "Macro F1": "66.02%", "Evidence Status": "LEGACY REPORTED"},
            {"Step": "Step 34", "Pipeline Architecture": "Error Correction Cascade", "Reported Acc": "85.33%", "Macro F1": "81.20%", "Evidence Status": "LEGACY REPORTED"},
            {"Step": "Step 54", "Pipeline Architecture": "Adaptive Few-Shot Personalization", "Reported Acc": "86.40%", "Macro F1": "82.50%", "Evidence Status": "LEGACY REPORTED"},
            {"Step": "Step 55", "Pipeline Architecture": "Zero-Leakage Personalization", "Reported Acc": "86.40%", "Macro F1": "82.50%", "Evidence Status": "LEGACY REPORTED (Contains Mock Logic)"},
            {"Step": "Step 56", "Pipeline Architecture": "Unseen-Run 85% Engine", "Reported Acc": "86.40%", "Macro F1": "82.50%", "Evidence Status": "LEGACY REPORTED (Contains Mock Logic)"},
            {"Step": "Step 57", "Pipeline Architecture": "90% Error Recovery Lab", "Reported Acc": "90.00%", "Macro F1": "86.10%", "Evidence Status": "LEGACY REPORTED (Contains Mock Logic)"},
            {"Step": "Step 58", "Pipeline Architecture": "Genuine Raw-EDF Rebuild", "Reported Acc": "67.67%", "Macro F1": "62.04%", "Evidence Status": "AUTHENTIC RECOMPUTED (Real EDF Data)"},
            {"Step": "Step 59", "Pipeline Architecture": "Subject-Adaptive Peak ERD Engine", "Reported Acc": "67.67%", "Macro F1": "62.24%", "Evidence Status": "AUTHENTIC RECOMPUTED (Champion Model)"}
        ]

        st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # TAB 3: MODEL COMPARISON
    # ------------------------------------------------------------------
    with tab_comp:
        st.markdown("### 🥊 Model Benchmark Comparison (R03+R07 Development Folds)")
        mdl_comp_csv = "models/step59/model_comparison.csv"
        if os.path.exists(mdl_comp_csv):
            df_comp = pd.read_csv(mdl_comp_csv)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
        else:
            st.info("Model comparison data loading...")

    # ------------------------------------------------------------------
    # TAB 4: CHECKPOINTS & ARTIFACTS
    # ------------------------------------------------------------------
    with tab_arts:
        st.markdown("### 🔍 Model Artifacts & Checkpoint Inspector")
        col_ckpt, col_json = st.columns(2)

        with col_ckpt:
            st.markdown("#### Step 59 Frozen Model Checkpoints")
            subs = [f"S{i:03d}" for i in range(1, 11)]
            sel_s = st.selectbox("Select Subject Model Checkpoint:", options=subs, index=0)
            s_dir = f"models/step59/selected_model/{sel_s}"
            if os.path.exists(s_dir):
                files_found = os.listdir(s_dir)
                st.write(f"<b>Files in `{s_dir}`:</b>", unsafe_allow_html=True)
                for ff in files_found:
                    st.markdown(f"- `{ff}` ({os.path.getsize(os.path.join(s_dir, ff))} bytes)")

        with col_json:
            st.markdown("#### Frozen Configuration (`final_freeze.json`)")
            ff_path = "models/step59/final_freeze.json"
            if os.path.exists(ff_path):
                with open(ff_path, "r", encoding="utf-8") as f:
                    st.json(json.load(f))

    render_scientific_disclosure_footer()
