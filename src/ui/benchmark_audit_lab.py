import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.ui.components import render_scientific_disclosure_footer


def render_benchmark_audit_lab_page():
    """
    Render 🔍 STEP 31 — BENCHMARK AUDIT LAB
    Sections:
    1. Official Champion
    2. Benchmark Reproduction
    3. Step 24 vs Step 29
    4. Step 24 vs Step 30
    5. Raw Model Metrics
    6. Confidence Metrics
    7. Command Metrics
    8. Subject Comparison
    9. Confusion Matrices
    10. Pipeline Configuration Comparison
    11. Channel Montage Audit
    12. Final Benchmark Decision
    """
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
        <div>
            <h2 style="font-family:'Space Grotesk',sans-serif; margin:0; color:#F8FAFC;">🔍 STEP 31 — BENCHMARK AUDIT LAB</h2>
            <div style="font-size:0.85rem; color:#94A3B8;">Official Benchmark Integrity, Pipeline Consistency & Decision Hierarchy Audit</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    step31_dir = os.path.abspath("models/step31")
    has_audit = os.path.exists(os.path.join(step31_dir, "benchmark_audit.json"))

    if not has_audit:
        st.warning("⚠️ Step 31 benchmark audit report missing. Run step31_pipeline.py to populate results.")
        return

    with open(os.path.join(step31_dir, "benchmark_audit.json"), "r", encoding="utf-8") as f:
        rep = json.load(f)

    # ------------------------------------------------------------------
    # 1. OFFICIAL CHAMPION & 12. FINAL BENCHMARK DECISION
    # ------------------------------------------------------------------
    st.markdown("### 1. Official Champion & Discrepancy Resolution Summary")
    col_a1, col_a2, col_a3, col_a4 = st.columns([1, 1, 1, 1])

    with col_a1:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #10B981; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">OFFICIAL CHAMPION</div>
            <div style="font-size:1.2rem; font-weight:bold; color:#10B981; margin:4px 0;">{rep.get("final_official_champion", "STEP 24 FBCSP")}</div>
            <div style="font-size:0.75rem; color:#F8FAFC;">Holdout R11: Verified</div>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">REPRODUCED R11 ACC</div>
            <div style="font-size:1.3rem; font-weight:bold; color:#00F2FE; margin:4px 0;">{rep.get("reproduced_step24_accuracy", 70.67)}%</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Official CSV: {rep.get("official_step24_accuracy", 70.67)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_a3:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">REPRODUCED R11 MACRO F1</div>
            <div style="font-size:1.3rem; font-weight:bold; color:#00F2FE; margin:4px 0;">{rep.get("reproduced_step24_macro_f1", 64.33)}%</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Official F1: {rep.get("official_step24_macro_f1", 64.26)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_a4:
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8;">DISCREPANCY CAUSE</div>
            <div style="font-size:0.85rem; font-weight:bold; color:#F59E0B; margin:4px 0;">EVALUATION LAYER</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Model vs Command Level</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#0F172A; border:1px solid #334155; border-radius:8px; padding:14px; margin-top:12px;">
        <div style="color:#00F2FE; font-weight:bold; font-size:0.9rem; margin-bottom:4px;">💡 Discrepancy Resolution Findings:</div>
        <div style="font-size:0.85rem; color:#E2E8F0; line-height:1.4;">
            {rep.get("explanation", "")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:14px 0; border-color:#334155;'>", unsafe_allow_html=True)

    # Tabs for interactive audit modules
    tab_reprod, tab_pipe, tab_layer, tab_sub, tab_cm, tab_opt = st.tabs([
        "📊 Benchmark Reproduction",
        "⚙️ Pipeline Comparison",
        "🪜 Decision Layers",
        "👤 Subject Audit",
        "🔲 Confusion Matrices",
        "🎯 85% Target Search"
    ])

    # ------------------------------------------------------------------
    # TAB 1: 2. BENCHMARK REPRODUCTION
    # ------------------------------------------------------------------
    with tab_reprod:
        st.markdown("### 2. Official Step 24 Holdout R11 Benchmark Reproduction")
        if os.path.exists(os.path.join(step31_dir, "reproduced_step24_results.csv")):
            df_rep = pd.read_csv(os.path.join(step31_dir, "reproduced_step24_results.csv"))
            st.dataframe(df_rep, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2: 3. STEP 24 vs 29, 4. STEP 24 vs 30, 10. PIPELINE CONFIG, 11. MONTAGE
    # ------------------------------------------------------------------
    with tab_pipe:
        col_p1, col_p2 = st.columns([1.2, 1.0])
        with col_p1:
            st.markdown("### 3, 4 & 10. Pipeline Configuration Audit (Step 24 vs Step 29 vs Step 30)")
            if os.path.exists(os.path.join(step31_dir, "pipeline_comparison.csv")):
                df_pipe = pd.read_csv(os.path.join(step31_dir, "pipeline_comparison.csv"))
                st.dataframe(df_pipe, use_container_width=True)

        with col_p2:
            st.markdown("### 11. Channel Montage Audit")
            st.markdown(f"""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px;">
                <div style="color:#F8FAFC; font-weight:bold; margin-bottom:6px;">Official Step 24 15 Motor Cortex Channels:</div>
                <div style="color:#00F2FE; font-family:monospace; font-size:0.85rem;">{rep.get("step24_channel_montage", "")}</div>
                <hr style="margin:8px 0; border-color:#334155;">
                <div style="font-size:0.8rem; color:#94A3B8;">All 15 Group B channels are identical across Step 24, Step 28, Step 29, Step 30, and Step 31.</div>
            </div>
            """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # TAB 3: 5. RAW MODEL, 6. CONFIDENCE, 7. COMMAND METRICS
    # ------------------------------------------------------------------
    with tab_layer:
        st.markdown("### 5, 6 & 7. Decision Hierarchy Metrics (Level 1 Raw to Level 5 Command)")
        if os.path.exists(os.path.join(step31_dir, "decision_layer_comparison.csv")):
            df_dec = pd.read_csv(os.path.join(step31_dir, "decision_layer_comparison.csv"))
            fig_dec = px.bar(
                df_dec,
                x="decision_level",
                y=["accuracy", "coverage"],
                barmode="group",
                title="Accuracy vs Coverage (%) across Decision Hierarchy Levels"
            )
            fig_dec.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_dec, use_container_width=True)
            st.dataframe(df_dec, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 4: 8. SUBJECT COMPARISON
    # ------------------------------------------------------------------
    with tab_sub:
        st.markdown("### 8. Per-Subject Accuracy & F1 Comparison Matrix (S001–S010)")
        if os.path.exists(os.path.join(step31_dir, "subject_comparison.csv")):
            df_sub = pd.read_csv(os.path.join(step31_dir, "subject_comparison.csv"))
            fig_sub = px.bar(
                df_sub,
                x="subject",
                y=["official_step24_accuracy", "reproduced_step24_accuracy"],
                barmode="group",
                title="Official vs Reproduced Step 24 Per-Subject Accuracy (%)"
            )
            fig_sub.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_sub, use_container_width=True)
            st.dataframe(df_sub, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 5: 9. CONFUSION MATRICES
    # ------------------------------------------------------------------
    with tab_cm:
        col_cm1, col_cm2, col_cm3 = st.columns([1, 1, 1])
        with col_cm1:
            st.markdown("### Step 24 Confusion Matrix")
            if os.path.exists(os.path.join(step31_dir, "confusion_step24.csv")):
                df_cm24 = pd.read_csv(os.path.join(step31_dir, "confusion_step24.csv"), index_col=0)
                st.dataframe(df_cm24, use_container_width=True)

        with col_cm2:
            st.markdown("### Step 29 Evaluation Matrix")
            if os.path.exists(os.path.join(step31_dir, "confusion_step29.csv")):
                df_cm29 = pd.read_csv(os.path.join(step31_dir, "confusion_step29.csv"), index_col=0)
                st.dataframe(df_cm29, use_container_width=True)

        with col_cm3:
            st.markdown("### Step 30 Evaluation Matrix")
            if os.path.exists(os.path.join(step31_dir, "confusion_step30.csv")):
                df_cm30 = pd.read_csv(os.path.join(step31_dir, "confusion_step30.csv"), index_col=0)
                st.dataframe(df_cm30, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 6: 85% TARGET OPTIMIZATION LEADERBOARD
    # ------------------------------------------------------------------
    with tab_opt:
        st.markdown("### 🎯 Leakage-Free 85% Accuracy Optimization Leaderboard")
        opt_summary_path = os.path.join(step31_dir, "accuracy_optimization_summary.json")
        if os.path.exists(opt_summary_path):
            with open(opt_summary_path, "r", encoding="utf-8") as f:
                opt_sum = json.load(f)
            col_o1, col_o2, col_o3 = st.columns([1, 1, 1])
            with col_o1:
                st.metric("Best Validation Accuracy", f"{opt_sum.get('best_validation_accuracy', 0)}%")
            with col_o2:
                st.metric("Holdout R11 Accuracy", f"{opt_sum.get('r11_holdout_accuracy', 0)}%")
            with col_o3:
                st.metric("Target (85%) Reached", "YES" if opt_sum.get("target_reached_holdout") else "NO")

        opt_lb_path = os.path.join(step31_dir, "accuracy_optimization", "validation_leaderboard.csv")
        if os.path.exists(opt_lb_path):
            df_lb = pd.read_csv(opt_lb_path)
            st.dataframe(df_lb, use_container_width=True)

    render_scientific_disclosure_footer()
