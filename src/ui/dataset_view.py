import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

from src.step8_pipeline import find_subject_runs
from src.ui.components import render_metric_card, render_scientific_disclosure_footer


def render_dataset_page():
    """
    Render PhysioNet EEGMMIDB Dataset Browser Page.
    Displays subject files, recording specifications, class distribution, and trial breakdown.
    """
    st.markdown("""
    <div style="background:linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border:2px solid #A855F7; border-radius:12px; padding:20px; margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="color:#A855F7; font-weight:800; font-size:0.85rem; letter-spacing:0.08em;">📁 DATASET BROWSER</div>
                <h1 style="font-family:'Space Grotesk',sans-serif; color:#F8FAFC; margin:4px 0 0 0; font-size:2.0rem;">
                    PHYSIONET EEG MOTOR MOVEMENT/IMAGERY DATABASE
                </h1>
                <div style="color:#94A3B8; font-size:0.9rem; margin-top:4px;">
                    Authentic 64-Channel EEG Recordings | Subjects S001–S010 | Motor Imagery Runs R03, R07, R11
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("SUBJECTS", "10 SUBJECTS", "S001 – S010", color="#A855F7")
    with c2:
        render_metric_card("EDF RECORDINGS", "30 EDF FILES", "R03, R07, R11 per Subject", color="#00F2FE")
    with c3:
        render_metric_card("TOTAL TRIALS", "900 TRIALS", "3.0s Epochs (480 samples)", color="#10B981")
    with c4:
        render_metric_card("MOTOR CHANNELS", "15 CHANNELS", "Group B Motor Array @ 160 Hz", color="#38BDF8")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: Class & Run Breakdown
    col_cls, col_run = st.columns(2)

    with col_cls:
        st.markdown("### 📊 Class Trial Distribution")
        df_class_dist = pd.DataFrame([
            {"Class": "REST (T0)", "Calibration (R03+R07)": 300, "Holdout Test (R11)": 150, "Total Genuine Trials": 450, "Percentage": "50.0%"},
            {"Class": "LEFT_FIST (T1)", "Calibration (R03+R07)": 151, "Holdout Test (R11)": 76, "Total Genuine Trials": 227, "Percentage": "25.2%"},
            {"Class": "RIGHT_FIST (T2)", "Calibration (R03+R07)": 149, "Holdout Test (R11)": 74, "Total Genuine Trials": 223, "Percentage": "24.8%"},
            {"Class": "TOTAL", "Calibration (R03+R07)": 600, "Holdout Test (R11)": 300, "Total Genuine Trials": 900, "Percentage": "100.0%"}
        ])
        st.dataframe(df_class_dist, use_container_width=True, hide_index=True)

    with col_run:
        st.markdown("### 🎬 Run Configuration Specifications")
        df_run_spec = pd.DataFrame([
            {"Run ID": "R03", "Task Description": "Motor Imagery: Left vs Right Fist", "Role": "Calibration / Development", "Trials per Sub": 30, "Total Trials": 300},
            {"Run ID": "R07", "Task Description": "Motor Imagery: Left vs Right Fist", "Role": "Calibration / Development", "Trials per Sub": 30, "Total Trials": 300},
            {"Run ID": "R11", "Task Description": "Motor Imagery: Left vs Right Fist", "Role": "Independent Holdout Test", "Trials per Sub": 30, "Total Trials": 300}
        ])
        st.dataframe(df_run_spec, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Section 3: Subject EDF File Inspector
    st.markdown("### 🔍 Subject EDF File Inspector")
    subs = [f"S{i:03d}" for i in range(1, 11)]
    sel_sub = st.selectbox("Select Subject to Inspect:", options=subs, index=0)

    runs_found = find_subject_runs(sel_sub, data_dir="data")
    sub_records = []
    for r_key in ["R03", "R07", "R11"]:
        f_path = runs_found.get(r_key)
        exists = os.path.exists(f_path) if f_path else False
        f_size = os.path.getsize(f_path) / 1024.0 if exists else 0.0
        sub_records.append({
            "Subject": sel_sub,
            "Run": r_key,
            "File Path": f_path,
            "File Status": "EXISTS (Genuine EDF)" if exists else "MISSING",
            "Size (KB)": round(f_size, 1),
            "Sampling Rate": "160 Hz",
            "Channels": 64
        })

    st.dataframe(pd.DataFrame(sub_records), use_container_width=True, hide_index=True)

    render_scientific_disclosure_footer()
