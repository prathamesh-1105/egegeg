import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.load_eeg import load_eeg_raw, get_eeg_info
from src.events import get_eeg_events
from src.feature_extraction import calculate_fft_psd, calculate_band_powers
from src.model_loader import ModelLoader


@st.cache_resource
def get_cached_model_loader(models_dir: str = "models") -> ModelLoader:
    """Cache ModelLoader instance to avoid disk re-scans."""
    return ModelLoader(models_dir=models_dir)


@st.cache_resource
def get_cached_step14_model(subject_id: str = "S001", models_dir: str = "models") -> dict:
    """Cache Step 14 Champion model artifacts in memory."""
    loader = get_cached_model_loader(models_dir=models_dir)
    return loader.load_step14_model(subject_id=subject_id)


@st.cache_data
def get_cached_edf_data(file_path: str):
    """Cache raw EDF file loading and event extraction."""
    raw = load_eeg_raw(file_path)
    info = get_eeg_info(raw, file_path)
    df_events = get_eeg_events(raw)
    return raw, info, df_events


@st.cache_data
def get_cached_psd_computation(signal_data: np.ndarray, sfreq: float):
    """Cache Power Spectral Density computation for signal window."""
    return calculate_fft_psd(signal_data, sfreq)


@st.cache_data
def get_cached_band_powers_computation(freqs: np.ndarray, psd: np.ndarray):
    """Cache band power integration from PSD."""
    return calculate_band_powers(freqs, psd)


def render_metric_card(title: str, value: str, subtitle: str = None, color: str = "#00F2FE"):
    """Render a styled glassmorphic metric card."""
    st.markdown(f"""
    <div style="background:#0F172A; border:1px solid #1E293B; border-top:3px solid {color}; border-radius:12px; padding:16px; margin-bottom:12px;">
        <div style="font-size:0.75rem; font-weight:700; color:#94A3B8; letter-spacing:0.05em; text-transform:uppercase;">{title}</div>
        <div style="font-size:1.6rem; font-weight:700; color:{color}; font-family:'Space Grotesk',sans-serif; margin-top:4px;">{value}</div>
        {f'<div style="font-size:0.8rem; color:#64748B; margin-top:2px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_status_pill(text: str, color: str = "#10B981"):
    """Render a pulsing status indicator pill."""
    st.markdown(f"""
    <span style="display:inline-flex; align-items:center; background:rgba(15,23,42,0.8); border:1px solid {color}; color:{color}; border-radius:20px; padding:4px 12px; font-size:0.8rem; font-weight:600;">
        <span style="height:8px; width:8px; background-color:{color}; border-radius:50%; display:inline-block; margin-right:8px; box-shadow:0 0 8px {color};"></span>
        {text}
    </span>
    """, unsafe_allow_html=True)


def render_evidence_badge(status_type: str = "AUTHENTIC RECOMPUTED"):
    """Render scientific evidence status badge (AUTHENTIC RECOMPUTED vs LEGACY REPORTED vs INCONSISTENT)."""
    status_type_upper = status_type.upper()

    if "AUTHENTIC" in status_type_upper or "RECOMPUTED" in status_type_upper:
        color = "#10B981"
        bg_color = "rgba(16, 185, 129, 0.15)"
        border_color = "#10B981"
        icon = "✅"
    elif "LEGACY" in status_type_upper or "REPORTED" in status_type_upper:
        color = "#F59E0B"
        bg_color = "rgba(245, 158, 11, 0.15)"
        border_color = "#F59E0B"
        icon = "📜"
    elif "INCONSISTENT" in status_type_upper or "FAIL" in status_type_upper:
        color = "#EF4444"
        bg_color = "rgba(239, 68, 68, 0.15)"
        border_color = "#EF4444"
        icon = "⚠️"
    else:
        color = "#94A3B8"
        bg_color = "rgba(148, 163, 184, 0.15)"
        border_color = "#94A3B8"
        icon = "ℹ️"

    st.markdown(f"""
    <span style="display:inline-flex; align-items:center; background:{bg_color}; border:1px solid {border_color}; color:{color}; border-radius:16px; padding:3px 10px; font-size:0.75rem; font-weight:700; letter-spacing:0.04em;">
        <span style="margin-right:6px;">{icon}</span>{status_type_upper}
    </span>
    """, unsafe_allow_html=True)


def render_scientific_disclosure_footer():
    """Render mandatory scientific & hardware disclosure footer."""
    st.markdown("""
    <div style="background:#0F172A; border:1px solid #334155; border-radius:8px; padding:14px; margin-top:28px; font-size:0.8rem; color:#94A3B8;">
        <div style="font-weight:700; color:#F8FAFC; margin-bottom:4px;">🔬 SCIENTIFIC & HARDWARE DISCLOSURE</div>
        <div>Hardware mode requires a compatible EEG acquisition device capable of providing raw multi-channel EEG. Dataset playback is used for software validation and is not equivalent to live human EEG acquisition.</div>
        <div style="margin-top:4px;">This system is an experimental academic BCI prototype and is not intended for medical diagnosis, treatment, or safety-critical control.</div>
    </div>
    """, unsafe_allow_html=True)

