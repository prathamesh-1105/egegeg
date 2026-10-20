import streamlit as st
import os

from src.load_eeg import load_eeg_raw, get_eeg_info, EEGFileNotFoundError
from src.events import get_eeg_events
from src.ui.styles import inject_custom_theme
from src.ui.sidebar import render_sidebar_navigation
from src.ui.dashboard import render_dashboard_page
from src.ui.live_bci import render_live_bci_page
from src.ui.speech_bci import render_speech_bci_page
from src.ui.eeg_explorer import render_eeg_explorer_page
from src.ui.model_lab import render_model_lab_page
from src.ui.experiments import render_experiments_page
from src.ui.validation_view import render_validation_page
from src.ui.dataset_view import render_dataset_page
from src.ui.hardware import render_hardware_page
from src.ui.system_view import render_system_page

# Set Streamlit Page Config
st.set_page_config(
    page_title="NEURO BCI — Real-Time Motor Imagery Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Sci-Fi Dark Theme CSS
inject_custom_theme()


@st.cache_data
def cached_load_eeg(file_path: str):
    """Loads and caches EDF raw object, metadata, and event annotations."""
    raw = load_eeg_raw(file_path)
    info = get_eeg_info(raw, file_path)
    df_events = get_eeg_events(raw)
    return raw, info, df_events


def main():
    edf_path = "data/S001R03.edf"

    try:
        raw, info, df_events = cached_load_eeg(edf_path)
    except Exception as err:
        st.error(f"Error loading EDF dataset: {err}")
        st.stop()

    page_key = render_sidebar_navigation()

    if page_key in ["overview", "dashboard"]:
        render_dashboard_page()
    elif page_key == "speech_bci":
        render_speech_bci_page()
    elif page_key == "live_bci":
        render_live_bci_page(raw, info, df_events)
    elif page_key == "eeg_explorer":
        render_eeg_explorer_page(raw, info, df_events)
    elif page_key in ["models", "model_lab"]:
        render_model_lab_page()
    elif page_key == "experiments":
        render_experiments_page()
    elif page_key == "validation":
        render_validation_page()
    elif page_key == "dataset":
        render_dataset_page()
    elif page_key == "hardware":
        render_hardware_page()
    elif page_key == "system":
        render_system_page()
    else:
        render_dashboard_page()


if __name__ == "__main__":
    main()
