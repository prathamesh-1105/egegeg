import streamlit as st
from src.ui.components import render_scientific_disclosure_footer


def render_documentation_page():
    """
    Render Documentation Page.
    Comprehensive BCI theory overview, mathematical foundations, dataset specification, and academic references.
    """
    st.markdown("## BCI PLATFORM DOCUMENTATION & THEORY")
    st.caption("Theoretical foundations, mathematical formulation, PhysioNet dataset details, and scientific disclosure.")

    st.markdown("""
    ### 1. Neurophysiological Foundation
    Motor Imagery (MI) EEG brain-computer interfaces detect event-related desynchronization (ERD) and event-related synchronization (ERS) over the primary motor cortex (C3, Cz, C4 channels).
    * **Left Fist Imagery**: ERD desynchronization over right motor cortex (C4 channel).
    * **Right Fist Imagery**: ERD desynchronization over left motor cortex (C3 channel).
    * **Rest State**: Baseline mu (8–13 Hz) and beta (13–30 Hz) rhythm power.

    ### 2. Mathematical Formulation
    * **Covariance Matrix Estimation**: For a multi-channel EEG window \\(X \\in \\mathbb{R}^{C \\times N}\\), the sample covariance matrix is estimated using Ledoit-Wolf shrinkage.
    * **Filter Bank Common Spatial Pattern (FBCSP)**: Multi-band spatial filtering across 7 sub-bands (4–40 Hz) extracting log-variance features.

    ### 3. Dataset Specification
    * **Source**: PhysioNet EEG Motor Movement/Imagery Dataset (64 channels, 160 Hz sampling rate).
    * **Active Cohort**: Subjects S001–S010.
    * **Channel Grouping**: Group B 15 Motor Strip channels (`FC3..CP4`).
    * **Holdout Validation**: Untouched run R11 evaluated post-freezing.

    ### 4. Verified Performance Hierarchy
    1. **Step 17B Filter Bank CSP (FBCSP)** (Current Champion): Accuracy `71.33%`, Macro F1 `66.02%` (PhysioNet R11 Holdout).
    2. **Step 14 Event-Aligned Riemannian**: Accuracy `62.33%`, Macro F1 `55.99%`.
    3. **Step 13B Class-Balanced Riemannian**: Accuracy `59.33%`, Macro F1 `53.04%`.
    4. **Step 10 Baseline Riemannian**: Accuracy `59.00%`, Macro F1 `52.20%`.

    ### 5. Hardware Status & Scientific Disclosure
    * **Hardware Compatibility**: OpenBCI Cyton + Daisy 16-Channel system planned. Current hardware status: `🔴 HARDWARE DISCONNECTED`.
    * **Important Disclosure**: "The 71.33% benchmark was obtained using recorded PhysioNet EEG data. It does not represent guaranteed accuracy on live human EEG acquired from OpenBCI hardware."
    """)

    render_scientific_disclosure_footer()
