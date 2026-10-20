# NEURO BCI / NeuroOS — Real-Time Motor Imagery Platform & EEG-Controlled Scrolling Demo

A high-performance Brain-Computer Interface (BCI) platform built to load, process, classify, and translate real 64-channel EEG recordings from the PhysioNet Motor Movement/Imagery Dataset (`EEGMMIDB`) into real-time software actions.

---

## 🌟 Key Features

1. **Authentic Data Replay & Zero-Leakage Validation**:
   - Replays genuine 64-channel EDF binary recordings using MNE-Python (subjects S001–S010 across baseline Run 03, consolidation Run 07, and untouched holdout Run 11).
   - Strict zero data leakage: test trials are strictly isolated from training calibration runs.

2. **Step 60/61 Riemannian-FBCSP Engine**:
   - Subject-specific adaptive filter banks ($\mu$ 8–13 Hz and $\beta$ 13–30 Hz).
   - Regularized Filter Bank Common Spatial Patterns (FBCSP) + Riemannian Tangent Space manifold covariance features.
   - Benchmark holdout accuracy: **70.67%** (Step 60 Champion) / **72.00%** (Step 61 Ensemble) on Run 11.

3. **NeuroOS EEG-Controlled Webpage Scrolling Demo**:
   - Translates classified motor imagery trials into real-time webpage scrolling:
     - **`LEFT_FIST`** $\rightarrow$ **SCROLL UP (↑)**
     - **`RIGHT_FIST`** $\rightarrow$ **SCROLL DOWN (↓)**
     - **`REST`** $\rightarrow$ **NEUTRAL / NO SCROLL (⏸)**
   - Continuous playback, pause, step single trial, and reset controls.
   - Live telemetry HUD displaying ground truth, real-time prediction, confidence score, aligned probabilities, and safety shield gating status.
   - Dedicated long-form scientific document that scrolls smoothly in response to decoded brain signals.

4. **Safety Shield & Artifact Guardrail**:
   - Signal Quality Gate (SQI) amplitude ($0.1\ \mu V - 500\ \mu V$) and flatline detection.
   - Confidence thresholding ($\ge 60\%$), entropy bounds, and dwell-time confirmation.
   - Cooldown guardrail (0.8s) and manual emergency disengage.

5. **Scientific Disclosure**:
   - Clearly labeled as **Recorded PhysioNet EEG Playback**.
   - Explicitly clarifies that motor-imagery trials (fist clenching imagery) are mapped via software to scroll commands, rather than reading "scrolling thoughts" or claiming live invasive brain control.

---

## 🚀 Quick Start & Run Commands

### 1. Launch FastAPI Backend Server
Runs REST endpoints and the real-time WebSocket telemetry stream (`/ws/neuro-os`):
```powershell
python api_server.py
```
*API runs at `http://localhost:8050` (Interactive Swagger Docs: `http://localhost:8050/docs`)*

### 2. Launch Vite React Frontend
In a separate terminal:
```powershell
cd frontend
npm run dev
```
*Frontend runs at `http://localhost:5173`*

### 3. Alternative Streamlit Visualizer (Optional)
```powershell
python -m streamlit run app.py
```
*Streamlit runs at `http://localhost:8501`*

---

## 🧪 Running Automated Tests

Run the complete test suite (12 unit and integration tests):
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

To run individual test files:
```powershell
# Scrolling demo unit & integration tests (9 tests)
python -m unittest tests/test_scrolling_demo.py

# Feature extraction pipeline tests
python tests/test_features.py

# Deep learning Conformer & Speech tests (3 tests)
python -m unittest tests/test_speech_decoder.py
```

---

## 🎓 Evaluator & Teacher Presentation Guide

1. Open **`http://localhost:5173`** in your browser.
2. The interface opens directly to the **EEG Scroll Demo** tab.
3. Review the **Scientific Disclosure & Command Mappings**:
   - Point out that playback uses authentic PhysioNet holdout recording `R11`.
   - Explain the software command mapping (`LEFT_FIST` $\rightarrow$ Scroll Up, `RIGHT_FIST` $\rightarrow$ Scroll Down, `REST` $\rightarrow$ Stationary).
4. Click **"STEP TRIAL"** to step through individual trials one by one:
   - Observe the **True Label** vs. **Predicted Class**.
   - Notice the **Confidence %** and posterior probability distribution.
   - Observe the document container scrolling smoothly up or down.
   - Observe that `REST` trials produce `NEUTRAL` and no scroll displacement.
5. Click **"START PLAYBACK"** to start continuous real-time replay:
   - Watch the document scroll continuously as trials stream from the backend.
   - Test changing the playback speed (0.5x, 1x, 2x).
   - Test **"PAUSE PLAYBACK"** and **"RESET"**.
6. Inspect the **Recent Trial Execution Audit Log** table at the bottom of the page to review the chronological history of trial predictions and execution outcomes.
