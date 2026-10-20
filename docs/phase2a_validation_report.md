# NeuroOS Phase 2A — End-to-End EEG Decoder and Scrolling Validation Report

**Author / Role:** Lead BCI Software Engineer, NeuroOS Core Team  
**Evaluation Date:** October 19, 2026  
**Dataset Provenance:** PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)  
**Evaluation Protocol:** Strict Zero-Leakage Run 11 Holdout across Cohort S001–S010 (300 Trials)  
**Active Decoder Champion:** Step 60 SOTA Riemannian Tangent Space + ElasticNet Ensemble (with Step 61 validation)  
**Test Suite Status:** 12 / 12 Tests Passing (100% Pass Rate)

---

## 1. Executive Summary & Verification Verdict

The primary objective of **NeuroOS Phase 2A** is to transition the offline motor-imagery decoding pipeline into a **reproducible, trustworthy, and scientifically honest EEG-controlled webpage scrolling demonstration**. 

Prior to this phase, preliminary claims indicated an approximate 72% holdout accuracy on PhysioNet Run 11 for the Step 61 model. In this audit and validation phase:
1. **Independent Empirical Verification**: We verified the Step 60/61 models on 300 untouched holdout trials across 10 subjects on Run 11 (`S001`–`S010`). The exact measured holdout accuracy is **72.00%** with a Macro F1 score of **67.23%**.
2. **Critical Bugs Identified and Corrected**:
   - A critical unit-scale mismatch in [src/neuro_os/signal_quality.py](file:///d:/PROJECTS/EEG%20MAj/src/neuro_os/signal_quality.py) where raw EDF values in Volts ($10^{-4}\text{ V}$) were compared directly against microvolt thresholds ($0.1\ \mu\text{V}$), falsely flagging 100% of channels as defective.
   - A trial pointer reset bug in [src/neuro_os/lsl_bridge.py](file:///d:/PROJECTS/EEG%20MAj/src/neuro_os/lsl_bridge.py) that forced `current_trial_idx = 0` on every step call.
   - Hardcoded artificial latency padding (`+ 22.0 ms` and `+ 8.5 ms`) in [api_server.py](file:///d:/PROJECTS/EEG%20MAj/api_server.py), which has now been replaced with high-resolution `time.perf_counter()` real-time measurements ($12.91\text{ ms}$ average inference time).
   - Dwell-based navigation conflicts in [src/neuro_os/command_router.py](file:///d:/PROJECTS/EEG%20MAj/src/neuro_os/command_router.py) where 3-window dwelling interfered with scrolling actions.
3. **Rigorous Signal / Intent / Command Separation**:
   - Ground truth labels are strictly partitioned for post-hoc evaluation and audit logging. The decoder relies purely on the 15-channel raw EEG voltage window ($3.0\text{ s} \times 160\text{ Hz}$).
   - Motor imagery intents are mapped to software actions:
     - `LEFT_FIST` $\rightarrow$ `SCROLL_UP` ($-180\text{ px}$)
     - `RIGHT_FIST` $\rightarrow$ `SCROLL_DOWN` ($+180\text{ px}$)
     - `REST` $\rightarrow$ `NEUTRAL` (Stationary)
   - Low confidence ($< 0.60$) or low margin ($< 0.10$) trials are rejected by the Safety Shield, resulting in zero scroll displacement.
4. **End-to-End Interactive Demonstration**:
   - Built a dedicated, scrollable document viewer loaded with peer-reviewed BCI specification text.
   - Integrated live Telemetry HUD displaying real-time predictions, probability distributions, live session counters (Processed, Accepted, Rejected, Correct, Incorrect, Live Accuracy, Selective Accuracy), and displacement tracking with boundary clamping ($0\text{ px}$ to $3000\text{ px}$).

---

## 2. Comprehensive Codebase & Model Artifact Audit

### 2.1 Dataset Inspection
- **Source**: PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), sampled at $160\text{ Hz}$ across 64 scalp electrodes.
- **Run Protocol Structure**:
  - `R03`, `R07`, `R11`: Motor imagery of left fist (T1) vs. right fist (T2) vs. rest baseline (T0).
  - `R04`, `R08`, `R12`: Motor imagery of both fists vs. both feet (not used in the 3-class fist paradigm).
  - **Zero-Leakage Split**: `R03` and `R07` are utilized for calibration / training (60 trials per subject). `R11` is reserved strictly as the untouched test holdout (30 trials per subject $\times 10$ subjects = 300 trials).
- **Channel Montage**: Group B 15-channel motor montage centered on the sensorimotor cortex:  
  `['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']`.

### 2.2 Model Architectures & Artifact Hierarchy
- **Step 58 Baseline**: Riemannian Tangent Space with default Logistic Regression. (Holdout: $60.33\%$, F1: $56.53\%$).
- **Step 59 Benchmark**: Filter Bank CSP with adaptive mu/beta sub-band selection + Linear SVM. (Holdout: $67.67\%$, F1: $62.24\%$).
- **Step 60 SOTA Champion**: Ledoit-Wolf regularized sample covariance on Riemannian manifold tangent space projected onto an ElasticNet / Ridge classifier with subject-specific spatial filters. (Holdout: $70.67\%$, F1: $65.72\%$).
- **Step 61 Multimodal Ensemble**: Soft-voting ensemble combining Riemannian Tangent Space probabilities with Filter Bank CSP power features. Saved under [models/step61/selected_model/](file:///d:/PROJECTS/EEG%20MAj/models/step61/selected_model/). (Holdout: **$72.00\%$**, Macro F1: **$67.23\%$**).
- **Ground Truth Storage**: Raw `.edf` files reside in `data/physionet/`. Event markers `T0`, `T1`, `T2` are extracted dynamically using MNE's `read_raw_edf` annotations.

---

## 3. Discovered Bugs and Specific Implementations

### Bug 1: Signal Quality Gate Scale Mismatch
- **Location**: [src/neuro_os/signal_quality.py](file:///d:/PROJECTS/EEG%20MAj/src/neuro_os/signal_quality.py#L90-L115)
- **Symptom**: All valid PhysioNet EEG signals were flagged as bad channels, causing artificial safety rejections.
- **Root Cause**: Raw EDF signals from PhysioNet load in Volts (typical amplitudes $\approx 10^{-5}\text{ to } 10^{-4}\text{ V}$). The gate had hardcoded thresholds expecting microvolts (`min_amplitude_uv = 0.1`, `max_amplitude_uv = 500.0`). Because $10^{-4}\text{ V} < 0.1$, every channel was classified as flatline.
- **Fix**: Implemented automatic unit detection in `assess_channel_quality()`. Signals with peak absolute amplitudes $< 0.05$ are scaled to microvolts ($\times 10^6$) before boundary evaluation.

### Bug 2: LSL Bridge Trial Index Reset
- **Location**: [src/neuro_os/lsl_bridge.py](file:///d:/PROJECTS/EEG%20MAj/src/neuro_os/lsl_bridge.py#L120-L135)
- **Symptom**: Replaying consecutive trials continually re-evaluated trial #0.
- **Root Cause**: In `set_subject_run(subject_id, run_id)`, `self.current_trial_idx = 0` was executed on every invocation even if the subject and run were identical.
- **Fix**: Added guard condition `if subject_id != self.current_subject or run_id != self.current_run:` before re-initializing the trial counter.

### Bug 3: Artificial Latency Injections in API Server
- **Location**: [api_server.py](file:///d:/PROJECTS/EEG%20MAj/api_server.py#L420-L450)
- **Symptom**: Reported inference latencies contained hardcoded delays (`+ 22.0 ms` and `+ 8.5 ms`).
- **Fix**: Removed all artificial offsets. Wrapped the pipeline execution in `time.perf_counter()` to return honest, reproducible timing metrics.

### Bug 4: Command Router Dwell State Interruption
- **Location**: [src/neuro_os/command_router.py](file:///d:/PROJECTS/EEG%20MAj/src/neuro_os/command_router.py#L55-L100)
- **Symptom**: In navigation mode, consecutive fist motor imageries were intercepted by a 3-window dwell state machine designed for UI button clicks (`SELECT`/`BACK`), delaying or suppressing scroll actions.
- **Fix**: Added explicit `command_mode="SCROLL"` in `route_intent()`. In `SCROLL` mode, accepted motor intents (`LEFT_FIST`, `RIGHT_FIST`) directly trigger scrolling actions without dwelling delays.

### Bug 5: Scrolling Boundary Clamping & Action Executor Store
- **Location**: [src/neuro_os/action_executor.py](file:///d:/PROJECTS/EEG%20MAj/src/neuro_os/action_executor.py#L45-L75)
- **Symptom**: Scroll offsets could become negative or exceed document boundaries.
- **Fix**: Implemented strict bounding between `0` and `3000` px with `180px` step increments, along with state-store tracking for audit logs.

---

## 4. Empirical Evaluation Metrics (Run 11 Holdout)

The empirical validation was executed across all 10 subjects on Run 11 (300 total trials) via [src/neuro_os/evaluate_pipeline.py](file:///d:/PROJECTS/EEG%20MAj/src/neuro_os/evaluate_pipeline.py). Results are recorded in [models/evaluation_report_r11.json](file:///d:/PROJECTS/EEG%20MAj/models/evaluation_report_r11.json):

### 4.1 Global Classifier Performance
| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Total Holdout Trials** | **300** | 30 trials per subject across S001–S010 |
| **Raw Holdout Accuracy** | **72.00%** | 216 correct predictions out of 300 |
| **Macro F1 Score** | **67.23%** | Unweighted mean across all 3 classes |
| **Inference Latency** | **12.91 ms** | P95 Latency: $16.60\text{ ms}$ |
| **Total Benchmark Duration** | **4.46 s** | Full 300-trial pipeline evaluation |

### 4.2 Class-wise Breakdown
| Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **REST (T0)** | 150 | 77.46% | **89.33%** | **82.97%** |
| **LEFT_FIST (T1)** | 76 | 74.58% | 57.89% | 65.19% |
| **RIGHT_FIST (T2)** | 74 | 55.88% | 51.35% | 53.52% |

### 4.3 Empirical Confusion Matrix
```
               Predicted REST    Predicted LEFT_FIST    Predicted RIGHT_FIST
True REST            134                 4                      12
True LEFT_FIST        14                44                      18
True RIGHT_FIST       25                11                      38
```

### 4.4 Safety Shield & Guardrail Performance
- **Rejection Thresholds**: Peak confidence $< 0.60$ OR probability margin $< 0.10$.
- **Total Accepted Trials**: 265 / 300 (**88.33% Coverage**)
- **Total Gated Rejections**: 35 / 300 (**11.67% Rejection Rate**)
- **Selective Accuracy on Accepted Subset**: **75.09%**  
  *(Enforcing the Safety Shield provides a **+3.09%** increase in classification accuracy for dispatched commands).*

### 4.5 Downstream Command & Scrolling Execution
- **Dispatched Commands**:
  - `SCROLL_UP`: 47
  - `SCROLL_DOWN`: 50
  - `NEUTRAL`: 203
- **Total Executed Scrolls**: 97
- **Correct Scrolls**: 68
- **Incorrect Scrolls**: 29
- **Scroll Command Accuracy**: **70.10%**
- **REST Neutrality Guarantee**: **92.67%** (139 out of 150 REST trials produced zero scrolling movement, preventing accidental drifting).

---

## 5. Webpage Scrolling Demonstration Architecture

The demonstration is integrated in [frontend/src/components/EEGScrollingDemo.jsx](file:///d:/PROJECTS/EEG%20MAj/frontend/src/components/EEGScrollingDemo.jsx):

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             NeuroOS Navigation UI                                │
├───────────────────────────────────────┬──────────────────────────────────────────┤
│           Telemetry HUD               │          Dedicated Webpage Reader        │
│                                       │                                          │
│  [Step 60/61 Decoder Card]            │  [Browser Chrome: https://neuro-os.internal]│
│  - Ground Truth vs. Predicted         │  ┌────────────────────────────────────┐  │
│  - Posterior Probabilities (Bar)      │  │                                    │  │
│  - Confidence: 78.4% [PASS]           │  │ NeuroOS: Sensorimotor Decoding     │  │
│                                       │  │ Peer-Reviewed Specification        │  │
│  [Live Session Telemetry]             │  │                                    │  │
│  - Status: PLAYING / PAUSED / IDLE    │  │ 1. Executive Abstract...           │  │
│  - Processed: 18 | Accepted: 16       │  │ 2. SMR Neurophysiology...          │  │
│  - Correct: 13 | Live Acc: 72.2%      │  │ 3. 15-Channel Topography...        │  │
│  - Selective Acc: 75.0%               │  │ 4. FBCSP Mathematical Model...     │  │
│  - Measured Latency: 12.9 ms          │  │ 5. Riemannian Tangent Space...     │  │
│                                       │  │ 6. Multi-Tiered Safety Shield...   │  │
│  [Scroll Displacement Bar]            │  │ 7. Holdout Benchmark Validation... │  │
│  - Current Offset: 720px / 1000px     │  │ 8. Conclusion...                   │  │
│  - Manual Nudge Buttons (Up/Down)     │  │                                    │  │
│                                       │  └────────────────────────────────────┘  │
├───────────────────────────────────────┴──────────────────────────────────────────┤
│  [Recent Trial Audit Log & Toggleable Verified R11 Benchmark Dashboard]          │
│  - Time | Trial # | True Label | Predicted | Confidence | Safety | Command | Action│
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Key UI Features
1. **Live Synchronous Scrolling**: `documentContainerRef.current.scrollTo({ top: newTop, behavior: 'smooth' })` responds directly to backend execution payloads.
2. **Action Directional Flash**: Visual floating badge dynamically signals `LEFT_FIST → SCROLL UP (-180px)`, `RIGHT_FIST → SCROLL DOWN (+180px)`, or `SAFETY REJECTED`.
3. **Session Counters**: Real-time accumulation of processed, accepted, rejected, correct, and incorrect trials.
4. **Verified R11 Benchmark Viewer**: Built-in toggle reveals the full 300-trial cross-subject confusion matrix and class metrics directly in the UI.

---

## 6. Unit & Integration Test Suite Verification

A dedicated test suite was implemented in [tests/test_scrolling_demo.py](file:///d:/PROJECTS/EEG%20MAj/tests/test_scrolling_demo.py) and executed via Python unittest. All 12 tests passed:

```
python -m unittest discover -s tests -p "test_*.py"
............
----------------------------------------------------------------------
Ran 12 tests in 8.302s

OK
```

### Tests Executed:
1. `test_action_executor_clamping`: Confirms displacement stays within $[0, 3000]\text{ px}$.
2. `test_command_router_scroll_mode`: Validates that `LEFT_FIST` maps to `SCROLL_UP`, `RIGHT_FIST` to `SCROLL_DOWN`, and `REST` to `NEUTRAL`.
3. `test_dataset_trial_iteration`: Asserts sequential iteration through EDF trials without pointer resetting.
4. `test_end_to_end_scroll_step_api`: Validates the `/api/neuro-os/scroll-demo/step` HTTP endpoint.
5. `test_evaluation_report_metrics`: Checks integrity of holdout accuracy ($\ge 70\%$) and latency ($\le 50\text{ ms}$) in `evaluation_report_r11.json`.
6. `test_real_latency_measurement`: Validates that inference timing does not use artificial sleep offsets.
7. `test_reset_scroll_demo`: Ensures `/api/neuro-os/scroll-demo/reset` clears all buffers and pointers.
8. `test_safety_shield_rejection_no_scroll`: Confirms low-confidence intents do not trigger scroll movement.
9. `test_signal_quality_auto_scaling`: Asserts that raw Volt inputs are correctly converted to microvolts without flatline false-positives.
10. `test_signal_quality_flatline`: Confirms genuine flatline signals are rejected.
11. `test_signal_quality_normal`: Confirms nominal EEG signals pass the quality gate.
12. `test_signal_quality_saturation`: Confirms clipped/saturated signals are rejected.

---

## 7. Reproduction & Execution Guide

### 7.1 Backend API Server Launch
```bash
# In workspace root
python api_server.py
# Server starts on http://localhost:8050
```

### 7.2 Frontend Development Server Launch
```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:5173
```

### 7.3 Re-running Empirical Holdout Evaluation
```bash
python src/neuro_os/evaluate_pipeline.py
# Computes metrics across 300 trials on R11 and outputs models/evaluation_report_r11.json
```

### 7.4 Running the Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 8. Known Limitations & Presentation Guidance

When presenting this demonstration to evaluators or stakeholders, adhere strictly to the following scientific disclosures:

1. **Software Mapping vs. Direct Cognitive Intent**:
   - The user is **not** thinking "scroll down" or reading webpage text with their mind.
   - The user rehearsed physical motor movements (clenching the left or right hand), which generates Event-Related Desynchronization (ERD) in the sensorimotor cortex. NeuroOS decodes this sensorimotor rhythm and maps the resulting intent to browser scroll events.
2. **Offline Recorded Playback vs. Live Headset**:
   - The demonstration uses real binary `.edf` files recorded from human subjects in the PhysioNet EEGMMIDB database.
   - It does **not** claim live headset acquisition. However, the software pipeline (LSL streaming $\rightarrow$ quality gate $\rightarrow$ spatial filtering $\rightarrow$ manifold projection $\rightarrow$ safety shield $\rightarrow$ command routing) is identical to a live acquisition setup.
3. **$72\%$ Accuracy Ceiling**:
   - $72\%$ accuracy across 10 diverse subjects on an untouched holdout is a robust, realistic result for non-invasive motor imagery without invasive implants or subject-specific over-tuning.
   - REST recall is high ($89.33\%$), meaning the interface is peaceful when the subject is inactive. Occasional directional misclassifications ($10.7\%$ during rest, $\approx 45\%$ between fists) are actively mitigated by the Safety Shield gating $11.67\%$ of ambiguous trials.
