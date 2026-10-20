# NeuroOS Phase 4 — Scientific & Methodological Accuracy Audit

**Author:** Senior BCI Researcher & Lead EEG Signal Processing Scientist  
**Date:** October 20, 2026  
**Status:** Complete & Empirically Verified  
**Dataset:** PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB, Subjects S001–S010, Runs R03, R07, R11)  
**Historical Baselines:** Step 60 SOTA Engine, Step 61 Multimodal Ensemble, Step 62 Surface Laplacian Champion  

---

## 1. Executive Summary & Audit Mandate

Before undertaking model optimization or architectural modifications in Phase 4, a comprehensive scientific audit of the NeuroOS code repository was conducted. The audit covered data loading, channel montage ordering, temporal event alignment, spatial filtering, feature extraction, cross-validation isolation, historical model artifacts, threshold calibration procedures, and evaluation metric definitions.

### Key Audit Findings at a Glance
1. **Critical Channel Montage Misalignment in Asymmetry & Laplacian Modules:**
   A severe discrepancy was identified between the 15-channel array produced by `src/step7_pipeline.py` (`load_and_preprocess_cached`) and the channel indices defined in `src/step33_pipeline.py` (`GROUP_B_CHANNELS`). This caused all historical asymmetry features, peak frequency estimations, and approximate Surface Laplacian derivations to operate on the **wrong electrodes**:
   - The code assumed index 8 was $C3$; it was actually $C2$.
   - The code assumed index 12 was $C4$; it was actually $CPz$.
   - The code assumed index 10 was $Cz$; it was actually $CP3$.
   - The code assumed index 1 was $FC3$; it was actually $FC1$.
   - The code assumed index 5 was $FC4$; it was actually $C3$.
   As a consequence, the historical "contralateral motor asymmetry" feature was actually computing the difference between Parietal Midline ($CPz$) and Right Central ($C2$), explaining the pervasive Left $\leftrightarrow$ Right motor confusion.
2. **Surface Laplacian Spatial Neighbor Corruption:**
   Because `scratch/run_phase2b_accuracy_experiments.py` used the corrupted channel index assumptions, the local spatial derivation subtracted unrelated electrodes from $C2$ and $CPz$ rather than performing true orthogonal derivation around $C3$, $C4$, and $Cz$.
3. **Threshold Peeking / Contamination on Run 11 (R11):**
   Historical scripts (e.g., `scratch/tune_step60_threshold.py`) tuned the REST decision threshold ($P_{rest} \ge 0.58 / 0.60 / 0.62$) by sweeping directly over the 300 test trials of `R11`. R11 cannot be regarded as a pristine, virgin holdout set. It must be treated as a historical comparison benchmark, with out-of-fold development validation serving as the primary model selection criterion.
4. **Subject-Specific (Personalized) vs. Cross-Subject Architecture:**
   The reported Step 60, 61, and 62 benchmarks (70.67%–72.67%) are **subject-specific models**, where each subject has their own spatial filters and classifiers trained on their own runs (`R03` + `R07`, 60 trials) and evaluated on their own `R11` (30 trials). When evaluated zero-shot across subjects without personalization, accuracy drops to ~39.3% (near random chance 33.3%). Both paradigms must be evaluated and reported with complete separation.
5. **Class Imbalance & REST Dominance:**
   PhysioNet Task 1 contains 50% REST (15 trials/run) and 50% Motor Active (7–8 LEFT_FIST, 7–8 RIGHT_FIST). High overall accuracy (72%+) was achieved historically by driving REST recall to >90% through threshold gating, while active motor recognition remained modest (LEFT recall: 56.58%, RIGHT recall: 52.70%).
6. **Zero-Phase (Non-Causal) vs. Causal Filtering in Replay:**
   The offline loader applies `scipy.signal.filtfilt` (bidirectional, zero-phase filtering) across the entire 125-second continuous recording before epoching. Real-time streaming requires causal processing or overlap buffering to maintain fidelity.

---

## 2. Empirical Verification of Pipeline Components

### 2.1 Raw Dataset & Event Annotation Structure
- **Recordings:** PhysioNet EEGMMIDB 64-channel EDF files sampled at 160 Hz.
- **Subjects Audited:** `S001` through `S010`.
- **Target Task (Task 1):** Real vs. Imagined opening and closing of left or right fist:
  - Run 03 (`R03`): Motor Imagery Run 1 (Calibration / Dev)
  - Run 07 (`R07`): Motor Imagery Run 2 (Calibration / Dev)
  - Run 11 (`R11`): Motor Imagery Run 3 (Historical Evaluation)
- **Event Annotation Timing:**
  Every file contains 30 event-related annotations:
  - `T0` (Rest): Exactly 15 events per run, each with duration = 4.2 seconds (total = 63.0s).
  - `T1` (Left Fist MI): 7 or 8 events per run, each with duration = 4.1 seconds (mean total = 30.8s).
  - `T2` (Right Fist MI): 7 or 8 events per run, each with duration = 4.1 seconds (mean total = 30.8s).
  - Total duration per run = 124.6 to 125.0 seconds.

### 2.2 Epoch Extraction & Timing Contract
In `src/step14_pipeline.py` (`extract_event_aligned_windows`):
- `window_sec`: 3.0 seconds (480 samples at 160 Hz).
- `onset_offset_sec`: 0.5 seconds.
- **Timing Mechanics:** For each event starting at `onset_sec`, the extracted window spans `[onset_sec + 0.5s, onset_sec + 3.5s]`.
- **Neurophysiological Justification:** The 0.5s offset accounts for sensory cue perception latency and avoids early visual evoked potentials (VEPs). The 3.0s duration captures the steady-state sensorimotor rhythm (SMR) desynchronization within the 4.1s execution window.
- **Verification Status:** Verified. No window exceeds event boundaries; all 30 trials per run are cleanly extracted without NaN or Inf values.

---

## 3. Discovered Defects & Scientific Vulnerabilities

### 3.1 Defect 1: Critical Channel Montage Index Misalignment

#### Evidence and Demonstration:
`src/step7_pipeline.py` defines the channel extraction function `load_and_preprocess_cached` using `group_b_targets`:
```python
group_b_targets = [
    "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C3", "C1", "CZ", "C2", "C4",
    "CP3", "CP1", "CPZ", "CP2", "CP4"
]
```
When an EDF file (e.g. `S001R03.edf`) is loaded and picked using these targets, the returned 15-channel array has the exact channel order:
```
Index  0: 'Fc3.'
Index  1: 'Fc1.'
Index  2: 'Fcz.'
Index  3: 'Fc2.'
Index  4: 'Fc4.'
Index  5: 'C3..'   <-- True C3
Index  6: 'C1..'
Index  7: 'Cz..'   <-- True Cz
Index  8: 'C2..'   <-- True C2
Index  9: 'C4..'   <-- True C4
Index 10: 'Cp3.'   <-- True CP3
Index 11: 'Cp1.'
Index 12: 'Cpz.'   <-- True CPz
Index 13: 'Cp2.'
Index 14: 'Cp4.'   <-- True CP4
```

However, in `src/step33_pipeline.py`, line 44, `GROUP_B_CHANNELS` was defined as:
```python
GROUP_B_CHANNELS = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CPz']
```
This erroneous list included 4 temporal channels (`FC5, FC6, C5, C6`) that were never extracted, while omitting 4 parietal channels (`CP3, CP1, CP2, CP4`).
Consequently, the index lookup variables evaluated to:
- `C3_IDX = GROUP_B_CHANNELS.index('C3')` $\rightarrow$ **8**
- `C4_IDX = GROUP_B_CHANNELS.index('C4')` $\rightarrow$ **12**
- `CZ_IDX = GROUP_B_CHANNELS.index('Cz')` $\rightarrow$ **10**
- `FC3_IDX = GROUP_B_CHANNELS.index('FC3')` $\rightarrow$ **1**
- `FC4_IDX = GROUP_B_CHANNELS.index('FC4')` $\rightarrow$ **5**

#### Impact on Feature Extraction:
In `src/step61_pipeline.py`, `src/step62_pipeline.py`, `src/step59_pipeline.py`, and `scratch/run_phase2b_accuracy_experiments.py`:
```python
vars_ch = np.var(win, axis=-1)
c3_v = vars_ch[C3_IDX]   # Index 8 -> Actually reading 'C2..' (Right central)!
c4_v = vars_ch[C4_IDX]   # Index 12 -> Actually reading 'Cpz.' (Parietal midline)!
fc3_v = vars_ch[FC3_IDX] # Index 1 -> Actually reading 'Fc1.' (Frontal midline)!
fc4_v = vars_ch[FC4_IDX] # Index 5 -> Actually reading 'C3..' (Left central)!

log_asym_motor = np.log(c4_v + 1e-8) - np.log(c3_v + 1e-8)  # log(CPz) - log(C2)!
log_asym_premotor = np.log(fc4_v + 1e-8) - np.log(fc3_v + 1e-8)  # log(C3) - log(FC1)!
ratio_motor = (c3_v - c4_v) / (c3_v + c4_v + 1e-6)  # (C2 - CPz) / (C2 + CPz)!
```
Furthermore, in `estimate_subject_peak_frequency` (`src/step59_pipeline.py`):
```python
c3_sig = X_calib[:, C3_IDX, :]  # Reads C2
c4_sig = X_calib[:, C4_IDX, :]  # Reads CPz
```
**Conclusion:** The historical models never compared left motor strip ($C3$) against right motor strip ($C4$). They contrasted parietal midline against right central cortex. This directly explains why previous reports documented extensive lateralization confusion ($Left \leftrightarrow Right$).

---

### 3.2 Defect 2: Corrupted Surface Laplacian Derivation

In `scratch/run_phase2b_accuracy_experiments.py`, lines 65–78:
```python
def apply_surface_laplacian_approx(X: np.ndarray) -> np.ndarray:
    X_out = X.copy()
    c3_neighbors = [1, 7, 9]    # Assumed FC3, C5, C1
    c4_neighbors = [5, 11, 13]   # Assumed FC4, C2, C6
    cz_neighbors = [3, 9, 11]    # Assumed FCz, C1, C2

    X_out[:, C3_IDX, :] = X[:, C3_IDX, :] - np.mean(X[:, c3_neighbors, :], axis=1)
    X_out[:, C4_IDX, :] = X[:, C4_IDX, :] - np.mean(X[:, c4_neighbors, :], axis=1)
    X_out[:, CZ_IDX, :] = X[:, CZ_IDX, :] - np.mean(X[:, cz_neighbors, :], axis=1)
    return X_out
```
Because of the index mismatch:
- `C3_IDX` (8, which is $C2$) had `mean(channel 1 [FC1], channel 7 [Cz], channel 9 [C4])` subtracted from it.
- `C4_IDX` (12, which is $CPz$) had `mean(channel 5 [C3], channel 11 [CP1], channel 13 [CP2])` subtracted from it.
- `CZ_IDX` (10, which is $CP3$) had `mean(channel 3 [FC2], channel 9 [C4], channel 11 [CP1])` subtracted from it.
True $C3$ (index 5) and True $C4$ (index 9) were never Laplaced.

#### Correct Orthogonal Surface Laplacian Montage:
On the actual 15-channel array `['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']`:
- **For $C3$ (Index 5):** Orthogonal neighbors are Anterior $FC3$ (Index 0), Medial $C1$ (Index 6), Posterior $CP3$ (Index 10).
- **For $C4$ (Index 9):** Orthogonal neighbors are Anterior $FC4$ (Index 4), Medial $C2$ (Index 8), Posterior $CP4$ (Index 14).
- **For $Cz$ (Index 7):** Orthogonal neighbors are Anterior $FCz$ (Index 2), Left $C1$ (Index 6), Right $C2$ (Index 8), Posterior $CPz$ (Index 12).
All required neighbors exist cleanly within the 15-channel contract.

---

### 3.3 Defect 3: Historical Threshold Peeking on R11

In `scratch/tune_step60_threshold.py`:
- Line 14: `thresholds = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65]`
- Lines 24–69: Iterated through all 10 subjects on `X_r11_s`, evaluated accuracy for every threshold, and reported:
  `BEST THRESHOLD: 0.60 with Accuracy: 70.67%`
- Later in `src/step62_pipeline.py`: Hardcoded `rest_p >= 0.58`.
- In `api_server.py`: Hardcoded `rest_prob > 0.62`.

#### Scientific Implications:
The threshold selection was fitted directly to maximize accuracy on `R11`. Therefore:
1. `R11` is **not an untouched test set** in historical reports.
2. Any comparison against historical numbers must acknowledge this contamination.
3. In Phase 4, all threshold decisions must be fitted strictly inside out-of-fold cross-validation folds (`R03` + `R07`), with `R11` used only for final, fixed evaluation.

---

### 3.4 Defect 4: Model Scope (Personalized vs. Cross-Subject Generalization)

- Step 60, Step 61, and Step 62 are **personalized ensembles**. A separate model was saved in `models/step62/selected_model/S001` through `models/step62/selected_model/S010`.
- In Phase 2C (`docs/phase2c_calibration_report.md`), when unseen subjects were evaluated using a single model (`S001`), accuracy collapsed to **39.33%** (near the 33.33% random baseline) with 79 false motor commands out of 150 idle REST epochs.
- Phase 4 must evaluate and report both paradigms honestly:
  - **A. Subject-Specific Evaluation (Benchmark Comparison):** Fitted on each subject's `R03`+`R07`, tested on `R11`.
  - **B. Leave-One-Subject-Out (LOSO) Cross-Validation:** Trained on 9 subjects, tested zero-shot on the 10th unseen subject.
  - **C. Calibrated / Population-Adapted Evaluation:** Population prior adapted with small, legitimate calibration sets (e.g., 5 or 10 trials per class) from `R03`+`R07`.

---

### 3.5 Defect 5: Metric Denominators & Imbalance Masking

- PhysioNet Task 1 trials per subject across `R11`:
  - `REST`: 15 trials (50.0%)
  - `LEFT_FIST`: 7 or 8 trials (~25.0%, total 76 across cohort)
  - `RIGHT_FIST`: 7 or 8 trials (~25.0%, total 74 across cohort)
  - Total trials across cohort: 300 trials.
- **Vulnerability:** An uncalibrated classifier that predicts `REST` on 70% of trials can easily achieve >70% overall accuracy while performing no better than random guessing on motor active trials.
- **Phase 4 Mandatory Reporting:**
  - Overall Accuracy ($N_{correct} / 300$)
  - Balanced Accuracy ($\frac{1}{3} \sum_{c} \text{Recall}_c$)
  - Macro F1 (unweighted mean of per-class F1)
  - Per-class Precision, Recall, and F1
  - REST false actuation rate ($FP_{REST} / 150$)
  - Left $\leftrightarrow$ Right motor confusion rate ($N_{L \leftrightarrow R} / 150$)

---

## 4. Preserved Baseline Benchmarks

The following baseline metrics from existing historical artifacts are preserved immutable:

| Benchmark Source | Configuration | Accuracy | Macro F1 | REST Recall | LEFT Recall | RIGHT Recall | Model Path |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Step 60 Historical** | Adaptive FBCSP + TS + RegLDA ($P_{rest} \ge 0.60$) | 70.67% | 65.72% | 86.67% | 57.89% | 48.65% | `models/step60/` |
| **Step 61 Historical** | Multimodal Voting (LDA + SVM, $P_{rest} \ge 0.60$) | 72.00% | 67.23% | 87.33% | 57.89% | 52.70% | `models/step61/` |
| **Step 62 Historical** | Surface Laplacian Tri-Ensemble ($P_{rest} \ge 0.60$) | **72.67%** | **67.31%** | **90.67%** | **56.58%** | **52.70%** | `models/step62/` |
| **Phase 2F Operational** | Step 62 in NeuroOS Streaming Pipeline ($P_{rest} \ge 0.62$) | 72.00% | 67.23% | 89.33% | 57.89% | 51.35% | `docs/phase2f_reliability_metrics.json` |

---

## 5. Phase 4 Action Plan

Based on these audit findings, Phase 4 research will execute the following structured plan:

1. **Fix Channel Index Alignment:** Correct `C3`, `C4`, `Cz`, `FC3`, `FC4`, `CP3`, `CP4`, `CPz` mappings to match the verified 15-channel montage `['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']`.
2. **Implement Anatomical Surface Laplacian:** True local orthogonal derivation using physically adjacent electrodes.
3. **True Bilateral ERD & Spectral Asymmetry:** Contralateral power subtraction $(C4 - C3)$ in individual $\mu$ (8–13 Hz) and $\beta$ (13–30 Hz) bands.
4. **Multi-Scale Temporal & Spatial Ensembles:** Compare window durations (2.0s, 2.5s, 3.0s), onset offsets (0.25s, 0.5s), and regularized covariance estimators.
5. **Class-Balanced Classifiers & Hierarchical Decoders:**
   - Shrinkage-Regularized LDA (Ledoit-Wolf).
   - Class-Weighted Regularized Logistic Regression.
   - Balanced RBF SVM.
   - Riemannian Tangent Space Classification.
   - Hierarchical Classifier: Stage 1 (Rest vs Active Motor) $\rightarrow$ Stage 2 (Left vs Right Fist).
6. **Strict Out-of-Fold Threshold Selection:** Fit all confidence and REST gating thresholds strictly on training folds within the Development set (`R03` + `R07`).
7. **Complete Evaluation Protocol:** Execute both within-subject cross-run evaluation and cross-subject LOSO evaluation with transparent, uninflated metrics.
