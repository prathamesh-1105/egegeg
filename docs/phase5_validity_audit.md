# NeuroOS Phase 5 — Scientific Experimental Validity Audit

**Project:** NeuroOS BCI Platform  
**Phase:** Phase 5 — Motor-Imagery Accuracy Breakthrough and Independent Validation  
**Date:** October 20, 2026  
**Auditor:** Senior EEG/BCI Researcher & Machine Learning Engineer  
**Status:** Audit Complete — Methodological Strengths & Residual Historical Exposures Documented  

---

> [!CAUTION]
> **Investigational Prototype & Non-Medical Device Disclaimer**  
> NeuroOS is an experimental, non-clinical research software prototype. It is not an FDA-cleared or CE-marked medical device, is not intended for diagnostic or therapeutic monitoring, and must not be used for life-critical control. All hardware integrations are strictly in a software-readiness state.

---

## 1. Audit Overview & Objectives

In Phase 5, we conducted a rigorous audit of the complete Phase 4 experimental pipeline, codebase, serialized artifacts, data partitioning protocols, and threshold calibration methods. The goal of this audit is to answer the eight core experimental validity questions without equivocation, explicitly separating validated methodology from historical limitations.

---

## 2. Findings on the Eight Specific Audit Questions

### 1. Exact Trial Counts and Class Counts for Every Split
The dataset consists of 10 subjects ($S001$ to $S010$) from PhysioNet EEGMMIDB across three motor-imagery runs ($R03, R07, R11$). Each run contains exactly 30 event-aligned trials, yielding 900 total trials:

- **Development Split ($R03 + R07$):**
  - **Total Trials:** $60 \text{ trials/subject} \times 10 \text{ subjects} = \mathbf{600 \text{ trials}}$.
  - **REST:** 300 trials (50.00% — exactly 30 trials per subject).
  - **LEFT_FIST:** 151 trials (25.17% — 15 or 16 trials per subject).
  - **RIGHT_FIST:** 149 trials (24.83% — 14, 15, or 16 trials per subject).
- **Holdout Evaluation Split ($R11$):**
  - **Total Trials:** $30 \text{ trials/subject} \times 10 \text{ subjects} = \mathbf{300 \text{ trials}}$.
  - **REST:** 150 trials (50.00% — exactly 15 trials per subject).
  - **LEFT_FIST:** 76 trials (25.33% — 7 or 8 trials per subject).
  - **RIGHT_FIST:** 74 trials (24.67% — 7 or 8 trials per subject).
- **Subject-Wise Breakdown:**
  - $S001$: Dev = 30 REST, 16 LEFT, 14 RIGHT; R11 = 15 REST, 7 LEFT, 8 RIGHT
  - $S002$: Dev = 30 REST, 15 LEFT, 15 RIGHT; R11 = 15 REST, 8 LEFT, 7 RIGHT
  - $S003$: Dev = 30 REST, 15 LEFT, 15 RIGHT; R11 = 15 REST, 7 LEFT, 8 RIGHT
  - $S004$: Dev = 30 REST, 14 LEFT, 16 RIGHT; R11 = 15 REST, 8 LEFT, 7 RIGHT
  - $S005$: Dev = 30 REST, 15 LEFT, 15 RIGHT; R11 = 15 REST, 8 LEFT, 7 RIGHT
  - $S006$: Dev = 30 REST, 15 LEFT, 15 RIGHT; R11 = 15 REST, 7 LEFT, 8 RIGHT
  - $S007$: Dev = 30 REST, 15 LEFT, 15 RIGHT; R11 = 15 REST, 8 LEFT, 7 RIGHT
  - $S008$: Dev = 30 REST, 14 LEFT, 16 RIGHT; R11 = 15 REST, 8 LEFT, 7 RIGHT
  - $S009$: Dev = 30 REST, 16 LEFT, 14 RIGHT; R11 = 15 REST, 7 LEFT, 8 RIGHT
  - $S010$: Dev = 30 REST, 16 LEFT, 14 RIGHT; R11 = 15 REST, 8 LEFT, 7 RIGHT

---

### 2. Cross-Validation Grouping: Subject-Grouped vs. Trial-Grouped vs. Window-Grouped
- **Development Cross-Validation (Phase A):**
  - Implemented as within-subject **5-Fold Stratified Cross-Validation** on each subject's 60 development trials ($R03 + R07$).
  - For each subject, 48 trials form the training partition and 12 trials form the out-of-fold validation partition.
  - Across all 10 subjects, this produces 50 cross-validation folds (10 subjects $\times$ 5 folds).
  - This is a **trial-grouped within-subject stratified split**, not a window-sliced split.
- **Cross-Subject Validation (Phase C):**
  - Implemented as **Leave-One-Subject-Out (LOSO)** cross-validation across the 10 subjects.
  - In each fold, training is performed on 9 subjects ($9 \times 60 = 540$ trials) and evaluated on the 10th unseen subject.
  - This is a true **subject-grouped split**.

---

### 3. Trial Window Independence & Overlap Inspection
- **Source Inspection:** Evaluated `src/step14_pipeline.py::extract_event_aligned_windows`:
  ```python
  aligned_onset = ev['onset_sec'] + onset_offset_sec  # offset = 0.5s
  start_sample = int(aligned_onset * sfreq)
  win_samples = int(window_sec * sfreq)              # 3.0s * 160Hz = 480 samples
  sig_data = raw_sub.get_data(start=start_sample, stop=end_sample)
  windows.append(sig_data)
  labels.append(label)
  ```
- **Confirmation:** Exactly **one discrete 3.0-second window** is extracted per event marker.
- **No Sliding Windows:** The pipeline does **not** slice multiple overlapping sub-windows from a single trial.
- **Independence:** No segment or slice of any trial in a validation fold appears in a training fold. Every window in the dataset represents a distinct physical task execution separated in time.

---

### 4. Preprocessing, Scaling, Feature Selection, and Threshold Isolation
- In `scratch/run_phase4_accuracy_research.py` (lines 576–615), within each of the 50 CV folds:
  - Adaptive filter bank peak frequency estimation (`estimate_peak_frequencies_correct`) is fitted strictly on `X_tr`.
  - Filter Bank CSP (`safe_fit_fbcsp`) spatial filters are fitted strictly on `X_tr` and `y_tr`.
  - Covariance estimators (`Covariances(estimator='lwf')`) and Riemannian Tangent Space (`TangentSpace(metric='riemann')`) mean references are fitted strictly on `X_tr`.
  - Feature selection (`SelectKBest(f_classif, k=...)`) is fitted strictly on `X_tr_fused` and `y_tr`.
  - Scaler (`StandardScaler()`) is fitted strictly on `sel.transform(X_tr_fused)`.
  - Classifiers (`LinearDiscriminantAnalysis`, `LogisticRegression`, `SVC`) are fitted strictly on `X_tr_sc` and `y_tr`.
  - Validation data `X_va` is passed only through `.transform()` calls.
- **Finding:** Inside each cross-validation fold, feature transformation and classifier parameters are isolated without data leakage from `X_va`.

---

### 5. Historical Exposure to R11 Labels & Hyperparameter Peeking (Critical Finding)
Did R11 holdout labels or scores influence architecture selection or thresholds?
- **Audited History:**
  1. In Step 60 (`scratch/tune_step60_threshold.py`), the decision threshold ($0.60$) was explicitly tuned by sweeping across the frozen $R11$ holdout set.
  2. In Phase 4, Candidate Exp 07 achieved **71.17% accuracy and 65.91% Macro F1** on Development CV, whereas Exp 08 achieved **70.67% accuracy and 65.28% Macro F1**. On Development CV, Exp 07 was the true champion.
  3. However, on Holdout $R11$, Exp 08 scored **73.33% accuracy and 68.60% Macro F1**, whereas Exp 07 scored 71.67% accuracy. Exp 08 was promoted based on its superior score on $R11$.
  4. The threshold $0.58$ was hardcoded into `candidate_definitions` for Exp 07 and Exp 08 based on prior experiments rather than determined strictly through out-of-fold calibration on Development CV.
- **Scientific Verdict:** The evaluation on $R11$ cannot be claimed as an unexposed, blinded benchmark because $R11$ scores guided the selection of Exp 08 over Exp 07 and informed the REST threshold.
- **Protocol Correction for Phase 5:** In Phase 5, all model selection, feature dimension selection ($k$), spatial filter choice, and threshold calibration must be driven **exclusively by Development Cross-Validation (R03 + R07)**. $R11$ will be evaluated in a single frozen pass only to verify whether development improvements generalize.

---

### 6. Reproducibility of the 73.33% Result
- **Audit Test Executed:** Evaluated the saved serialized model artifacts in `models/research_phase4/exp_08_k28_ensemble/selected_model/` across all 10 subjects on $R11$.
- **Result:**
  - Accuracy: **73.33%** ($220 / 300$)
  - Macro F1: **68.60%**
  - REST Recall: **90.00%** ($135 / 150$)
  - LEFT_FIST Recall: **59.21%** ($45 / 76$)
  - RIGHT_FIST Recall: **54.05%** ($40 / 74$)
  - Confusion Matrix:
    $$\begin{pmatrix} 135 & 7 & 8 \\ 13 & 45 & 18 \\ 22 & 12 & 40 \end{pmatrix}$$
  - Per-Subject Accuracies: S001: 53.33%, S002: 76.67%, S003: 66.67%, S004: 76.67%, S005: 73.33%, S006: 56.67%, S007: 86.67%, S008: 86.67%, S009: 70.00%, S010: 86.67%.
- **Finding:** The 73.33% result is 100% bitwise reproducible from the saved artifacts and codebase.

---

### 7. Selection of the 0.58 REST Threshold
- The $0.58$ threshold was **not** selected purely out-of-fold from development data. It was carried forward from historical tuning routines that had evaluated multiple thresholds against $R11$.
- In Phase 5, threshold calibration will be computed dynamically within each development training fold using an explicit ROC/Youden index or desired false actuation ceiling on training-fold out-of-fold predictions.

---

### 8. Full Calibration vs. R11 Evaluation Independence
- Calibration experiments in Phase 4 used subsets of $R03 + R07$ (e.g., 5, 10, or 20 trials per class) for calibration, and evaluated on $R11$.
- Because $R03 + R07$ and $R11$ represent distinct recording runs recorded at different times in the experimental session, the calibration trials and the evaluation trials are physically independent recordings. Metric definitions (Overall Accuracy, Macro F1, Balanced Accuracy) were consistent across both evaluations.

---

## 3. Summary of Confirmed Strengths & Addressed Risks

| Dimension | Confirmed Status | Addressed Risk in Phase 5 |
| :--- | :--- | :--- |
| **Window Leakage** | ZERO leakage (1 window per trial, no overlapping sub-windows) | None — trial independence verified. |
| **Transform Fitting** | FBCSP, Covariance, Scaler, Feature Selector fitted strictly on $X_{tr}$ | None — fold isolation confirmed. |
| **Model Selection** | Exp 08 was promoted over Exp 07 based on $R11$ holdout score | **Strict Dev-CV model selection enforced in Phase 5.** |
| **Threshold Tuning** | $0.58$ threshold originated from historical $R11$ grid searches | **Out-of-fold threshold calibration inside Dev-CV.** |
| **Zero-Shot LOSO** | Cross-subject evaluation cleanly isolated on 9-train/1-test | Will be the primary benchmark for generalization. |
| **Reproducibility** | Exp 08 73.33% verified from raw data & saved artifacts | Automated test suite added to prevent regressions. |

---

## 4. Conclusion & Directive for Phase 5

The experimental foundation is sound with respect to trial independence and feature transformation isolation. However, to eliminate the subtle bias introduced by prior $R11$ exposure, Phase 5 will:
1. Treat Development CV ($R03 + R07$, $N=600$) as the sole arbiter for model architecture, feature configuration, and threshold tuning.
2. Focus specifically on breaking the active motor bottleneck: increasing LEFT_FIST and RIGHT_FIST recall (currently 54%–59%) and reducing motor lateralization confusion (currently 20%).
3. Develop true cross-subject alignment methods (e.g. Euclidean covariance alignment) to close the zero-shot LOSO gap without test-label leakage.
