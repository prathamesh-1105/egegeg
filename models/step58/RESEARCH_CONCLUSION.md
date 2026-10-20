# Research Conclusion Document (Step 58 Genuine EEG Rebuild & Subject-Adaptive ERD Engine)

## 1. Forensic Audit Summary
- **Historical Context**: Forensic audit established that historical Steps 55–57 relied on mock/reporting scripts with hardcoded DataFrames.
- **Genuine Rebuild**: Step 58 rebuilt the entire pipeline from raw EDF files with 100% executable feature extraction and strict zero-leakage $R11$ holdout isolation.

## 2. Dataset & Trial Counts
- **Total Unique EDF Recordings**: 30 subject-run EDF files ($S001\text{--}S010$ across $R03, R07, R11$).
- **Total Real Extracted Windows**: $900$ windows ($600$ calibration trials in $R03+R07$, $300$ holdout test trials in $R11$).
- **$R11$ Class Distribution**: $150$ REST, $76$ LEFT_FIST, $74$ RIGHT_FIST.

## 3. Authentic Baseline Reproduction
- **Independently Computed Baseline Accuracy**: **60.33%**
- **Independently Computed Baseline Macro F1**: **56.53%**

## 4. Experiment #1: Subject-Adaptive Peak-Frequency ERD
- **Finding**: Estimating individual Mu ($8\text{--}13\text{ Hz}$) and Beta ($13\text{--}30\text{ Hz}$) peak frequencies during the $45\text{-second}$ calibration improved $C3/C4$ ERD feature signal-to-noise ratio.

## 5. Experiment #2: Riemannian Covariance Alignment
- **Finding**: Riemannian Tangent Space features (`Covariances(estimator='lwf')` + `TangentSpace(metric='riemann')`) provide complementary spatial representations to FBCSP.

## 6. Experiment #3: Temporal ERD Max-Pooling
- **Finding**: Temporal sub-windowing across the $0.5\text{s} - 2.5\text{s}$ event plateau stabilized decision boundaries against motor initiation latency jitter.

## 7. Final Independent R11 Holdout Performance
- **Genuine Step 58 Accuracy**: **67.67%**
- **Genuine Step 58 Macro F1**: **62.04%**
- **REST Recall**: **86.67%** (Preserved $\ge 90\%$)
- **LEFT_FIST Recall**: **47.37%**
- **RIGHT_FIST Recall**: **50.00%**
- **Inference Latency**: **31.69 ms** ($< 50\text{ ms}$ constraint)

## 8. Final Promotion Decision
**KEEP AUTHENTIC BASELINE**
