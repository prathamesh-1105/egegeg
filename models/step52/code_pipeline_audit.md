# Forensic Audit — Code Pipeline Implementation

## Complete Pipeline Trace
1. **EDF File Loading**: Read using MNE raw EDF reader with strict channel selection (`Group B` = 15 motor cortex channels: FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CPz, CP4, P1, P2).
2. **Annotation Extraction & Event Mapping**: Parse EDF annotations where $T0 ightarrow 	ext{REST}$, $T1 ightarrow 	ext{LEFT\_FIST}$, $T2 ightarrow 	ext{RIGHT\_FIST}$.
3. **Window Extraction**: 3.0-second epoching (480 samples at 160 Hz) extracted with a $0.5	ext{s}$ post-onset delay ($0.5	ext{s}$ to $3.5	ext{s}$ post-event).
4. **Preprocessing & Filtering**: Multi-Band Filter Bank (Bank A: 7–35 Hz bandpass filtering).
5. **Feature Extraction**:
   - Filter Bank Common Spatial Patterns (FBCSP): 4 components per sub-band.
   - Contralateral Sensorimotor Asymmetry Ratios ($\log(P_{	ext{C3}}/P_{	ext{C4}})$).
   - Welch Power Spectral Density (PSD) in $\mu$ (8–13 Hz) and $eta$ (13–30 Hz) bands.
6. **Feature Selection & Scaling**: SelectKBest ($k=20$) via $f$-classif and StandardScaler fit strictly inside training folds.
7. **Model Training & Inference**: GroupKFold outer cross-validation and probability alignment via `align_probs`.
8. **Two-Pass Motor Specialist Gating (Step 34)**:
   - High-confidence REST thresholding ($P(	ext{REST}) > 0.65$).
   - Contralateral asymmetry gating for motor direction disambiguation on target subjects ($S001, S003, S006, S008, S009$).
