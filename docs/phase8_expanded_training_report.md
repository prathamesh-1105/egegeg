# NeuroOS Phase 8 — Expanded 20-Subject Dataset Cohort & Iterative Training Report

**Date:** 2026-11-02 12:56:05  
**Cohort Scope:** 20 Subjects (`S001` - `S020`), 120 Authentic PhysioNet EDF Recordings  
**Total EEG Windows Analyzed:** 3600 Windows (3.0s window, 160 Hz, 15 Motor Cortex Channels)  
**Cohort Mean Holdout Accuracy:** **58.5%**  
**Cohort Mean Macro F1-Score:** **49.55%**  

---

## 1. Executive Summary

To address user requirements for larger dataset scale and iterative training, the NeuroOS BCI pipeline was expanded from 10 subjects to **20 subjects** (`S001`–`S020`) using high-speed multithreaded downloads from PhysioNet AWS S3. Each subject incorporates all 6 motor execution and pure motor imagery runs (`R03`, `R04`, `R07`, `R08`, `R11`, `R12`), creating a balanced, authentic benchmark with over 3,600 discrete physiological trials.

Training was executed on two parallel iterative loops:
1. **Deep Learning Loop (PyTorch SpatialSpectralNet)**: Multi-epoch dense neural network with AdamW, Cosine Annealing, and Class-Weighted Cross Entropy.
2. **Multimodal Calibration Loop (20 Subjects)**: Full tri-ensemble (Shrinkage Regularized LDA + Calibrated RBF-SVM + L2 Logistic Regression) combined with Anatomical Surface Laplacian, Contralateral Spectral Asymmetry, and Riemannian Tangent Space Covariances.

---

## 2. Iterative Deep Learning Convergence Loop

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Val Macro F1 (%) | Learning Rate |
|:-----:|:----------:|:-------------:|:--------:|:-----------:|:----------------:|:-------------:|
| 01 | 1.1027 | 39.67% | 1.0576 | 46.00% | 41.19% | 0.001995 |
| 02 | 1.0683 | 45.50% | 1.0484 | 48.75% | 43.25% | 0.001978 |
| 03 | 1.0423 | 47.75% | 1.0435 | 49.50% | 44.00% | 0.001951 |
| 04 | 1.0356 | 48.04% | 1.0493 | 50.67% | 43.27% | 0.001914 |
| 05 | 1.0236 | 49.08% | 1.0374 | 51.75% | 46.73% | 0.001867 |
| 06 | 1.0188 | 49.67% | 1.0420 | 50.33% | 45.72% | 0.00181 |
| 07 | 1.0129 | 49.71% | 1.0403 | 52.00% | 46.32% | 0.001744 |
| 08 | 1.0130 | 50.12% | 1.0386 | 51.92% | 46.79% | 0.001671 |
| 09 | 0.9994 | 50.33% | 1.0358 | 50.50% | 45.41% | 0.00159 |
| 10 | 0.9922 | 51.62% | 1.0395 | 51.58% | 45.76% | 0.001503 |
| 11 | 1.0009 | 50.67% | 1.0337 | 51.50% | 46.23% | 0.00141 |
| 12 | 0.9959 | 51.29% | 1.0284 | 51.67% | 46.80% | 0.001312 |
| 13 | 0.9942 | 51.00% | 1.0352 | 50.58% | 45.21% | 0.001212 |
| 14 | 0.9847 | 51.58% | 1.0306 | 51.83% | 47.38% | 0.001109 |
| 15 | 0.9826 | 52.25% | 1.0352 | 52.67% | 47.56% | 0.001005 |
| 16 | 0.9856 | 51.88% | 1.0379 | 52.42% | 46.88% | 0.000901 |
| 17 | 0.9768 | 52.08% | 1.0317 | 51.92% | 46.98% | 0.000798 |
| 18 | 0.9750 | 54.08% | 1.0371 | 53.17% | 48.11% | 0.000698 |
| 19 | 0.9758 | 52.54% | 1.0405 | 52.50% | 47.06% | 0.0006 |
| 20 | 0.9701 | 53.71% | 1.0370 | 52.17% | 46.77% | 0.000508 |
| 21 | 0.9733 | 53.21% | 1.0390 | 52.50% | 46.70% | 0.00042 |
| 22 | 0.9574 | 54.37% | 1.0383 | 52.67% | 47.14% | 0.000339 |
| 23 | 0.9579 | 53.54% | 1.0400 | 51.83% | 46.23% | 0.000266 |
| 24 | 0.9675 | 53.00% | 1.0396 | 51.17% | 45.55% | 0.0002 |
| 25 | 0.9646 | 54.79% | 1.0390 | 52.17% | 46.76% | 0.000143 |
| 26 | 0.9578 | 53.71% | 1.0385 | 52.75% | 47.56% | 9.6e-05 |
| 27 | 0.9645 | 53.75% | 1.0450 | 53.00% | 47.26% | 5.9e-05 |
| 28 | 0.9562 | 52.88% | 1.0385 | 52.67% | 47.39% | 3.2e-05 |
| 29 | 0.9623 | 52.88% | 1.0420 | 52.67% | 47.25% | 1.5e-05 |
| 30 | 0.9672 | 53.71% | 1.0387 | 53.17% | 47.75% | 1e-05 |

**Peak Validation Macro F1:** **48.11%**

---

## 3. Per-Subject Calibration Performance Across 20 Subjects

| Subject | Dev Trials | Holdout Trials | Accuracy (%) | Macro F1 (%) | Left Recall (%) | Right Recall (%) | Rest Recall (%) |
|:-------:|:----------:|:--------------:|:------------:|:------------:|:---------------:|:----------------:|:---------------:|
| `S001` | 120 | 60 | 56.67% | 48.84% | 35.71% | 25.0% | 83.33% |
| `S002` | 120 | 60 | 60.0% | 49.92% | 62.5% | 14.29% | 80.0% |
| `S003` | 120 | 60 | 38.33% | 36.21% | 20.0% | 66.67% | 33.33% |
| `S004` | 120 | 60 | 81.67% | 76.9% | 62.5% | 71.43% | 96.67% |
| `S005` | 120 | 60 | 56.67% | 42.62% | 20.0% | 20.0% | 93.33% |
| `S006` | 120 | 60 | 50.0% | 22.22% | 0.0% | 0.0% | 100.0% |
| `S007` | 120 | 60 | 76.67% | 75.34% | 80.0% | 60.0% | 83.33% |
| `S008` | 120 | 60 | 56.67% | 47.99% | 43.75% | 21.43% | 80.0% |
| `S009` | 120 | 60 | 55.0% | 46.99% | 46.67% | 20.0% | 76.67% |
| `S010` | 120 | 60 | 45.0% | 38.04% | 37.5% | 14.29% | 63.33% |
| `S011` | 120 | 60 | 45.0% | 41.95% | 25.0% | 50.0% | 53.33% |
| `S012` | 120 | 60 | 55.0% | 39.98% | 13.33% | 20.0% | 93.33% |
| `S013` | 120 | 60 | 58.33% | 44.81% | 33.33% | 13.33% | 93.33% |
| `S014` | 120 | 60 | 61.67% | 52.23% | 68.75% | 14.29% | 80.0% |
| `S015` | 120 | 60 | 68.33% | 60.94% | 56.25% | 35.71% | 90.0% |
| `S016` | 120 | 60 | 53.33% | 51.74% | 40.0% | 73.33% | 50.0% |
| `S017` | 120 | 60 | 71.67% | 61.53% | 60.0% | 26.67% | 100.0% |
| `S018` | 120 | 60 | 53.33% | 42.51% | 21.43% | 25.0% | 83.33% |
| `S019` | 120 | 60 | 61.67% | 58.26% | 46.67% | 53.33% | 73.33% |
| `S020` | 120 | 60 | 65.0% | 51.9% | 20.0% | 40.0% | 100.0% |

| **COHORT MEAN** | — | — | **58.5%** | **49.55%** | **39.67%** | **33.24%** | **80.33%** |

---

## 4. Architectural Enhancements
- **Dataset Scale**: Doubled cohort size (10 → 20 subjects, 60 → 120 EDF files, 3,600 authentic windows).
- **Dual Modality Inclusion**: Seamlessly combines motor execution runs (`R03, R07, R11`) and pure motor imagery runs (`R04, R08, R12`).
- **Artifact Serialization**: Full model bundles saved in `models/research_phase8/p8_expanded_cohort_champion/selected_model/` for production inference.
- **NeuroOS Integration**: Accessible in the UI via the `Phase8-ExpandedCohort-Ensemble` model profile with full subject support (`S001` - `S020`).