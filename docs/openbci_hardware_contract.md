# NeuroOS OpenBCI Hardware Contract & Spatial Interpolation Specification

## 1. Executive Summary

NeuroOS's champion machine learning engine (**Step 60**) is trained on a 15-channel sensorimotor electrode grid (`FC3` through `CP4`) at a 160Hz sampling rate. Standard consumer/research BCI headsets, such as the 8-channel **OpenBCI Cyton**, physically record 8 channels.

To make NeuroOS hardware-ready without retraining or modifying Step 60, we implemented an explicit, scientifically rigorous preprocessing adapter (`src/neuro_os/channel_adapter.py`) utilizing **3D Spherical Spline Spatial Interpolation** (Perrin et al., 1989).

---

## 2. Channel Contract Specifications

### **Hardware Input (8 Physical OpenBCI Cyton Channels)**
The 8-channel OpenBCI Cyton must be placed according to the standard 10-20 International System:

| Channel Index | Electrode Name | Anatomical Region | Function |
| :---: | :---: | :---: | :---: |
| 1 | **F3** | Frontal Left | Frontal boundary anchor |
| 2 | **F4** | Frontal Right | Frontal boundary anchor |
| 3 | **C3** | Central Left | **Physical Motor Cortex (Hand Area)** |
| 4 | **Cz** | Central Vertex | **Physical Motor Cortex (Center)** |
| 5 | **C4** | Central Right | **Physical Motor Cortex (Hand Area)** |
| 6 | **P3** | Parietal Left | Parietal boundary anchor |
| 7 | **Pz** | Parietal Midline | Parietal boundary anchor |
| 8 | **P4** | Parietal Right | Parietal boundary anchor |

### **Model Input (15 Target Step 60 Channels)**
The 15 target channels required by Step 60 in exact deterministic order:

```
[ FC3, FC1, FCz, FC2, FC4,
  C3,  C1,  Cz,  C2,  C4,
  CP3, CP1, CPz, CP2, CP4 ]
```

---

## 3. Physical vs. Reconstructed Channels

```
                        FRONTAL ANCHORS
                          [F3]  [F4]
                           |     |
                 +---------v-----v---------+
                 |   FC3   FC1 FCz FC2 FC4 | <-- Reconstructed (Frontocentral)
                 |                         |
PHYSICAL MOTOR ->|  [C3]   C1 [Cz] C2 [C4] | <-- Physical: C3, Cz, C4; Reconstructed: C1, C2
                 |                         |
                 |   CP3   CP1 CPz CP2 CP4 | <-- Reconstructed (Centroparietal)
                 +---------^-----^---------+
                           |     |
                          [P3]  [P4]
                        PARIETAL ANCHORS
```

* **Physically Measured (3 channels)**: `C3`, `Cz`, `C4`
* **Spatially Reconstructed (12 channels)**: `FC3`, `FC1`, `FCz`, `FC2`, `FC4`, `C1`, `C2`, `CP3`, `CP1`, `CPz`, `CP2`, `CP4`

---

## 4. Scientific Findings & Information Loss

We performed a controlled offline experiment on **300 genuine unseen holdout trials** ($R11$ across subjects $S001\text{--}S010$):

| Benchmark Condition | Accuracy | Macro F1 | REST Recall | LEFT Recall | RIGHT Recall | Processing Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground Truth (15 Physical Channels)** | **67.67%** | **62.24%** | **85.33%** | **57.89%** | **41.89%** | 9.33 ms |
| **Reconstructed (8-ch OpenBCI Interpolated)** | **46.33%** | **45.10%** | **46.00%** | **44.74%** | **48.65%** | **9.42 ms** |

### **Key Scientific Takeaways**:
1. **Mathematical Overhead**: The precomputed spherical spline transformation executes in **0.08 ms** (microsecond scale), contributing zero perceptible latency.
2. **Information Quality Loss**: Reconstructing 12 of 15 channels from 8 physical positions causes a **21.34% accuracy drop** (67.67% $\rightarrow$ 46.33%). High-frequency sensorimotor rhythm (SMR) spatial localized power shifts (ERD/ERS) are smoothed across neighboring electrodes by spherical splines.
3. **Hardware Recommendation**:
   * For **Product Demonstrations / Research Prototypes**: The 8-channel OpenBCI Cyton adapter works out-of-the-box.
   * For **Production Real-World BCI Deployment**: Use **OpenBCI Cyton + Daisy (16 Physical Channels)** or retrain a dedicated 8-channel Step 60 variant.

---

## 5. Hardware Validation Protocol

When physical OpenBCI hardware is available:
1. Connect OpenBCI Cyton via Bluetooth dongle.
2. Open OpenBCI GUI and enable **LSL Streaming** (`Stream Type: EEG`).
3. Place electrodes on subject head: `F3`, `F4`, `C3`, `Cz`, `C4`, `P3`, `Pz`, `P4` (Refer to 10-20 chart; ensure impedance < 10 kΩ).
4. Run `python api_server.py`.
5. Open React HUD at `http://localhost:5173`, switch source to **LSL HARDWARE (OpenBCI)**.
6. Observe live 8-ch $\rightarrow$ 15-ch telemetry and real-time execution.
