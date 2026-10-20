# NeuroOS 16-Channel OpenBCI Cyton + Daisy Hardware Contract

## 1. Executive Overview

This contract specifies the physical electrode configuration, channel mapping, and signal specifications for deploying NeuroOS with a **16-Channel OpenBCI Cyton + Daisy** expansion board.

Unlike 8-channel setups that require spatial interpolation, the 16-channel Cyton + Daisy configuration captures all 15 sensorimotor electrodes physically, achieving **100.0% information preservation** and **0.0% spatial interpolation accuracy loss**.

---

## 2. 16-Channel Physical Electrode Layout

| Physical Channel | Electrode Name | 10-20 Position | Step 60 Model Mapping | Status |
| :---: | :---: | :---: | :---: | :---: |
| 1 | **FC3** | Frontocentral Left | `FC3` | Physical |
| 2 | **FC1** | Frontocentral Left Midline | `FC1` | Physical |
| 3 | **FCz** | Frontocentral Vertex | `FCz` | Physical |
| 4 | **FC2** | Frontocentral Right Midline | `FC2` | Physical |
| 5 | **FC4** | Frontocentral Right | `FC4` | Physical |
| 6 | **C3** | Central Left (Motor Hand) | `C3` | Physical |
| 7 | **C1** | Central Left Midline | `C1` | Physical |
| 8 | **Cz** | Central Vertex | `Cz` | Physical |
| 9 | **C2** | Central Right Midline | `C2` | Physical |
| 10 | **C4** | Central Right (Motor Hand) | `C4` | Physical |
| 11 | **CP3** | Centroparietal Left | `CP3` | Physical |
| 12 | **CP1** | Centroparietal Left Midline | `CP1` | Physical |
| 13 | **CPz** | Centroparietal Vertex | `CPz` | Physical |
| 14 | **CP2** | Centroparietal Right Midline | `CP2` | Physical |
| 15 | **CP4** | Centroparietal Right | `CP4` | Physical |
| 16 | **O1** | Occipital Left | *Unused by Model* | Aux Alpha Baseline |

---

## 3. Unused Channel Rationale

* **Channel 16 (`O1`)**: Placed over the occipital lobe to monitor visual alpha rhythm (8-12 Hz) or serve as a systemic impedance reference. It is excluded from Step 60's sensorimotor feature extraction array, which consumes exactly 15 motor cortex channels (`FC3` through `CP4`).

---

## 4. Signal & Hardware Requirements

* **Sampling Rate**: 160.0 Hz (or 250 Hz re-sampled to 160 Hz via anti-aliasing sinc filter).
* **Sample Format**: `float32` values in microvolts ($\mu V$).
* **Input Window Length**: 480 samples per channel (3.0 seconds).
* **Impedance Requirement**: $< 10\text{ k}\Omega$ per electrode before starting session.
