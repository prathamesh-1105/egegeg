# OpenBCI Live Hardware Test Plan & Bench Protocol (Phase 2E)

**Document Version:** 1.0.0  
**Status:** BENCH READY (Awaiting Physical Hardware Procurement)  
**Target Hardware:** OpenBCI Cyton (8-Channel) + Daisy (16-Channel) Module  
**Interface:** Lab Streaming Layer (LSL) / BrainFlow / OpenBCI GUI  
**Champion Decoder:** Step 62 Surface Laplacian Multimodal Balanced Ensemble (Frozen Base)

---

## 1. Executive Summary & Hardware Contract

This document specifies the exact, step-by-step physical bench-testing procedure for integrating OpenBCI Cyton + Daisy hardware with **NeuroOS**. 

### Critical Scientific Guardrail
> [!IMPORTANT]
> **No physical EEG headset is currently connected to this bench workstation.**
> All Phase 2E tests executed to date are **software unit, mock-stream, and recorded PhysioNet EDF tests**.
> Physical hardware validation, true electrode impedance measurement, and live subject accuracy **CANNOT** be claimed until this bench protocol is executed on live hardware.

---

## 2. Hardware & Cable Prerequisites

### A. Core Equipment
1. **OpenBCI Cyton 8-Channel Biosensing Board** (PIC32MX250F128B microcontroller + ADS1299 24-bit ADC).
2. **OpenBCI Daisy 8-Channel Expansion Module** (Stacked on Cyton via header pins, expanding system to 16 physical channels).
3. **OpenBCI Programmable USB Dongle** (RFDuino BLE wireless communication, FTDI FT231X UART interface).
4. **Standard Micro-USB Cable** or direct USB-A dongle connection.
5. **Power Source:** 4x AA battery pack (6.0V nominal) or single 3.7V LiPo battery (1000mAh+) with standard JST connector. *Never power headset directly from mains USB charger during human EEG recording for electrical isolation and low noise floor.*

### B. Electrodes & Application Supplies
1. **Electrode Type:**
   - Gold-cup electrodes (10mm disc, Ag/AgCl or Au-plated) with touchproof DIN 42802 connectors, OR
   - Dry comb Ag/AgCl polymer electrodes mounted in an adjustable headset cap.
2. **Skin Prep & Conductive Media:**
   - Nuprep abrasive skin exfoliating gel.
   - Ten20 conductive paste (for gold cups) or alcohol wipes (70% isopropyl for dry electrode scalp degreasing).
   - Cotton swabs and medical tape/gauze.
3. **Reference & Bias Clips:**
   - 2x Earclip electrodes: Reference (A1) and Bias / Ground (A2 or earlobe/mastoid).

---

## 3. Host Workstation Software & Driver Prerequisites

1. **FTDI Virtual COM Port (VCP) Drivers:**
   - Ensure FTDI D2XX/VCP driver is installed on Windows.
   - Device Manager check: Ports (COM & LPT) $\to$ `USB Serial Port (COMx)`.
   - Latency timer setting in Advanced Port Settings: Set from default 16ms to **1ms** for minimal packet jitter.
2. **OpenBCI GUI (v5.x+) or BrainFlow LSL Bridge:**
   - OpenBCI GUI downloaded and verified.
   - Live streaming widget configured: `Networking` $\to$ Protocol: `LSL` $\to$ Stream Type: `EEG` $\to$ Data Type: `float32`.
   - LSL stream name: `openbci_eeg` (Type: `EEG`).
3. **Workstation Python Environment:**
   - Python 3.11 with `pylsl`, `numpy`, `scipy`, `mne`, `fastapi`, `uvicorn`.
   - PyLSL wheels require `liblsl.dll` bundled (included in standard `pip install pylsl`).

---

## 4. Physical Electrode Montage & 10-20 Grid Configuration

### The Ultracortex Default Montage Hazard
> [!WARNING]
> The default factory configuration of the OpenBCI Ultracortex Mark IV headset provides a broad cortical montage (`Fp1, Fp2, C3, C4, P3, P4, O1, O2, F7, F8, F3, F4, T3, T4, T5, T6`).
> **This default cap layout LACKS frontocentral and centroparietal motor grid positions.**
> Connecting an Ultracortex cap with the factory layout will be **REJECTED** by NeuroOS (`CONNECTED_INVALID`).
> **Custom 10-20 electrode placement on the motor strip is MANDATORY.**

### Exact Required 16-Channel Pinout

| Cyton / Daisy Pin | 10-20 Position | Brain Region | Role in NeuroOS |
|---|---|---|---|
| **Cyton Ch 1** | **FC3** | Left Premotor Cortex | Model Input Channel 1 |
| **Cyton Ch 2** | **FC1** | Left Supplementary Motor Area | Model Input Channel 2 |
| **Cyton Ch 3** | **FCz** | Central Supplementary Motor Area | Model Input Channel 3 |
| **Cyton Ch 4** | **FC2** | Right Supplementary Motor Area | Model Input Channel 4 |
| **Cyton Ch 5** | **FC4** | Right Premotor Cortex | Model Input Channel 5 |
| **Cyton Ch 6** | **C3** | Left Primary Motor Cortex (Hand Area) | Model Input Channel 6 |
| **Cyton Ch 7** | **C1** | Left Central Paracentral Gyrus | Model Input Channel 7 |
| **Cyton Ch 8** | **Cz** | Vertex Primary Motor Cortex (Foot Area) | Model Input Channel 8 |
| **Daisy Ch 9** | **C2** | Right Central Paracentral Gyrus | Model Input Channel 9 |
| **Daisy Ch 10** | **C4** | Right Primary Motor Cortex (Hand Area) | Model Input Channel 10 |
| **Daisy Ch 11** | **CP3** | Left Primary Somatosensory Cortex | Model Input Channel 11 |
| **Daisy Ch 12** | **CP1** | Left Parietal Postcentral Gyrus | Model Input Channel 12 |
| **Daisy Ch 13** | **CPz** | Central Somatosensory Strip | Model Input Channel 13 |
| **Daisy Ch 14** | **CP2** | Right Parietal Postcentral Gyrus | Model Input Channel 14 |
| **Daisy Ch 15** | **CP4** | Right Primary Somatosensory Cortex | Model Input Channel 15 |
| **Daisy Ch 16** | **O1** | Left Occipital Lobe | Auxiliary (Alpha / Artifact Reference) |
| **Cyton SRB2** | **A1** | Left Earlobe / Mastoid | Hardware Reference Electrode |
| **Cyton BIAS** | **A2** | Right Earlobe / Mastoid | Hardware Common-Mode Ground / BIAS |

---

## 5. Sampling Rate & Resampling Verification

1. **Nominal Hardware Broadcast Rate:**
   - Standalone Cyton (8 channels): **250.0 Hz**
   - Cyton + Daisy (16 channels): **125.0 Hz** (radio packet bandwidth limit over RFDuino)
2. **NeuroOS Target Rate:** Exact **160.0 Hz** (480 samples per 3.0s window).
3. **Resampling Engine:**
   - NeuroOS `LSLEEGSource` automatically applies polyphase / sinc anti-aliased resampling from 125.0 Hz to 160.0 Hz.
   - Verification Check:
     - Inlet Nominal Rate: `125.0 Hz`
     - NeuroOS Internal Buffer: Resampled to `160.0 Hz`
     - Window Output Shape: Strictly `(1, 15, 480)` float32 microvolts.

---

## 6. Signal Quality & Physical Impedance Audit

### A. True Contact Impedance Check (ADS1299 Hardware)
1. In OpenBCI GUI, open the **Hardware Settings / Impedance** tab.
2. Enable 6 nA lead-off current injection at 31.2 Hz.
3. Observe measured electrode contact impedance in $\text{k}\Omega$:
   - **Target:** $\le 10.0\ \text{k}\Omega$ across all 15 motor channels.
   - **Marginal:** $10.0 - 25.0\ \text{k}\Omega$ (acceptable with active skin prep).
   - **Failed:** $> 25.0\ \text{k}\Omega$ (clean scalp, reapply Ten20 paste, reseat cup).
4. Disable impedance test mode before streaming live EEG (impedance current introduces 31.2 Hz artifact).

### B. Software Quality Gate Verification (NeuroOS)
Once live streaming begins, NeuroOS monitors incoming windows:
- **Flatline check:** Channel variance must be $\ge 1\times 10^{-12}\ \mu\text{V}^2$.
- **Disconnection check:** Peak amplitude must be $\ge 0.1\ \mu\text{V}$.
- **Artifact check:** Peak amplitude must not exceed $500.0\ \mu\text{V}$.
- **Sudden pop check:** Sample-to-sample jump must not exceed $150.0\ \mu\text{V}$.
- **Montage check:** All 15 required electrodes must be detected.

---

## 7. Safe Subject Calibration Protocol

Once the stream is verified in state `CONNECTED_VALID`:
1. **Subject Briefing:**
   - Seat user in an ergonomic chair with armrests.
   - Instruct user to keep eyes open, face relaxed, and avoid jaw clenching, swallowing, or head movements during 3.0s trials.
2. **Phase 1: REST (10 Windows = 30 seconds):**
   - Visual cue: Fixation cross.
   - User maintains quiet visual fixation without motor imagery.
   - NeuroOS records 10 validated windows into `collected_windows["REST"]`.
3. **Phase 2: LEFT_FIST (10 Windows = 30 seconds):**
   - Visual cue: Left arrow / fist icon.
   - User performs sustained kinesthetic motor imagery of squeezing the left hand.
   - NeuroOS records 10 validated windows into `collected_windows["LEFT_FIST"]`.
4. **Phase 3: RIGHT_FIST (10 Windows = 30 seconds):**
   - Visual cue: Right arrow / fist icon.
   - User performs sustained kinesthetic motor imagery of squeezing the right hand.
   - NeuroOS records 10 validated windows into `collected_windows["RIGHT_FIST"]`.
5. **Completion & Metadata Storage:**
   - Session completes automatically at 100%.
   - Metadata is saved to `data/calibration_sessions/<session_id>_meta.json`.
   - Base Step 62 model weights remain strictly frozen.

---

## 8. Disconnect, Reconnect & Latency Verification

1. **Unplug / Signal Loss Test:**
   - Unplug USB dongle or power off headset while stream is running.
   - **Expected behavior:** State transitions to `DISCONNECTED` within 1.0s. Buffer flushes. No fake data is emitted. UI displays amber `HARDWARE DISCONNECTED` badge.
2. **Reconnection Test:**
   - Reinsert dongle / power on headset. Click **Scan & Reconnect LSL** in UI.
   - **Expected behavior:** Stream rediscovered, montage re-verified, state transitions to `CONNECTED_VALID` without requiring server restart.
3. **Latency Verification:**
   - Hardware-to-inlet transport latency: $\le 20\ \text{ms}$.
   - NeuroOS pipeline latency: $\le 6\ \text{ms}$ (resampling + quality gate + ML decoder).
   - UI broadcast latency: $\le 3\ \text{ms}$.
   - Total round-trip latency: $\le 30\ \text{ms}$.

---

## 9. Criteria for Accepting or Rejecting First Live Session

| Metric | Acceptance Threshold | Rejection Threshold |
|---|---|---|
| **Detected Channels** | All 15 Motor Channels Present | $\ge 1$ Missing Channel |
| **Physical Impedance** | $< 15\ \text{k}\Omega$ on all motor channels | $> 25\ \text{k}\Omega$ or saturated |
| **Sampling Rate Jitter** | Measured Rate $125 \pm 2\ \text{Hz}$ | Dropouts $> 5\%$ of packets |
| **Window Quality Rate** | $\ge 90\%$ windows pass quality gate | $> 25\%$ windows rejected |
| **REST Baseline Recall** | Step 62 REST recall $\ge 80\%$ | Continuous false triggers |
| **Motor Desynchronization** | Observed C3/C4 ERD during imagery | Flat PSD / zero ERD response |
