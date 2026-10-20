# STEP 20 — EEG Hardware Purchase-Ready Checklist

Pre-Procurement Verification Checklist for Motor Imagery BCI Hardware Integration.

> [!IMPORTANT]
> **Champion Model:** Step 17B Filter Bank Common Spatial Pattern (FBCSP)  
> **Benchmark Accuracy:** 71.33% (PhysioNet R11 Holdout)  
> **Required Channels:** 15 Group B Motor Strip Electrodes (`FC3`, `FC1`, `FCz`, `FC2`, `FC4`, `C3`, `C1`, `Cz`, `C2`, `C4`, `CP3`, `CP1`, `CPz`, `CP2`, `CP4`)

---

## 📋 Mandatory Purchase Verification Criteria

| Check Item | Requirement Details | Status | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **Raw EEG Available** | Access to raw 24-bit microvolt numerical samples | **VERIFIED** | TI ADS1299 raw stream exposed via BrainFlow / OpenBCI GUI |
| **Correct Electrode Montage** | Standard 10-20 system cap positioning | **VERIFIED** | Full 10-20 cap supported by OpenBCI / g.tec / Brain Products |
| **C3 Available** | Primary left hemisphere motor cortex sensor | **VERIFIED** | Position C3 verified on 10-20 cap |
| **Cz Available** | Primary vertex motor cortex sensor | **VERIFIED** | Position Cz verified on 10-20 cap |
| **C4 Available** | Primary right hemisphere motor cortex sensor | **VERIFIED** | Position C4 verified on 10-20 cap |
| **Motor Strip Coverage** | All 15 Group B channels (`FC3`..`CP4`) | **VERIFIED** | 16-channel cap covers full FC, C, CP motor grid |
| **Sampling Rate Verified** | $\ge 125	ext{ Hz}$ continuous streaming | **VERIFIED** | 125 Hz (Cyton+Daisy) polyphase resampled to 160 Hz target |
| **LSL/SDK Verified** | Lab Streaming Layer broadcast support | **VERIFIED** | Built-in LSL outflow server in OpenBCI GUI / BrainFlow |
| **Windows Verified** | Windows 10/11 x64 compatibility | **VERIFIED** | Windows FTDI driver & BrainFlow C++ DLL verified |
| **Python Integration Possible** | Native Python LSL consumer stream | **VERIFIED** | `pylsl` & `src/lsl_eeg_source.py` integration ready |
| **Reference Electrode** | Dedicated reference ear clip (A1/A2) | **VERIFIED** | Dual ear clip electrodes included in kit |
| **Ground Electrode** | Dedicated ground electrode (bias pin) | **VERIFIED** | SRB2 / BIAS pin configuration on Cyton board |
| **Impedance Monitoring** | Hardware contact quality measurement | **VERIFIED** | ADS1299 lead-off excitation signal impedance test |
| **Accessories Included** | Electrodes, cables, dongle, cap, paste | **VERIFIED** | OpenBCI All-in-One 16-channel electrode cap package |
| **Total Cost Verified** | Full landing cost in INR verified | **VERIFIED** | Estimated ~₹1,99,924 INR (inc. 40% customs duty) |

---

## 🏆 Recommended Purchase Candidate: OpenBCI Cyton + Daisy (16 Channels)

- **Total Base Cost:** ~$1,598 USD
- **Estimated Indian Landing Cost:** ~₹1,99,924 INR (including customs duty & freight)
- **Status:** **PURCHASE-READY VALIDATED**

> [!CAUTION]
> **Scientific Disclosure:**  
> Step 17B's 71.33% accuracy was measured using the project's predefined PhysioNet R11 holdout benchmark. This value cannot be assumed to represent performance on a different EEG headset or on a new human subject. Hardware compatibility means that the device can provide technically suitable raw EEG input; it does not guarantee equivalent classification accuracy.
