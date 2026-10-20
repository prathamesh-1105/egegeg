# NeuroOS Hardware Setup & Bench-Testing Guide (OpenBCI Cyton + Daisy)

**Document Version:** 3.0.0 (Master Phase Production Release)  
**Date:** October 20, 2026  
**Target Hardware:** OpenBCI Cyton (8-ch) + Daisy (16-ch expansion)  
**Transmission Layer:** Lab Streaming Layer (LSL) over Local Network  
**Target Architecture:** Step 62 Surface Laplacian 15-Channel Sensorimotor Grid  

---

## 1. Hardware Overview & Requirements

NeuroOS includes a production-ready Lab Streaming Layer (LSL) hardware bridge (`LSLEEGSource`) designed to interface with an OpenBCI Cyton + Daisy 16-channel system.

### Hardware Specifications
- **Base Board:** OpenBCI Cyton (8 channels, 24-bit Texas Instruments ADS1299 ADC, PIC32MX250F128B MCU).
- **Expansion Board:** OpenBCI Daisy Module (adds 8 channels for 16 total channels).
- **Nominal Sampling Rate:**
  - Cyton 8-channel: $250.0\text{ Hz}$
  - Cyton + Daisy 16-channel: $125.0\text{ Hz}$ (due to serial RF bandwidth limits over Bluetooth/dongle)
- **Host Interface:** OpenBCI RF USB Dongle (FTDI FT231X USB-to-UART bridge).
- **Electrode Type:** Ag/AgCl wet cup electrodes with conductive Ten20 paste (recommended for high signal-to-noise ratio), or dry comb electrodes on custom 10-20 headwear.

---

## 2. Electrode Montage & Channel Contract

### 2.1 Required 15-Channel Motor Strip Grid

The Step 62 Surface Laplacian Ensemble classifier strictly requires 15 continuous sensorimotor recording channels centered over the bilateral primary motor cortex and supplementary motor areas:

```
                  Frontal Polar
                     (Anterior)
                         │
     FC3     FC1        FCz        FC2     FC4      [Frontocentral Motor Band]
      │       │          │          │       │
      C3      C1         Cz         C2      C4      [Primary Motor Strip / SMR]
      │       │          │          │       │
     CP3     CP1        CPz        CP2     CP4      [Centroparietal Sensory Band]
                         │
                     Occipital
                    (Posterior)

  Reference: Earlobe A1 (Left) or Linked Mastoids
  Ground:    Earlobe A2 (Right) or Forehead Fpz
```

### 2.2 OpenBCI 16-Channel Physical Channel Mapping

When wiring electrodes to the Cyton and Daisy pin headers, use the following standardized channel assignment:

| Physical Header | Board | Channel Index | Standard 10-20 Label | Function |
|---|---|---|---|---|
| **Pin 1** | Cyton | Ch 1 | `FC3` | Left Frontocentral (Laplacian neighbor) |
| **Pin 2** | Cyton | Ch 2 | `FC1` | Left Pre-motor |
| **Pin 3** | Cyton | Ch 3 | `FCz` | Midline Pre-motor |
| **Pin 4** | Cyton | Ch 4 | `FC2` | Right Pre-motor |
| **Pin 5** | Cyton | Ch 5 | `FC4` | Right Frontocentral (Laplacian neighbor) |
| **Pin 6** | Cyton | Ch 6 | `C3` | **Left Motor Cortex (Contralateral Right Fist)** |
| **Pin 7** | Cyton | Ch 7 | `C1` | Left Central |
| **Pin 8** | Cyton | Ch 8 | `Cz` | Midline Central (Feet/Rest focus) |
| **Pin 9** | Daisy | Ch 9 | `C2` | Right Central |
| **Pin 10** | Daisy | Ch 10 | `C4` | **Right Motor Cortex (Contralateral Left Fist)** |
| **Pin 11** | Daisy | Ch 11 | `CP3` | Left Centroparietal (Laplacian neighbor) |
| **Pin 12** | Daisy | Ch 12 | `CP1` | Left Somatosensory |
| **Pin 13** | Daisy | Ch 13 | `CPz` | Midline Somatosensory |
| **Pin 14** | Daisy | Ch 14 | `CP2` | Right Somatosensory |
| **Pin 15** | Daisy | Ch 15 | `CP4` | Right Centroparietal (Laplacian neighbor) |
| **Pin 16** | Daisy | Ch 16 | `Aux/GND` | Unused / Ambient EOG reference |
| **REF** | Cyton | Reference | `A1` | Left Ear Clip / Mastoid |
| **BIAS** | Cyton | Ground | `A2` | Right Ear Clip / Fpz |

---

## 3. Montage Rejection: Why the Standard Ultracortex Cap is Rejected

> [!WARNING]
> **CRITICAL HARDWARE RESTRICTION:**
> NeuroOS explicitly rejects the default OpenBCI Ultracortex Mark IV 16-channel headwear montage.

### Explanation:
The factory default Ultracortex Mark IV headset spreads its 16 dry electrode nodes broadly across the entire scalp for general cognitive neuroscience:
`Fp1, Fp2, C3, C4, T5, T6, O1, O2, F7, F8, F3, F4, T3, T4, P3, P4`

This layout only provides **two** central motor channels (`C3` and `C4`). It completely lacks the 13 required frontocentral and centroparietal channels (`FC3, FC1, FCz, FC2, FC4, C1, Cz, C2, CP3, CP1, CPz, CP2, CP4`).

The Step 62 champion model requires local five-point Surface Laplacian derivations around C3 and C4:
$$\text{Lap}(C3) = C3 - \frac{1}{4}(FC3 + C1 + C5 + CP3)$$
$$\text{Lap}(C4) = C4 - \frac{1}{4}(FC4 + C2 + C6 + CP4)$$

Without physical electrodes at these locations, spatial Laplacian derivations cannot be computed. Unvalidated spatial interpolation over missing electrodes introduces severe spatial blurring and spurious phase shifts.

**Contract Enforcement:** When NeuroOS detects an Ultracortex Mark IV default cap configuration, the LSL bridge immediately transitions to `CONNECTED_INVALID`, disengages the Safety Shield, and displays a clear explanation in the UI.

To use an Ultracortex Mark IV frame with NeuroOS, the mechanical dry electrode pods must be relocated into the 15 10-20 sensorimotor locations listed in Section 2.

---

## 4. Sampling Rate Conversion & Anti-Aliased Resampling

The Step 62 champion pipeline was trained on PhysioNet recordings sampled at exactly $160.0\text{ Hz}$ with spectral bandpass filters calibrated between $1.0\text{ Hz}$ and $45.0\text{ Hz}$.

The OpenBCI Cyton + Daisy transmits 16 channels at a nominal rate of $125.0\text{ Hz}$. An 8-channel Cyton transmits at $250.0\text{ Hz}$.

### Mathematical Resampling Pipeline
NeuroOS implements real-time polyphase resampling (`LSLEEGSource._resample_chunk`) using `scipy.signal.resample`:

1. **Upsampling from 125 Hz to 160 Hz:**
   $$\text{Ratio} = \frac{160}{125} = \frac{32}{25} = 1.28$$
   - Each input buffer chunk of $N_{\text{in}}$ samples is transformed to the frequency domain via FFT.
   - Sinc-interpolation in the frequency domain upsamples the signal to $N_{\text{out}} = \text{round}(N_{\text{in}} \times 1.28)$ samples.
   - An intrinsic anti-aliasing low-pass filter eliminates spectral imaging artifacts above the original Nyquist frequency ($62.5\text{ Hz}$).

2. **Downsampling from 250 Hz to 160 Hz:**
   $$\text{Ratio} = \frac{160}{250} = \frac{16}{25} = 0.64$$
   - High-frequency spectral content above the target Nyquist frequency ($80.0\text{ Hz}$) is truncated in the Fourier domain to prevent aliasing.

3. **Buffer Synchronization:**
   Resampled samples enter a thread-safe ring buffer capped at 1,920 samples (~12 seconds) to protect against memory growth during UI stalls.

---

## 5. Bench-Testing Checklist (Step-by-Step)

Follow this protocol before attempting live human EEG acquisition:

### Stage 1: Synthetic Signal Generation Bench Test
- [ ] Connect OpenBCI Cyton + Daisy to a hardware signal generator or use the built-in ADS1299 internal test signal (1 Hz, $\pm 187.5\mu\text{V}$ square wave).
- [ ] Plug the OpenBCI USB Dongle into the host computer. Set the switch to `GPIO_6` (default PC communication).
- [ ] Launch the OpenBCI GUI. Verify that all 16 channels stream with valid amplitude.
- [ ] In the OpenBCI GUI, open the **Networking Widget** and select **LSL**. Set stream name to `OpenBCI_EEG`, data type to `EEG`, and channel count to 16. Start the stream.

### Stage 2: NeuroOS LSL Stream Discovery Test
- [ ] In a terminal, run:
  ```powershell
  py -3.11 -c "from src.neuro_os.lsl_bridge import LSLEEGSource; s = LSLEEGSource(); print('Status:', s.get_info())"
  ```
- [ ] Verify that `LSLEEGSource` transitions from `SEARCHING` to `CONNECTED_VALID`.
- [ ] Verify that `measured_sampling_rate_hz` detects nominal 125.0 Hz and reports resampling to 160.0 Hz.
- [ ] Verify that `missing_channels` is empty.

### Stage 3: Signal Quality Gate Calibration Test
- [ ] Disconnect one electrode lead: verify that `NeuroOSSignalQualityGate` immediately detects a flatline or high-amplitude disconnected artifact and flags the channel name.
- [ ] Verify that the Safety Shield disengages and no commands can be triggered while a channel is bad.
- [ ] Reconnect the lead: verify that the gate returns to `EXCELLENT` within 1 sliding window.

### Stage 4: Live Interactive Workspace Test
- [ ] Start the NeuroOS backend (`py -3.11 -m uvicorn api_server:app --port 8050`).
- [ ] Start the Vite frontend (`npm run dev` in `frontend/`).
- [ ] In the dashboard, switch the EEG Source from `SIMULATED_EDF` to `LSL_HARDWARE`.
- [ ] Verify that the Hardware Status panel shows `HARDWARE_VALID_STREAM` and indicates `LIVE HARDWARE`.
- [ ] Execute an Emergency Lock override to ensure the hardware kill-switch functions before placing electrodes on a subject.
