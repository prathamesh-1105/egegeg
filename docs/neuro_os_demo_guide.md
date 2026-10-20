# NeuroOS Local Demonstration & Verification Guide

**Document Version:** 3.0.0 (Master Phase Production Release)  
**Date:** October 20, 2026  
**Audience:** Evaluators, BCI Engineers, Clinical Demonstrators, QA Reviewers  

---

## 1. Quick-Start Launch Instructions

NeuroOS is composed of a FastAPI / WebSocket backend engine and a modern React / Vite user interface.

### Prerequisites
- Python 3.11+ installed (`py -3.11`).
- Node.js 18+ and npm installed.
- Modern web browser (Chrome, Edge, Firefox).

### Step 1: Start the Backend API & Pipeline Server
Open a terminal in the repository root (`d:\PROJECTS\EEG MAj`):
```powershell
py -3.11 -m uvicorn api_server:app --host 127.0.0.1 --port 8050 --reload
```
Verify the server starts:
- REST API: `http://localhost:8050/api/neuro-os/status`
- Interactive API Docs: `http://localhost:8050/docs`
- WebSocket Telemetry: `ws://localhost:8050/ws/neuro-os`

### Step 2: Start the React Frontend Dashboard
Open a second terminal in the `frontend/` directory:
```powershell
cd frontend
npm run dev
```
Open your browser and navigate to:
```
http://localhost:5173
```

---

## 2. Navigating the Multi-Command BCI Center

The dashboard loads directly into the **Configurable Multi-Command BCI Interface**.

### Key Interface Sections:
1. **Ethical Disclaimer & System Badges (Top Banner):**
   - Displays clear notices that actions are generated from recorded PhysioNet EEG replay, not live mind-reading.
   - Distinct badges indicate:
     - `OFFLINE REPLAY (PhysioNet S001 R11)` (Mode A) or `SIMULATION (CONTINUOUS STREAM)` (Mode B).
     - `HARDWARE DISCONNECTED (Standby)` (or `LIVE HARDWARE` if LSL stream is connected).
     - `STEP 62 FROZEN` champion model indicator.
2. **Command Profile Selector Cards:**
   - **Profile A (Scrolling):** Maps `LEFT_FIST` $\to$ Scroll Up, `RIGHT_FIST` $\to$ Scroll Down, `REST` $\to$ Neutral.
   - **Profile B (Document Navigation):** Maps `LEFT_FIST` $\to$ Previous Section, `RIGHT_FIST` $\to$ Next Section, `REST` $\to$ Neutral.
   - **Profile C (Presentation Demo):** Maps `LEFT_FIST` $\to$ Previous Slide, `RIGHT_FIST` $\to$ Next Slide, `REST` $\to$ Neutral.
3. **Session Control Toolbar:**
   - `Start Session`: Begins continuous playback.
   - `Pause`: Halts streaming and disengages confirmation.
   - `Single Step`: Steps forward one trial/window synchronously.
   - `Reset Target`: Resets scroll position, document index, slide counter, and trial pointer.
   - `Mode A: Replay (1-Win)` vs `Mode B: Stream (Dwell)`: Instant execution mode toggle.
   - `Cooldown` & `Confirmation` dropdowns.
   - **GIANT EMERGENCY LOCK BUTTON:** Instantly blocks all action execution.
4. **Interactive Sandbox Demo Workspaces:**
   - Dedicated interactive viewport corresponding to the active profile.
5. **Real-Time BCI Telemetry & Signal Quality Bar:**
   - Shows live decoded class, confidence bar, class probabilities (`REST`, `LEFT_FIST`, `RIGHT_FIST`), safety status, and signal quality metrics.
6. **Command Reliability & Session Analytics:**
   - Displays live processed trials, classification accuracy, safety rejection rate, action accuracy, false action rate, and latency percentiles.
7. **Recent Action Audit Trail Table:**
   - Detailed chronological ledger showing exact timestamps, trial numbers, decoupled ground truth, predicted class, confidence, mapped command, action outcome, state parameter deltas (e.g. `scroll_offset_px: +180 px`), and latency.
   - Action buttons: **Export JSON**, **Export CSV**, and **Run Deterministic Replay Test**.

---

## 3. Step-by-Step Demonstration Walkthroughs

### Walkthrough 1: Profile A — Document Scrolling
1. Click the **Profile A — Scrolling** card.
2. Ensure **Mode A: Replay (1-Win)** is selected.
3. Click **Single Step** multiple times.
4. Observe the interactive article in the center pane:
   - When `RIGHT_FIST` is decoded and verified, the container smoothly scrolls downward by $+180\text{ px}$.
   - When `LEFT_FIST` is decoded and verified, the container scrolls upward by $-180\text{ px}$.
   - When `REST` is predicted, the container maintains its position with strict zero displacement.
5. Verify the **State Delta** column in the audit table displays `scroll_offset_px: +180 px`.

---

### Walkthrough 2: Profile B — Section-by-Section Clinical Reader
1. Click the **Profile B — Document Navigation** card.
2. The demo target switches to the **Clinical Literature Reader**.
3. Click **Single Step** or **Start Session**:
   - `RIGHT_FIST` advances to the next clinical chapter (e.g., from Section 1 to Section 2).
   - `LEFT_FIST` returns to the preceding chapter.
   - State mutations clamp cleanly at Section 0 and Section 5 without index out-of-bounds exceptions.
4. Verify the state delta reports `active_section_idx: +1 index`.

---

### Walkthrough 3: Profile C — Hands-Free Presentation Slide Viewer
1. Click the **Profile C — Presentation Demo** card.
2. The demo target switches to the **Technical Slide Deck Viewer**.
3. Click **Single Step**:
   - `RIGHT_FIST` advances to the next slide frame (`NEXT_SLIDE`).
   - `LEFT_FIST` returns to the prior slide frame (`PREVIOUS_SLIDE`).
   - Bounds are clamped between Slide 1 and Slide 5.

---

### Walkthrough 4: Mode B — Continuous Sliding Stream & Command Release
1. Switch the mode toggle from `Mode A: Replay` to **Mode B: Stream (Dwell)**.
2. Observe that Confirmation automatically sets to `2 Windows (Confirmed)`.
3. Click **Single Step**:
   - Window 1 on a motor cue enters `TEMPORAL_CONFIRMATION_PENDING (1/2 windows)` and fires no action.
   - Window 2 confirms the intent and triggers the action.
   - Window 3 (overlapping window on the same sustained cue) is suppressed by **Repeat Suppression** until the subject returns to `REST`. This prevents runaway slide skips on a single motor imagery thought.

---

### Walkthrough 5: Safety Guardrails & Emergency Lock
1. While streaming or stepping, click the red **EMERGENCY LOCK** button.
2. The button glows with an alert pulse: `EMERGENCY LOCK: ACTIVE`.
3. Click **Single Step** repeatedly:
   - Observe that every step is strictly rejected: `SAFETY_SHIELD_MANUAL_LOCK_ACTIVE`.
   - The action executor is disengaged; zero state mutations occur.
4. Click the lock button again to disengage the emergency lockout. Normal operations resume safely.

---

### Walkthrough 6: Audit Export & Deterministic Replay Verification
1. Scroll down to the **Recent Action Audit Trail** table.
2. Click **Export JSON**:
   - Your browser downloads `neuro_os_audit_<session_id>.json`.
   - Open the JSON file to inspect the complete 22-field trial audit records and summary metrics.
3. Click **Export CSV**:
   - Your browser downloads `neuro_os_audit_<session_id>.csv`.
   - Open the CSV file in Excel or text editor to verify the clean flattened schema.
4. Click **Run Deterministic Replay Test**:
   - The system executes a dual-run replay benchmark in synchronous mode.
   - A modal dialog appears displaying bit-exact match rates:
     - Prediction Match: **100.0%**
     - Safety Decision Match: **100.0%**
     - Command Routing Match: **100.0%**
     - State Mutation Match: **100.0%**

---

## 4. Verification Commands Cheat-Sheet

To verify all backend components from the terminal:

```powershell
# Run the entire automated test suite (90 tests)
py -3.11 -m pytest tests/

# Run the Master Phase Integration test suite (12 tests)
py -3.11 -m pytest tests/test_master_phase_integration.py

# Verify the frontend production build
cd frontend
npm run build
```
