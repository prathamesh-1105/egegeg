import time
from typing import Dict, Any, List, Optional, Callable


class NeuroOSActionExecutor:
    """
    NeuroOS Software Action Execution & Adapter Engine (Phase 2D).
    
    Executes software actions triggered by validated BCI commands against safe,
    isolated demo targets:
    1. SCROLL_DEMO (Profile A: Webpage and document vertical scrolling)
    2. DOCUMENT_NAVIGATION (Profile B: Dedicated section-by-section clinical reader)
    3. PRESENTATION (Profile C: Local simulated slide viewer)
    4. NAVIGATION, MEDIA, DOCUMENTS, SYSTEM, SMART_ROOM (Legacy adapters)

    SAFETY EXECUTION BOUNDARY:
    Strictly sandboxed execution. Does NOT send global OS keyboard shortcuts,
    does NOT control external applications, and does NOT execute system-level commands.
    All actions mutate isolated in-memory state models and stream telemetry to the UI.
    """

    ADAPTER_MODES = [
        "SCROLL_DEMO",
        "DOCUMENT_NAVIGATION",
        "PRESENTATION",
        "NAVIGATION",
        "MEDIA",
        "DOCUMENTS",
        "SYSTEM",
        "SMART_ROOM"
    ]

    DOCUMENT_SECTIONS = [
        {
            "id": "sec-1",
            "number": 1,
            "title": "1. Executive Summary & Clinical Intent",
            "read_time": "3 min",
            "summary": "Non-invasive brain-computer interfacing translating sensorimotor rhythm (SMR) neural patterns into software commands without semantic conflation.",
            "content": "NeuroOS Phase 2D introduces a configurable multi-command dispatch architecture for assistive communication. Rather than making unwarranted claims of mind-reading or natural-language intention recognition, the system treats classifier outputs as discrete 3-class statistical predictions (REST, LEFT_FIST, RIGHT_FIST) routed through deterministic software profiles."
        },
        {
            "id": "sec-2",
            "number": 2,
            "title": "2. Sensorimotor Rhythms (Mu & Beta Desynchronization)",
            "read_time": "4 min",
            "summary": "Physiological foundations of Event-Related Desynchronization (ERD) over contralateral motor cortex.",
            "content": "Motor imagery of the left hand elicits contralateral power attenuation (ERD) in the mu (8-12 Hz) and beta (16-24 Hz) frequency bands over electrode C4, accompanied by ipsilateral event-related synchronization (ERS) over C3. Right hand motor imagery elicits the symmetric inverse pattern over C3. Idle REST maintains high rhythmic alpha power across the central sulcus."
        },
        {
            "id": "sec-3",
            "number": 3,
            "title": "3. Surface Laplacian Spatial Filtering",
            "read_time": "5 min",
            "summary": "Discrete five-point finite difference approximations to mitigate volume conduction blur.",
            "content": "Non-invasive scalp potentials suffer severe spatial smearing due to skull and scalp electrical resistivity. The Step 62 champion model applies localized Surface Laplacian spatial derivations around C3 (referenced to FC3, C1, C5, CP3) and C4 (referenced to FC4, C2, C6, CP4) to sharpen localized current source density prior to FBCSP and Riemannian covariance estimation."
        },
        {
            "id": "sec-4",
            "number": 4,
            "title": "4. Riemannian Manifold Geometry & Covariance Matrices",
            "read_time": "4 min",
            "summary": "SPD covariance matrices projected onto Euclidean tangent spaces via affine-invariant Riemannian metrics.",
            "content": "Spatial covariance matrices reside on the manifold of Symmetric Positive Definite (SPD) tensors. By computing the Riemannian geometric mean and projecting matrices to the tangent space, affine-invariant Riemannian distance features eliminate non-linear distortion, providing resilient complementary features to classical FBCSP."
        },
        {
            "id": "sec-5",
            "number": 5,
            "title": "5. Safety Guardrails & Zero False-Actuation",
            "read_time": "3 min",
            "summary": "Causal temporal probability smoothing, confidence and margin thresholding, and continuous REST disengage.",
            "content": "To prevent accidental actuation during patient usage, all raw classifier probabilities pass through the NeuroOS Safety Shield. The shield enforces signal quality gates (SQI), moving-average temporal smoothing (w=3), minimum confidence (p >= 0.60), class margin (margin >= 0.12), Shannon entropy limits (H <= 1.30), and an automatic emergency disengage after 5 consecutive REST windows."
        },
        {
            "id": "sec-6",
            "number": 6,
            "title": "6. Assistive BCI Applications & Future Horizons",
            "read_time": "4 min",
            "summary": "Practical multi-profile mappings for accessibility: scrolling, document navigation, and slide presentation.",
            "content": "Phase 2D verifies that a single stable Step 62 champion model can power multiple specialized assistive interfaces through modular software profiles. Future phases will integrate the OpenBCI hardware bridge for physical streaming once bench validation and impedance protocols are completed."
        }
    ]

    PRESENTATION_SLIDES = [
        {
            "id": 1,
            "slide_number": 1,
            "title": "NeuroOS: Safe Multi-Command BCI Interface",
            "subtitle": "Phase 2D Architecture: Configurable Profiles & Guardrails",
            "category": "ARCHITECTURE OVERVIEW",
            "bullets": [
                "PhysioNet EEGMMIDB genuine EDF replay (Subjects S001-S010)",
                "Step 62 Surface Laplacian Champion: 72.67% R11 accuracy, 90.67% REST recall",
                "Explicit separation between statistical decoder outputs and executed actions",
                "No claims of direct natural-language intention decoding"
            ]
        },
        {
            "id": 2,
            "slide_number": 2,
            "title": "Profile A — Document Scrolling",
            "subtitle": "Continuous Reading Assistance for Assistive Technology",
            "category": "COMMAND PROFILE A",
            "bullets": [
                "LEFT_FIST  -> SCROLL_UP   (-180px smooth displacement)",
                "RIGHT_FIST -> SCROLL_DOWN (+180px smooth displacement)",
                "REST       -> NEUTRAL     (Strict zero-displacement idle state)",
                "Upper and lower bounds clamped to container viewport"
            ]
        },
        {
            "id": 3,
            "slide_number": 3,
            "title": "Profile B — Document Navigation",
            "subtitle": "Structured Section-by-Section Reader for Clinical Literature",
            "category": "COMMAND PROFILE B",
            "bullets": [
                "LEFT_FIST  -> PREVIOUS_SECTION (Jump to preceding clinical chapter)",
                "RIGHT_FIST -> NEXT_SECTION     (Advance to following clinical chapter)",
                "REST       -> NEUTRAL          (Maintain active section without disruption)",
                "Indexed state management bounded between Section 1 and Section 6"
            ]
        },
        {
            "id": 4,
            "slide_number": 4,
            "title": "Profile C — Presentation Demo",
            "subtitle": "Hands-Free Slide Viewer for Accessibility Demonstrations",
            "category": "COMMAND PROFILE C",
            "bullets": [
                "LEFT_FIST  -> PREVIOUS_SLIDE (Return to prior slide)",
                "RIGHT_FIST -> NEXT_SLIDE     (Advance to next slide)",
                "REST       -> NEUTRAL        (Keep active presentation frame static)",
                "Isolated client simulation; zero global OS keyboard shortcuts"
            ]
        },
        {
            "id": 5,
            "slide_number": 5,
            "title": "Comprehensive Safety Shield & Guardrails",
            "subtitle": "Multi-Layer Fail-Safe Verification Pipeline",
            "category": "SAFETY & VERIFICATION",
            "bullets": [
                "Stale Prediction Guard: duplicate prediction IDs blocked from repeat action",
                "Configurable Cooldown: suppresses rapid trigger bursts (default: 0.8s)",
                "Optional Temporal Confirmation: requires N consecutive consistent windows",
                "Continuous REST Disengage & Immediate Emergency Lock Override"
            ]
        }
    ]

    def __init__(self, mode: str = "SCROLL_DEMO"):
        self.mode = mode if mode in self.ADAPTER_MODES else "SCROLL_DEMO"
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.state_store: Dict[str, Any] = {
            "scroll_demo": {
                "scroll_offset_px": 0,
                "max_scroll_px": 3000,
                "step_px": 180,
                "last_direction": "NONE"
            },
            "document_navigation": {
                "active_section_idx": 0,
                "total_sections": len(self.DOCUMENT_SECTIONS),
                "sections": self.DOCUMENT_SECTIONS,
                "last_direction": "NONE"
            },
            "presentation": {
                "active_slide": 1,
                "total_slides": len(self.PRESENTATION_SLIDES),
                "slides": self.PRESENTATION_SLIDES,
                "last_direction": "NONE"
            },
            "navigation": {"active_tab": "DASHBOARD", "scroll_position": 0, "scroll_offset_px": 0},
            "documents": {"active_slide": 1, "total_slides": 12, "zoom_pct": 100},
            "media": {"playing": False, "track": 1, "volume": 80},
            "smart_room": {"light_on": False, "fan_on": False, "temperature_c": 22},
            "system": {"muted": False, "locked": False}
        }

    def set_adapter_mode(self, mode: str):
        """Sets active adapter mode."""
        clean_mode = str(mode).upper()
        if clean_mode in self.ADAPTER_MODES:
            self.mode = clean_mode
        elif clean_mode in ["SCROLL", "SCROLLING"]:
            self.mode = "SCROLL_DEMO"
        elif clean_mode in ["DOC_NAV", "DOCUMENT", "SECTION_READER"]:
            self.mode = "DOCUMENT_NAVIGATION"
        elif clean_mode in ["SLIDES", "SLIDE_VIEWER"]:
            self.mode = "PRESENTATION"

    def register_callback(self, callback_func: Callable[[Dict[str, Any]], None]):
        """Registers a callback function triggered upon action execution."""
        self.callbacks.append(callback_func)

    def execute_command(self, command: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a validated BCI command against the active adapter mode.
        Returns detailed action execution result dictionary.
        REST and NEUTRAL are guaranteed to execute zero state mutations.
        """
        if command == "NEUTRAL" or not command:
            return {
                "executed": False,
                "command": "NEUTRAL",
                "action": "NONE",
                "action_name": "NONE",
                "details": "No action required (Neutral / Idle)",
                "adapter_mode": self.mode,
                "state_change": {"parameter": "none", "before": None, "after": None, "delta": 0, "unit": "none"}
            }

        now = time.time()
        meta = metadata or {}
        action_name = "NONE"
        details = ""

        # Capture pre-execution state for delta tracking
        pre_param = None
        pre_val = None
        unit = "none"
        if self.mode == "SCROLL_DEMO":
            pre_param = "scroll_offset_px"
            pre_val = self.state_store["scroll_demo"]["scroll_offset_px"]
            unit = "px"
        elif self.mode == "DOCUMENT_NAVIGATION":
            pre_param = "active_section_idx"
            pre_val = self.state_store["document_navigation"]["active_section_idx"]
            unit = "index"
        elif self.mode == "PRESENTATION":
            pre_param = "active_slide"
            pre_val = self.state_store["presentation"]["active_slide"]
            unit = "slide"

        # ----------------------------------------------------------------------
        # 1. PROFILE A — SCROLL DEMO ADAPTER
        # ----------------------------------------------------------------------
        if self.mode == "SCROLL_DEMO":
            demo_st = self.state_store["scroll_demo"]
            if command in ["PREVIOUS", "SCROLL_UP", "WEB_SCROLL_UP"]:
                action_name = "SCROLL_UP"
                old_off = demo_st["scroll_offset_px"]
                demo_st["scroll_offset_px"] = max(0, old_off - demo_st["step_px"])
                demo_st["last_direction"] = "UP"
                details = f"Scrolled webpage UP: {old_off}px -> {demo_st['scroll_offset_px']}px"
            elif command in ["NEXT", "SCROLL_DOWN", "WEB_SCROLL_DOWN"]:
                action_name = "SCROLL_DOWN"
                old_off = demo_st["scroll_offset_px"]
                demo_st["scroll_offset_px"] = min(demo_st["max_scroll_px"], old_off + demo_st["step_px"])
                demo_st["last_direction"] = "DOWN"
                details = f"Scrolled webpage DOWN: {old_off}px -> {demo_st['scroll_offset_px']}px"
            elif command in ["SELECT", "RESET", "SCROLL_RESET"]:
                action_name = "SCROLL_RESET"
                demo_st["scroll_offset_px"] = 0
                demo_st["last_direction"] = "TOP"
                details = "Reset scroll position to top (0px)"
            elif command in ["BACK", "SCROLL_TOP"]:
                action_name = "SCROLL_TOP"
                demo_st["scroll_offset_px"] = 0
                demo_st["last_direction"] = "TOP"
                details = "Jumped to top of page (0px)"

        # ----------------------------------------------------------------------
        # 2. PROFILE B — DOCUMENT NAVIGATION ADAPTER (Dedicated Section Reader)
        # ----------------------------------------------------------------------
        elif self.mode == "DOCUMENT_NAVIGATION":
            doc_st = self.state_store["document_navigation"]
            total_sec = doc_st["total_sections"]

            if command in ["PREVIOUS_SECTION", "PREVIOUS", "SCROLL_UP"]:
                action_name = "PREVIOUS_SECTION"
                old_idx = doc_st["active_section_idx"]
                doc_st["active_section_idx"] = max(0, old_idx - 1)
                doc_st["last_direction"] = "PREVIOUS"
                sec_title = doc_st["sections"][doc_st["active_section_idx"]]["title"]
                details = f"Navigated to previous section: '{sec_title}' (Index: {doc_st['active_section_idx'] + 1}/{total_sec})"

            elif command in ["NEXT_SECTION", "NEXT", "SCROLL_DOWN"]:
                action_name = "NEXT_SECTION"
                old_idx = doc_st["active_section_idx"]
                doc_st["active_section_idx"] = min(total_sec - 1, old_idx + 1)
                doc_st["last_direction"] = "NEXT"
                sec_title = doc_st["sections"][doc_st["active_section_idx"]]["title"]
                details = f"Navigated to next section: '{sec_title}' (Index: {doc_st['active_section_idx'] + 1}/{total_sec})"

            elif command in ["RESET_SECTION", "SELECT", "RESET"]:
                action_name = "RESET_SECTION"
                doc_st["active_section_idx"] = 0
                doc_st["last_direction"] = "NONE"
                details = "Reset document reader to Section 1"

        # ----------------------------------------------------------------------
        # 3. PROFILE C — PRESENTATION DEMO ADAPTER (Simulated Slide Viewer)
        # ----------------------------------------------------------------------
        elif self.mode == "PRESENTATION":
            pres_st = self.state_store["presentation"]
            total_slides = pres_st["total_slides"]

            if command in ["PREVIOUS_SLIDE", "PREVIOUS", "SCROLL_UP"]:
                action_name = "PREVIOUS_SLIDE"
                old_slide = pres_st["active_slide"]
                pres_st["active_slide"] = max(1, old_slide - 1)
                pres_st["last_direction"] = "PREVIOUS"
                slide_title = pres_st["slides"][pres_st["active_slide"] - 1]["title"]
                details = f"Moved to previous slide #{pres_st['active_slide']}: '{slide_title}'"

            elif command in ["NEXT_SLIDE", "NEXT", "SCROLL_DOWN"]:
                action_name = "NEXT_SLIDE"
                old_slide = pres_st["active_slide"]
                pres_st["active_slide"] = min(total_slides, old_slide + 1)
                pres_st["last_direction"] = "NEXT"
                slide_title = pres_st["slides"][pres_st["active_slide"] - 1]["title"]
                details = f"Moved to next slide #{pres_st['active_slide']}: '{slide_title}'"

            elif command in ["RESET_SLIDE", "SELECT", "RESET"]:
                action_name = "RESET_SLIDE"
                pres_st["active_slide"] = 1
                pres_st["last_direction"] = "NONE"
                details = "Reset presentation to Slide #1"

        # ----------------------------------------------------------------------
        # 4. LEGACY NAVIGATION ADAPTER
        # ----------------------------------------------------------------------
        elif self.mode == "NAVIGATION":
            if command in ["PREVIOUS", "SCROLL_UP"]:
                action_name = "NAVIGATE_PREVIOUS"
                self.state_store["navigation"]["scroll_position"] = max(0, self.state_store["navigation"]["scroll_position"] - 1)
                self.state_store["navigation"]["scroll_offset_px"] = max(0, self.state_store["navigation"].get("scroll_offset_px", 0) - 150)
                details = f"Scrolled up to item #{self.state_store['navigation']['scroll_position']}"
            elif command in ["NEXT", "SCROLL_DOWN"]:
                action_name = "NAVIGATE_NEXT"
                self.state_store["navigation"]["scroll_position"] += 1
                self.state_store["navigation"]["scroll_offset_px"] = self.state_store["navigation"].get("scroll_offset_px", 0) + 150
                details = f"Scrolled down to item #{self.state_store['navigation']['scroll_position']}"
            elif command == "SELECT":
                action_name = "NAVIGATE_SELECT"
                details = f"Opened & selected item #{self.state_store['navigation']['scroll_position']}"
            elif command == "BACK":
                action_name = "NAVIGATE_BACK"
                details = "Returned to previous menu / Escaped"

        # ----------------------------------------------------------------------
        # 5. LEGACY MEDIA ADAPTER
        # ----------------------------------------------------------------------
        elif self.mode == "MEDIA":
            if command == "PREVIOUS":
                action_name = "MEDIA_PREVIOUS_TRACK"
                self.state_store["media"]["track"] = max(1, self.state_store["media"]["track"] - 1)
                details = f"Skipped to previous track #{self.state_store['media']['track']}"
            elif command == "NEXT":
                action_name = "MEDIA_NEXT_TRACK"
                self.state_store["media"]["track"] += 1
                details = f"Skipped to next track #{self.state_store['media']['track']}"
            elif command == "SELECT":
                action_name = "MEDIA_TOGGLE_PLAY"
                self.state_store["media"]["playing"] = not self.state_store["media"]["playing"]
                state_str = "PLAYING" if self.state_store["media"]["playing"] else "PAUSED"
                details = f"Toggled media playback -> {state_str}"
            elif command == "BACK":
                action_name = "MEDIA_MUTE_TOGGLE"
                self.state_store["system"]["muted"] = not self.state_store["system"]["muted"]
                details = "Toggled media mute"

        # ----------------------------------------------------------------------
        # 6. LEGACY DOCUMENTS ADAPTER
        # ----------------------------------------------------------------------
        elif self.mode == "DOCUMENTS":
            if command == "PREVIOUS":
                action_name = "DOCS_PREVIOUS_SLIDE"
                self.state_store["documents"]["active_slide"] = max(1, self.state_store["documents"]["active_slide"] - 1)
                details = f"Moved to previous slide #{self.state_store['documents']['active_slide']}"
            elif command == "NEXT":
                action_name = "DOCS_NEXT_SLIDE"
                self.state_store["documents"]["active_slide"] = min(self.state_store["documents"]["total_slides"], self.state_store["documents"]["active_slide"] + 1)
                details = f"Moved to next slide #{self.state_store['documents']['active_slide']}"
            elif command == "SELECT":
                action_name = "DOCS_PRESENTATION_MODE"
                details = f"Entered fullscreen presentation on Slide #{self.state_store['documents']['active_slide']}"
            elif command == "BACK":
                action_name = "DOCS_EXIT_PRESENTATION"
                details = "Exited slide presentation mode"

        # ----------------------------------------------------------------------
        # 7. LEGACY SYSTEM ADAPTER
        # ----------------------------------------------------------------------
        elif self.mode == "SYSTEM":
            if command == "PREVIOUS":
                action_name = "SYSTEM_VOLUME_DOWN"
                self.state_store["media"]["volume"] = max(0, self.state_store["media"]["volume"] - 10)
                details = f"Decreased volume to {self.state_store['media']['volume']}%"
            elif command == "NEXT":
                action_name = "SYSTEM_VOLUME_UP"
                self.state_store["media"]["volume"] = min(100, self.state_store["media"]["volume"] + 10)
                details = f"Increased volume to {self.state_store['media']['volume']}%"
            elif command == "SELECT":
                action_name = "SYSTEM_LOCK_SCREEN"
                self.state_store["system"]["locked"] = True
                details = "Locked NeuroOS screen for safety"
            elif command == "BACK":
                action_name = "SYSTEM_GO_HOME"
                self.state_store["system"]["locked"] = False
                details = "Returned to NeuroOS main home screen"

        # ----------------------------------------------------------------------
        # 8. LEGACY SMART ROOM ADAPTER
        # ----------------------------------------------------------------------
        elif self.mode == "SMART_ROOM":
            if command == "PREVIOUS":
                action_name = "SMART_ROOM_TEMP_DOWN"
                self.state_store["smart_room"]["temperature_c"] -= 1
                details = f"Thermostat set to {self.state_store['smart_room']['temperature_c']}°C"
            elif command == "NEXT":
                action_name = "SMART_ROOM_TEMP_UP"
                self.state_store["smart_room"]["temperature_c"] += 1
                details = f"Thermostat set to {self.state_store['smart_room']['temperature_c']}°C"
            elif command == "SELECT":
                action_name = "SMART_ROOM_TOGGLE_LIGHT"
                self.state_store["smart_room"]["light_on"] = not self.state_store["smart_room"]["light_on"]
                state_str = "ON" if self.state_store["smart_room"]["light_on"] else "OFF"
                details = f"Smart Light turned {state_str}"
            elif command == "BACK":
                action_name = "SMART_ROOM_TOGGLE_FAN"
                self.state_store["smart_room"]["fan_on"] = not self.state_store["smart_room"]["fan_on"]
                state_str = "ON" if self.state_store["smart_room"]["fan_on"] else "OFF"
                details = f"Smart Fan turned {state_str}"

        # Compute post-execution state change delta
        post_val = None
        delta = 0
        if pre_param:
            if self.mode == "SCROLL_DEMO":
                post_val = self.state_store["scroll_demo"]["scroll_offset_px"]
            elif self.mode == "DOCUMENT_NAVIGATION":
                post_val = self.state_store["document_navigation"]["active_section_idx"]
            elif self.mode == "PRESENTATION":
                post_val = self.state_store["presentation"]["active_slide"]
            delta = (post_val - pre_val) if (post_val is not None and pre_val is not None) else 0

        state_change = {
            "parameter": pre_param or "none",
            "before": pre_val,
            "after": post_val,
            "delta": delta,
            "unit": unit
        }

        res = {
            "executed": action_name != "NONE",
            "timestamp": now,
            "iso_time": time.strftime("%H:%M:%S"),
            "adapter_mode": self.mode,
            "command": command,
            "action": action_name,
            "action_name": action_name,
            "details": details or f"Command executed in {self.mode} adapter",
            "state_change": state_change,
            "system_state": self.state_store
        }

        self.execution_history.append(res)
        if len(self.execution_history) > 50:
            self.execution_history.pop(0)

        # Notify UI callbacks
        for cb in self.callbacks:
            try:
                cb(res)
            except Exception as e:
                print(f"Error executing action callback: {e}")

        return res
