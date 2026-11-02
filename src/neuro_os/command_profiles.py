"""
NeuroOS Configurable Command Profiles (Phase 2D)
Defines reusable software action mappings for Step 62 motor-imagery decoder predictions.

CRITICAL SCIENTIFIC & ETHICAL BOUNDARY:
These profiles are software mappings of 3-class motor-imagery predictions (REST, LEFT_FIST, RIGHT_FIST).
They map statistical electrophysiological patterns to software commands.
They do NOT imply natural-language semantic understanding or direct human intention reading.
REST is strictly mapped to NEUTRAL across all profiles.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class CommandProfile:
    """
    BCI Command Profile Configuration.
    Maps 3-class motor imagery predictions to application-specific software actions.
    """
    id: str
    name: str
    tag: str
    description: str
    target_environment: str
    adapter_mode: str
    class_to_command: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def map_intent(self, intent: str) -> str:
        """
        Maps a safety-accepted motor-imagery intent to a software command.
        REST strictly maps to NEUTRAL.
        Any unknown or unmapped intent maps to NEUTRAL.
        """
        if intent == "REST" or not intent:
            return "NEUTRAL"
        return self.class_to_command.get(intent, "NEUTRAL")

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary representation."""
        return asdict(self)


# Profile A — Scrolling (Default)
PROFILE_SCROLLING = CommandProfile(
    id="scrolling",
    name="Profile A — Scrolling",
    tag="PROFILE_A",
    description="Vertical document and webpage scrolling (default NeuroOS profile)",
    target_environment="SCROLLABLE_DOCUMENT",
    adapter_mode="SCROLL_DEMO",
    class_to_command={
        "LEFT_FIST": "SCROLL_UP",
        "RIGHT_FIST": "SCROLL_DOWN",
        "REST": "NEUTRAL"
    },
    metadata={
        "scroll_step_px": 180,
        "max_scroll_px": 3000,
        "intent_semantics": {
            "LEFT_FIST": "Scroll upwards toward previous paragraphs",
            "RIGHT_FIST": "Scroll downwards toward following paragraphs",
            "REST": "Maintain current scroll offset (idle)"
        }
    }
)

# Profile B — Document navigation
PROFILE_DOCUMENT_NAVIGATION = CommandProfile(
    id="document_navigation",
    name="Profile B — Document navigation",
    tag="PROFILE_B",
    description="Section-by-section reader navigation in structured clinical documents",
    target_environment="DOCUMENT_READER",
    adapter_mode="DOCUMENT_NAVIGATION",
    class_to_command={
        "LEFT_FIST": "PREVIOUS_SECTION",
        "RIGHT_FIST": "NEXT_SECTION",
        "REST": "NEUTRAL"
    },
    metadata={
        "intent_semantics": {
            "LEFT_FIST": "Navigate to previous chapter or section",
            "RIGHT_FIST": "Navigate to next chapter or section",
            "REST": "Stay on active section (idle reading)"
        }
    }
)

# Profile C — Presentation demo
PROFILE_PRESENTATION = CommandProfile(
    id="presentation",
    name="Profile C — Presentation demo",
    tag="PROFILE_C",
    description="Simulated slide deck switcher for hands-free presentations",
    target_environment="PRESENTATION_VIEWER",
    adapter_mode="PRESENTATION",
    class_to_command={
        "LEFT_FIST": "PREVIOUS_SLIDE",
        "RIGHT_FIST": "NEXT_SLIDE",
        "REST": "NEUTRAL"
    },
    metadata={
        "total_slides": 5,
        "intent_semantics": {
            "LEFT_FIST": "Move to previous presentation slide",
            "RIGHT_FIST": "Move to next presentation slide",
            "REST": "Keep active slide visible (idle)"
        }
    }
)

# Profile D — Web Browser Control (Phase 9 Neuro Browser)
PROFILE_WEB_CONTROL = CommandProfile(
    id="web_control",
    name="Profile D — Web Browser Control",
    tag="PROFILE_D",
    description="EEG-driven physical web browser scrolling and article navigation",
    target_environment="WEB_BROWSER",
    adapter_mode="SCROLL_DEMO",
    class_to_command={
        "LEFT_FIST": "SCROLL_UP",
        "RIGHT_FIST": "SCROLL_DOWN",
        "REST": "NEUTRAL"
    },
    metadata={
        "scroll_step_px": 180,
        "max_scroll_px": 5000,
        "intent_semantics": {
            "LEFT_FIST": "Scroll webpage upward (-180px)",
            "RIGHT_FIST": "Scroll webpage downward (+180px)",
            "REST": "Maintain current page viewport position (idle)"
        }
    }
)

# Available Profiles Registry
COMMAND_PROFILES: Dict[str, CommandProfile] = {
    "scrolling": PROFILE_SCROLLING,
    "document_navigation": PROFILE_DOCUMENT_NAVIGATION,
    "presentation": PROFILE_PRESENTATION,
    "web_control": PROFILE_WEB_CONTROL
}

# Alias mapping for backwards compatibility
PROFILE_ALIASES: Dict[str, str] = {
    "scroll": "scrolling",
    "scroll_demo": "scrolling",
    "scrolling": "scrolling",
    "default": "scrolling",
    "profile_a": "scrolling",
    "doc_nav": "document_navigation",
    "document": "document_navigation",
    "document_navigation": "document_navigation",
    "profile_b": "document_navigation",
    "slides": "presentation",
    "slide_viewer": "presentation",
    "presentation": "presentation",
    "presentation_demo": "presentation",
    "profile_c": "presentation",
    "web_control": "web_control",
    "web_browser": "web_control",
    "neuro_browser": "web_control",
    "browser": "web_control",
    "web-control": "web_control",
    "profile_d": "web_control"
}

DEFAULT_PROFILE_ID = "scrolling"


def get_command_profile(profile_id: Optional[str] = None) -> CommandProfile:
    """
    Retrieves a CommandProfile by ID or alias.
    Defaults to PROFILE_SCROLLING if None or not found.
    """
    if not profile_id:
        return COMMAND_PROFILES[DEFAULT_PROFILE_ID]
    
    canonical_id = PROFILE_ALIASES.get(str(profile_id).lower(), DEFAULT_PROFILE_ID)
    return COMMAND_PROFILES.get(canonical_id, COMMAND_PROFILES[DEFAULT_PROFILE_ID])


def list_available_profiles() -> List[Dict[str, Any]]:
    """Returns list of all available command profiles with metadata."""
    return [p.to_dict() for p in COMMAND_PROFILES.values()]

