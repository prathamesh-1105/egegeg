"""
NeuroOS — Brain-Controlled Computer Engine
Package initialization for NeuroOS core modules:
- safety_shield: Temporal probability smoothing, confidence gating & emergency lock
- command_router: Intent translation, dwell time confirmation & cooldown
- action_executor: Modular software action adapter system
- event_logger: Explainable JSON telemetry auditor
- lsl_bridge: EEGSource abstraction (PhysioNet EDF dataset & OpenBCI LSL stream)
- session_manager: Session lifecycle state machine and real-time telemetry pipeline
- channel_adapter: OpenBCI 8-channel to 15-channel sensorimotor spatial reconstruction adapter
- hardware_profiles: Hardware profile registry (SIMULATED, OPENBCI_8CH, OPENBCI_16CH)
- signal_quality: Hardware signal quality gate & artifact disengage warnings
- calibration_manager: Real hardware calibration data collection protocol
"""

from src.neuro_os.safety_shield import NeuroOSSafetyShield
from src.neuro_os.command_router import NeuroOSCommandRouter
from src.neuro_os.action_executor import NeuroOSActionExecutor
from src.neuro_os.event_logger import NeuroOSEventLogger
from src.neuro_os.lsl_bridge import EEGSource, DatasetEEGSource, LSLEEGSource, StreamState
from src.neuro_os.session_manager import SessionState, NeuroOSSessionManager
from src.neuro_os.channel_adapter import OpenBCIChannelAdapter, AdapterValidationError
from src.neuro_os.hardware_profiles import (
    HardwareProfileRegistry,
    HardwareProfile,
    STEP62_TARGET_CHANNELS,
    validate_channel_montage,
    normalize_channel_name
)
from src.neuro_os.signal_quality import NeuroOSSignalQualityGate
from src.neuro_os.calibration_manager import NeuroOSCalibrationManager, CalibrationState
from src.neuro_os.command_profiles import (
    CommandProfile,
    COMMAND_PROFILES,
    get_command_profile,
    list_available_profiles
)

from src.neuro_os.session_auditor import NeuroOSSessionAuditor, SessionAuditRecord
from src.neuro_os.deterministic_replayer import DeterministicReplayer

__all__ = [
    "NeuroOSSafetyShield",
    "NeuroOSCommandRouter",
    "NeuroOSActionExecutor",
    "NeuroOSEventLogger",
    "EEGSource",
    "DatasetEEGSource",
    "LSLEEGSource",
    "SessionState",
    "NeuroOSSessionManager",
    "OpenBCIChannelAdapter",
    "AdapterValidationError",
    "HardwareProfileRegistry",
    "HardwareProfile",
    "NeuroOSSignalQualityGate",
    "NeuroOSCalibrationManager",
    "CommandProfile",
    "COMMAND_PROFILES",
    "get_command_profile",
    "list_available_profiles",
    "NeuroOSSessionAuditor",
    "SessionAuditRecord",
    "DeterministicReplayer"
]
