import mne
import pandas as pd

# Standard event mapping for PhysioNet Motor Movement / Imagery Dataset
EVENT_MAPPING = {
    "T0": "Rest",
    "T1": "Left Fist",
    "T2": "Right Fist"
}

EVENT_COLORS = {
    "T0": "#4A5568",  # Slate grey for Rest
    "T1": "#3182CE",  # Vibrant Blue for Left Fist
    "T2": "#DD6B20"   # Vibrant Orange for Right Fist
}


def get_eeg_events(raw: mne.io.Raw) -> pd.DataFrame:
    """
    Extracts annotations from the MNE Raw object and builds a structured DataFrame.

    Parameters:
        raw (mne.io.Raw): MNE Raw instance.

    Returns:
        pd.DataFrame: Contains columns ['onset_sec', 'duration_sec', 'end_sec', 'event_code', 'description', 'color']
    """
    annotations = raw.annotations
    if len(annotations) == 0:
        return pd.DataFrame(columns=['onset_sec', 'duration_sec', 'end_sec', 'event_code', 'description', 'color'])

    events_list = []
    for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
        code = str(desc).strip()
        label = EVENT_MAPPING.get(code, f"Unknown ({code})")
        color = EVENT_COLORS.get(code, "#A0AEC0")

        events_list.append({
            "onset_sec": float(onset),
            "duration_sec": float(duration),
            "end_sec": float(onset + duration),
            "event_code": code,
            "description": label,
            "color": color
        })

    df_events = pd.DataFrame(events_list)
    return df_events


def get_event_summary(df_events: pd.DataFrame) -> dict:
    """
    Computes summary counts and total durations per event type.

    Parameters:
        df_events (pd.DataFrame): DataFrame produced by get_eeg_events().

    Returns:
        dict: Event counts and mapped descriptions.
    """
    if df_events.empty:
        return {}

    summary = {}
    for code, label in EVENT_MAPPING.items():
        subset = df_events[df_events['event_code'] == code]
        count = len(subset)
        total_time = subset['duration_sec'].sum() if count > 0 else 0.0
        summary[code] = {
            "label": label,
            "count": count,
            "total_duration_sec": round(total_time, 2)
        }
    return summary
