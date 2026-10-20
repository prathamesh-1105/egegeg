import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def create_single_channel_plot(
    raw: mne.io.Raw,
    channel_name: str,
    start_time: float,
    end_time: float,
    df_events: pd.DataFrame
) -> go.Figure:
    """
    Creates an interactive Plotly waveform visualization for a single selected EEG channel.

    Parameters:
        raw (mne.io.Raw): MNE raw data object.
        channel_name (str): Selected channel label (e.g., 'C3..').
        start_time (float): Window start time in seconds.
        end_time (float): Window end time in seconds.
        df_events (pd.DataFrame): DataFrame containing event annotations.

    Returns:
        go.Figure: Dark-themed interactive Plotly figure.
    """
    sfreq = raw.info['sfreq']
    start_idx = int(max(0, start_time * sfreq))
    end_idx = int(min(len(raw.times), end_time * sfreq))

    # Slice time array and EEG signal
    times = raw.times[start_idx:end_idx]
    # Extract channel data and scale from Volts to microvolts (uV)
    ch_idx = raw.ch_names.index(channel_name)
    signal_uv = raw.get_data(picks=[ch_idx], start=start_idx, stop=end_idx)[0] * 1e6

    fig = go.Figure()

    # Add Event Background Shading Spans
    if not df_events.empty:
        # Filter events overlapping current window
        overlapping_events = df_events[
            (df_events['end_sec'] >= start_time) & (df_events['onset_sec'] <= end_time)
        ]

        added_legend_codes = set()

        for _, ev in overlapping_events.iterrows():
            ev_start = max(start_time, ev['onset_sec'])
            ev_end = min(end_time, ev['end_sec'])
            code = ev['event_code']
            desc = ev['description']
            color = ev['color']

            show_in_legend = code not in added_legend_codes
            added_legend_codes.add(code)

            # Add vertical shaded region for event duration
            fig.add_vrect(
                x0=ev_start,
                x1=ev_end,
                fillcolor=color,
                opacity=0.22,
                layer="below",
                line_width=1,
                line_color=color,
                annotation_text=f"{code}: {desc}" if show_in_legend else None,
                annotation_position="top left",
                annotation_font=dict(size=10, color="#E2E8F0")
            )

    # Plot EEG Signal Trace
    fig.add_trace(go.Scatter(
        x=times,
        y=signal_uv,
        mode='lines',
        name=f"Channel {channel_name}",
        line=dict(color='#00F2FE', width=1.5),
        hovertemplate="<b>Time:</b> %{x:.3f} s<br><b>Amplitude:</b> %{y:.2f} μV<extra></extra>"
    ))

    # Update Dark Futuristic Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        title=dict(
            text=f"Real EEG Signal Waveform — Channel: {channel_name}",
            font=dict(size=18, color="#F8FAFC", family="sans-serif")
        ),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="#334155",
            zerolinecolor="#475569",
            showgrid=True,
            range=[start_time, end_time]
        ),
        yaxis=dict(
            title="EEG Amplitude (μV)",
            gridcolor="#334155",
            zerolinecolor="#475569",
            showgrid=True
        ),
        margin=dict(l=60, r=40, t=60, b=50),
        height=450,
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="#334155",
            font=dict(color="#E2E8F0")
        )
    )

    return fig


def create_event_timeline_plot(df_events: pd.DataFrame, total_duration: float) -> go.Figure:
    """
    Creates an interactive event timeline displaying T0, T1, and T2 task segments.

    Parameters:
        df_events (pd.DataFrame): DataFrame containing event annotations.
        total_duration (float): Total recording duration in seconds.

    Returns:
        go.Figure: Gantt-style timeline plot.
    """
    fig = go.Figure()

    if df_events.empty:
        return fig

    seen_codes = set()

    for _, ev in df_events.iterrows():
        code = ev['event_code']
        desc = ev['description']
        color = ev['color']
        show_legend = code not in seen_codes
        seen_codes.add(code)

        fig.add_trace(go.Bar(
            x=[ev['duration_sec']],
            y=[desc],
            base=[ev['onset_sec']],
            orientation='h',
            marker=dict(color=color, cornerradius=4),
            name=f"{code} ({desc})",
            showlegend=show_legend,
            hovertemplate=(
                f"<b>Task:</b> {desc} ({code})<br>"
                f"<b>Start:</b> %{{base:.2f}} s<br>"
                f"<b>End:</b> {ev['end_sec']:.2f} s<br>"
                f"<b>Duration:</b> %{{x:.2f}} s<extra></extra>"
            )
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        title=dict(
            text="Experiment Task Event Timeline (PhysioNet Annotations)",
            font=dict(size=18, color="#F8FAFC")
        ),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="#334155",
            range=[0, total_duration],
            showgrid=True
        ),
        yaxis=dict(
            title="Event Task",
            gridcolor="#334155",
            autorange="reversed"
        ),
        barmode='stack',
        height=320,
        margin=dict(l=100, r=40, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.7)"
        )
    )

    return fig


def create_multi_channel_plot(
    raw: mne.io.Raw,
    start_time: float,
    end_time: float,
    selected_channels: list,
    df_events: pd.DataFrame
) -> go.Figure:
    """
    Creates a stacked multi-channel EEG recording visualization.

    Parameters:
        raw (mne.io.Raw): MNE raw data object.
        start_time (float): Window start time in seconds.
        end_time (float): Window end time in seconds.
        selected_channels (list): List of channel names to display.
        df_events (pd.DataFrame): DataFrame of event annotations.

    Returns:
        go.Figure: Stacked channel Plotly visualization.
    """
    sfreq = raw.info['sfreq']
    start_idx = int(max(0, start_time * sfreq))
    end_idx = int(min(len(raw.times), end_time * sfreq))

    times = raw.times[start_idx:end_idx]

    if not selected_channels:
        selected_channels = raw.ch_names[:16]

    ch_indices = [raw.ch_names.index(ch) for ch in selected_channels if ch in raw.ch_names]
    data_uv = raw.get_data(picks=ch_indices, start=start_idx, stop=end_idx) * 1e6  # Shape: (n_ch, n_samples)

    # Compute offset scale for stacked plot
    channel_stds = np.std(data_uv, axis=1)
    mean_std = np.mean(channel_stds) if len(channel_stds) > 0 else 50.0
    spacing = max(mean_std * 5.0, 100.0)  # uV vertical separation per channel

    fig = go.Figure()

    # Event Shading Spans
    if not df_events.empty:
        overlapping_events = df_events[
            (df_events['end_sec'] >= start_time) & (df_events['onset_sec'] <= end_time)
        ]
        for _, ev in overlapping_events.iterrows():
            ev_start = max(start_time, ev['onset_sec'])
            ev_end = min(end_time, ev['end_sec'])
            fig.add_vrect(
                x0=ev_start,
                x1=ev_end,
                fillcolor=ev['color'],
                opacity=0.15,
                layer="below",
                line_width=0
            )

    yticks = []
    yticklabels = []

    # Stack channels vertically
    n_ch = len(selected_channels)
    for i, ch_name in enumerate(selected_channels):
        offset = (n_ch - 1 - i) * spacing
        signal_offset = data_uv[i] + offset
        yticks.append(offset)
        yticklabels.append(ch_name)

        fig.add_trace(go.Scatter(
            x=times,
            y=signal_offset,
            mode='lines',
            name=ch_name,
            line=dict(width=1.0, color='#38BDF8'),
            hovertemplate=f"<b>Channel {ch_name}</b><br>Time: %{{x:.3f}} s<br>Value: %{{y:.2f}} μV<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        title=dict(
            text=f"Multi-Channel EEG Display ({len(selected_channels)} Channels Stacked)",
            font=dict(size=18, color="#F8FAFC")
        ),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="#334155",
            range=[start_time, end_time],
            showgrid=True
        ),
        yaxis=dict(
            title="EEG Channels",
            tickmode='array',
            tickvals=yticks,
            ticktext=yticklabels,
            gridcolor="#334155",
            showgrid=True
        ),
        showlegend=False,
        height=max(500, len(selected_channels) * 28),
        margin=dict(l=80, r=40, t=60, b=50)
    )

    return fig
