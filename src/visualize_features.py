import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_fft_psd_plot(freqs: np.ndarray, psd: np.ndarray, channel_name: str) -> go.Figure:
    """
    Creates an interactive Power Spectral Density (PSD) plot with shaded frequency bands.

    Parameters:
        freqs (np.ndarray): Array of frequency bins in Hz.
        psd (np.ndarray): Power Spectral Density values in uV^2/Hz.
        channel_name (str): Channel label.

    Returns:
        go.Figure: Interactive Plotly line plot.
    """
    fig = go.Figure()

    # Limit view to 0 - 45 Hz range
    mask = freqs <= 45.0
    f_sub = freqs[mask]
    psd_sub = psd[mask]

    # Add Frequency Band Shaded Background Regions
    band_regions = [
        ("Delta (0.5–4 Hz)", 0.5, 4.0, "rgba(59, 130, 246, 0.15)", "#3B82F6"),
        ("Theta (4–8 Hz)", 4.0, 8.0, "rgba(16, 185, 129, 0.15)", "#10B981"),
        ("Alpha (8–13 Hz)", 8.0, 13.0, "rgba(245, 158, 11, 0.15)", "#F59E0B"),
        ("Beta (13–30 Hz)", 13.0, 30.0, "rgba(239, 68, 68, 0.15)", "#EF4444"),
        ("Gamma (30–45 Hz)", 30.0, 45.0, "rgba(168, 85, 247, 0.15)", "#A855F7")
    ]

    for band_name, f0, f1, fillcolor, linecolor in band_regions:
        fig.add_vrect(
            x0=f0, x1=f1,
            fillcolor=fillcolor,
            layer="below",
            line_width=1,
            line_color=linecolor,
            annotation_text=band_name,
            annotation_position="top left",
            annotation_font=dict(size=10, color="#E2E8F0")
        )

    # Plot Spectral Power Density Curve
    fig.add_trace(go.Scatter(
        x=f_sub,
        y=psd_sub,
        mode='lines',
        name=f"Channel {channel_name} PSD",
        line=dict(color='#00F2FE', width=2.2),
        hovertemplate="<b>Freq:</b> %{x:.2f} Hz<br><b>Power:</b> %{y:.4f} μV²/Hz<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        title=dict(
            text=f"Frequency Power Spectrum (Welch PSD) — Channel: {channel_name}",
            font=dict(size=18, color="#F8FAFC")
        ),
        xaxis=dict(
            title="Frequency (Hz)",
            gridcolor="#334155",
            range=[0, 45],
            showgrid=True
        ),
        yaxis=dict(
            title="Power Spectral Density (μV² / Hz)",
            gridcolor="#334155",
            showgrid=True
        ),
        margin=dict(l=60, r=40, t=60, b=50),
        height=420
    )

    return fig


def create_band_power_bar_chart(band_powers: dict) -> go.Figure:
    """
    Creates a bar chart displaying power distribution across standard EEG frequency bands.

    Parameters:
        band_powers (dict): Dictionary mapping band names to calculated power values.

    Returns:
        go.Figure: Interactive bar chart.
    """
    bands = list(band_powers.keys())
    powers = [band_powers[b] for b in bands]

    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#A855F7"]

    fig = go.Figure(data=[
        go.Bar(
            x=bands,
            y=powers,
            marker_color=colors,
            text=[f"{p:.2f} μV²" for p in powers],
            textposition='auto',
            hovertemplate="<b>Band:</b> %{x}<br><b>Power:</b> %{y:.4f} μV²<extra></extra>"
        )
    ])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        title=dict(
            text="EEG Frequency Band Power Distribution",
            font=dict(size=16, color="#F8FAFC")
        ),
        xaxis=dict(title="Frequency Band", gridcolor="#334155"),
        yaxis=dict(title="Power (μV²)", gridcolor="#334155"),
        margin=dict(l=50, r=30, t=50, b=40),
        height=380
    )

    return fig


def create_beta_filter_plot(
    times: np.ndarray,
    raw_signal: np.ndarray,
    beta_signal: np.ndarray,
    channel_name: str
) -> go.Figure:
    """
    Creates a stacked dual waveform chart comparing Raw EEG vs 13–30 Hz Beta Band EEG.

    Parameters:
        times (np.ndarray): Time vector in seconds.
        raw_signal (np.ndarray): Unfiltered EEG microvolts.
        beta_signal (np.ndarray): 13–30 Hz band-pass filtered EEG microvolts.
        channel_name (str): Selected channel label.

    Returns:
        go.Figure: Subplot figure.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"RAW EEG Signal ({channel_name}) — All Frequencies Included",
            f"BETA-BAND EEG ({channel_name}) — 13–30 Hz Band-Pass Filtered"
        )
    )

    # Subplot 1: Raw EEG Signal
    fig.add_trace(
        go.Scatter(
            x=times,
            y=raw_signal,
            mode='lines',
            name="Raw EEG",
            line=dict(color='#00F2FE', width=1.5),
            hovertemplate="Time: %{x:.3f} s<br>Raw Amplitude: %{y:.2f} μV<extra></extra>"
        ),
        row=1, col=1
    )

    # Subplot 2: Beta Band Signal
    fig.add_trace(
        go.Scatter(
            x=times,
            y=beta_signal,
            mode='lines',
            name="Beta Band (13–30 Hz)",
            line=dict(color='#EF4444', width=1.6),
            hovertemplate="Time: %{x:.3f} s<br>Beta Amplitude: %{y:.2f} μV<extra></extra>"
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        height=520,
        margin=dict(l=60, r=40, t=60, b=50),
        showlegend=False
    )

    fig.update_xaxes(title_text="Time (seconds)", gridcolor="#334155", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (μV)", gridcolor="#334155", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (μV)", gridcolor="#334155", row=2, col=1)

    return fig
