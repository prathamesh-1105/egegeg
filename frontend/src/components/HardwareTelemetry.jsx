import React, { useState, useEffect } from 'react';
import {
  Cpu,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Activity,
  Wifi,
  Radio,
  RefreshCw,
  Sliders,
  Layers,
  AlertOctagon,
  Clock,
  Zap,
  Info
} from 'lucide-react';

export default function HardwareTelemetry() {
  const [hwData, setHwData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [actionMsg, setActionMsg] = useState(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch('http://localhost:8050/api/hardware/status');
      if (res.ok) {
        const data = await res.json();
        setHwData(data);
      }
    } catch (err) {
      console.error('Error fetching hardware status:', err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleReconnect = async () => {
    setReconnecting(true);
    setActionMsg(null);
    try {
      const res = await fetch('http://localhost:8050/api/hardware/reconnect', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setHwData(data);
        setActionMsg('LSL stream scan completed.');
      }
    } catch (err) {
      setActionMsg('Failed to trigger hardware reconnection.');
    } finally {
      setReconnecting(false);
      setTimeout(() => setActionMsg(null), 4000);
    }
  };

  const handleSourceSelect = async (sourceType) => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8050/api/hardware/select-source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_type: sourceType })
      });
      if (res.ok) {
        const data = await res.json();
        setHwData(data);
        setActionMsg(`Switched source to ${sourceType}`);
      }
    } catch (err) {
      setActionMsg('Error switching source.');
    } finally {
      setLoading(false);
      setTimeout(() => setActionMsg(null), 3000);
    }
  };

  const handleProfileSelect = async (profileId) => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8050/api/hardware/select-profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: profileId })
      });
      if (res.ok) {
        const data = await res.json();
        setHwData(data.hardware_status);
        setActionMsg(`Selected hardware profile: ${profileId}`);
      }
    } catch (err) {
      setActionMsg('Error selecting profile.');
    } finally {
      setLoading(false);
      setTimeout(() => setActionMsg(null), 3000);
    }
  };

  // Determine state styling and badges across the 6 distinct states:
  // 1. RECORDED_SIMULATION
  // 2. HARDWARE_DISCONNECTED
  // 3. HARDWARE_INVALID_CONFIG
  // 4. HARDWARE_VALID_STREAM
  // 5. CALIBRATION_IN_PROGRESS
  // 6. CALIBRATION_COMPLETE
  const uiState = hwData?.ui_readiness_state || 'RECORDED_SIMULATION';

  const getStateConfig = (st) => {
    switch (st) {
      case 'RECORDED_SIMULATION':
        return {
          title: 'RECORDED PHYSIONET SIMULATION',
          subtitle: 'Genuine PhysioNet EEGMMIDB Motor Imagery Replay (Software Mode)',
          badgeColor: '#00F2FE',
          badgeBg: 'rgba(0, 242, 254, 0.1)',
          borderColor: '#00F2FE',
          icon: Sliders,
          isHardwareReady: false,
          readyLabel: 'SIMULATED PLAYBACK',
          readyBg: '#1E293B',
          readyColor: '#94A3B8'
        };
      case 'HARDWARE_DISCONNECTED':
        return {
          title: 'HARDWARE DISCONNECTED',
          subtitle: 'No active Lab Streaming Layer (LSL) EEG stream detected on local network',
          badgeColor: '#F59E0B',
          badgeBg: 'rgba(245, 158, 11, 0.1)',
          borderColor: '#F59E0B',
          icon: Wifi,
          isHardwareReady: false,
          readyLabel: 'DISCONNECTED',
          readyBg: '#451A03',
          readyColor: '#F59E0B'
        };
      case 'HARDWARE_INVALID_CONFIG':
        return {
          title: 'HARDWARE CONNECTED — INVALID CONFIGURATION',
          subtitle: 'LSL stream found but channel count or physical electrode montage violates contract',
          badgeColor: '#EF4444',
          badgeBg: 'rgba(239, 68, 68, 0.1)',
          borderColor: '#EF4444',
          icon: AlertOctagon,
          isHardwareReady: false,
          readyLabel: 'INVALID MONTAGE',
          readyBg: '#450A0A',
          readyColor: '#EF4444'
        };
      case 'HARDWARE_VALID_STREAM':
        return {
          title: 'HARDWARE CONNECTED — VALID STREAM',
          subtitle: 'OpenBCI Cyton + Daisy 16-channel motor strip stream validated and resampled',
          badgeColor: '#10B981',
          badgeBg: 'rgba(16, 185, 129, 0.1)',
          borderColor: '#10B981',
          icon: CheckCircle2,
          isHardwareReady: true,
          readyLabel: 'BENCH READY',
          readyBg: '#064E3B',
          readyColor: '#10B981'
        };
      case 'CALIBRATION_IN_PROGRESS':
        return {
          title: 'HARDWARE CALIBRATION IN PROGRESS',
          subtitle: `Recording ${hwData?.calibration_status?.active_class || 'REST'} windows for calibration`,
          badgeColor: '#38BDF8',
          badgeBg: 'rgba(56, 189, 248, 0.15)',
          borderColor: '#38BDF8',
          icon: Clock,
          isHardwareReady: false,
          readyLabel: 'CALIBRATING',
          readyBg: '#0C4A6E',
          readyColor: '#38BDF8'
        };
      case 'CALIBRATION_COMPLETE':
        return {
          title: 'CALIBRATION SESSION COMPLETED',
          subtitle: 'Subject baseline recorded and metadata stored separately from base model',
          badgeColor: '#10B981',
          badgeBg: 'rgba(16, 185, 129, 0.1)',
          borderColor: '#10B981',
          icon: ShieldCheck,
          isHardwareReady: true,
          readyLabel: 'CALIBRATED',
          readyBg: '#064E3B',
          readyColor: '#10B981'
        };
      default:
        return {
          title: 'STREAM STATUS UNKNOWN',
          subtitle: 'Awaiting backend telemetry...',
          badgeColor: '#64748B',
          badgeBg: 'rgba(100, 116, 139, 0.1)',
          borderColor: '#334155',
          icon: Info,
          isHardwareReady: false,
          readyLabel: 'STANDBY',
          readyBg: '#0F172A',
          readyColor: '#64748B'
        };
    }
  };

  const stateCfg = getStateConfig(uiState);
  const StateIcon = stateCfg.icon;

  return (
    <div style={{ maxWidth: '1440px', margin: '24px auto', padding: '0 24px' }}>
      {/* Action Notification Toast */}
      {actionMsg && (
        <div style={{
          background: 'rgba(15, 23, 42, 0.95)',
          border: '1px solid #00F2FE',
          borderRadius: '8px',
          padding: '10px 16px',
          marginBottom: '16px',
          color: '#00F2FE',
          fontSize: '0.85rem',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <Zap size={14} />
          {actionMsg}
        </div>
      )}

      {/* Hero Header & Diagnostics Banner */}
      <div style={{
        background: 'linear-gradient(135deg, #090D16 0%, #111827 100%)',
        border: `2px solid ${stateCfg.borderColor}`,
        borderRadius: '16px',
        padding: '28px',
        marginBottom: '24px',
        boxShadow: `0 8px 32px ${stateCfg.badgeBg}`
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '20px' }}>
          <div>
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 12px',
              borderRadius: '20px',
              background: stateCfg.badgeBg,
              color: stateCfg.badgeColor,
              fontSize: '0.75rem',
              fontWeight: 800,
              fontFamily: 'JetBrains Mono',
              letterSpacing: '0.04em',
              marginBottom: '12px',
              border: `1px solid ${stateCfg.borderColor}50`
            }}>
              <StateIcon size={14} />
              {uiState}
            </div>

            <h1 style={{ fontFamily: 'Space Grotesk', margin: '0 0 6px 0', color: '#F8FAFC', fontSize: '2.1rem', letterSpacing: '-0.02em' }}>
              {stateCfg.title}
            </h1>

            <div style={{ color: '#94A3B8', fontSize: '0.92rem', maxWidth: '750px' }}>
              {stateCfg.subtitle}
            </div>

            {hwData?.is_simulated && (
              <div style={{
                marginTop: '12px',
                padding: '8px 12px',
                background: 'rgba(0, 242, 254, 0.08)',
                border: '1px solid rgba(0, 242, 254, 0.3)',
                borderRadius: '8px',
                fontSize: '0.78rem',
                color: '#38BDF8',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <Info size={14} />
                <b>SIMULATION NOTICE:</b> Currently streaming genuine PhysioNet EEGMMIDB recorded data. Physical headset is not connected.
              </div>
            )}

            {hwData?.last_error && (
              <div style={{
                marginTop: '12px',
                padding: '8px 12px',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '8px',
                fontSize: '0.78rem',
                color: '#F87171',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <AlertTriangle size={14} />
                <b>STREAM ERROR:</b> {hwData.last_error}
              </div>
            )}
          </div>

          {/* Quick Status Cards */}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {/* Readiness Level Card */}
            <div className="glass-card" style={{ padding: '14px 20px', borderColor: stateCfg.borderColor, textAlign: 'center', minWidth: '130px' }}>
              <div style={{ fontSize: '0.7rem', color: '#94A3B8', fontWeight: 700 }}>READINESS LEVEL</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 800, color: stateCfg.badgeColor, marginTop: '4px' }}>
                {hwData?.readiness_level || 'SOFTWARE_READY'}
              </div>
              <div style={{ fontSize: '0.65rem', color: '#64748B', marginTop: '2px' }}>
                {stateCfg.readyLabel}
              </div>
            </div>

            {/* Active Source Card */}
            <div className="glass-card" style={{ padding: '14px 20px', borderColor: '#334155', textAlign: 'center', minWidth: '130px' }}>
              <div style={{ fontSize: '0.7rem', color: '#94A3B8', fontWeight: 700 }}>ACTIVE SOURCE</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#F8FAFC', marginTop: '4px' }}>
                {hwData?.selected_source === 'SIMULATED_EDF' ? 'PhysioNet EDF' : 'OpenBCI LSL'}
              </div>
              <div style={{ fontSize: '0.65rem', color: '#64748B', marginTop: '2px' }}>
                {hwData?.is_simulated ? 'Recorded Dataset' : 'Network Stream'}
              </div>
            </div>

            {/* Sampling Rate Card */}
            <div className="glass-card" style={{ padding: '14px 20px', borderColor: '#334155', textAlign: 'center', minWidth: '130px' }}>
              <div style={{ fontSize: '0.7rem', color: '#94A3B8', fontWeight: 700 }}>SAMPLING RATE</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#00F2FE', marginTop: '4px', fontFamily: 'JetBrains Mono' }}>
                160.0 Hz
              </div>
              <div style={{ fontSize: '0.65rem', color: '#64748B', marginTop: '2px' }}>
                Measured: {hwData?.measured_sampling_rate_hz || 160.0} Hz
              </div>
            </div>
          </div>
        </div>

        {/* Source Switcher & Hardware Controls Bar */}
        <div style={{
          marginTop: '24px',
          paddingTop: '20px',
          borderTop: '1px solid rgba(51, 65, 85, 0.4)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.8rem', color: '#94A3B8', fontWeight: 600 }}>SOURCE SELECTION:</span>
            <button
              onClick={() => handleSourceSelect('SIMULATED_EDF')}
              disabled={loading}
              style={{
                padding: '6px 14px',
                borderRadius: '8px',
                border: hwData?.selected_source === 'SIMULATED_EDF' ? '1px solid #00F2FE' : '1px solid #334155',
                background: hwData?.selected_source === 'SIMULATED_EDF' ? 'rgba(0, 242, 254, 0.15)' : 'rgba(15, 23, 42, 0.6)',
                color: hwData?.selected_source === 'SIMULATED_EDF' ? '#00F2FE' : '#94A3B8',
                fontWeight: 700,
                fontSize: '0.78rem',
                cursor: 'pointer'
              }}
            >
              PhysioNet EDF Simulation
            </button>

            <button
              onClick={() => handleSourceSelect('LSL_HARDWARE')}
              disabled={loading}
              style={{
                padding: '6px 14px',
                borderRadius: '8px',
                border: hwData?.selected_source === 'LSL_HARDWARE' ? '1px solid #10B981' : '1px solid #334155',
                background: hwData?.selected_source === 'LSL_HARDWARE' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(15, 23, 42, 0.6)',
                color: hwData?.selected_source === 'LSL_HARDWARE' ? '#10B981' : '#94A3B8',
                fontWeight: 700,
                fontSize: '0.78rem',
                cursor: 'pointer'
              }}
            >
              OpenBCI Cyton + Daisy (LSL)
            </button>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              onClick={handleReconnect}
              disabled={reconnecting}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 16px',
                borderRadius: '8px',
                border: '1px solid #3B82F6',
                background: 'linear-gradient(135deg, rgba(37, 99, 235, 0.2) 0%, rgba(30, 58, 138, 0.2) 100%)',
                color: '#60A5FA',
                fontWeight: 700,
                fontSize: '0.82rem',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              <RefreshCw size={14} className={reconnecting ? 'spin-animation' : ''} />
              {reconnecting ? 'Scanning LSL...' : 'Scan & Reconnect LSL'}
            </button>
          </div>
        </div>
      </div>

      {/* Main Diagnostics 2-Column Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px', marginBottom: '24px' }}>
        {/* Left Column: 6-Point Hardware Verification & Safety Gates */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#F8FAFC', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={18} color="#10B981" />
            Phase 2E Hardware Verification & Safety Protocol
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {hwData?.safety_checks?.map((check, idx) => {
              const isPass = check.gate === 'PASSED';
              const isInfo = check.gate === 'UNAVAILABLE_WITHOUT_HARDWARE';
              return (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '12px 14px',
                    background: 'rgba(15, 23, 42, 0.7)',
                    border: `1px solid ${isPass ? '#1E293B' : (isInfo ? '#334155' : 'rgba(239, 68, 68, 0.4)')}`,
                    borderRadius: '8px'
                  }}
                >
                  <div>
                    <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#F8FAFC' }}>
                      {idx + 1}. {check.check}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#94A3B8', fontFamily: 'JetBrains Mono', marginTop: '2px' }}>
                      {check.status}
                    </div>
                  </div>
                  <div
                    className={isPass ? 'neural-badge badge-emerald' : (isInfo ? 'neural-badge badge-cyan' : 'neural-badge badge-red')}
                    style={{ fontSize: '0.68rem', padding: '3px 8px' }}
                  >
                    {isPass ? <CheckCircle2 size={12} /> : (isInfo ? <Info size={12} /> : <AlertTriangle size={12} />)}
                    {check.gate}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Stream Continuity & Buffer Telemetry */}
          <div style={{
            marginTop: '20px',
            padding: '16px',
            background: 'rgba(15, 23, 42, 0.5)',
            border: '1px solid #1E293B',
            borderRadius: '10px'
          }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '8px' }}>
              Buffer Continuity & Stream Freshness
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', textAlign: 'center' }}>
              <div style={{ background: '#090D16', padding: '10px', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.65rem', color: '#64748B' }}>BUFFER SAMPLES</div>
                <div style={{ fontSize: '1.0rem', fontWeight: 800, color: '#00F2FE', fontFamily: 'JetBrains Mono' }}>
                  {hwData?.samples_in_buffer || 0} / 480
                </div>
              </div>
              <div style={{ background: '#090D16', padding: '10px', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.65rem', color: '#64748B' }}>DISCONTINUITIES</div>
                <div style={{ fontSize: '1.0rem', fontWeight: 800, color: (hwData?.discontinuity_count || 0) > 0 ? '#F59E0B' : '#10B981', fontFamily: 'JetBrains Mono' }}>
                  {hwData?.discontinuity_count || 0}
                </div>
              </div>
              <div style={{ background: '#090D16', padding: '10px', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.65rem', color: '#64748B' }}>DROPPED SAMPLES</div>
                <div style={{ fontSize: '1.0rem', fontWeight: 800, color: (hwData?.dropped_samples_count || 0) > 0 ? '#EF4444' : '#10B981', fontFamily: 'JetBrains Mono' }}>
                  {hwData?.dropped_samples_count || 0}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: 15-Channel Motor Grid Contract & Amplitude Diagnostics */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#F8FAFC', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} color="#00F2FE" />
            15-Channel Motor Grid Contract (Step 62 Target)
          </div>
          <div style={{ fontSize: '0.75rem', color: '#94A3B8', marginBottom: '16px' }}>
            Exact sensorimotor electrode montage required by Step 62 champion model. Units: Microvolts (μV).
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
            {hwData?.expected_channel_names?.map((ch) => {
              const isMissing = hwData?.missing_channels?.includes(ch);
              const isGood = !isMissing;
              return (
                <div
                  key={ch}
                  style={{
                    background: 'rgba(15, 23, 42, 0.85)',
                    border: `1px solid ${isGood ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.6)'}`,
                    borderRadius: '8px',
                    padding: '10px 8px',
                    textAlign: 'center'
                  }}
                >
                  <div style={{ fontSize: '0.85rem', fontWeight: 800, color: '#F8FAFC' }}>{ch}</div>
                  <div style={{
                    fontSize: '0.72rem',
                    fontWeight: 700,
                    color: isGood ? '#10B981' : '#EF4444',
                    fontFamily: 'JetBrains Mono',
                    margin: '3px 0'
                  }}>
                    {isGood ? 'VERIFIED' : 'MISSING'}
                  </div>
                  <div style={{ fontSize: '0.62rem', color: '#64748B' }}>
                    {isGood ? '0.1-500 μV Bound' : 'Montage Reject'}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Scientific Notice Banner */}
          <div style={{
            marginTop: '20px',
            padding: '12px 14px',
            background: 'rgba(15, 23, 42, 0.6)',
            border: '1px solid #334155',
            borderRadius: '8px',
            fontSize: '0.74rem',
            color: '#94A3B8',
            lineHeight: 1.45
          }}>
            <div style={{ fontWeight: 700, color: '#F8FAFC', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Info size={14} color="#00F2FE" />
              Scientific Measurement Integrity Policy
            </div>
            Electrode amplitude bounds (0.1 μV to 500 μV) and flatline detection are configurable <b>engineering signal-sanity checks</b>, NOT physical electrode impedance measurements.
            True contact impedance measurement requires hardware ADS1299 current stimulation on compatible OpenBCI boards.
          </div>
        </div>
      </div>
    </div>
  );
}
