import React, { useState, useEffect } from 'react';
import {
  Sliders,
  Play,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  Activity,
  Layers,
  Clock,
  Sparkles,
  Radio,
  FileText
} from 'lucide-react';

export default function CalibrationStudio() {
  const [calibStatus, setCalibStatus] = useState(null);
  const [selectedSubject, setSelectedSubject] = useState('S001');
  const [budget, setBudget] = useState(10);
  const [isRunning, setIsRunning] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [autoProgress, setAutoProgress] = useState(false);
  const [recentSessions, setRecentSessions] = useState([]);
  const [lastMessage, setLastMessage] = useState(null);

  // Fetch status on mount
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch('http://localhost:8050/api/calibration/status');
      if (res.ok) {
        const data = await res.json();
        setCalibStatus(data);
        if (data.recent_sessions) {
          setRecentSessions(data.recent_sessions);
        }
      }
    } catch (err) {
      console.error('Failed to fetch calibration status:', err);
    }
  };

  const handleStartCalibration = async () => {
    try {
      const res = await fetch('http://localhost:8050/api/calibration/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject_id: selectedSubject,
          budget: Number(budget),
          hardware_profile_id: 'OPENBCI_CYTON_DAISY_16CH'
        })
      });
      if (res.ok) {
        const data = await res.json();
        setCalibStatus(data);
        setIsRunning(true);
        setStepIdx(0);
        setLastMessage(`Started calibration for ${selectedSubject} with target ${budget} windows/class.`);
      }
    } catch (err) {
      console.error('Error starting calibration:', err);
    }
  };

  const handleStepCalibration = async () => {
    try {
      const nextIdx = stepIdx + 1;
      setStepIdx(nextIdx);
      const res = await fetch(`http://localhost:8050/api/calibration/step?subject_id=${selectedSubject}&trial_idx=${stepIdx}`, {
        method: 'POST'
      });
      if (res.ok) {
        const data = await res.json();
        setCalibStatus(data);
        if (data.state === 'COMPLETED') {
          setIsRunning(false);
          setAutoProgress(false);
          setLastMessage(`Calibration completed successfully! Metadata saved.`);
          fetchStatus();
        }
      }
    } catch (err) {
      console.error('Error stepping calibration:', err);
    }
  };

  // Auto-progress timer
  useEffect(() => {
    let timer;
    if (autoProgress && isRunning && calibStatus?.state !== 'COMPLETED') {
      timer = setInterval(() => {
        handleStepCalibration();
      }, 700);
    }
    return () => clearInterval(timer);
  }, [autoProgress, isRunning, stepIdx, calibStatus?.state]);

  const handleResetCalibration = async () => {
    try {
      const res = await fetch('http://localhost:8050/api/calibration/reset', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setIsRunning(false);
        setAutoProgress(false);
        setStepIdx(0);
        setLastMessage(`Calibration reset: Restored frozen Step 62 base model.`);
        fetchStatus();
      }
    } catch (err) {
      console.error('Error resetting calibration:', err);
    }
  };

  const activeState = calibStatus?.state || 'IDLE';
  const activeClass = calibStatus?.active_class || 'NONE';
  const collectedCounts = calibStatus?.collected_counts || { REST: 0, LEFT_FIST: 0, RIGHT_FIST: 0 };
  const targetBudget = calibStatus?.target_windows_per_class || budget;

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '24px 20px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Header Banner */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)',
        border: '1px solid rgba(59, 130, 246, 0.3)',
        borderRadius: '16px',
        padding: '24px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <Sliders size={26} color="#00F2FE" />
              <h1 style={{ margin: 0, fontSize: '1.75rem', fontWeight: 800, letterSpacing: '0.02em', color: '#F8FAFC' }}>
                NeuroOS Personalized EEG Calibration Studio
              </h1>
              <span style={{
                background: 'rgba(0, 242, 254, 0.15)',
                border: '1px solid #00F2FE',
                color: '#00F2FE',
                fontSize: '0.72rem',
                fontWeight: 700,
                padding: '3px 10px',
                borderRadius: '999px'
              }}>
                Phase 2C
              </span>
            </div>
            <p style={{ margin: 0, fontSize: '0.88rem', color: '#94A3B8', maxWidth: '900px', lineHeight: 1.5 }}>
              Adaptive subject calibration testing suite. Evaluates sample efficiency (5, 10, 20, 30 trials/class) and compares subject-specific normalization, probability calibration, and rejection tuning against the frozen <strong>Step 62 Surface Laplacian Champion</strong>.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              onClick={handleResetCalibration}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                background: 'rgba(239, 68, 68, 0.12)',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                color: '#F87171',
                padding: '10px 18px',
                borderRadius: '10px',
                fontSize: '0.84rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              <RotateCcw size={16} />
              Reset to Base Model
            </button>
          </div>
        </div>

        {/* Mandatory Research Notice */}
        <div style={{
          marginTop: '16px',
          background: 'rgba(234, 179, 8, 0.1)',
          border: '1px solid rgba(234, 179, 8, 0.3)',
          borderRadius: '10px',
          padding: '12px 16px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          fontSize: '0.82rem',
          color: '#FDE047'
        }}>
          <AlertTriangle size={20} color="#FACC15" style={{ flexShrink: 0 }} />
          <div>
            <strong>Offline Simulated Calibration Notice:</strong> This session replays authentic recorded PhysioNet EEGMMIDB runs (R03/R07) to test calibration algorithms offline. It does not claim real-time live human brain control. A physical OpenBCI Cyton & Daisy 16-channel headset is required for live subject calibration.
          </div>
        </div>
      </div>

      {/* Grid: Calibration Controls & Progress Machine */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '20px' }}>
        
        {/* Panel 1: Session Setup */}
        <div style={{
          background: 'rgba(15, 23, 42, 0.85)',
          border: '1px solid rgba(51, 65, 85, 0.6)',
          borderRadius: '14px',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', borderBottom: '1px solid rgba(51, 65, 85, 0.6)', paddingBottom: '12px' }}>
            <Cpu size={18} color="#38BDF8" />
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#F1F5F9' }}>
              Calibration Protocol Configuration
            </h3>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: '#94A3B8', marginBottom: '6px' }}>
                Subject Profile (PhysioNet EEGMMIDB)
              </label>
              <select
                value={selectedSubject}
                onChange={(e) => setSelectedSubject(e.target.value)}
                disabled={isRunning}
                style={{
                  width: '100%',
                  background: 'rgba(30, 41, 59, 0.9)',
                  border: '1px solid rgba(71, 85, 105, 0.6)',
                  color: '#F8FAFC',
                  padding: '10px 14px',
                  borderRadius: '8px',
                  fontSize: '0.9rem',
                  outline: 'none'
                }}
              >
                {Array.from({ length: 10 }, (_, i) => `S${String(i + 1).padStart(3, '0')}`).map((s) => (
                  <option key={s} value={s}>{s} (16-channel motor strip)</option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: '#94A3B8', marginBottom: '6px' }}>
                Target Calibration Budget (Trials per Class)
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
                {[5, 10, 20, 30].map((b) => (
                  <button
                    key={b}
                    onClick={() => setBudget(b)}
                    disabled={isRunning}
                    style={{
                      background: budget === b ? 'rgba(0, 242, 254, 0.2)' : 'rgba(30, 41, 59, 0.8)',
                      border: budget === b ? '1px solid #00F2FE' : '1px solid rgba(71, 85, 105, 0.5)',
                      color: budget === b ? '#00F2FE' : '#94A3B8',
                      padding: '8px',
                      borderRadius: '8px',
                      fontWeight: 700,
                      cursor: 'pointer'
                    }}
                  >
                    {b} / class
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: '#94A3B8', marginBottom: '6px' }}>
                Hardware Reference Montage
              </label>
              <div style={{
                background: 'rgba(30, 41, 59, 0.6)',
                border: '1px solid rgba(51, 65, 85, 0.5)',
                padding: '10px 14px',
                borderRadius: '8px',
                fontSize: '0.82rem',
                color: '#CBD5E1',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <Radio size={16} color="#38BDF8" />
                <span>OpenBCI Cyton + Daisy (15 Active + CPz Ground)</span>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '10px', marginTop: '8px' }}>
              <button
                onClick={handleStartCalibration}
                disabled={isRunning}
                style={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  background: isRunning ? 'rgba(51, 65, 85, 0.5)' : 'linear-gradient(135deg, #00F2FE 0%, #3B82F6 100%)',
                  border: 'none',
                  color: isRunning ? '#64748B' : '#070B14',
                  padding: '12px',
                  borderRadius: '10px',
                  fontWeight: 700,
                  fontSize: '0.9rem',
                  cursor: isRunning ? 'not-allowed' : 'pointer'
                }}
              >
                <Play size={18} />
                {isRunning ? 'Calibration in Progress' : 'Start Simulated Session'}
              </button>
            </div>
          </div>
        </div>

        {/* Panel 2: Live Calibration State Machine */}
        <div style={{
          background: 'rgba(15, 23, 42, 0.85)',
          border: '1px solid rgba(51, 65, 85, 0.6)',
          borderRadius: '14px',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(51, 65, 85, 0.6)', paddingBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={18} color="#10B981" />
              <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#F1F5F9' }}>
                Calibration Execution Machine
              </h3>
            </div>
            <span style={{
              background: activeState === 'COMPLETED' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(59, 130, 246, 0.2)',
              border: `1px solid ${activeState === 'COMPLETED' ? '#10B981' : '#3B82F6'}`,
              color: activeState === 'COMPLETED' ? '#34D399' : '#60A5FA',
              padding: '2px 8px',
              borderRadius: '6px',
              fontSize: '0.74rem',
              fontWeight: 700
            }}>
              {activeState}
            </span>
          </div>

          {/* Prompt Cue Box */}
          <div style={{
            background: activeClass === 'LEFT_FIST'
              ? 'rgba(59, 130, 246, 0.15)'
              : activeClass === 'RIGHT_FIST'
              ? 'rgba(16, 185, 129, 0.15)'
              : activeClass === 'REST'
              ? 'rgba(148, 163, 184, 0.12)'
              : 'rgba(30, 41, 59, 0.6)',
            border: `1px solid ${
              activeClass === 'LEFT_FIST' ? '#3B82F6' : activeClass === 'RIGHT_FIST' ? '#10B981' : 'rgba(100, 116, 139, 0.4)'
            }`,
            borderRadius: '12px',
            padding: '20px',
            textAlign: 'center',
            minHeight: '120px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center'
          }}>
            <div style={{ fontSize: '0.78rem', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '6px' }}>
              Current Paced Stimulus Cue
            </div>
            <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#F8FAFC' }}>
              {activeClass === 'REST' && '🧘 REST & RELAX'}
              {activeClass === 'LEFT_FIST' && '⬅️ IMAGINE LEFT FIST'}
              {activeClass === 'RIGHT_FIST' && '➡️ IMAGINE RIGHT FIST'}
              {activeClass === 'COMPLETED' && '🎉 CALIBRATION COMPLETE'}
              {activeClass === 'NONE' && '⏸️ IDLE - CLICK START'}
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748B', marginTop: '6px' }}>
              {activeState.startsWith('COLLECTING') && `Recording genuine 3.0s window #${stepIdx + 1} for ${selectedSubject}...`}
            </div>
          </div>

          {/* Class Progress Badges */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
            {[
              { label: 'REST', count: collectedCounts.REST || 0, color: '#94A3B8' },
              { label: 'LEFT FIST', count: collectedCounts.LEFT_FIST || 0, color: '#60A5FA' },
              { label: 'RIGHT FIST', count: collectedCounts.RIGHT_FIST || 0, color: '#34D399' }
            ].map((c) => (
              <div key={c.label} style={{
                background: 'rgba(30, 41, 59, 0.6)',
                border: '1px solid rgba(51, 65, 85, 0.5)',
                borderRadius: '8px',
                padding: '10px',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '0.72rem', color: c.color, fontWeight: 700 }}>{c.label}</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#F8FAFC', marginTop: '4px' }}>
                  {c.count} / {targetBudget}
                </div>
              </div>
            ))}
          </div>

          {/* Stepping Actions */}
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              onClick={handleStepCalibration}
              disabled={!isRunning || activeState === 'COMPLETED'}
              style={{
                flex: 1,
                background: 'rgba(59, 130, 246, 0.2)',
                border: '1px solid #3B82F6',
                color: '#60A5FA',
                padding: '10px',
                borderRadius: '8px',
                fontWeight: 600,
                fontSize: '0.84rem',
                cursor: !isRunning || activeState === 'COMPLETED' ? 'not-allowed' : 'pointer'
              }}
            >
              Step Window (+1)
            </button>
            <button
              onClick={() => setAutoProgress(!autoProgress)}
              disabled={!isRunning || activeState === 'COMPLETED'}
              style={{
                flex: 1,
                background: autoProgress ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                border: `1px solid ${autoProgress ? '#EF4444' : '#10B981'}`,
                color: autoProgress ? '#F87171' : '#34D399',
                padding: '10px',
                borderRadius: '8px',
                fontWeight: 600,
                fontSize: '0.84rem',
                cursor: !isRunning || activeState === 'COMPLETED' ? 'not-allowed' : 'pointer'
              }}
            >
              {autoProgress ? 'Pause Auto' : 'Auto Stream'}
            </button>
          </div>
        </div>
      </div>

      {/* Metadata & Session Logs */}
      <div style={{
        background: 'rgba(15, 23, 42, 0.85)',
        border: '1px solid rgba(51, 65, 85, 0.6)',
        borderRadius: '14px',
        padding: '20px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
          <FileText size={18} color="#F59E0B" />
          <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#F1F5F9' }}>
            Saved Calibration Sessions & Base Model Artifact Status
          </h3>
        </div>

        {lastMessage && (
          <div style={{
            background: 'rgba(59, 130, 246, 0.1)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            color: '#93C5FD',
            fontSize: '0.82rem',
            padding: '8px 12px',
            borderRadius: '8px',
            marginBottom: '14px'
          }}>
            {lastMessage}
          </div>
        )}

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '12px'
        }}>
          {recentSessions.length > 0 ? (
            recentSessions.map((s, idx) => (
              <div key={idx} style={{
                background: 'rgba(30, 41, 59, 0.5)',
                border: '1px solid rgba(51, 65, 85, 0.5)',
                borderRadius: '8px',
                padding: '12px',
                fontSize: '0.78rem'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94A3B8', marginBottom: '6px' }}>
                  <span style={{ fontFamily: 'monospace', color: '#38BDF8' }}>{s.session_id}</span>
                  <span>{s.subject_id || 'S001'}</span>
                </div>
                <div style={{ color: '#E2E8F0', marginBottom: '4px' }}>
                  Target: <strong>{s.target_windows_per_class}</strong> windows/class
                </div>
                <div style={{ display: 'flex', gap: '8px', color: '#64748B' }}>
                  <span>REST: {s.collected_counts?.REST || 0}</span>
                  <span>LEFT: {s.collected_counts?.LEFT_FIST || 0}</span>
                  <span>RIGHT: {s.collected_counts?.RIGHT_FIST || 0}</span>
                </div>
                <div style={{ marginTop: '6px', fontSize: '0.72rem', color: '#10B981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <ShieldCheck size={14} /> Base model preserved intact
                </div>
              </div>
            ))
          ) : (
            <div style={{ color: '#64748B', fontSize: '0.84rem' }}>No recent calibration sessions saved yet.</div>
          )}
        </div>
      </div>

    </div>
  );
}
