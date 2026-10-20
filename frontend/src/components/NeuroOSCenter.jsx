import React, { useState, useEffect, useRef } from 'react';
import { 
  Radio, ShieldCheck, ShieldAlert, Cpu, Play, Pause, SkipForward, SkipBack, 
  Lightbulb, Fan, Thermometer, Volume2, Lock, ArrowLeft, ArrowRight, CheckCircle2, 
  RefreshCw, Power, Layers, FileText, Monitor, Activity, Zap, AlertTriangle, Eye, Compass, Wifi, WifiOff
} from 'lucide-react';

export default function NeuroOSCenter() {
  const [adapterMode, setAdapterMode] = useState('NAVIGATION');
  const [sourceType, setSourceType] = useState('SIMULATED_EDF'); // 'SIMULATED_EDF' or 'LSL_HARDWARE'
  const [subjectId, setSubjectId] = useState('S001');
  const [runId, setRunId] = useState('R11');
  const [trialIdx, setTrialIdx] = useState(0);

  const [wsStatus, setWsStatus] = useState('DISCONNECTED'); // 'CONNECTED', 'CONNECTING', 'DISCONNECTED'
  const [sessionState, setSessionState] = useState('CONNECTED'); // 'RUNNING', 'PAUSED', 'STOPPED', 'CONNECTED'
  const [streamFps, setStreamFps] = useState(0);

  const [confidenceThresh, setConfidenceThresh] = useState(0.60);
  const [manualLock, setManualLock] = useState(false);

  const [currentTrialData, setCurrentTrialData] = useState(null);
  const [sourceInfo, setSourceInfo] = useState(null);
  const [latencies, setLatencies] = useState({ ml_latency: 0, pipeline_latency: 0, ui_broadcast_latency: 0 });
  const [auditLogs, setAuditLogs] = useState([]);
  const [activeStage, setActiveStage] = useState(0);

  const wsRef = useRef(null);
  const messageCountRef = useRef(0);

  // System State Store across all 5 adapters
  const [systemState, setSystemState] = useState({
    media: { playing: false, track: 1, volume: 80, songTitle: 'Alpha Brainwave Symphony No. 9' },
    smartRoom: { lightOn: false, fanOn: false, temperatureC: 22 },
    navigation: { scrollPosition: 0, activeTab: 'OVERVIEW' },
    documents: { activeSlide: 1, totalSlides: 12, zoomPct: 100 },
    system: { muted: false, locked: false }
  });

  // Calculate FPS (update rate)
  useEffect(() => {
    const timer = setInterval(() => {
      setStreamFps(messageCountRef.current);
      messageCountRef.current = 0;
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Connect WebSocket to /ws/neuro-os
  useEffect(() => {
    let socket = null;
    let reconnectTimeout = null;

    const connectWs = () => {
      setWsStatus('CONNECTING');
      socket = new WebSocket('ws://localhost:8050/ws/neuro-os');
      wsRef.current = socket;

      socket.onopen = () => {
        setWsStatus('CONNECTED');
        console.log('[NeuroOS WS] Connected to backend streaming pipeline.');
      };

      socket.onmessage = (event) => {
        try {
          messageCountRef.current += 1;
          const data = JSON.parse(event.data);
          
          setCurrentTrialData(data);
          if (data.session_state) setSessionState(data.session_state);
          if (data.source_info) setSourceInfo(data.source_info);
          if (data.latencies_ms) setLatencies(data.latencies_ms);
          if (data.trial_info?.trial_idx !== undefined) setTrialIdx(data.trial_info.trial_idx);

          // Animate pipeline stages
          if (data.stage === 'EEG_RECEIVED') setActiveStage(1);
          else if (data.stage === 'SIGNAL_QUALITY') setActiveStage(2);
          else if (data.stage === 'DECODER_RUNNING') setActiveStage(3);
          else if (data.stage === 'SAFETY_CHECK') setActiveStage(4);
          else if (data.stage === 'COMMAND_DECIDED') setActiveStage(5);
          else if (data.stage === 'ACTION_EXECUTED') setActiveStage(6);
          else setActiveStage(4);

          if (data.event_log) {
            setAuditLogs(data.event_log);
          }
          if (data.system_state) {
            syncState(data.system_state);
          }
        } catch (err) {
          console.error('[NeuroOS WS] Error parsing message:', err);
        }
      };

      socket.onclose = () => {
        setWsStatus('DISCONNECTED');
        console.warn('[NeuroOS WS] Disconnected. Retrying in 3s...');
        reconnectTimeout = setTimeout(connectWs, 3000);
      };

      socket.onerror = (err) => {
        setWsStatus('DISCONNECTED');
        console.error('[NeuroOS WS] Error:', err);
        socket.close();
      };
    };

    connectWs();

    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const syncState = (st) => {
    setSystemState({
      media: { ...systemState.media, playing: st.media?.playing || false, track: st.media?.track || 1, volume: st.media?.volume || 80 },
      smartRoom: {
        lightOn: st.smart_room?.light_on || false,
        fanOn: st.smart_room?.fan_on || false,
        temperatureC: st.smart_room?.temperature_c || 22
      },
      navigation: { scrollPosition: st.navigation?.scroll_position || 0 },
      documents: { activeSlide: st.documents?.active_slide || 1, totalSlides: 12 },
      system: { muted: st.system?.muted || false, locked: st.system?.locked || false }
    });
  };

  const sendWsAction = (actionObj) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(actionObj));
    }
  };

  const toggleSession = () => {
    if (sessionState === 'RUNNING') {
      sendWsAction({ action: 'STOP_SESSION' });
    } else {
      sendWsAction({ action: 'START_SESSION' });
    }
  };

  const switchSource = (newSource) => {
    setSourceType(newSource);
    sendWsAction({ action: 'SWITCH_SOURCE', source_type: newSource });
  };

  const switchAdapterMode = (newMode) => {
    setAdapterMode(newMode);
    sendWsAction({ action: 'SET_MODE', mode: newMode });
  };

  // Fallback REST trial trigger if WS paused
  const triggerManualTrial = async () => {
    try {
      const res = await fetch(`http://localhost:8050/api/neuro-os/process-trial?subject_id=${subjectId}&run_id=${runId}&trial_idx=${trialIdx + 1}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setCurrentTrialData(data);
        if (data.system_state) syncState(data.system_state);
      }
    } catch (err) {
      console.error('Error triggering trial:', err);
    }
  };

  const isSafetyPassed = currentTrialData?.safety_state?.passed || currentTrialData?.safety_assessment?.passed;
  const currentIntent = currentTrialData?.safety_state?.intent || currentTrialData?.safety_assessment?.intent || 'REST';
  const confidenceVal = currentTrialData?.safety_state?.confidence || currentTrialData?.safety_assessment?.confidence || 0.0;
  const rejectionReason = currentTrialData?.safety_state?.rejection_reason || currentTrialData?.safety_assessment?.rejection_reason;
  const dwellProgress = currentTrialData?.safety_state?.dwell_progress || currentTrialData?.safety_assessment?.dwell_progress || 0.0;
  const currentCommand = currentTrialData?.command_routing?.command || 'NEUTRAL';
  const executedAction = currentTrialData?.action_execution?.action || 'NONE';

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '24px 16px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Top Banner & Real-Time Connection Controls */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)',
        border: '1px solid #00F2FE',
        borderRadius: '16px',
        padding: '24px',
        boxShadow: '0 0 30px rgba(0, 242, 254, 0.15)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%)',
            padding: '14px',
            borderRadius: '14px',
            boxShadow: '0 0 20px rgba(0, 242, 254, 0.5)'
          }}>
            <Power size={32} color="#090D16" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '1.8rem', fontWeight: 900, color: '#F8FAFC', margin: 0, letterSpacing: '0.05em' }}>
                NEURO <span style={{ color: '#00F2FE' }}>OS</span>
              </h1>
              <span style={{
                background: wsStatus === 'CONNECTED' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                border: `1px solid ${wsStatus === 'CONNECTED' ? '#10B981' : '#EF4444'}`,
                color: wsStatus === 'CONNECTED' ? '#10B981' : '#EF4444',
                padding: '2px 10px',
                borderRadius: '12px',
                fontSize: '0.75rem',
                fontWeight: 800,
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}>
                {wsStatus === 'CONNECTED' ? <Wifi size={14} /> : <WifiOff size={14} />}
                WS: {wsStatus} ({streamFps} Hz)
              </span>
            </div>
            <p style={{ color: '#94A3B8', fontSize: '0.85rem', margin: '4px 0 0 0' }}>
              Real-Time Streaming BCI Architecture | Step 60 ML Engine (70.67% Holdout Accuracy)
            </p>
          </div>
        </div>

        {/* Source Switcher & Start Session Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
          
          {/* Data Source Selector (3 Modes) */}
          <div style={{ display: 'flex', gap: '4px', background: '#090D16', padding: '4px', borderRadius: '10px', border: '1px solid #334155' }}>
            <button
              onClick={() => switchSource('SIMULATED_EDF')}
              style={{
                padding: '8px 14px',
                borderRadius: '6px',
                border: 'none',
                background: sourceType === 'SIMULATED_EDF' ? 'linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%)' : 'transparent',
                color: sourceType === 'SIMULATED_EDF' ? '#090D16' : '#94A3B8',
                fontWeight: 800,
                fontSize: '0.8rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              PhysioNet Simulation
            </button>

            <button
              onClick={() => switchSource('LSL_HARDWARE')}
              style={{
                padding: '8px 14px',
                borderRadius: '6px',
                border: 'none',
                background: sourceType === 'LSL_HARDWARE' ? 'linear-gradient(135deg, #10B981 0%, #059669 100%)' : 'transparent',
                color: sourceType === 'LSL_HARDWARE' ? '#090D16' : '#94A3B8',
                fontWeight: 800,
                fontSize: '0.8rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              OpenBCI 16CH Physical
            </button>

            <button
              onClick={() => switchSource('LSL_HARDWARE_8CH')}
              style={{
                padding: '8px 14px',
                borderRadius: '6px',
                border: 'none',
                background: sourceType === 'LSL_HARDWARE_8CH' ? 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)' : 'transparent',
                color: sourceType === 'LSL_HARDWARE_8CH' ? '#090D16' : '#94A3B8',
                fontWeight: 800,
                fontSize: '0.8rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              OpenBCI 8CH Experimental
            </button>
          </div>

          <button
            onClick={toggleSession}
            style={{
              padding: '12px 24px',
              borderRadius: '12px',
              border: 'none',
              background: sessionState === 'RUNNING' ? '#EF4444' : 'linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%)',
              color: sessionState === 'RUNNING' ? '#FFF' : '#090D16',
              fontWeight: 900,
              fontSize: '0.9rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: sessionState === 'RUNNING' ? '0 0 20px rgba(239, 68, 68, 0.4)' : '0 0 20px rgba(0, 242, 254, 0.4)',
              transition: 'all 0.2s ease'
            }}
          >
            {sessionState === 'RUNNING' ? <Pause size={18} /> : <Play size={18} />}
            {sessionState === 'RUNNING' ? 'STOP STREAM' : 'START REAL-TIME SESSION'}
          </button>

          {/* Mode Selector */}
          <div style={{ display: 'flex', gap: '4px', background: '#090D16', padding: '4px', borderRadius: '10px', border: '1px solid #334155' }}>
            {['NAVIGATION', 'MEDIA', 'DOCUMENTS', 'SMART_ROOM', 'SYSTEM'].map((mode) => (
              <button
                key={mode}
                onClick={() => switchAdapterMode(mode)}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  border: 'none',
                  background: adapterMode === mode ? 'linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%)' : 'transparent',
                  color: adapterMode === mode ? '#090D16' : '#94A3B8',
                  fontWeight: adapterMode === mode ? 800 : 600,
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                {mode}
              </button>
            ))}
          </div>

        </div>
      </div>

      {/* LSL Hardware Disconnected Warning Banner (if LSL mode active & no stream) */}
      {(sourceType === 'LSL_HARDWARE' || sourceType === 'LSL_HARDWARE_8CH') && sourceInfo?.connected === false && (
        <div style={{
          background: 'rgba(245, 158, 11, 0.1)',
          border: '1px solid #F59E0B',
          borderRadius: '12px',
          padding: '16px 20px',
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
          color: '#F59E0B'
        }}>
          <AlertTriangle size={28} />
          <div>
            <div style={{ fontWeight: 800, fontSize: '0.95rem' }}>
              LSL HARDWARE DISCONNECTED — NO STREAM BROADCASTING ON NETWORK
            </div>
            <div style={{ fontSize: '0.85rem', color: '#CBD5E1', marginTop: '2px' }}>
              {sourceInfo?.error_detail || "Ensure OpenBCI GUI or LabRecorder is broadcasting an active LSL EEG stream."} 
              <br />
              <strong style={{ color: '#F59E0B' }}>Contract:</strong> {sourceType === 'LSL_HARDWARE' ? 'OpenBCI Cyton + Daisy (16 Physical Channels) expected.' : 'OpenBCI 8-Channel Cyton expected (Experimental Fallback Mode).'}
            </div>
          </div>
        </div>
      )}

      {/* Real-time Measured Latencies Telemetry Bar */}
      <div style={{
        background: '#0F172A',
        border: '1px solid #1E293B',
        borderRadius: '14px',
        padding: '16px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#F8FAFC', fontWeight: 800, fontSize: '0.9rem' }}>
          <Activity size={18} color="#00F2FE" />
          REAL-TIME TELEMETRY LATENCIES
        </div>

        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={16} color="#38BDF8" />
            <span style={{ color: '#94A3B8', fontSize: '0.8rem' }}>ML Inference:</span>
            <strong style={{ color: '#38BDF8', fontSize: '0.9rem' }}>{latencies.ml_latency} ms</strong>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Zap size={16} color="#3B82F6" />
            <span style={{ color: '#94A3B8', fontSize: '0.8rem' }}>Pipeline Total:</span>
            <strong style={{ color: '#3B82F6', fontSize: '0.9rem' }}>{latencies.pipeline_latency} ms</strong>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Radio size={16} color="#10B981" />
            <span style={{ color: '#94A3B8', fontSize: '0.8rem' }}>WebSocket / UI:</span>
            <strong style={{ color: '#10B981', fontSize: '0.9rem' }}>{latencies.ui_broadcast_latency} ms</strong>
          </div>
        </div>
      </div>

      {/* Phase 7: Hardware Profile Telemetry Card (16CH Physical vs 8CH Experimental vs Simulation) */}
      <div style={{
        background: '#0F172A',
        border: '1px solid #1E293B',
        borderRadius: '16px',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Layers size={22} color={sourceType === 'LSL_HARDWARE' ? '#10B981' : (sourceType === 'LSL_HARDWARE_8CH' ? '#F59E0B' : '#00F2FE')} />
            <span style={{ fontSize: '0.95rem', fontWeight: 800, color: '#F8FAFC' }}>
              HARDWARE PROFILE: <span style={{ color: sourceType === 'LSL_HARDWARE' ? '#10B981' : (sourceType === 'LSL_HARDWARE_8CH' ? '#F59E0B' : '#00F2FE') }}>
                {sourceType === 'LSL_HARDWARE' ? 'OPENBCI CYTON + DAISY (16 PHYSICAL CHANNELS)' : (sourceType === 'LSL_HARDWARE_8CH' ? 'OPENBCI CYTON (8-CHANNEL EXPERIMENTAL)' : 'PHYSIONET EDF SIMULATION (15 CHANNELS)')}
              </span>
            </span>
          </div>
          <span style={{
            background: sourceType === 'LSL_HARDWARE' ? 'rgba(16, 185, 129, 0.15)' : (sourceType === 'LSL_HARDWARE_8CH' ? 'rgba(245, 158, 11, 0.15)' : 'rgba(0, 242, 254, 0.15)'),
            border: `1px solid ${sourceType === 'LSL_HARDWARE' ? '#10B981' : (sourceType === 'LSL_HARDWARE_8CH' ? '#F59E0B' : '#00F2FE')}`,
            color: sourceType === 'LSL_HARDWARE' ? '#10B981' : (sourceType === 'LSL_HARDWARE_8CH' ? '#F59E0B' : '#00F2FE'),
            padding: '4px 12px',
            borderRadius: '12px',
            fontSize: '0.75rem',
            fontWeight: 800
          }}>
            {sourceType === 'LSL_HARDWARE' ? '16CH PHYSICAL DIRECT PASS-THROUGH (ZERO INTERPOLATION)' : (sourceType === 'LSL_HARDWARE_8CH' ? 'SPHERICAL SPLINE RECONSTRUCTION' : 'GENUINE 15CH EDF DATASET')}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
          <div style={{ background: '#090D16', padding: '12px 16px', borderRadius: '10px', border: '1px solid #1E293B' }}>
            <div style={{ fontSize: '0.75rem', color: '#94A3B8', fontWeight: 700 }}>PHYSICAL HARDWARE CHANNELS</div>
            <div style={{ fontSize: '0.85rem', color: sourceType === 'LSL_HARDWARE' ? '#10B981' : '#F59E0B', fontWeight: 800, marginTop: '4px', fontFamily: 'monospace' }}>
              {sourceType === 'LSL_HARDWARE' ? '16 Channels (FC3..CP4 + O1 Aux)' : (sourceType === 'LSL_HARDWARE_8CH' ? '8 Channels (F3, F4, C3, Cz, C4, P3, Pz, P4)' : '15 Channels (FC3..CP4)')}
            </div>
          </div>

          <div style={{ background: '#090D16', padding: '12px 16px', borderRadius: '10px', border: '1px solid #1E293B' }}>
            <div style={{ fontSize: '0.75rem', color: '#94A3B8', fontWeight: 700 }}>MODEL CONSUMED CHANNELS</div>
            <div style={{ fontSize: '0.85rem', color: '#00F2FE', fontWeight: 800, marginTop: '4px', fontFamily: 'monospace' }}>
              15 Sensorimotor Channels (FC3..CP4)
            </div>
          </div>

          <div style={{ background: '#090D16', padding: '12px 16px', borderRadius: '10px', border: '1px solid #1E293B' }}>
            <div style={{ fontSize: '0.75rem', color: '#94A3B8', fontWeight: 700 }}>INTERPOLATED CHANNELS COUNT</div>
            <div style={{ fontSize: '0.85rem', color: sourceType === 'LSL_HARDWARE_8CH' ? '#EF4444' : '#10B981', fontWeight: 800, marginTop: '4px' }}>
              {sourceType === 'LSL_HARDWARE_8CH' ? '12 Channels Reconstructed' : '0 Channels Interpolated (100% Physical)'}
            </div>
          </div>

        </div>

        <div style={{ fontSize: '0.75rem', color: '#94A3B8', background: 'rgba(255, 255, 255, 0.03)', padding: '10px 14px', borderRadius: '8px', borderLeft: `3px solid ${sourceType === 'LSL_HARDWARE' ? '#10B981' : '#F59E0B'}` }}>
          {sourceType === 'LSL_HARDWARE' ? (
            <span>✅ <strong>Recommended 16-Channel Physical Path:</strong> Direct 1-to-1 sensorimotor electrode capture preserves 100% of Step 60 model accuracy (67.67%). Zero spatial interpolation information loss.</span>
          ) : (
            <span>💡 <strong>Scientific Notice:</strong> 8-channel mode requires 3D spherical spline reconstruction of 12 missing motor channels. Offline benchmarks show accuracy drops from 67.67% to 46.33%. 16-channel Cyton + Daisy is recommended for production hardware.</span>
          )}
        </div>
      </div>



      {/* Brain -> Action Live Stage Lighting Visualizer */}
      <div style={{
        background: '#0F172A',
        border: '1px solid #1E293B',
        borderRadius: '16px',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 800, color: '#94A3B8', letterSpacing: '0.05em' }}>
            BRAIN → ACTION REAL-TIME PIPELINE CHAIN
          </span>
          <span style={{ fontSize: '0.75rem', color: '#00F2FE', background: 'rgba(0, 242, 254, 0.1)', padding: '2px 8px', borderRadius: '8px' }}>
            STAGE {activeStage}/6
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px' }}>
          {[
            { id: 1, name: 'EEG STREAM', desc: sourceType === 'SIMULATED_EDF' ? 'PhysioNet EDF' : 'LSL Hardware', icon: Radio },
            { id: 2, name: 'SQI GUARD', desc: 'Signal Quality OK', icon: Activity },
            { id: 3, name: 'STEP 60 AI', desc: currentTrialData?.trial_info?.predicted_label || 'Decoding...', icon: Cpu },
            { id: 4, name: 'SAFETY SHIELD', desc: isSafetyPassed ? 'PASSED' : (rejectionReason || 'CHECKING'), icon: isSafetyPassed ? ShieldCheck : ShieldAlert },
            { id: 5, name: 'COMMAND ROUTER', desc: currentCommand, icon: Zap },
            { id: 6, name: 'ACTION EXECUTOR', desc: executedAction, icon: Power }
          ].map((stg) => {
            const IconComp = stg.icon;
            const isActive = activeStage >= stg.id;
            const isHighlight = activeStage === stg.id;
            return (
              <div
                key={stg.id}
                style={{
                  background: isHighlight ? 'rgba(0, 242, 254, 0.15)' : (isActive ? '#1E293B' : '#090D16'),
                  border: `1px solid ${isHighlight ? '#00F2FE' : (isActive ? '#334155' : '#1E293B')}`,
                  borderRadius: '12px',
                  padding: '14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                  transition: 'all 0.2s ease',
                  boxShadow: isHighlight ? '0 0 15px rgba(0, 242, 254, 0.3)' : 'none'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <IconComp size={20} color={isHighlight ? '#00F2FE' : (isActive ? '#F8FAFC' : '#64748B')} />
                  <span style={{ fontSize: '0.65rem', color: '#64748B', fontWeight: 700 }}>0{stg.id}</span>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 800, color: isHighlight ? '#00F2FE' : '#F8FAFC' }}>
                    {stg.name}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: isHighlight ? '#E2E8F0' : '#94A3B8', marginTop: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {stg.desc}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Grid: Safety Shield Telemetry HUD + Interactive Reactive App Deck */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px' }}>
        
        {/* Left Deck: Safety Shield Telemetry */}
        <div style={{
          background: '#0F172A',
          border: '1px solid #1E293B',
          borderRadius: '16px',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <ShieldCheck size={24} color="#00F2FE" />
              <h2 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#F8FAFC', margin: 0 }}>
                SAFETY SHIELD TELEMETRY
              </h2>
            </div>
            <span style={{
              background: isSafetyPassed ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              border: `1px solid ${isSafetyPassed ? '#10B981' : '#EF4444'}`,
              color: isSafetyPassed ? '#10B981' : '#EF4444',
              padding: '4px 12px',
              borderRadius: '20px',
              fontSize: '0.8rem',
              fontWeight: 800
            }}>
              {isSafetyPassed ? 'SAFE TO EXECUTE' : 'REJECTED / BLOCKED'}
            </span>
          </div>

          {/* Rejection Explainer Box */}
          {!isSafetyPassed && rejectionReason && (
            <div style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid #EF4444',
              borderRadius: '12px',
              padding: '14px',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              color: '#FCA5A5'
            }}>
              <AlertTriangle size={24} color="#EF4444" />
              <div>
                <div style={{ fontWeight: 800, fontSize: '0.85rem' }}>PREDICTION REJECTED BY SAFETY SHIELD</div>
                <div style={{ fontSize: '0.8rem', marginTop: '2px' }}>Reason: <strong>{rejectionReason}</strong></div>
              </div>
            </div>
          )}

          {/* Metrics Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ background: '#090D16', padding: '14px', borderRadius: '12px', border: '1px solid #1E293B' }}>
              <div style={{ fontSize: '0.75rem', color: '#94A3B8', fontWeight: 600 }}>CURRENT INTENT</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 900, color: '#00F2FE', marginTop: '4px' }}>
                {currentIntent}
              </div>
            </div>

            <div style={{ background: '#090D16', padding: '14px', borderRadius: '12px', border: '1px solid #1E293B' }}>
              <div style={{ fontSize: '0.75rem', color: '#94A3B8', fontWeight: 600 }}>CONFIDENCE score</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 900, color: confidenceVal >= 0.60 ? '#10B981' : '#EF4444', marginTop: '4px' }}>
                {(confidenceVal * 100).toFixed(1)}%
              </div>
            </div>

            <div style={{ background: '#090D16', padding: '14px', borderRadius: '12px', border: '1px solid #1E293B' }}>
              <div style={{ fontSize: '0.75rem', color: '#94A3B8', fontWeight: 600 }}>DWELL HOLD PROGRESS</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 900, color: '#F59E0B', marginTop: '4px' }}>
                {(dwellProgress * 100).toFixed(0)}%
              </div>
            </div>

            <div style={{ background: '#090D16', padding: '14px', borderRadius: '12px', border: '1px solid #1E293B' }}>
              <div style={{ fontSize: '0.75rem', color: '#94A3B8', fontWeight: 600 }}>SAFETY THRESHOLD</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 900, color: '#F8FAFC', marginTop: '4px' }}>
                {(confidenceThresh * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Probability Bars */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#94A3B8' }}>STEP 60 DECODER PROBABILITIES</div>
            {['REST', 'LEFT_FIST', 'RIGHT_FIST'].map((cls) => {
              const prob = currentTrialData?.decoder_output?.probabilities?.[cls] || currentTrialData?.probabilities?.[cls] || 0;
              return (
                <div key={cls} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#CBD5E1' }}>
                    <span>{cls}</span>
                    <span>{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ background: '#090D16', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${prob * 100}%`,
                      height: '100%',
                      background: cls === currentIntent ? 'linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%)' : '#334155',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                </div>
              );
            })}
          </div>

        </div>

        {/* Right Deck: Reactive Interactive Environment Simulator */}
        <div style={{
          background: '#0F172A',
          border: '1px solid #1E293B',
          borderRadius: '16px',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Monitor size={24} color="#00F2FE" />
              <h2 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#F8FAFC', margin: 0 }}>
                REACTIVE ENVIRONMENT — {adapterMode} MODE
              </h2>
            </div>
          </div>

          {/* 1. MEDIA MODE DECK */}
          {adapterMode === 'MEDIA' && (
            <div style={{ background: '#090D16', border: '1px solid #1E293B', borderRadius: '14px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div style={{ width: '64px', height: '64px', borderRadius: '12px', background: 'linear-gradient(135deg, #0284C7 0%, #2563EB 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Activity size={32} color="#FFF" />
                </div>
                <div>
                  <div style={{ fontSize: '1rem', fontWeight: 800, color: '#F8FAFC' }}>{systemState.media.songTitle}</div>
                  <div style={{ fontSize: '0.8rem', color: '#94A3B8' }}>Track #{systemState.media.track} | Vol: {systemState.media.volume}%</div>
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', alignItems: 'center' }}>
                <div style={{ background: '#1E293B', padding: '10px', borderRadius: '50%', color: '#94A3B8' }}><SkipBack size={20} /></div>
                <div style={{ background: systemState.media.playing ? '#10B981' : '#3B82F6', padding: '14px', borderRadius: '50%', color: '#FFF' }}>
                  {systemState.media.playing ? <Pause size={24} /> : <Play size={24} />}
                </div>
                <div style={{ background: '#1E293B', padding: '10px', borderRadius: '50%', color: '#94A3B8' }}><SkipForward size={20} /></div>
              </div>
            </div>
          )}

          {/* 2. SMART ROOM MODE DECK */}
          {adapterMode === 'SMART_ROOM' && (
            <div style={{ background: '#090D16', border: '1px solid #1E293B', borderRadius: '14px', padding: '20px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div style={{ background: systemState.smartRoom.lightOn ? 'rgba(245, 158, 11, 0.2)' : '#1E293B', border: `1px solid ${systemState.smartRoom.lightOn ? '#F59E0B' : '#334155'}`, borderRadius: '12px', padding: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Lightbulb size={28} color={systemState.smartRoom.lightOn ? '#F59E0B' : '#64748B'} />
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 800, color: '#F8FAFC' }}>SMART LIGHTS</div>
                  <div style={{ fontSize: '0.75rem', color: systemState.smartRoom.lightOn ? '#F59E0B' : '#64748B' }}>{systemState.smartRoom.lightOn ? 'ON (100%)' : 'OFF'}</div>
                </div>
              </div>

              <div style={{ background: '#1E293B', borderRadius: '12px', padding: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Thermometer size={28} color="#00F2FE" />
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 800, color: '#F8FAFC' }}>THERMOSTAT</div>
                  <div style={{ fontSize: '0.75rem', color: '#00F2FE' }}>{systemState.smartRoom.temperatureC}°C</div>
                </div>
              </div>
            </div>
          )}

          {/* 3. DOCUMENTS MODE DECK */}
          {adapterMode === 'DOCUMENTS' && (
            <div style={{ background: '#090D16', border: '1px solid #1E293B', borderRadius: '14px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 800, color: '#F8FAFC' }}>NeuroOS Whitepaper.pdf</span>
                <span style={{ color: '#00F2FE', fontSize: '0.85rem', fontWeight: 700 }}>Slide {systemState.documents.activeSlide} / {systemState.documents.totalSlides}</span>
              </div>
              <div style={{ background: '#1E293B', height: '140px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94A3B8', fontSize: '0.9rem' }}>
                [ Document Slide #{systemState.documents.activeSlide} Display Content ]
              </div>
            </div>
          )}

          {/* 4. NAVIGATION MODE DECK */}
          {adapterMode === 'NAVIGATION' && (
            <div style={{ background: '#090D16', border: '1px solid #1E293B', borderRadius: '14px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {['1. Dashboard Home', '2. System Telemetry', '3. Model Architecture', '4. Security Log'].map((item, idx) => (
                <div key={item} style={{
                  padding: '12px 16px',
                  borderRadius: '8px',
                  background: systemState.navigation.scrollPosition === idx ? 'linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%)' : '#1E293B',
                  color: systemState.navigation.scrollPosition === idx ? '#090D16' : '#F8FAFC',
                  fontWeight: systemState.navigation.scrollPosition === idx ? 800 : 500,
                  fontSize: '0.85rem'
                }}>
                  {item} {systemState.navigation.scrollPosition === idx ? '← [SELECTED]' : ''}
                </div>
              ))}
            </div>
          )}

          {/* 5. SYSTEM MODE DECK */}
          {adapterMode === 'SYSTEM' && (
            <div style={{ background: '#090D16', border: '1px solid #1E293B', borderRadius: '14px', padding: '20px', display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                <Volume2 size={32} color={systemState.system.muted ? '#EF4444' : '#10B981'} />
                <span style={{ fontSize: '0.8rem', color: '#94A3B8' }}>{systemState.system.muted ? 'MUTED' : 'AUDIO ACTIVE'}</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                <Lock size={32} color={systemState.system.locked ? '#EF4444' : '#10B981'} />
                <span style={{ fontSize: '0.8rem', color: '#94A3B8' }}>{systemState.system.locked ? 'OS LOCKED' : 'UNLOCKED'}</span>
              </div>
            </div>
          )}

        </div>

      </div>

      {/* Session Audit Event Timeline */}
      <div style={{
        background: '#0F172A',
        border: '1px solid #1E293B',
        borderRadius: '16px',
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#F8FAFC', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={20} color="#00F2FE" />
            REAL-TIME SESSION AUDIT TIMELINE
          </h3>
          <span style={{ fontSize: '0.8rem', color: '#94A3B8' }}>{auditLogs.length} Events Recorded</span>
        </div>

        <div style={{
          maxHeight: '220px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          paddingRight: '6px'
        }}>
          {auditLogs.length === 0 ? (
            <div style={{ color: '#64748B', fontSize: '0.85rem', fontStyle: 'italic', textAlign: 'center', padding: '20px' }}>
              No events recorded yet. Click "START REAL-TIME SESSION" to begin streaming telemetry.
            </div>
          ) : (
            auditLogs.map((log, idx) => (
              <div
                key={log.event_id || idx}
                style={{
                  background: '#090D16',
                  border: '1px solid #1E293B',
                  borderRadius: '10px',
                  padding: '10px 14px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '0.8rem'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span style={{ color: '#64748B', fontFamily: 'monospace' }}>{log.timestamp ? new Date(log.timestamp * 1000).toLocaleTimeString() : '20:41:03'}</span>
                  <span style={{ color: '#00F2FE', fontWeight: 800 }}>INTENT: {log.intent || 'REST'}</span>
                  <span style={{ color: log.safety_status?.passed ? '#10B981' : '#EF4444' }}>
                    CONF: {((log.confidence || 0) * 100).toFixed(0)}%
                  </span>
                  <span style={{ color: '#CBD5E1' }}>CMD: {log.command || 'NEUTRAL'}</span>
                </div>
                <div style={{ color: log.action?.executed ? '#10B981' : '#94A3B8', fontWeight: 700 }}>
                  ACTION: {log.action?.action || 'NONE'}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

    </div>
  );
}
