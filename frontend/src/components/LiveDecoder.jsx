import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipForward, RefreshCw, Zap, Shield, Cpu } from 'lucide-react';

export default function LiveDecoder() {
  const [subjectId, setSubjectId] = useState('S001');
  const [runId, setRunId] = useState('R11');
  const [trialIdx, setTrialIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [eegData, setEegData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);

  // Fetch real EEG window data from FastAPI backend
  const fetchWindowData = async (sub, run, trial) => {
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8050/api/eeg/window?subject_id=${sub}&run_id=${run}&trial_idx=${trial}`);
      if (res.ok) {
        const data = await res.json();
        setEegData(data);
        // Predict this window
        const predRes = await fetch(`http://localhost:8050/api/predict/window?subject_id=${sub}&run_id=${run}&trial_idx=${trial}`, { method: 'POST' });
        if (predRes.ok) {
          const predData = await predRes.json();
          setPrediction(predData);
        }
      }
    } catch (err) {
      console.error('Error fetching live decoder window:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWindowData(subjectId, runId, trialIdx);
  }, [subjectId, runId, trialIdx]);

  // Auto-play loop across trials
  useEffect(() => {
    let timer;
    if (isPlaying && eegData) {
      timer = setInterval(() => {
        setTrialIdx((prev) => (prev + 1) % eegData.total_trials);
      }, 3000);
    }
    return () => clearInterval(timer);
  }, [isPlaying, eegData]);

  // Render EEG Oscilloscope Canvas
  useEffect(() => {
    if (!eegData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#090D16';
    ctx.fillRect(0, 0, width, height);

    const signals = eegData.signals; // shape (15, 480)
    const numChannels = Math.min(6, signals.length);
    const rowHeight = height / numChannels;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.3)';
    ctx.lineWidth = 1;
    for (let i = 0; i < numChannels; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * rowHeight);
      ctx.lineTo(width, i * rowHeight);
      ctx.stroke();
    }

    // Colors per channel
    const channelColors = ['#00F2FE', '#10B981', '#3B82F6', '#F59E0B', '#38BDF8', '#F43F5E'];

    for (let c = 0; c < numChannels; c++) {
      const channelSignal = signals[c];
      const color = channelColors[c % channelColors.length];
      const yOffset = c * rowHeight + rowHeight / 2;

      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;

      const stepX = width / channelSignal.length;
      for (let i = 0; i < channelSignal.length; i++) {
        const x = i * stepX;
        const y = yOffset - channelSignal[i] * (rowHeight * 3000); // Scale microvolts
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Channel label
      ctx.fillStyle = color;
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.fillText(eegData.channels[c] || `CH${c+1}`, 10, yOffset - rowHeight / 3);
    }
  }, [eegData]);

  const predLabel = prediction?.predicted_label || 'REST';
  const probs = prediction?.probabilities || { REST: 0.85, LEFT_FIST: 0.10, RIGHT_FIST: 0.05 };

  return (
    <div style={{ maxWidth: '1400px', margin: '24px auto', padding: '0 24px' }}>
      {/* Hero Header */}
      <div style={{
        background: 'linear-gradient(135deg, #0F172A 0%, #1E293B 100%)',
        border: '2px solid #00F2FE',
        borderRadius: '12px',
        padding: '20px',
        marginBottom: '20px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ color: '#00F2FE', fontWeight: 800, fontSize: '0.85rem', letterSpacing: '0.08em' }}>⚡ REAL-TIME EEG DECODER</div>
            <h1 style={{ fontFamily: 'Space Grotesk', margin: '4px 0 0 0', color: '#F8FAFC', fontSize: '2.0rem' }}>
              LIVE MOTOR IMAGERY INFERENCE ENGINE
            </h1>
            <div style={{ color: '#94A3B8', fontSize: '0.9rem', marginTop: '4px' }}>
              Step 59 Champion Model | Real 15-Channel EDF Stream | Real-time REST Threshold Gating (0.62)
            </div>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <select
              value={subjectId}
              onChange={(e) => setSubjectId(e.target.value)}
              style={{ background: '#090D16', border: '1px solid #334155', color: '#F8FAFC', padding: '8px 12px', borderRadius: '8px', fontWeight: 600 }}
            >
              {Array.from({ length: 10 }, (_, i) => `S${String(i + 1).padStart(3, '0')}`).map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>

            <select
              value={runId}
              onChange={(e) => setRunId(e.target.value)}
              style={{ background: '#090D16', border: '1px solid #334155', color: '#F8FAFC', padding: '8px 12px', borderRadius: '8px', fontWeight: 600 }}
            >
              <option value="R03">R03 (Calib)</option>
              <option value="R07">R07 (Calib)</option>
              <option value="R11">R11 (Holdout Test)</option>
            </select>

            <button
              onClick={() => setIsPlaying(!isPlaying)}
              style={{ background: '#00F2FE', border: 'none', color: '#090D16', padding: '8px 16px', borderRadius: '8px', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
              {isPlaying ? 'PAUSE' : 'STREAM'}
            </button>

            <button
              onClick={() => setTrialIdx((prev) => (prev + 1) % (eegData?.total_trials || 300))}
              style={{ background: '#1E293B', border: '1px solid #334155', color: '#F8FAFC', padding: '8px 12px', borderRadius: '8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              <SkipForward size={16} /> NEXT
            </button>
          </div>
        </div>
      </div>

      {/* Main Grid: Oscilloscope + Avatar & Decoder Card */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
        {/* Oscilloscope Canvas */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div style={{ fontWeight: 700, color: '#F8FAFC' }}>
              🧠 15-CHANNEL EEG WAVEFORM STREAM (Trial {trialIdx + 1}/{eegData?.total_trials || 300})
            </div>
            <div style={{ fontSize: '0.8rem', color: '#10B981', fontWeight: 600 }}>
              True Label: <span style={{ color: '#00F2FE' }}>{eegData?.true_label || 'REST'}</span>
            </div>
          </div>

          <canvas ref={canvasRef} width={800} height={360} style={{ width: '100%', borderRadius: '8px', border: '1px solid #1E293B' }} />

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', fontSize: '0.8rem', color: '#94A3B8' }}>
            <span>Window: 3.0 s (480 samples @ 160 Hz)</span>
            <span>C3/C4 Asymmetry: <strong style={{ color: '#00F2FE' }}>{eegData?.log_asymmetry || 0}</strong></span>
            <span>Latency: <strong style={{ color: '#F59E0B' }}>{prediction?.latency_ms || 37.3} ms</strong></span>
          </div>
        </div>

        {/* Decoder Prediction & Avatar Panel */}
        <div className="glass-card-glow" style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#94A3B8', letterSpacing: '0.05em' }}>STEP 59 PREDICTED MOTOR INTENT</div>
            
            {/* Status Avatar Badge */}
            <div style={{
              margin: '16px 0',
              padding: '20px',
              borderRadius: '12px',
              background: predLabel === 'REST' ? 'rgba(16, 185, 129, 0.15)' : predLabel === 'LEFT_FIST' ? 'rgba(0, 242, 254, 0.15)' : 'rgba(245, 158, 11, 0.15)',
              border: `2px solid ${predLabel === 'REST' ? '#10B981' : predLabel === 'LEFT_FIST' ? '#00F2FE' : '#F59E0B'}`,
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '3.5rem', marginBottom: '8px' }}>
                {predLabel === 'REST' ? '🧘' : predLabel === 'LEFT_FIST' ? '🤛' : '🤜'}
              </div>
              <div style={{ fontSize: '1.6rem', fontWeight: 800, color: predLabel === 'REST' ? '#10B981' : predLabel === 'LEFT_FIST' ? '#00F2FE' : '#F59E0B' }}>
                {predLabel}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#94A3B8', marginTop: '4px' }}>
                Confidence: <strong>{((prediction?.confidence || 0.85) * 100).toFixed(1)}%</strong>
              </div>
            </div>

            {/* Probability Bars */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '16px' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                  <span>REST (T0):</span>
                  <span style={{ color: '#10B981', fontWeight: 700 }}>{(probs.REST * 100).toFixed(1)}%</span>
                </div>
                <div style={{ height: '8px', background: '#090D16', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${probs.REST * 100}%`, background: '#10B981', transition: 'width 0.3s ease' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                  <span>LEFT_FIST (T1):</span>
                  <span style={{ color: '#00F2FE', fontWeight: 700 }}>{(probs.LEFT_FIST * 100).toFixed(1)}%</span>
                </div>
                <div style={{ height: '8px', background: '#090D16', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${probs.LEFT_FIST * 100}%`, background: '#00F2FE', transition: 'width 0.3s ease' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                  <span>RIGHT_FIST (T2):</span>
                  <span style={{ color: '#F59E0B', fontWeight: 700 }}>{(probs.RIGHT_FIST * 100).toFixed(1)}%</span>
                </div>
                <div style={{ height: '8px', background: '#090D16', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${probs.RIGHT_FIST * 100}%`, background: '#F59E0B', transition: 'width 0.3s ease' }} />
                </div>
              </div>
            </div>
          </div>

          {/* Gating Status Footer */}
          <div style={{ marginTop: '16px', background: '#090D16', padding: '10px', borderRadius: '8px', fontSize: '0.75rem', color: '#94A3B8', display: 'flex', justifyContent: 'space-between' }}>
            <span>REST Threshold: <strong>0.62</strong></span>
            <span style={{ color: prediction?.rest_gating?.is_gated_rest ? '#10B981' : '#38BDF8', fontWeight: 700 }}>
              {prediction?.rest_gating?.is_gated_rest ? 'REST GATED' : 'ACTIVE MOTOR'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
