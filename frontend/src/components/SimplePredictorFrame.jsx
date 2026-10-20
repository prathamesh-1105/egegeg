import React, { useState, useEffect, useRef } from 'react';
import { ChevronRight, ChevronLeft, Play, Pause, CheckCircle2, XCircle, Zap, RefreshCw, Cpu, Brain, User } from 'lucide-react';
import Brain3DCanvas from './Brain3DCanvas';
import Avatar3DCanvas from './Avatar3DCanvas';

export default function SimplePredictorFrame() {
  const [subjectId, setSubjectId] = useState('S001');
  const [runId, setRunId] = useState('R11');
  const [trialIdx, setTrialIdx] = useState(0);
  const [eegData, setEegData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [visMode, setVisMode] = useState('dual');
  
  // Track session stats
  const [stats, setStats] = useState({ total: 0, correct: 0 });

  const canvasRef = useRef(null);

  // Normalizer for labels: REST, LEFT_FIST -> LEFT, RIGHT_FIST -> RIGHT
  const formatLabel = (label) => {
    if (!label) return 'LOADING...';
    const clean = label.toUpperCase();
    if (clean.includes('REST')) return 'REST';
    if (clean.includes('LEFT')) return 'LEFT';
    if (clean.includes('RIGHT')) return 'RIGHT';
    return clean;
  };

  // Color mapping for classes
  const getLabelColor = (label) => {
    const formatted = formatLabel(label);
    if (formatted === 'REST') return { color: '#10B981', bg: 'rgba(16, 185, 129, 0.15)', border: '#10B981' };
    if (formatted === 'LEFT') return { color: '#38BDF8', bg: 'rgba(56, 189, 248, 0.15)', border: '#38BDF8' };
    if (formatted === 'RIGHT') return { color: '#3B82F6', bg: 'rgba(59, 130, 246, 0.15)', border: '#3B82F6' };
    return { color: '#94A3B8', bg: 'rgba(148, 163, 184, 0.15)', border: '#94A3B8' };
  };

  // Fetch EEG data and run prediction for trialIdx
  const loadAndPredictTrial = async (sub, run, idx) => {
    setLoading(true);
    try {
      // 1. Fetch EEG Window Data
      const eegRes = await fetch(`http://localhost:8050/api/eeg/window?subject_id=${sub}&run_id=${run}&trial_idx=${idx}`);
      if (!eegRes.ok) throw new Error('Failed to fetch EEG data');
      const eegJson = await eegRes.json();
      setEegData(eegJson);

      // 2. Fetch Prediction
      const predRes = await fetch(`http://localhost:8050/api/predict/window?subject_id=${sub}&run_id=${run}&trial_idx=${idx}`, {
        method: 'POST',
      });
      if (!predRes.ok) throw new Error('Failed to fetch prediction');
      const predJson = await predRes.json();
      setPrediction(predJson);

      // Update session statistics
      setStats((prev) => ({
        total: prev.total + 1,
        correct: prev.correct + (predJson.is_correct ? 1 : 0),
      }));
    } catch (err) {
      console.error('Error fetching EEG trial/prediction:', err);
    } finally {
      setLoading(false);
    }
  };

  // Trigger load when subject, run, or trialIdx changes
  useEffect(() => {
    loadAndPredictTrial(subjectId, runId, trialIdx);
  }, [subjectId, runId, trialIdx]);

  // Auto-play interval
  useEffect(() => {
    let timer;
    if (isPlaying && eegData) {
      timer = setInterval(() => {
        setTrialIdx((prev) => (prev + 1) % eegData.total_trials);
      }, 2500);
    }
    return () => clearInterval(timer);
  }, [isPlaying, eegData]);

  // Draw simple oscilloscope canvas
  useEffect(() => {
    if (!eegData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0B1120';
    ctx.fillRect(0, 0, width, height);

    const signals = eegData.signals; // shape (15, 480)
    if (!signals || signals.length === 0) return;

    const numChannels = Math.min(5, signals.length);
    const rowHeight = height / numChannels;

    const colors = ['#00F2FE', '#10B981', '#38BDF8', '#3B82F6', '#F59E0B'];

    for (let c = 0; c < numChannels; c++) {
      const chData = signals[c];
      const color = colors[c % colors.length];
      const yOffset = c * rowHeight + rowHeight / 2;

      // Draw grid line
      ctx.strokeStyle = 'rgba(51, 65, 85, 0.2)';
      ctx.beginPath();
      ctx.moveTo(0, c * rowHeight);
      ctx.lineTo(width, c * rowHeight);
      ctx.stroke();

      // Draw channel line
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;

      const stepX = width / chData.length;
      for (let i = 0; i < chData.length; i++) {
        const x = i * stepX;
        const y = yOffset - chData[i] * (rowHeight * 2500);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.fillText(eegData.channels[c] || `CH${c + 1}`, 8, yOffset - rowHeight / 3);
    }
  }, [eegData]);

  const handleNext = () => {
    if (!eegData) return;
    setTrialIdx((prev) => (prev + 1) % eegData.total_trials);
  };

  const handlePrev = () => {
    if (!eegData) return;
    setTrialIdx((prev) => (prev - 1 + eegData.total_trials) % eegData.total_trials);
  };

  const resetStats = () => {
    setStats({ total: 0, correct: 0 });
  };

  const trueFormatted = formatLabel(prediction?.true_label || eegData?.true_label);
  const predFormatted = formatLabel(prediction?.predicted_label);
  const isCorrect = prediction?.is_correct ?? (trueFormatted === predFormatted);
  const confidencePct = prediction ? (prediction.confidence * 100).toFixed(1) : '0.0';

  const probs = prediction?.probabilities || { REST: 0.33, LEFT_FIST: 0.33, RIGHT_FIST: 0.33 };

  const trueStyle = getLabelColor(trueFormatted);
  const predStyle = getLabelColor(predFormatted);

  return (
    <div style={{ maxWidth: '1200px', margin: '30px auto', padding: '0 20px', fontFamily: 'Inter, system-ui, sans-serif' }}>
      
      {/* Top Header Bar */}
      <div style={{
        display: 'flex',
        justify: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        background: '#0F172A',
        padding: '16px 24px',
        borderRadius: '12px',
        border: '1px solid #1E293B'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'rgba(0, 242, 254, 0.15)',
            border: '1px solid #00F2FE',
            borderRadius: '8px',
            padding: '8px',
            display: 'flex'
          }}>
            <Cpu size={24} color="#00F2FE" />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.25rem', color: '#F8FAFC', fontWeight: 700 }}>
              EEG Motor Imagery Predictor
            </h2>
            <div style={{ fontSize: '0.8rem', color: '#94A3B8' }}>
              Subject-Adaptive FBCSP Engine (Step 59 Model)
            </div>
          </div>
        </div>

        {/* Selectors */}
        <div style={{ display: 'flex', gap: '10px' }}>
          <select
            value={subjectId}
            onChange={(e) => { setSubjectId(e.target.value); setTrialIdx(0); resetStats(); }}
            style={{
              background: '#090D16',
              border: '1px solid #334155',
              color: '#F8FAFC',
              padding: '6px 12px',
              borderRadius: '8px',
              fontWeight: 600,
              fontSize: '0.85rem'
            }}
          >
            {Array.from({ length: 10 }, (_, i) => `S${String(i + 1).padStart(3, '0')}`).map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>

          <select
            value={runId}
            onChange={(e) => { setRunId(e.target.value); setTrialIdx(0); resetStats(); }}
            style={{
              background: '#090D16',
              border: '1px solid #334155',
              color: '#F8FAFC',
              padding: '6px 12px',
              borderRadius: '8px',
              fontWeight: 600,
              fontSize: '0.85rem'
            }}
          >
            <option value="R11">R11 (Test Holdout)</option>
            <option value="R03">R03 (Calib 1)</option>
            <option value="R07">R07 (Calib 2)</option>
          </select>
        </div>
      </div>

      {/* Main Single Frame Card */}
      <div style={{
        background: 'rgba(15, 23, 42, 0.9)',
        backdropFilter: 'blur(16px)',
        borderRadius: '16px',
        border: '1px solid #334155',
        boxShadow: '0 12px 40px rgba(0, 0, 0, 0.5)',
        padding: '30px',
        position: 'relative',
        overflow: 'hidden'
      }}>

        {/* Top Status Bar: Trial Index & Session Accuracy */}
        <div style={{
          display: 'flex',
          justify: 'space-between',
          alignItems: 'center',
          marginBottom: '24px',
          paddingBottom: '16px',
          borderBottom: '1px solid #1E293B'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem', color: '#94A3B8' }}>
            <span style={{ background: '#1E293B', padding: '4px 10px', borderRadius: '6px', color: '#00F2FE', fontWeight: 700 }}>
              Trial #{trialIdx + 1} / {eegData?.total_trials || 30}
            </span>
            <span>Subject: <strong style={{ color: '#F8FAFC' }}>{subjectId}</strong></span>
          </div>

          {stats.total > 0 && (
            <div style={{ fontSize: '0.85rem', color: '#94A3B8' }}>
              Session Accuracy:{' '}
              <strong style={{ color: '#10B981' }}>
                {stats.correct}/{stats.total} ({((stats.correct / stats.total) * 100).toFixed(0)}%)
              </strong>
            </div>
          )}
        </div>

        {/* Side-by-Side Actual vs Predicted Comparison Frame */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '20px',
          marginBottom: '28px'
        }}>
          {/* Actual Class Frame */}
          <div style={{
            background: trueStyle.bg,
            border: `2px solid ${trueStyle.border}`,
            borderRadius: '14px',
            padding: '24px',
            textAlign: 'center',
            boxShadow: `0 0 20px ${trueStyle.bg}`
          }}>
            <div style={{ fontSize: '0.75rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#94A3B8', fontWeight: 700, marginBottom: '8px' }}>
              ACTUAL CLASS
            </div>
            <div style={{ fontSize: '2.5rem', fontWeight: 900, color: trueStyle.color, letterSpacing: '0.05em' }}>
              {loading ? '...' : trueFormatted}
            </div>
            <div style={{ fontSize: '0.8rem', color: '#CBD5E1', marginTop: '6px' }}>
              Ground Truth Label
            </div>
          </div>

          {/* Predicted Class Frame */}
          <div style={{
            background: predStyle.bg,
            border: `2px solid ${predStyle.border}`,
            borderRadius: '14px',
            padding: '24px',
            textAlign: 'center',
            boxShadow: `0 0 20px ${predStyle.bg}`
          }}>
            <div style={{ fontSize: '0.75rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#94A3B8', fontWeight: 700, marginBottom: '8px' }}>
              PREDICTED CLASS
            </div>
            <div style={{ fontSize: '2.5rem', fontWeight: 900, color: predStyle.color, letterSpacing: '0.05em' }}>
              {loading ? '...' : predFormatted}
            </div>
            <div style={{ fontSize: '0.85rem', color: '#00F2FE', fontWeight: 700, marginTop: '6px' }}>
              Confidence: {confidencePct}%
            </div>
          </div>
        </div>

        {/* Right / Wrong Result Indicator Banner */}
        <div style={{
          display: 'flex',
          justify: 'center',
          alignItems: 'center',
          gap: '12px',
          padding: '16px 24px',
          borderRadius: '12px',
          marginBottom: '28px',
          background: isCorrect ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
          border: `2px solid ${isCorrect ? '#10B981' : '#F43F5E'}`,
          boxShadow: `0 0 25px ${isCorrect ? 'rgba(16, 185, 129, 0.25)' : 'rgba(244, 63, 94, 0.25)'}`
        }}>
          {isCorrect ? (
            <>
              <CheckCircle2 size={32} color="#10B981" />
              <div>
                <div style={{ fontSize: '1.4rem', fontWeight: 900, color: '#10B981', letterSpacing: '0.05em' }}>
                  RIGHT (CORRECT)
                </div>
                <div style={{ fontSize: '0.8rem', color: '#A7F3D0' }}>
                  Model prediction matched actual motor imagery label!
                </div>
              </div>
            </>
          ) : (
            <>
              <XCircle size={32} color="#F43F5E" />
              <div>
                <div style={{ fontSize: '1.4rem', fontWeight: 900, color: '#F43F5E', letterSpacing: '0.05em' }}>
                  WRONG (INCORRECT)
                </div>
                <div style={{ fontSize: '0.8rem', color: '#FECDD3' }}>
                  Model predicted {predFormatted} but actual was {trueFormatted}.
                </div>
              </div>
            </>
          )}
        </div>

        {/* 3D WebGL Neural Cortex & Avatar Simulation */}
        <div style={{ marginBottom: '28px' }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '12px'
          }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#F8FAFC', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Brain size={18} color="#00F2FE" />
              Real-Time 3D Neuro-Physical Visualization
            </div>

            <div style={{ display: 'flex', gap: '6px', background: '#090D16', padding: '3px', borderRadius: '8px', border: '1px solid #1E293B' }}>
              <button
                onClick={() => setVisMode('dual')}
                style={{
                  background: visMode === 'dual' ? '#1E293B' : 'transparent',
                  color: visMode === 'dual' ? '#00F2FE' : '#94A3B8',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '4px 10px',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                ⚡ DUAL 3D
              </button>
              <button
                onClick={() => setVisMode('brain')}
                style={{
                  background: visMode === 'brain' ? '#1E293B' : 'transparent',
                  color: visMode === 'brain' ? '#00F2FE' : '#94A3B8',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '4px 10px',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                🧠 CORTEX
              </button>
              <button
                onClick={() => setVisMode('avatar')}
                style={{
                  background: visMode === 'avatar' ? '#1E293B' : 'transparent',
                  color: visMode === 'avatar' ? '#00F2FE' : '#94A3B8',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '4px 10px',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                🤖 AVATAR
              </button>
            </div>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: visMode === 'dual' ? '1fr 1fr' : '1fr',
            gap: '16px'
          }}>
            {(visMode === 'dual' || visMode === 'brain') && (
              <Brain3DCanvas
                predictedClass={predFormatted}
                height={320}
                isProcessing={loading}
              />
            )}
            {(visMode === 'dual' || visMode === 'avatar') && (
              <Avatar3DCanvas
                predictedClass={predFormatted}
                actualClass={trueFormatted}
                confidence={prediction?.confidence || 0.85}
                height={320}
                activeChannel="15-Channel Motor Strip"
              />
            )}
          </div>
        </div>

        {/* Probability Breakdown Percentage Bars */}
        <div style={{
          background: '#090D16',
          borderRadius: '12px',
          padding: '20px',
          marginBottom: '28px',
          border: '1px solid #1E293B'
        }}>
          <div style={{ fontSize: '0.8rem', color: '#94A3B8', fontWeight: 700, marginBottom: '14px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Class Probability Distribution
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {/* REST Bar */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px', fontWeight: 600 }}>
                <span style={{ color: '#10B981' }}>REST</span>
                <span style={{ color: '#F8FAFC' }}>{((probs.REST || 0) * 100).toFixed(1)}%</span>
              </div>
              <div style={{ background: '#1E293B', height: '10px', borderRadius: '5px', overflow: 'hidden' }}>
                <div style={{
                  width: `${(probs.REST || 0) * 100}%`,
                  background: 'linear-gradient(90deg, #10B981, #059669)',
                  height: '100%',
                  transition: 'width 0.4s ease'
                }} />
              </div>
            </div>

            {/* LEFT FIST Bar */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px', fontWeight: 600 }}>
                <span style={{ color: '#38BDF8' }}>LEFT (FIST)</span>
                <span style={{ color: '#F8FAFC' }}>{((probs.LEFT_FIST || 0) * 100).toFixed(1)}%</span>
              </div>
              <div style={{ background: '#1E293B', height: '10px', borderRadius: '5px', overflow: 'hidden' }}>
                <div style={{
                  width: `${(probs.LEFT_FIST || 0) * 100}%`,
                  background: 'linear-gradient(90deg, #38BDF8, #0284C7)',
                  height: '100%',
                  transition: 'width 0.4s ease'
                }} />
              </div>
            </div>

            {/* RIGHT FIST Bar */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px', fontWeight: 600 }}>
                <span style={{ color: '#3B82F6' }}>RIGHT (FIST)</span>
                <span style={{ color: '#F8FAFC' }}>{((probs.RIGHT_FIST || 0) * 100).toFixed(1)}%</span>
              </div>
              <div style={{ background: '#1E293B', height: '10px', borderRadius: '5px', overflow: 'hidden' }}>
                <div style={{
                  width: `${(probs.RIGHT_FIST || 0) * 100}%`,
                  background: 'linear-gradient(90deg, #3B82F6, #1D4ED8)',
                  height: '100%',
                  transition: 'width 0.4s ease'
                }} />
              </div>
            </div>
          </div>
        </div>

        {/* Minimal EEG Waveform Display */}
        <div style={{
          background: '#090D16',
          borderRadius: '12px',
          padding: '16px',
          marginBottom: '28px',
          border: '1px solid #1E293B'
        }}>
          <div style={{ fontSize: '0.75rem', color: '#64748B', fontWeight: 700, marginBottom: '8px', textTransform: 'uppercase' }}>
            Live Trial EEG Oscilloscope (3.0 sec Window)
          </div>
          <canvas
            ref={canvasRef}
            width={800}
            height={110}
            style={{ width: '100%', height: '110px', borderRadius: '8px', display: 'block' }}
          />
        </div>

        {/* Action Controls: Next Button, Prev Button, Auto Play */}
        <div style={{ display: 'flex', gap: '14px', justifyContent: 'center', alignItems: 'center' }}>
          <button
            onClick={handlePrev}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: '#1E293B',
              border: '1px solid #334155',
              color: '#F8FAFC',
              padding: '12px 20px',
              borderRadius: '10px',
              fontWeight: 700,
              fontSize: '0.95rem',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
          >
            <ChevronLeft size={20} />
            Previous
          </button>

          {/* Prominent Glowing Next Button */}
          <button
            onClick={handleNext}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              background: 'linear-gradient(135deg, #00F2FE 0%, #0284C7 100%)',
              border: 'none',
              color: '#090D16',
              padding: '14px 36px',
              borderRadius: '10px',
              fontWeight: 900,
              fontSize: '1.1rem',
              cursor: 'pointer',
              boxShadow: '0 0 25px rgba(0, 242, 254, 0.4)',
              transition: 'all 0.2s ease',
              letterSpacing: '0.02em'
            }}
          >
            NEXT TRIAL
            <ChevronRight size={24} />
          </button>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: isPlaying ? 'rgba(244, 63, 94, 0.2)' : 'rgba(16, 185, 129, 0.2)',
              border: `1px solid ${isPlaying ? '#F43F5E' : '#10B981'}`,
              color: isPlaying ? '#F43F5E' : '#10B981',
              padding: '12px 20px',
              borderRadius: '10px',
              fontWeight: 700,
              fontSize: '0.95rem',
              cursor: 'pointer'
            }}
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? 'Pause Auto' : 'Auto Play'}
          </button>
        </div>

      </div>
    </div>
  );
}
