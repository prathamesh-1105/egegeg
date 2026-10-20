import React, { useState, useEffect, useRef } from 'react';
import { Mic, Zap, Activity, Cpu, Sparkles, CheckCircle2, AlertCircle } from 'lucide-react';

export default function SpeechBCICenter() {
  const [targetWord, setTargetWord] = useState('Hello');
  const [noiseLevel, setNoiseLevel] = useState(1.0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [classes, setClasses] = useState(['Hello', 'Hi', 'Yes', 'No', 'Rest']);
  const [benchmark, setBenchmark] = useState(null);
  const canvasRef = useRef(null);

  // Fetch classes and benchmark on load
  useEffect(() => {
    fetch('http://localhost:8050/api/speech/classes')
      .then(res => res.json())
      .then(data => {
        if (data.classes) setClasses(data.classes);
      })
      .catch(err => console.log('Speech classes fetch:', err));

    fetch('http://localhost:8050/api/speech/benchmark')
      .then(res => res.json())
      .then(data => setBenchmark(data))
      .catch(err => console.log('Speech benchmark fetch:', err));
  }, []);

  const handleRunInference = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8050/api/speech/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_word: targetWord, noise_level: noiseLevel })
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data);
      }
    } catch (err) {
      console.error('Error simulating speech trial:', err);
    } finally {
      setLoading(false);
    }
  };

  // Run initial trial on mount
  useEffect(() => {
    handleRunInference();
  }, []);

  // Draw EEG Oscilloscope for Speech Trials
  useEffect(() => {
    if (!result?.signals || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Dark grid background
    ctx.fillStyle = '#090D16';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#1E293B';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const signals = result.signals;
    const nCh = Math.min(signals.length, 8); // Top 8 speech channels
    const chHeight = height / nCh;

    const colors = ['#00F2FE', '#38BDF8', '#60A5FA', '#3B82F6', '#0284C7', '#10B981', '#F59E0B', '#F43F5E'];

    for (let i = 0; i < nCh; i++) {
      const sig = signals[i];
      const chName = result.channels ? result.channels[i] : `CH ${i+1}`;
      const color = colors[i % colors.length];

      ctx.fillStyle = '#64748B';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(chName, 8, i * chHeight + 14);

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();

      const step = width / (sig.length - 1);
      const midY = i * chHeight + chHeight / 2;

      for (let s = 0; s < sig.length; s++) {
        const val = sig[s];
        const y = midY - val * 6.0;
        if (s === 0) ctx.moveTo(0, y);
        else ctx.lineTo(s * step, y);
      }
      ctx.stroke();
    }
  }, [result]);

  return (
    <div style={{ maxWidth: '1400px', margin: '24px auto', padding: '0 24px' }}>
      {/* Hero Header */}
      <div style={{
        background: 'linear-gradient(135deg, #090D16 0%, #0F172A 100%)',
        border: '1px solid rgba(56, 189, 248, 0.3)',
        borderRadius: '14px',
        padding: '24px',
        marginBottom: '24px',
        boxShadow: '0 10px 30px rgba(56, 189, 248, 0.08)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div className="neural-badge badge-blue" style={{ marginBottom: '8px' }}>
              <Sparkles size={14} /> SOTA SPATIAL-TEMPORAL TRANSFORMER
            </div>
            <h1 style={{ fontFamily: 'Space Grotesk', margin: '4px 0 0 0', color: '#F8FAFC', fontSize: '2.0rem', letterSpacing: '-0.02em' }}>
              IMAGINED SPEECH & INNER WORD DECODING
            </h1>
            <div style={{ color: '#94A3B8', fontSize: '0.92rem', marginTop: '6px' }}>
              Real-time decoding of covert/silent speech brainwaves using <b>EEGConformer (Multi-Head Self-Attention)</b>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <div className="glass-card" style={{ padding: '10px 18px', textAlign: 'center', borderColor: '#818CF8' }}>
              <div style={{ fontSize: '0.7rem', color: '#94A3B8', fontWeight: 700 }}>VOCABULARY</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#00F2FE' }}>5 WORDS</div>
              <div style={{ fontSize: '0.65rem', color: '#64748B' }}>Hello, Hi, Yes, No, Rest</div>
            </div>

            <div className="glass-card" style={{ padding: '10px 18px', textAlign: 'center', borderColor: '#10B981' }}>
              <div style={{ fontSize: '0.7rem', color: '#94A3B8', fontWeight: 700 }}>VALIDATION ACC</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#10B981' }}>86.7%</div>
              <div style={{ fontSize: '0.65rem', color: '#64748B' }}>Transformer SOTA</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid: Controls + Decoder Output */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1.9fr', gap: '24px', marginBottom: '24px' }}>
        {/* Controls Card */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#F8FAFC', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Mic size={18} color="#00F2FE" />
            Trial Simulation Parameters
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '0.8rem', color: '#94A3B8', fontWeight: 600, marginBottom: '8px' }}>
              TARGET IMAGINED WORD:
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
              {classes.map((word) => {
                const isSelected = targetWord === word;
                return (
                  <button
                    key={word}
                    onClick={() => setTargetWord(word)}
                    style={{
                      background: isSelected ? 'linear-gradient(135deg, #0284C7 0%, #2563EB 100%)' : '#0F172A',
                      color: isSelected ? '#FFFFFF' : '#94A3B8',
                      border: isSelected ? '1px solid #38BDF8' : '1px solid #334155',
                      borderRadius: '8px',
                      padding: '10px 0',
                      fontWeight: 700,
                      fontSize: '0.85rem',
                      cursor: 'pointer',
                      boxShadow: isSelected ? '0 0 12px rgba(56, 189, 248, 0.4)' : 'none',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    {word}
                  </button>
                );
              })}
            </div>
          </div>

          <div style={{ marginBottom: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94A3B8', fontWeight: 600, marginBottom: '6px' }}>
              <span>SCALP EEG NOISE FACTOR:</span>
              <span style={{ color: '#00F2FE' }}>{noiseLevel.toFixed(1)}x</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={noiseLevel}
              onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#64748B', marginTop: '4px' }}>
              <span>0.1 (Laboratory Grade)</span>
              <span>1.0 (Realistic Scalp)</span>
              <span>2.0 (High Artifact)</span>
            </div>
          </div>

          <button
            onClick={handleRunInference}
            disabled={loading}
            className="cyber-btn"
            style={{ width: '100%', justifyContent: 'center', padding: '12px' }}
          >
            <Zap size={18} />
            {loading ? 'RUNNING EEGCONFORMER...' : 'DECODE IMAGINED SPEECH'}
          </button>

          {/* Decoded Word Hero Output */}
          {result && (
            <div style={{
              marginTop: '24px',
              padding: '16px',
              borderRadius: '12px',
              background: 'rgba(15, 23, 42, 0.85)',
              border: result.is_correct ? '1px solid #10B981' : '1px solid #F59E0B',
              boxShadow: result.is_correct ? '0 0 20px rgba(16, 185, 129, 0.2)' : 'none'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#94A3B8' }}>DECODED INNER WORD</span>
                <span className={`neural-badge ${result.is_correct ? 'badge-emerald' : 'badge-amber'}`}>
                  {result.is_correct ? 'MATCHED TARGET' : 'PREDICTION'}
                </span>
              </div>
              <div style={{ fontFamily: 'Space Grotesk', fontSize: '2.2rem', fontWeight: 800, color: '#00F2FE', margin: '4px 0' }}>
                "{result.predicted_word}"
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94A3B8' }}>
                <span>Confidence: <b style={{ color: '#F8FAFC' }}>{(result.confidence * 100).toFixed(1)}%</b></span>
                <span>Latency: <b style={{ color: '#10B981' }}>{result.latency_ms} ms</b></span>
              </div>
            </div>
          )}
        </div>

        {/* Oscilloscope + Class Probabilities */}
        <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#F8FAFC', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity size={16} color="#00F2FE" />
                Speech ROI Multi-Channel EEG Oscilloscope (Broca / Wernicke / Mouth)
              </div>
              <span style={{ fontSize: '0.7rem', color: '#64748B', fontFamily: 'JetBrains Mono' }}>160 Hz • 4.0s Window</span>
            </div>
            <div style={{ borderRadius: '10px', overflow: 'hidden', border: '1px solid #1E293B' }}>
              <canvas ref={canvasRef} width={760} height={210} style={{ width: '100%', height: '210px', display: 'block' }} />
            </div>
          </div>

          {/* Probabilities Bars */}
          {result?.probabilities && (
            <div>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '10px' }}>
                Transformer Softmax Output Probabilities
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {Object.entries(result.probabilities).map(([word, prob]) => {
                  const pct = Math.round(prob * 100);
                  const isTop = word === result.predicted_word;
                  return (
                    <div key={word} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <span style={{ width: '50px', fontSize: '0.8rem', fontWeight: 700, color: isTop ? '#00F2FE' : '#94A3B8' }}>
                        {word}
                      </span>
                      <div style={{ flex: 1, height: '8px', background: '#090D16', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{
                          width: `${pct}%`,
                          height: '100%',
                          background: isTop ? 'linear-gradient(90deg, #00F2FE, #6366F1)' : '#334155',
                          borderRadius: '4px',
                          boxShadow: isTop ? '0 0 10px rgba(0, 242, 254, 0.5)' : 'none',
                          transition: 'width 0.3s ease'
                        }} />
                      </div>
                      <span style={{ width: '45px', textAlign: 'right', fontSize: '0.75rem', fontFamily: 'JetBrains Mono', color: isTop ? '#00F2FE' : '#64748B' }}>
                        {pct}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ROI Channel Power & IEEE Benchmark Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        {/* Speech ROI Channel Power */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#F8FAFC', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={16} color="#38BDF8" />
            Speech Cortical ROI Activation Heatmap (RMS Power)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
            {result?.roi_channel_power && Object.entries(result.roi_channel_power).map(([ch, power]) => {
              const intensity = Math.min(1, Math.max(0.1, power));
              return (
                <div
                  key={ch}
                  style={{
                    background: `rgba(59, 130, 246, ${intensity * 0.4})`,
                    border: `1px solid rgba(56, 189, 248, ${intensity * 0.8})`,
                    borderRadius: '8px',
                    padding: '8px',
                    textAlign: 'center'
                  }}
                >
                  <div style={{ fontSize: '0.75rem', fontWeight: 800, color: '#F8FAFC' }}>{ch}</div>
                  <div style={{ fontSize: '0.65rem', fontFamily: 'JetBrains Mono', color: '#00F2FE', marginTop: '2px' }}>
                    {power.toFixed(2)}
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#64748B', marginTop: '12px' }}>
            Electrodes F7, FC5 (Broca's Language Production) • T7, T8 (Wernicke's Processing) • C3, C4 (Motor Mouth)
          </div>
        </div>

        {/* IEEE Architecture Benchmark Table */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#F8FAFC', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sparkles size={16} color="#10B981" />
            IEEE Architecture Comparison & Benchmark
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #1E293B', color: '#94A3B8', textAlign: 'left' }}>
                <th style={{ padding: '8px 4px' }}>Model Architecture</th>
                <th style={{ padding: '8px 4px' }}>Accuracy</th>
                <th style={{ padding: '8px 4px' }}>Macro F1</th>
                <th style={{ padding: '8px 4px' }}>Latency</th>
              </tr>
            </thead>
            <tbody>
              {benchmark?.benchmarks?.map((bm, idx) => (
                <tr key={idx} style={{
                  borderBottom: '1px solid #1E293B',
                  background: bm.model.includes('EEGConformer') ? 'rgba(0, 242, 254, 0.05)' : 'transparent'
                }}>
                  <td style={{ padding: '10px 4px', fontWeight: bm.model.includes('EEGConformer') ? 700 : 500, color: bm.model.includes('EEGConformer') ? '#00F2FE' : '#F8FAFC' }}>
                    {bm.model}
                  </td>
                  <td style={{ padding: '10px 4px', color: '#10B981', fontWeight: 700 }}>{bm.val_acc}</td>
                  <td style={{ padding: '10px 4px', color: '#818CF8' }}>{bm.val_f1}</td>
                  <td style={{ padding: '10px 4px', color: '#94A3B8', fontFamily: 'JetBrains Mono' }}>{bm.latency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
