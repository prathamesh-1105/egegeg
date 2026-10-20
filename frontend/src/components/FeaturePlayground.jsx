import React, { useState } from 'react';
import { Sliders, Zap, RotateCcw, CheckCircle2 } from 'lucide-react';

export default function FeaturePlayground() {
  const [c3Power, setC3Power] = useState(12.5);
  const [c4Power, setC4Power] = useState(8.2);
  const [asymmetry, setAsymmetry] = useState(-0.42);
  const [muErdPct, setMuErdPct] = useState(-24.5);
  const [betaPower, setBetaPower] = useState(15.0);
  const [erdSlope, setErdSlope] = useState(-0.18);
  const [restThreshold, setRestThreshold] = useState(0.62);

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8050/api/predict/features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          c3_power: c3Power,
          c4_power: c4Power,
          c3_c4_asymmetry: asymmetry,
          mu_erd_pct: muErdPct,
          beta_power: betaPower,
          erd_slope: erdSlope,
          rest_threshold: restThreshold
        })
      });
      if (res.ok) {
        const data = await res.json();
        setPrediction(data);
      }
    } catch (err) {
      console.error('Error predicting custom features:', err);
    } finally {
      setLoading(false);
    }
  };

  // Preset Configurations
  const applyPreset = (type) => {
    if (type === 'LEFT') {
      setAsymmetry(-1.25);
      setMuErdPct(-45.0);
      setErdSlope(-0.35);
      setC3Power(18.2);
      setC4Power(6.1);
    } else if (type === 'RIGHT') {
      setAsymmetry(1.15);
      setMuErdPct(-42.0);
      setErdSlope(0.32);
      setC3Power(5.8);
      setC4Power(17.9);
    } else {
      setAsymmetry(0.02);
      setMuErdPct(5.0);
      setErdSlope(-0.02);
      setC3Power(12.0);
      setC4Power(12.1);
    }
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '24px auto', padding: '0 24px' }}>
      {/* Hero Header */}
      <div style={{
        background: 'linear-gradient(135deg, #090D16 0%, #0F172A 100%)',
        border: '1px solid rgba(56, 189, 248, 0.3)',
        borderRadius: '12px',
        padding: '20px',
        marginBottom: '20px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ color: '#38BDF8', fontWeight: 800, fontSize: '0.85rem', letterSpacing: '0.08em' }}>🎛️ INTERACTIVE FEATURE PLAYGROUND</div>
            <h1 style={{ fontFamily: 'Space Grotesk', margin: '4px 0 0 0', color: '#F8FAFC', fontSize: '2.0rem' }}>
              EEG FEATURE EXTRACTION & LIVE MODEL PREDICTOR
            </h1>
            <div style={{ color: '#94A3B8', fontSize: '0.9rem', marginTop: '4px' }}>
              Adjust Motor Cortex Power, C3/C4 Asymmetry, ERD Suppression, and Thresholds to Predict Motor Intent
            </div>
          </div>

          {/* Quick Presets */}
          <div style={{ display: 'flex', gap: '8px' }}>
            <button onClick={() => applyPreset('REST')} style={{ background: '#1E293B', border: '1px solid #10B981', color: '#10B981', padding: '8px 14px', borderRadius: '8px', fontWeight: 700, cursor: 'pointer' }}>
              🧘 Rest Preset
            </button>
            <button onClick={() => applyPreset('LEFT')} style={{ background: '#1E293B', border: '1px solid #00F2FE', color: '#00F2FE', padding: '8px 14px', borderRadius: '8px', fontWeight: 700, cursor: 'pointer' }}>
              🤛 Left Fist Preset
            </button>
            <button onClick={() => applyPreset('RIGHT')} style={{ background: '#1E293B', border: '1px solid #F59E0B', color: '#F59E0B', padding: '8px 14px', borderRadius: '8px', fontWeight: 700, cursor: 'pointer' }}>
              🤜 Right Fist Preset
            </button>
          </div>
        </div>
      </div>

      {/* Main Grid: Sliders vs Live Output */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Feature Sliders */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ fontWeight: 700, color: '#F8FAFC', marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>🧠 EEG FEATURE PARAMETERS</span>
            <button onClick={handlePredict} style={{ background: 'linear-gradient(135deg, #0284C7 0%, #2563EB 100%)', border: 'none', color: '#F8FAFC', padding: '8px 16px', borderRadius: '8px', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Zap size={16} /> RUN PREDICTION
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            {/* C3 vs C4 Asymmetry */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span style={{ color: '#CBD5E1' }}>C3 / C4 Log Asymmetry:</span>
                <strong style={{ color: '#00F2FE' }}>{asymmetry}</strong>
              </div>
              <input type="range" min="-2.5" max="2.5" step="0.05" value={asymmetry} onChange={(e) => setAsymmetry(parseFloat(e.target.value))} style={{ width: '100%' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#64748B' }}>
                <span>Left Fist ERD (Negative)</span>
                <span>Rest (Zero)</span>
                <span>Right Fist ERD (Positive)</span>
              </div>
            </div>

            {/* Mu ERD % */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span style={{ color: '#CBD5E1' }}>Mu Band ERD Suppression (%):</span>
                <strong style={{ color: '#10B981' }}>{muErdPct}%</strong>
              </div>
              <input type="range" min="-80" max="40" step="1" value={muErdPct} onChange={(e) => setMuErdPct(parseFloat(e.target.value))} style={{ width: '100%' }} />
            </div>

            {/* ERD Slope */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span style={{ color: '#CBD5E1' }}>Temporal ERD Slope (uV²/s):</span>
                <strong style={{ color: '#F59E0B' }}>{erdSlope}</strong>
              </div>
              <input type="range" min="-0.8" max="0.8" step="0.02" value={erdSlope} onChange={(e) => setErdSlope(parseFloat(e.target.value))} style={{ width: '100%' }} />
            </div>

            {/* C3 Band Power */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span style={{ color: '#CBD5E1' }}>C3 Motor Channel Power (uV²):</span>
                <strong style={{ color: '#38BDF8' }}>{c3Power} uV²</strong>
              </div>
              <input type="range" min="1.0" max="30.0" step="0.5" value={c3Power} onChange={(e) => setC3Power(parseFloat(e.target.value))} style={{ width: '100%' }} />
            </div>

            {/* C4 Band Power */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span style={{ color: '#CBD5E1' }}>C4 Motor Channel Power (uV²):</span>
                <strong style={{ color: '#38BDF8' }}>{c4Power} uV²</strong>
              </div>
              <input type="range" min="1.0" max="30.0" step="0.5" value={c4Power} onChange={(e) => setC4Power(parseFloat(e.target.value))} style={{ width: '100%' }} />
            </div>

            {/* REST Decision Threshold */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span style={{ color: '#CBD5E1' }}>REST Probability Threshold:</span>
                <strong style={{ color: '#F43F5E' }}>{restThreshold}</strong>
              </div>
              <input type="range" min="0.40" max="0.85" step="0.01" value={restThreshold} onChange={(e) => setRestThreshold(parseFloat(e.target.value))} style={{ width: '100%' }} />
            </div>
          </div>
        </div>

        {/* Prediction Results */}
        <div className="glass-card-glow" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#94A3B8', letterSpacing: '0.05em' }}>STEP 59 CUSTOM FEATURE INFERENCE</div>

            {prediction ? (
              <div style={{
                margin: '20px 0',
                padding: '24px',
                borderRadius: '12px',
                background: prediction.predicted_label === 'REST' ? 'rgba(16, 185, 129, 0.15)' : prediction.predicted_label === 'LEFT_FIST' ? 'rgba(0, 242, 254, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                border: `2px solid ${prediction.predicted_label === 'REST' ? '#10B981' : prediction.predicted_label === 'LEFT_FIST' ? '#00F2FE' : '#F59E0B'}`,
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '4.0rem', marginBottom: '8px' }}>
                  {prediction.predicted_label === 'REST' ? '🧘' : prediction.predicted_label === 'LEFT_FIST' ? '🤛' : '🤜'}
                </div>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: prediction.predicted_label === 'REST' ? '#10B981' : prediction.predicted_label === 'LEFT_FIST' ? '#00F2FE' : '#F59E0B' }}>
                  {prediction.predicted_label}
                </div>
                <div style={{ fontSize: '0.85rem', color: '#94A3B8', marginTop: '6px' }}>
                  Confidence: <strong>{(prediction.confidence * 100).toFixed(1)}%</strong> | Latency: <strong>{prediction.latency_ms} ms</strong>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#94A3B8' }}>
                Adjust feature sliders and click <strong>RUN PREDICTION</strong> above to evaluate the model.
              </div>
            )}

            {/* Probabilities Output */}
            {prediction && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '20px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                    <span>REST (T0):</span>
                    <strong style={{ color: '#10B981' }}>{(prediction.probabilities.REST * 100).toFixed(1)}%</strong>
                  </div>
                  <div style={{ height: '10px', background: '#090D16', borderRadius: '5px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${prediction.probabilities.REST * 100}%`, background: '#10B981', transition: 'width 0.3s ease' }} />
                  </div>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                    <span>LEFT_FIST (T1):</span>
                    <strong style={{ color: '#00F2FE' }}>{(prediction.probabilities.LEFT_FIST * 100).toFixed(1)}%</strong>
                  </div>
                  <div style={{ height: '10px', background: '#090D16', borderRadius: '5px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${prediction.probabilities.LEFT_FIST * 100}%`, background: '#00F2FE', transition: 'width 0.3s ease' }} />
                  </div>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                    <span>RIGHT_FIST (T2):</span>
                    <strong style={{ color: '#F59E0B' }}>{(prediction.probabilities.RIGHT_FIST * 100).toFixed(1)}%</strong>
                  </div>
                  <div style={{ height: '10px', background: '#090D16', borderRadius: '5px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${prediction.probabilities.RIGHT_FIST * 100}%`, background: '#F59E0B', transition: 'width 0.3s ease' }} />
                  </div>
                </div>
              </div>
            )}
          </div>

          <div style={{ marginTop: '20px', background: '#090D16', padding: '12px', borderRadius: '8px', fontSize: '0.8rem', color: '#94A3B8' }}>
            💡 <strong>BCI Science Insight:</strong> Left Fist imagery causes Event-Related Desynchronization (ERD) over the right motor cortex (C4), making C3/C4 log asymmetry negative.
          </div>
        </div>
      </div>
    </div>
  );
}
