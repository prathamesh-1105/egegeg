import React, { useState, useEffect } from 'react';
import { LineChart, BarChart2, Database, Search } from 'lucide-react';

export default function SignalExplorer() {
  const [subject, setSubject] = useState('S001');
  const [run, setRun] = useState('R03');
  const [trial, setTrial] = useState(0);
  const [eegData, setEegData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await fetch(`http://localhost:8050/api/eeg/window?subject_id=${subject}&run_id=${run}&trial_idx=${trial}`);
        if (res.ok) {
          const data = await res.json();
          setEegData(data);
        }
      } catch (err) {
        console.error('Error fetching EEG Explorer data:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [subject, run, trial]);

  return (
    <div style={{ maxWidth: '1400px', margin: '24px auto', padding: '0 24px' }}>
      {/* Hero Header */}
      <div style={{
        background: 'linear-gradient(135deg, #0F172A 0%, #1E293B 100%)',
        border: '2px solid #38BDF8',
        borderRadius: '12px',
        padding: '20px',
        marginBottom: '20px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ color: '#38BDF8', fontWeight: 800, fontSize: '0.85rem', letterSpacing: '0.08em' }}>📊 EEG SIGNAL EXPLORER</div>
            <h1 style={{ fontFamily: 'Space Grotesk', margin: '4px 0 0 0', color: '#F8FAFC', fontSize: '2.0rem' }}>
              RAW EDF SIGNAL & POWER SPECTRUM VISUALIZER
            </h1>
            <div style={{ color: '#94A3B8', fontSize: '0.9rem', marginTop: '4px' }}>
              Inspect 15-Channel Motor Arrays, Welch PSD (0–40 Hz), and C3/C4 Sensorimotor Rhythm Asymmetry
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <select value={subject} onChange={(e) => setSubject(e.target.value)} style={{ background: '#090D16', border: '1px solid #334155', color: '#F8FAFC', padding: '8px 12px', borderRadius: '8px', fontWeight: 600 }}>
              {Array.from({ length: 10 }, (_, i) => `S${String(i + 1).padStart(3, '0')}`).map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>

            <select value={run} onChange={(e) => setRun(e.target.value)} style={{ background: '#090D16', border: '1px solid #334155', color: '#F8FAFC', padding: '8px 12px', borderRadius: 800, fontWeight: 600 }}>
              <option value="R03">R03 (Left vs Right Fist)</option>
              <option value="R07">R07 (Left vs Right Fist)</option>
              <option value="R11">R11 (Holdout Test)</option>
            </select>

            <input
              type="number"
              min="0"
              max={eegData?.total_trials ? eegData.total_trials - 1 : 29}
              value={trial}
              onChange={(e) => setTrial(parseInt(e.target.value) || 0)}
              style={{ background: '#090D16', border: '1px solid #334155', color: '#F8FAFC', padding: '8px 12px', borderRadius: '8px', width: '80px', fontWeight: 600 }}
            />
          </div>
        </div>
      </div>

      {/* Explorer Content */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Signal Specs & Metrics */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ fontWeight: 700, color: '#F8FAFC', marginBottom: '16px' }}>
            📈 TRIAL SIGNAL PROPERTIES ({subject} {run} - Trial #{trial + 1})
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px', background: '#090D16', borderRadius: '8px' }}>
              <span style={{ color: '#94A3B8' }}>True Ground Truth Label:</span>
              <strong style={{ color: '#00F2FE' }}>{eegData?.true_label || 'REST'}</strong>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px', background: '#090D16', borderRadius: '8px' }}>
              <span style={{ color: '#94A3B8' }}>C3 Motor Channel Variance:</span>
              <strong style={{ color: '#10B981' }}>{eegData?.c3_power || 0} uV²</strong>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px', background: '#090D16', borderRadius: '8px' }}>
              <span style={{ color: '#94A3B8' }}>C4 Motor Channel Variance:</span>
              <strong style={{ color: '#38BDF8' }}>{eegData?.c4_power || 0} uV²</strong>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px', background: '#090D16', borderRadius: '8px' }}>
              <span style={{ color: '#94A3B8' }}>Log C4/C3 Asymmetry Ratio:</span>
              <strong style={{ color: '#F59E0B' }}>{eegData?.log_asymmetry || 0}</strong>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px', background: '#090D16', borderRadius: '8px' }}>
              <span style={{ color: '#94A3B8' }}>Sampling Rate & Epoch Length:</span>
              <strong style={{ color: '#38BDF8' }}>160 Hz (480 samples / 3.0 s)</strong>
            </div>
          </div>
        </div>

        {/* PSD Frequency Spectrum Table */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ fontWeight: 700, color: '#F8FAFC', marginBottom: '16px' }}>
            ⚡ WELCH PSD POWER SPECTRUM (C3 vs C4)
          </div>

          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #334155', color: '#94A3B8', textAlign: 'left' }}>
                  <th style={{ padding: '8px' }}>Freq (Hz)</th>
                  <th style={{ padding: '8px', color: '#10B981' }}>C3 PSD (uV²/Hz)</th>
                  <th style={{ padding: '8px', color: '#38BDF8' }}>C4 PSD (uV²/Hz)</th>
                </tr>
              </thead>
              <tbody>
                {eegData?.psd?.frequencies ? (
                  eegData.psd.frequencies.slice(0, 15).map((f, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #1E293B' }}>
                      <td style={{ padding: '8px', fontWeight: 600, color: '#F8FAFC' }}>{f.toFixed(1)} Hz</td>
                      <td style={{ padding: '8px', color: '#10B981' }}>{eegData.psd.c3_psd[i].toFixed(2)}</td>
                      <td style={{ padding: '8px', color: '#38BDF8' }}>{eegData.psd.c4_psd[i].toFixed(2)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="3" style={{ padding: '16px', textAlign: 'center', color: '#94A3B8' }}>Loading PSD data...</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
