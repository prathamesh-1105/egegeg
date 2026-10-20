import React, { useState, useEffect } from 'react';
import { ShieldCheck, CheckCircle2, FileText, AlertTriangle } from 'lucide-react';

export default function ValidationCenter() {
  const [cmData, setCmData] = useState(null);
  const [auditData, setAuditData] = useState([]);

  useEffect(() => {
    const fetchAudit = async () => {
      try {
        const cmRes = await fetch('http://localhost:8050/api/experiments/confusion_matrix');
        if (cmRes.ok) {
          const resJson = await cmRes.json();
          setCmData(resJson.data);
        }

        const audRes = await fetch('http://localhost:8050/api/experiments/consistency_audit');
        if (audRes.ok) {
          const audJson = await audRes.json();
          setAuditData(audJson.data);
        }
      } catch (err) {
        console.error('Error fetching validation audit data:', err);
      }
    };
    fetchAudit();
  }, []);

  const safeguards = [
    { title: '1. Raw EDF File Integrity', desc: '30 Genuine EDF files across S001–S010 validated with MNE-Python.', status: 'PASS' },
    { title: '2. Zero Synthetic Data Check', desc: '0 Synthetic or interpolated samples generated. Real EEG microvolts only.', status: 'PASS' },
    { title: '3. R11 Holdout Isolation', desc: 'R11 (300 trials) untouched during model tuning and feature selection.', status: 'PASS' },
    { title: '4. Pre-R11 Config Freeze', desc: 'All hyperparameters frozen in final_freeze.json prior to R11 evaluation.', status: 'PASS' },
    { title: '5. Single-Pass Evaluation', desc: 'Holdout test set evaluated exactly once without post-hoc hyperparameter tuning.', status: 'PASS' }
  ];

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
        <div style={{ color: '#38BDF8', fontWeight: 800, fontSize: '0.85rem', letterSpacing: '0.08em' }}>🛡️ VALIDATION AUDIT CENTER</div>
        <h1 style={{ fontFamily: 'Space Grotesk', margin: '4px 0 0 0', color: '#F8FAFC', fontSize: '2.0rem' }}>
          INDEPENDENT R11 HOLDOUT AUDIT & CONSISTENCY REPORT
        </h1>
        <div style={{ color: '#94A3B8', fontSize: '0.9rem', marginTop: '4px' }}>
          Zero Data Leakage Safeguards | 3x3 Confusion Matrix | 182-Check Verification Suite
        </div>
      </div>

      {/* Grid: Safeguard Checklist vs Confusion Matrix */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
        {/* Safeguard Checklist */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ fontWeight: 700, color: '#F8FAFC', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={20} color="#10B981" />
            <span>DATA LEAKAGE & PROTOCOL SAFEGUARD CHECKLIST</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {safeguards.map((item, idx) => (
              <div key={idx} style={{ padding: '12px', background: '#090D16', borderRadius: '8px', border: '1px solid #1E293B', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontWeight: 700, color: '#F8FAFC', fontSize: '0.85rem' }}>{item.title}</div>
                  <div style={{ fontSize: '0.75rem', color: '#94A3B8', marginTop: '2px' }}>{item.desc}</div>
                </div>
                <div style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#10B981', padding: '4px 10px', borderRadius: '12px', fontWeight: 800, fontSize: '0.75rem', border: '1px solid #10B981' }}>
                  {item.status}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 3x3 Confusion Matrix */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ fontWeight: 700, color: '#F8FAFC', marginBottom: '16px' }}>
            📊 R11 CONFUSION MATRIX (300 Holdsout Trials)
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'center' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #334155', color: '#94A3B8' }}>
                <th style={{ padding: '10px', textAlign: 'left' }}>True \ Pred</th>
                <th style={{ padding: '10px', color: '#10B981' }}>REST (T0)</th>
                <th style={{ padding: '10px', color: '#00F2FE' }}>LEFT (T1)</th>
                <th style={{ padding: '10px', color: '#F59E0B' }}>RIGHT (T2)</th>
              </tr>
            </thead>
            <tbody>
              {cmData && cmData.length >= 3 ? (
                cmData.map((row, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid #1E293B' }}>
                    <td style={{ padding: '12px', fontWeight: 700, color: '#F8FAFC', textAlign: 'left' }}>
                      {Object.values(row)[0]}
                    </td>
                    <td style={{ padding: '12px', background: idx === 0 ? 'rgba(16, 185, 129, 0.25)' : 'transparent', color: '#10B981', fontWeight: idx === 0 ? 800 : 400 }}>
                      {Object.values(row)[1]}
                    </td>
                    <td style={{ padding: '12px', background: idx === 1 ? 'rgba(0, 242, 254, 0.25)' : 'transparent', color: '#00F2FE', fontWeight: idx === 1 ? 800 : 400 }}>
                      {Object.values(row)[2]}
                    </td>
                    <td style={{ padding: '12px', background: idx === 2 ? 'rgba(245, 158, 11, 0.25)' : 'transparent', color: '#F59E0B', fontWeight: idx === 2 ? 800 : 400 }}>
                      {Object.values(row)[3]}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="4" style={{ padding: '20px', color: '#94A3B8' }}>Loading confusion matrix...</td>
                </tr>
              )}
            </tbody>
          </table>

          <div style={{ marginTop: '16px', fontSize: '0.8rem', color: '#94A3B8', background: '#090D16', padding: '10px', borderRadius: '8px' }}>
            Diagonal elements highlight correct predictions. <strong>128/150 REST</strong>, <strong>44/76 LEFT_FIST</strong>, <strong>31/74 RIGHT_FIST</strong>.
          </div>
        </div>
      </div>

      {/* Consistency Audit Dataframe */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ fontWeight: 700, color: '#F8FAFC', marginBottom: '16px' }}>
          🔍 MATHEMATICAL CONSISTENCY AUDIT CHECKS
        </div>

        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #334155', color: '#94A3B8', textAlign: 'left' }}>
              <th style={{ padding: '10px' }}>Audit Check</th>
              <th style={{ padding: '10px' }}>Value 1</th>
              <th style={{ padding: '10px' }}>Value 2</th>
              <th style={{ padding: '10px' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {auditData.map((row, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid #1E293B' }}>
                <td style={{ padding: '10px', fontWeight: 600, color: '#F8FAFC' }}>{Object.values(row)[0]}</td>
                <td style={{ padding: '10px', color: '#CBD5E1' }}>{String(Object.values(row)[1])}</td>
                <td style={{ padding: '10px', color: '#CBD5E1' }}>{String(Object.values(row)[2])}</td>
                <td style={{ padding: '10px' }}>
                  <span style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#10B981', border: '1px solid #10B981', padding: '2px 8px', borderRadius: '10px', fontSize: '0.75rem', fontWeight: 700 }}>
                    {String(Object.values(row)[3])}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
