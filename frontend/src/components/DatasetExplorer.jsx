import React, { useState, useEffect } from 'react';
import { Database, FileText, CheckCircle2, Layers, Cpu } from 'lucide-react';

export default function DatasetExplorer() {
  const [datasetData, setDatasetData] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8050/api/dataset/summary')
      .then(res => res.json())
      .then(data => setDatasetData(data))
      .catch(err => console.error('Error fetching dataset summary:', err));
  }, []);

  const channels = [
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
  ];

  return (
    <div style={{ maxWidth: '1400px', margin: '24px auto', padding: '0 24px' }}>
      {/* Hero Header */}
      <div style={{
        background: 'linear-gradient(135deg, #090D16 0%, #0F172A 100%)',
        border: '1px solid rgba(56, 189, 248, 0.3)',
        borderRadius: '14px',
        padding: '24px',
        marginBottom: '24px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div className="neural-badge badge-blue" style={{ marginBottom: '8px' }}>
              <Database size={14} /> BENCHMARK DATASET REGISTRY
            </div>
            <h1 style={{ fontFamily: 'Space Grotesk', margin: '4px 0 0 0', color: '#F8FAFC', fontSize: '2.0rem', letterSpacing: '-0.02em' }}>
              PHYSIONET EEG MOTOR MOVEMENT/IMAGERY DATABASE
            </h1>
            <div style={{ color: '#94A3B8', fontSize: '0.92rem', marginTop: '6px' }}>
              Genuine 64-Channel Multi-Subject Scalp EEG Recordings | Strict Holdout Test Partitioning
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <div className="glass-card" style={{ padding: '12px 18px', borderColor: '#38BDF8', textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', color: '#94A3B8', fontWeight: 700 }}>SUBJECTS</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#38BDF8' }}>10 SUBJECTS</div>
              <div style={{ fontSize: '0.65rem', color: '#64748B' }}>S001 – S010</div>
            </div>

            <div className="glass-card" style={{ padding: '12px 18px', borderColor: '#00F2FE', textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', color: '#94A3B8', fontWeight: 700 }}>EDF RECORDINGS</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#00F2FE' }}>30 FILES</div>
              <div style={{ fontSize: '0.65rem', color: '#64748B' }}>R03, R07, R11</div>
            </div>

            <div className="glass-card" style={{ padding: '12px 18px', borderColor: '#10B981', textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', color: '#94A3B8', fontWeight: 700 }}>TOTAL TRIALS</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#10B981' }}>900 TRIALS</div>
              <div style={{ fontSize: '0.65rem', color: '#64748B' }}>3.0s Epochs @ 160Hz</div>
            </div>
          </div>
        </div>
      </div>

      {/* Grid: Runs breakdown + Channel Topography */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1.3fr', gap: '24px' }}>
        {/* Experimental Runs Breakdown */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#F8FAFC', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={18} color="#00F2FE" />
            Experimental Task Protocols & Split Configuration
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ padding: '14px', background: 'rgba(15, 23, 42, 0.7)', border: '1px solid #1E293B', borderRadius: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 700, color: '#00F2FE' }}>Run R03 (Motor Imagery Baseline)</span>
                <span className="neural-badge badge-cyan">TRAIN (300 TRIALS)</span>
              </div>
              <div style={{ fontSize: '0.8rem', color: '#94A3B8', marginTop: '6px' }}>
                Imagined opening and closing of either the left fist or the right fist. Target cue presentation duration 4.1s.
              </div>
            </div>

            <div style={{ padding: '14px', background: 'rgba(15, 23, 42, 0.7)', border: '1px solid #1E293B', borderRadius: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 700, color: '#38BDF8' }}>Run R07 (Motor Imagery Consolidation)</span>
                <span className="neural-badge badge-blue">TRAIN (300 TRIALS)</span>
              </div>
              <div style={{ fontSize: '0.8rem', color: '#94A3B8', marginTop: '6px' }}>
                Secondary calibration session replicating Task 2 motor imagery with intra-session neural variability.
              </div>
            </div>

            <div style={{ padding: '14px', background: 'rgba(15, 23, 42, 0.7)', border: '1px solid rgba(16, 185, 129, 0.4)', borderRadius: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 700, color: '#10B981' }}>Run R11 (Strict Holdout Test Run)</span>
                <span className="neural-badge badge-emerald">HOLDOUT (300 TRIALS)</span>
              </div>
              <div style={{ fontSize: '0.8rem', color: '#94A3B8', marginTop: '6px' }}>
                Isolated test session. Evaluated exactly once under frozen model parameters to guarantee absolute zero data leakage.
              </div>
            </div>
          </div>
        </div>

        {/* 15 Group B Motor Channel Layout */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#F8FAFC', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={18} color="#38BDF8" />
            Group B Sensorimotor Electrode Array (10-20 Standard)
          </div>
          <div style={{ fontSize: '0.85rem', color: '#94A3B8', marginBottom: '14px' }}>
            15 bilateral electrodes centered over the primary motor cortex (M1) and supplementary motor area (SMA):
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '10px' }}>
            {channels.map((ch) => {
              const isKey = ch === 'C3' || ch === 'C4' || ch === 'Cz';
              return (
                <div
                  key={ch}
                  style={{
                    background: isKey ? 'rgba(0, 242, 254, 0.15)' : 'rgba(15, 23, 42, 0.8)',
                    border: `1px solid ${isKey ? '#00F2FE' : '#334155'}`,
                    borderRadius: '8px',
                    padding: '12px 6px',
                    textAlign: 'center'
                  }}
                >
                  <div style={{ fontSize: '1.0rem', fontWeight: 800, color: isKey ? '#00F2FE' : '#F8FAFC' }}>
                    {ch}
                  </div>
                  <div style={{ fontSize: '0.65rem', color: isKey ? '#38BDF8' : '#64748B', marginTop: '2px' }}>
                    {isKey ? 'PRIMARY M1' : 'PARIETAL'}
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748B', marginTop: '16px' }}>
            * C3 & C4 capture contralateral Event-Related Desynchronization (ERD) during left/right fist motor imagery.
          </div>
        </div>
      </div>
    </div>
  );
}
