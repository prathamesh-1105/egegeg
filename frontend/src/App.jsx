import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import MultiCommandCenter from './components/MultiCommandCenter';
import EEGScrollingDemo from './components/EEGScrollingDemo';
import CalibrationStudio from './components/CalibrationStudio';
import NeuroOSCenter from './components/NeuroOSCenter';
import SpeechBCICenter from './components/SpeechBCICenter';
import SimplePredictorFrame from './components/SimplePredictorFrame';
import LiveDecoder from './components/LiveDecoder';
import FeaturePlayground from './components/FeaturePlayground';
import SignalExplorer from './components/SignalExplorer';
import ModelCenter from './components/ModelCenter';
import ValidationCenter from './components/ValidationCenter';
import HardwareTelemetry from './components/HardwareTelemetry';
import DatasetExplorer from './components/DatasetExplorer';

export default function App() {
  const [activeTab, setActiveTab] = useState('multi-command');
  const [statusData, setStatusData] = useState(null);
  const [apiStatus, setApiStatus] = useState('CHECKING');

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('http://localhost:8050/api/status');
        if (res.ok) {
          const data = await res.json();
          setStatusData(data);
          setApiStatus('ONLINE');
        } else {
          setApiStatus('OFFLINE');
        }
      } catch (err) {
        console.error('FastAPI Backend connection error:', err);
        setApiStatus('OFFLINE');
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 8000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-dark)', color: '#F8FAFC', display: 'flex', flexDirection: 'column' }}>
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} apiStatus={apiStatus} />

      <main style={{ flex: 1, paddingBottom: '30px' }}>
        {activeTab === 'multi-command' && <MultiCommandCenter />}
        {activeTab === 'scroll-demo' && <EEGScrollingDemo />}
        {activeTab === 'calibration' && <CalibrationStudio />}
        {activeTab === 'neuro-os' && <NeuroOSCenter />}
        {activeTab === 'speech' && <SpeechBCICenter />}
        {activeTab === 'simple' && <SimplePredictorFrame />}
        {activeTab === 'live' && <LiveDecoder />}
        {activeTab === 'features' && <FeaturePlayground />}
        {activeTab === 'explorer' && <SignalExplorer />}
        {activeTab === 'models' && <ModelCenter statusData={statusData} />}
        {activeTab === 'validation' && <ValidationCenter />}
        {activeTab === 'hardware' && <HardwareTelemetry />}
        {activeTab === 'dataset' && <DatasetExplorer />}
      </main>

      {/* Cyberpunk Footer */}
      <footer style={{
        textAlign: 'center',
        padding: '24px 20px',
        borderTop: '1px solid rgba(51, 65, 85, 0.5)',
        background: 'rgba(9, 13, 22, 0.95)',
        fontSize: '0.82rem',
        color: '#64748B'
      }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '16px', flexWrap: 'wrap', marginBottom: '8px' }}>
          <span style={{ color: '#00F2FE', fontWeight: 700 }}>NEURO BCI v2.5</span>
          <span>•</span>
          <span>PhysioNet EEGMMIDB Motor Imagery</span>
          <span>•</span>
          <span>Spatial-Temporal Transformer (EEGConformer)</span>
          <span>•</span>
          <span>Step 61 Multimodal Riemannian-FBCSP Ensemble</span>
        </div>
        <div style={{ fontSize: '0.74rem', color: '#475569' }}>
          Real 15-Channel Microvolt Signal Processing • OpenBCI Cyton & Daisy Compatible • Strict Zero Leakage Architecture
        </div>
      </footer>
    </div>
  );
}
