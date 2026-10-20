import React, { useState } from 'react';
import {
  Activity,
  Sliders,
  LineChart,
  Cpu,
  ShieldCheck,
  Zap,
  Power,
  Mic,
  Radio,
  Database,
  Compass,
  Layers,
  ChevronDown,
  Sparkles
} from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, apiStatus }) {
  const [showMoreMenu, setShowMoreMenu] = useState(false);

  // Core high-frequency tabs (simple, immediate, intuitive)
  const primaryTabs = [
    { id: 'multi-command', label: 'Live BCI Control', icon: Layers, badge: 'FLAGSHIP' },
    { id: 'scroll-demo', label: 'EEG Scroller', icon: Compass },
    { id: 'models', label: 'AI Models & Benchmarks', icon: Cpu, badge: 'PHASE 7' },
    { id: 'calibration', label: 'Calibration Studio', icon: Sliders },
    { id: 'hardware', label: 'Hardware & Signals', icon: Radio },
  ];

  // Secondary research & exploration tools
  const secondaryTabs = [
    { id: 'validation', label: 'Scientific Validation Audit', icon: ShieldCheck, desc: 'Zero-leakage proof & logs' },
    { id: 'live', label: 'Live Signal Decoder', icon: Activity, desc: 'Real-time classification pipeline' },
    { id: 'explorer', label: 'Microvolt EEG Explorer', icon: LineChart, desc: 'Channel waveforms & PSD' },
    { id: 'speech', label: 'Imagined Speech BCI', icon: Mic, desc: 'Inner speech decoding lab' },
    { id: 'neuro-os', label: 'NeuroOS Architecture', icon: Power, desc: 'System orchestrator & shield' },
    { id: 'features', label: 'Feature Playground', icon: Sliders, desc: 'Spectral & asymmetry tuning' },
    { id: 'dataset', label: 'PhysioNet Dataset Hub', icon: Database, desc: '109-subject corpus inventory' },
    { id: 'simple', label: '3D Direction Predictor', icon: Zap, desc: 'Interactive 3D direction view' }
  ];

  const isSecondaryActive = secondaryTabs.some(t => t.id === activeTab);
  const activeSecondaryItem = secondaryTabs.find(t => t.id === activeTab);

  return (
    <nav style={{
      background: 'rgba(9, 13, 22, 0.94)',
      backdropFilter: 'blur(16px)',
      WebkitBackdropFilter: 'blur(16px)',
      borderBottom: '1px solid rgba(51, 65, 85, 0.5)',
      padding: '10px 24px',
      position: 'sticky',
      top: 0,
      zIndex: 100,
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.35)'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        maxWidth: '1600px',
        margin: '0 auto',
        gap: '16px',
        flexWrap: 'wrap'
      }}>
        {/* Brand Logo */}
        <div
          onClick={() => setActiveTab('multi-command')}
          style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}
        >
          <div style={{
            background: 'linear-gradient(135deg, #00F2FE 0%, #2563EB 100%)',
            borderRadius: '10px',
            padding: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 16px rgba(0, 242, 254, 0.4)'
          }}>
            <Zap size={22} color="#070B14" />
          </div>
          <div>
            <div style={{ fontFamily: 'Space Grotesk, sans-serif', fontSize: '1.25rem', fontWeight: 800, color: '#F8FAFC', letterSpacing: '0.03em', lineHeight: 1.1 }}>
              NEURO <span style={{ color: '#00F2FE' }}>BCI</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#94A3B8', fontWeight: 500 }}>
              Motor Imagery Control Platform
            </div>
          </div>
        </div>

        {/* Primary Clean Navigation Tabs */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          background: 'rgba(15, 23, 42, 0.8)',
          padding: '4px',
          borderRadius: '12px',
          border: '1px solid rgba(51, 65, 85, 0.5)'
        }}>
          {primaryTabs.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveTab(item.id);
                  setShowMoreMenu(false);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 14px',
                  borderRadius: '8px',
                  border: 'none',
                  background: isActive
                    ? 'linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%)'
                    : 'transparent',
                  color: isActive ? '#00F2FE' : '#94A3B8',
                  boxShadow: isActive ? '0 0 12px rgba(0, 242, 254, 0.2)' : 'none',
                  borderBottom: isActive ? '2px solid #00F2FE' : '2px solid transparent',
                  fontWeight: isActive ? 700 : 500,
                  fontSize: '0.84rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  whiteSpace: 'nowrap'
                }}
              >
                <Icon size={16} color={isActive ? '#00F2FE' : '#64748B'} />
                <span>{item.label}</span>
                {item.badge && (
                  <span style={{
                    fontSize: '0.62rem',
                    fontWeight: 800,
                    padding: '2px 6px',
                    borderRadius: '4px',
                    background: isActive ? 'rgba(0, 242, 254, 0.2)' : 'rgba(56, 189, 248, 0.1)',
                    color: isActive ? '#00F2FE' : '#38BDF8',
                    letterSpacing: '0.04em'
                  }}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}

          {/* More Tools Dropdown */}
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setShowMoreMenu(!showMoreMenu)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '8px 12px',
                borderRadius: '8px',
                border: 'none',
                background: isSecondaryActive
                  ? 'rgba(56, 189, 248, 0.15)'
                  : (showMoreMenu ? 'rgba(51, 65, 85, 0.5)' : 'transparent'),
                color: isSecondaryActive ? '#38BDF8' : '#94A3B8',
                borderBottom: isSecondaryActive ? '2px solid #38BDF8' : '2px solid transparent',
                fontWeight: isSecondaryActive ? 700 : 500,
                fontSize: '0.84rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                whiteSpace: 'nowrap'
              }}
            >
              <Sparkles size={15} color={isSecondaryActive ? '#38BDF8' : '#64748B'} />
              <span>{isSecondaryActive ? activeSecondaryItem?.label : 'Research Tools'}</span>
              <ChevronDown size={14} color="#64748B" style={{ transform: showMoreMenu ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
            </button>

            {showMoreMenu && (
              <div style={{
                position: 'absolute',
                top: '100%',
                right: 0,
                marginTop: '8px',
                background: '#0F172A',
                border: '1px solid rgba(51, 65, 85, 0.8)',
                borderRadius: '12px',
                padding: '8px',
                width: '300px',
                boxShadow: '0 12px 36px rgba(0, 0, 0, 0.7)',
                zIndex: 200,
                display: 'flex',
                flexDirection: 'column',
                gap: '4px'
              }}>
                <div style={{ padding: '6px 10px', fontSize: '0.72rem', fontWeight: 700, color: '#64748B', letterSpacing: '0.06em' }}>
                  ADVANCED RESEARCH SUITES
                </div>
                {secondaryTabs.map((item) => {
                  const Icon = item.icon;
                  const isActive = activeTab === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => {
                        setActiveTab(item.id);
                        setShowMoreMenu(false);
                      }}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '10px',
                        padding: '8px 10px',
                        borderRadius: '8px',
                        border: 'none',
                        background: isActive ? 'rgba(0, 242, 254, 0.12)' : 'transparent',
                        color: isActive ? '#00F2FE' : '#F8FAFC',
                        textAlign: 'left',
                        cursor: 'pointer',
                        transition: 'background 0.15s ease',
                        width: '100%'
                      }}
                      onMouseEnter={(e) => {
                        if (!isActive) e.currentTarget.style.background = 'rgba(30, 41, 59, 0.7)';
                      }}
                      onMouseLeave={(e) => {
                        if (!isActive) e.currentTarget.style.background = 'transparent';
                      }}
                    >
                      <div style={{
                        padding: '6px',
                        borderRadius: '6px',
                        background: isActive ? 'rgba(0, 242, 254, 0.2)' : 'rgba(51, 65, 85, 0.4)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <Icon size={16} color={isActive ? '#00F2FE' : '#94A3B8'} />
                      </div>
                      <div>
                        <div style={{ fontSize: '0.84rem', fontWeight: isActive ? 700 : 600 }}>{item.label}</div>
                        <div style={{ fontSize: '0.7rem', color: '#64748B' }}>{item.desc}</div>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Right Status Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(15, 23, 42, 0.9)',
            padding: '6px 12px',
            borderRadius: '20px',
            border: '1px solid rgba(51, 65, 85, 0.6)'
          }}>
            <span
              className={apiStatus === 'ONLINE' ? 'pulse-indicator-emerald' : ''}
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: apiStatus === 'ONLINE' ? '#10B981' : '#EF4444'
              }}
            />
            <span style={{
              fontSize: '0.74rem',
              fontWeight: 700,
              color: apiStatus === 'ONLINE' ? '#10B981' : '#EF4444',
              fontFamily: 'JetBrains Mono, monospace'
            }}>
              API {apiStatus}
            </span>
          </div>

          <div style={{
            background: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(16, 185, 129, 0.4)',
            color: '#10B981',
            padding: '5px 12px',
            borderRadius: '16px',
            fontSize: '0.74rem',
            fontWeight: 700,
            letterSpacing: '0.04em'
          }}>
            PHASE 7 ENGINE • READY
          </div>
        </div>
      </div>
    </nav>
  );
}
