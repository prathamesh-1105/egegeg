import streamlit as st

def inject_custom_theme():
    """
    Injects custom CSS styling for a futuristic, cinematic sci-fi BCI interface.
    """
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        /* Force Global Dark Background */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], .stApp {
            background-color: #040711 !important;
            color: #F8FAFC !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 50% 0%, rgba(0, 242, 254, 0.05) 0%, transparent 65%),
                        radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.04) 0%, transparent 55%),
                        #040711 !important;
            background-attachment: fixed;
        }

        /* Hide Streamlit default chrome while keeping sidebar toggle button accessible */
        #MainMenu, footer {
            visibility: hidden;
            height: 0;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #040711;
        }
        ::-webkit-scrollbar-thumb {
            background: #1E293B;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #00F2FE;
        }

        /* Top Navigation Header Bar */
        .top-nav-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(11, 17, 32, 0.95);
            border: 1px solid rgba(56, 189, 248, 0.2);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(16px);
            padding: 14px 24px;
            border-radius: 16px;
            margin-bottom: 20px;
        }

        .nav-logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.4rem;
            letter-spacing: -0.02em;
            color: #F8FAFC;
        }
        .nav-logo-icon {
            font-size: 1.6rem;
            filter: drop-shadow(0 0 10px rgba(0, 242, 254, 0.8));
        }
        .nav-logo-text {
            background: linear-gradient(135deg, #FFFFFF 0%, #00F2FE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #10B981;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #10B981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10B981;
            animation: pulse-animation 2s infinite;
        }
        @keyframes pulse-animation {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* Hero Section */
        .hero-wrapper {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(11, 17, 32, 0.95) 100%);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 24px;
            padding: 36px 40px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }

        .hero-badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: #38BDF8;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 14px;
        }

        .hero-main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.03em;
            margin: 0 0 12px 0;
            background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 50%, #00F2FE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-tagline {
            font-size: 1.2rem;
            font-weight: 500;
            color: #00F2FE;
            margin-bottom: 12px;
        }

        .hero-description {
            font-size: 0.98rem;
            color: #94A3B8;
            line-height: 1.6;
            margin-bottom: 24px;
            max-width: 580px;
        }

        /* Glassmorphic Metric Strip Cards */
        .status-strip-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }
        .status-card {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(56, 189, 248, 0.2);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 16px 20px;
            transition: all 0.25s ease;
        }
        .status-card:hover {
            border-color: rgba(0, 242, 254, 0.5);
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 242, 254, 0.15);
        }
        .status-card-label {
            font-size: 0.72rem;
            font-weight: 700;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }
        .status-card-val {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: #F8FAFC;
        }

        /* HUD Prediction Panel */
        .hud-pred-panel {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.95) 0%, rgba(11, 17, 32, 0.98) 100%);
            border: 2px solid #00F2FE;
            border-radius: 20px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 0 30px rgba(0, 242, 254, 0.2);
            position: relative;
        }
        .hud-pred-title {
            color: #94A3B8;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 8px;
        }
        .hud-pred-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #00F2FE;
            margin: 8px 0;
            text-shadow: 0 0 20px rgba(0, 242, 254, 0.5);
        }

        /* Animated Pipeline Nodes */
        .sci-pipeline-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(11, 17, 32, 0.9);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 16px;
            padding: 18px 24px;
            margin-bottom: 28px;
        }
        .sci-node {
            background: #0F172A;
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 12px 18px;
            text-align: center;
            font-size: 0.82rem;
            font-weight: 700;
            color: #F8FAFC;
            box-shadow: 0 4px 14px rgba(0,0,0,0.4);
            transition: all 0.3s ease;
        }
        .sci-node.active {
            border-color: #00F2FE;
            color: #00F2FE;
            box-shadow: 0 0 16px rgba(0, 242, 254, 0.35);
        }
        .sci-arrow {
            color: #64748B;
            font-size: 1.2rem;
            font-weight: bold;
            padding: 0 4px;
        }

        /* Disclaimer & Footer */
        .disclaimer-box {
            background: rgba(15, 23, 42, 0.7);
            border-left: 4px solid #38BDF8;
            border-radius: 8px;
            padding: 14px 20px;
            font-size: 0.85rem;
            color: #CBD5E1;
            margin-top: 32px;
            margin-bottom: 24px;
        }
        .footer-wrapper {
            text-align: center;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 24px;
            padding-bottom: 32px;
            color: #64748B;
            font-size: 0.82rem;
        }
        .footer-brand {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            color: #94A3B8;
            margin-bottom: 4px;
        }

        /* Radio Buttons Navigation Segmented Control Styling */
        div[data-testid="stRadio"] > div {
            background-color: rgba(11, 17, 32, 0.9);
            padding: 6px;
            border-radius: 14px;
            border: 1px solid rgba(56, 189, 248, 0.2);
            gap: 6px;
        }
        div[data-testid="stRadio"] label {
            background-color: transparent;
            color: #94A3B8 !important;
            padding: 8px 18px !important;
            border-radius: 10px;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 0.88rem;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }
        div[data-testid="stRadio"] label:hover {
            color: #F8FAFC !important;
            background: rgba(30, 41, 59, 0.6);
        }
        div[data-testid="stRadio"] label[data-checked="true"] {
            background-color: #1E293B !important;
            color: #00F2FE !important;
            border-color: rgba(0, 242, 254, 0.4) !important;
            box-shadow: 0 4px 14px rgba(0, 242, 254, 0.2);
        }

        /* Streamlit Button Styling */
        .stButton button {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-radius: 10px !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            padding: 10px 20px !important;
            transition: all 0.2s ease !important;
        }
        .stButton button:hover {
            border-color: #00F2FE !important;
            color: #00F2FE !important;
            box-shadow: 0 4px 16px rgba(0, 242, 254, 0.25) !important;
            transform: translateY(-1px) !important;
        }

        /* Section Headings */
        .sci-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            margin-top: 28px;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .sci-header::before {
            content: '';
            display: inline-block;
            width: 4px;
            height: 20px;
            background: #00F2FE;
            border-radius: 2px;
            box-shadow: 0 0 8px #00F2FE;
        }
    </style>
    """, unsafe_allow_html=True)
