import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.step7_pipeline import EXPECTED_CLASSES
from src.command_state_machine import BCICommandStateMachine
from src.ui.avatar_3d import render_3d_avatar_html
from src.ui.components import render_scientific_disclosure_footer
from src.ui.live_bci_helper import LiveBCIPipeline


def render_reliability_lab_page():
    """
    Render 🧠 STEP 28 — REAL-TIME RELIABILITY LAB
    Sections:
    1. System Status
    2. Active Model
    3. Live Prediction HUD
    4. Confidence Monitor
    5. Command State
    6. Prediction Probability Graph
    7. Command Timeline
    8. False Activation Metrics
    9. Stability Metrics
    10. Temporal Strategy Comparison
    11. Global vs Personalized Comparison
    12. Latency Monitor
    13. Dataset Playback
    14. 3D Avatar
    """
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
        <div>
            <h2 style="font-family:'Space Grotesk',sans-serif; margin:0; color:#F8FAFC;">🧠 STEP 28 — REAL-TIME RELIABILITY LAB</h2>
            <div style="font-size:0.85rem; color:#94A3B8;">Real-Time BCI Reliability, Command Stability & False-Action Reduction Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    step28_dir = os.path.abspath("models/step28")

    # Load artifacts if available
    has_artifacts = os.path.exists(os.path.join(step28_dir, "summary_report.json"))

    # ------------------------------------------------------------------
    # 1. SYSTEM STATUS & 2. ACTIVE MODEL SELECTOR
    # ------------------------------------------------------------------
    col_s1, col_s2, col_s3 = st.columns([1.5, 1.5, 1.0])

    with col_s1:
        st.markdown("**1. SYSTEM STATUS**")
        st.markdown("""
        <div style="background:#0F172A; border:1px solid #10B981; border-radius:8px; padding:10px 14px; display:flex; align-items:center; justify-content:space-between;">
            <div>
                <span style="color:#10B981; font-weight:800; font-size:1.0rem;">🟢 SYSTEM READY</span>
                <div style="color:#94A3B8; font-size:0.75rem;">15-Channel Motor Cortex Stream Active</div>
            </div>
            <span style="background:#064E3B; color:#34D399; padding:4px 8px; border-radius:4px; font-size:0.75rem; font-weight:bold;">CAUSAL</span>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown("**2. ACTIVE MODEL**")
        model_choice = st.selectbox(
            "Active Model Choice:",
            options=["🏆 GLOBAL STEP 24 FBCSP", "🎯 PERSONALIZED STEP 27 FBCSP"],
            index=0,
            key="s28_model_choice",
            label_visibility="collapsed"
        )

    with col_s3:
        st.markdown("**BCI MODE**")
        st.markdown("""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:10px; text-align:center;">
            <span style="color:#00F2FE; font-weight:bold;">REAL-TIME STREAM</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:14px 0; border-color:#334155;'>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 13. DATASET PLAYBACK & SUBJECT SELECTION
    # ------------------------------------------------------------------
    st.markdown("### 13. Dataset Stream Playback")
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns([1, 1, 1.2, 1.2, 1.2])

    with col_p1:
        sub_sel = st.selectbox("Subject:", options=[f"S{i:03d}" for i in range(1, 11)], index=0, key="s28_sub_sel")
    with col_p2:
        run_sel = st.selectbox("Run Recording:", options=["R11", "R03", "R07"], index=0, key="s28_run_sel")
    with col_p3:
        strategy_sel = st.selectbox(
            "Temporal Confirmation:",
            options=["HYSTERESIS", "NO_SMOOTHING", "2_CONSECUTIVE", "3_CONSECUTIVE", "2_OF_3_MAJORITY", "3_OF_5_CONFIRMATION", "PROB_ACCUMULATION"],
            index=0,
            key="s28_strat_sel"
        )
    with col_p4:
        playback_speed = st.select_slider("Playback Speed:", options=[0.5, 1.0, 2.0, 4.0], value=1.0, key="s28_speed_sel")
    with col_p5:
        debug_mode = st.checkbox("🛠️ Developer Debug Mode", value=False, key="s28_debug_chk")

    # Session State for Playback Stream
    state_key = f"s28_state_{sub_sel}_{run_sel}"
    if "s28_curr_key" not in st.session_state or st.session_state.s28_curr_key != state_key:
        st.session_state.s28_curr_key = state_key
        st.session_state.s28_win_idx = 0
        st.session_state.s28_is_playing = False
        st.session_state.s28_stream_history = []
        st.session_state.s28_csm = BCICommandStateMachine(strategy=strategy_sel)

    # Initialize LiveBCIPipeline
    try:
        model_type_str = "STEP 24" if "GLOBAL" in model_choice else "STEP 17B"
        pipeline = LiveBCIPipeline(
            subject_id=sub_sel,
            run_id=run_sel,
            data_dir="data",
            models_dir="models",
            model_type=model_type_str
        )
        total_windows = pipeline.get_total_windows()
    except Exception as err:
        st.error(f"Error initializing EEG stream pipeline: {err}")
        st.stop()

    # Playback Control Buttons
    c_b1, c_b2, c_b3, c_b4, c_b5 = st.columns([3.5, 1, 1, 1, 1])

    with c_b1:
        st.session_state.s28_win_idx = st.slider(
            "Playback Window Step:",
            min_value=0,
            max_value=max(0, total_windows - 1),
            value=min(st.session_state.s28_win_idx, max(0, total_windows - 1)),
            step=1,
            key="s28_slider_step"
        )

    with c_b2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.s28_is_playing:
            if st.button("⏸ PAUSE", use_container_width=True, key="s28_btn_pause"):
                st.session_state.s28_is_playing = False
                st.rerun()
        else:
            if st.button("▶ PLAY", use_container_width=True, key="s28_btn_play"):
                st.session_state.s28_is_playing = True
                st.rerun()

    with c_b3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⏮ PREV", use_container_width=True, key="s28_btn_prev"):
            st.session_state.s28_win_idx = max(0, st.session_state.s28_win_idx - 1)
            st.rerun()

    with c_b4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⏭ NEXT", use_container_width=True, key="s28_btn_next"):
            st.session_state.s28_win_idx = min(total_windows - 1, st.session_state.s28_win_idx + 1)
            st.rerun()

    with c_b5:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⟳ RESET", use_container_width=True, key="s28_btn_reset"):
            st.session_state.s28_win_idx = 0
            st.session_state.s28_is_playing = False
            st.session_state.s28_stream_history = []
            st.session_state.s28_csm.reset()
            st.rerun()

    # Predict current window
    win_res = pipeline.predict_window(
        st.session_state.s28_win_idx,
        past_raw_history=[h["raw_prediction"] for h in st.session_state.s28_stream_history]
    )

    if "error" in win_res:
        st.error(f"Stream Prediction Error: {win_res['status_message']}")
        st.stop()

    raw_probs = win_res["probabilities"]
    gt_label = win_res["actual_label"]

    # Process through Command State Machine
    csm_res = st.session_state.s28_csm.process_window(raw_probs, hardware_connected=True, signal_quality_ok=True)
    final_action = csm_res["final_action"]

    # Record history entry
    hist_entry = {
        "win_idx": st.session_state.s28_win_idx,
        "raw_prediction": win_res["raw_prediction"],
        "smoothed_prediction": win_res["smoothed_prediction"],
        "final_action": final_action,
        "ground_truth": gt_label,
        "P_REST": raw_probs.get("REST", 0.0),
        "P_LEFT": raw_probs.get("LEFT_FIST", 0.0),
        "P_RIGHT": raw_probs.get("RIGHT_FIST", 0.0),
        "max_prob": csm_res["max_prob"],
        "margin": csm_res["margin"],
        "entropy": csm_res["entropy"],
        "confidence_accepted": csm_res["confidence_accepted"],
        "latency_ms": win_res["latency"]["total_computation_ms"]
    }
    if not any(h["win_idx"] == st.session_state.s28_win_idx for h in st.session_state.s28_stream_history):
        st.session_state.s28_stream_history.append(hist_entry)

    # ------------------------------------------------------------------
    # 3. LIVE PREDICTION HUD, 4. CONFIDENCE MONITOR, 5. COMMAND STATE, 14. 3D AVATAR
    # ------------------------------------------------------------------
    st.markdown("<hr style='margin:14px 0; border-color:#334155;'>", unsafe_allow_html=True)
    col_hud, col_conf, col_state, col_avatar = st.columns([1.1, 1.1, 1.1, 1.3])

    with col_hud:
        st.markdown("### 3. Prediction HUD")
        p_rest = raw_probs.get("REST", 0.33)
        p_left = raw_probs.get("LEFT_FIST", 0.33)
        p_right = raw_probs.get("RIGHT_FIST", 0.33)

        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; font-size:0.85rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span>REST:</span> <strong style="color:#38BDF8;">{p_rest*100:.1f}%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span>LEFT FIST:</span> <strong style="color:#10B981;">{p_left*100:.1f}%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span>RIGHT FIST:</span> <strong style="color:#F59E0B;">{p_right*100:.1f}%</strong>
            </div>
            <hr style="margin:8px 0; border-color:#334155;">
            <div>Raw: <strong style="color:#F8FAFC;">{win_res['raw_prediction']}</strong></div>
            <div>Ground Truth: <strong style="color:#38BDF8;">{gt_label}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with col_conf:
        st.markdown("### 4. Confidence Monitor")
        conf_status = "HIGH" if csm_res["confidence_accepted"] else "ABSTAIN / UNCERTAIN"
        conf_color = "#10B981" if csm_res["confidence_accepted"] else "#EF4444"

        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; font-size:0.85rem;">
            <div style="font-weight:bold; color:{conf_color}; font-size:1.0rem; margin-bottom:6px;">
                CONFIDENCE: {conf_status}
            </div>
            <div>P_max: <strong style="color:#F8FAFC;">{csm_res['max_prob']:.3f}</strong> (Threshold: ≥ 0.70)</div>
            <div>Margin: <strong style="color:#F8FAFC;">{csm_res['margin']:.3f}</strong> (Threshold: ≥ 0.05)</div>
            <div>Entropy: <strong style="color:#F8FAFC;">{csm_res['entropy']:.3f}</strong> (Threshold: ≤ 1.20)</div>
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">{csm_res['reason']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_state:
        st.markdown("### 5. Command State")
        cmd_color_map = {"REST": "#38BDF8", "LEFT_FIST": "#10B981", "RIGHT_FIST": "#F59E0B", "UNCERTAIN": "#64748B", "SIGNAL_WARNING": "#EF4444", "DISCONNECTED": "#EF4444"}
        cmd_c = cmd_color_map.get(final_action, "#00F2FE")

        st.markdown(f"""
        <div style="background:#0F172A; border:2px solid {cmd_c}; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:0.75rem; color:#94A3B8; font-weight:bold;">FINAL COMMAND STATE</div>
            <div style="font-size:1.5rem; font-weight:800; color:{cmd_c}; font-family:'Space Grotesk',sans-serif; margin:6px 0;">{final_action}</div>
            <div style="font-size:0.8rem; color:#F8FAFC;">Duration: <strong>{csm_res['command_duration_sec']} sec</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with col_avatar:
        st.markdown("### 14. 3D Avatar Viewport")
        try:
            avatar_html = render_3d_avatar_html(
                predicted_class=final_action,
                actual_class=gt_label,
                confidence=csm_res["max_prob"],
                height=260,
                active_channel="C3"
            )
            st.components.v1.html(avatar_html, height=265, scrolling=False)
        except Exception:
            st.info(f"Avatar state: {final_action}")

    # ------------------------------------------------------------------
    # 6. PREDICTION PROBABILITY GRAPH & 7. COMMAND TIMELINE
    # ------------------------------------------------------------------
    st.markdown("<hr style='margin:14px 0; border-color:#334155;'>", unsafe_allow_html=True)
    c_g1, c_g2 = st.columns([1.4, 1.0])

    with c_g1:
        st.markdown("### 6. Real-Time Probability Stream Graph")
        if st.session_state.s28_stream_history:
            df_hist = pd.DataFrame(st.session_state.s28_stream_history)
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(x=df_hist["win_idx"], y=df_hist["P_REST"], mode='lines+markers', name='REST', line=dict(color='#38BDF8', width=2)))
            fig_prob.add_trace(go.Scatter(x=df_hist["win_idx"], y=df_hist["P_LEFT"], mode='lines+markers', name='LEFT_FIST', line=dict(color='#10B981', width=2)))
            fig_prob.add_trace(go.Scatter(x=df_hist["win_idx"], y=df_hist["P_RIGHT"], mode='lines+markers', name='RIGHT_FIST', line=dict(color='#F59E0B', width=2)))
            
            # Threshold lines
            fig_prob.add_hline(y=0.70, line_dash="dash", line_color="#EF4444", annotation_text="Activation Threshold (0.70)")
            fig_prob.add_hline(y=0.45, line_dash="dot", line_color="#F59E0B", annotation_text="Deactivation Threshold (0.45)")

            fig_prob.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                height=320,
                margin=dict(l=35, r=20, t=30, b=35),
                xaxis=dict(title="Window Index", gridcolor="#334155"),
                yaxis=dict(title="Probability", range=[0.0, 1.05], gridcolor="#334155")
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("Step through stream windows to populate real-time probability graph.")

    with c_g2:
        st.markdown("### 7. Command Sequence Timeline")
        if st.session_state.s28_stream_history:
            df_hist = pd.DataFrame(st.session_state.s28_stream_history)
            fig_tl = go.Figure()
            
            # Color map for timeline bars
            color_map = {"REST": "#38BDF8", "LEFT_FIST": "#10B981", "RIGHT_FIST": "#F59E0B", "UNCERTAIN": "#64748B"}
            bar_colors = [color_map.get(a, "#00F2FE") for a in df_hist["final_action"]]

            fig_tl.add_trace(go.Bar(
                x=df_hist["win_idx"],
                y=[1]*len(df_hist),
                marker_color=bar_colors,
                hovertext=df_hist["final_action"]
            ))
            fig_tl.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                height=320,
                margin=dict(l=20, r=20, t=30, b=35),
                xaxis=dict(title="Window Step", gridcolor="#334155"),
                yaxis=dict(visible=False)
            )
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.info("Timeline will render active command blocks.")

    # ------------------------------------------------------------------
    # 8. FALSE ACTIVATION & 9. STABILITY & 12. LATENCY BENCHMARKS
    # ------------------------------------------------------------------
    st.markdown("<hr style='margin:14px 0; border-color:#334155;'>", unsafe_allow_html=True)
    st.markdown("### Step 28 Benchmark Results & Reliability Metrics")

    col_m1, col_m2, col_m3, col_m4 = st.columns([1, 1, 1, 1])

    if has_artifacts:
        with open(os.path.join(step28_dir, "summary_report.json"), "r") as f:
            summary = json.load(f)

        with col_m1:
            st.markdown("""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.75rem; color:#94A3B8;">CHAMPION MODEL</div>
                <div style="font-size:1.2rem; font-weight:bold; color:#00F2FE; margin:4px 0;">{}</div>
                <div style="font-size:0.8rem; color:#10B981;">Window Acc: {}%</div>
            </div>
            """.format(summary.get("best_model", "GLOBAL STEP 24"), summary.get("window_accuracy", 56.0)), unsafe_allow_html=True)

        with col_m2:
            st.markdown("""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.75rem; color:#94A3B8;">8. FALSE ACTIVATION RATE</div>
                <div style="font-size:1.4rem; font-weight:bold; color:#10B981; margin:4px 0;">{}%</div>
                <div style="font-size:0.75rem; color:#94A3B8;">False Switches: {}%</div>
            </div>
            """.format(summary.get("false_activation_rate", 2.67), summary.get("false_switch_rate", 0.0)), unsafe_allow_html=True)

        with col_m3:
            st.markdown("""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.75rem; color:#94A3B8;">9. STABILITY SCORE</div>
                <div style="font-size:1.4rem; font-weight:bold; color:#00F2FE; margin:4px 0;">{} / 100</div>
                <div style="font-size:0.75rem; color:#94A3B8;">Strategy: {}</div>
            </div>
            """.format(summary.get("command_stability", 59.84), summary.get("best_temporal_strategy", "HYSTERESIS")), unsafe_allow_html=True)

        with col_m4:
            st.markdown("""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.75rem; color:#94A3B8;">12. COMPUTATION LATENCY</div>
                <div style="font-size:1.4rem; font-weight:bold; color:#10B981; margin:4px 0;">{} ms</div>
                <div style="font-size:0.75rem; color:#94A3B8;">P95: {} ms</div>
            </div>
            """.format(summary.get("total_computation_latency_ms", 0.73), summary.get("p95_latency_ms", 1.66)), unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 10. TEMPORAL STRATEGY COMPARISON & 11. GLOBAL vs PERSONALIZED COMPARISON
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    t_tab1, t_tab2 = st.tabs(["10. Temporal Strategy Comparison", "11. Global vs Personalized Comparison"])

    with t_tab1:
        if os.path.exists(os.path.join(step28_dir, "temporal_strategy_results.csv")):
            df_strat_res = pd.read_csv(os.path.join(step28_dir, "temporal_strategy_results.csv"))
            st.dataframe(df_strat_res, use_container_width=True)
        else:
            st.info("Run step28_pipeline.py to populate temporal strategy comparison.")

    with t_tab2:
        if os.path.exists(os.path.join(step28_dir, "command_results.csv")):
            df_cmd_res = pd.read_csv(os.path.join(step28_dir, "command_results.csv"))
            st.dataframe(df_cmd_res, use_container_width=True)
        else:
            st.info("Run step28_pipeline.py to populate command results.")

    if debug_mode:
        st.markdown("<hr style='margin:16px 0; border-color:#EF4444;'>", unsafe_allow_html=True)
        st.markdown("### 🛠️ Developer Diagnostics")
        st.json(csm_res)

    render_scientific_disclosure_footer()
