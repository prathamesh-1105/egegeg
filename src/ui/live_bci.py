import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.step7_pipeline import EXPECTED_CLASSES
from src.ui.live_bci_helper import LiveBCIPipeline, COMPATIBLE_TASK1_RUNS, INCOMPATIBLE_TASK3_4_RUNS
from src.ui.avatar_3d import render_3d_avatar_html
from src.ui.components import render_scientific_disclosure_footer
from src.model_integrity import ModelIntegrityVerifier
from src.ac_ir_controller import ACIRController



def render_live_bci_page(raw, info, df_events):
    """
    Render Live BCI Control Center (Requirement 4, 5, 6, 12, 13, 15).
    Primary BCI demonstration screen featuring Step 17B FBCSP model, 3D Avatar,
    real predict_proba HUD, interactive playback controls, latency profiling, and error handling.
    """
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
        <div>
            <h2 style="font-family:'Space Grotesk',sans-serif; margin:0; color:#F8FAFC;">🧠 LIVE BCI CONTROL CENTER</h2>
            <div style="font-size:0.85rem; color:#94A3B8;">Real-Time FBCSP Inference & 3D Avatar Control</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 1. TOP STATUS BAR (Requirement 4)
    # ------------------------------------------------------------------
    col_st1, col_st2, col_st3 = st.columns([1.2, 1.2, 1.0])

    with col_st1:
        st.markdown("**INPUT SOURCE**")
        eeg_source_mode = st.radio(
            "Input Source:",
            options=["PhysioNet Dataset", "LSL Hardware"],
            index=0,
            horizontal=True,
            key="live_bci_source_mode",
            label_visibility="collapsed"
        )

    with col_st2:
        st.markdown("**ACTIVE MODEL**")
        selected_model_option = st.selectbox(
            "Active Model:",
            options=["🏆 STEP 61 SOTA CHAMPION (72.00% Acc, 4.1ms)", "🎯 STEP 60 RIEMANNIAN-FBCSP (70.67% Acc)", "🏆 STEP 59 AUTHENTIC (67.67% Acc, 37.3ms)", "🎯 STEP 58 SUBJECT-ADAPTIVE"],
            index=0,
            key="live_bci_model_select",
            label_visibility="collapsed"
        )

    with col_st3:
        st.markdown("**STATUS**")
        if eeg_source_mode == "LSL Hardware":
            st.markdown("""
            <div style="background:#0F172A; border:1px solid #EF4444; border-radius:6px; padding:6px 12px; font-size:0.85rem; text-align:center;">
                <span style="color:#EF4444; font-weight:bold;">● DISCONNECTED</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0F172A; border:1px solid #10B981; border-radius:6px; padding:6px 12px; font-size:0.85rem; text-align:center;">
                <span style="color:#10B981; font-weight:bold;">● READY</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:12px 0; border-color:#334155;'>", unsafe_allow_html=True)

    if eeg_source_mode == "LSL Hardware":
        st.error("⚠️ HARDWARE INCOMPATIBLE / DISCONNECTED: No physical OpenBCI LSL stream detected on local network. Switch to 'PhysioNet Dataset' to evaluate dataset playback.")
        st.stop()

    # ------------------------------------------------------------------
    # 2. SUBJECT / RUN SELECTOR & SESSION STATE AUDIT (Requirement 5, 16)
    # ------------------------------------------------------------------
    available_subs = [f"S{i:03d}" for i in range(1, 11)]
    available_runs = ["R11", "R03", "R07", "R04", "R08", "R12"]

    col_c1, col_c2, col_c3, col_c4 = st.columns([1, 1, 1, 1])
    with col_c1:
        selected_sub = st.selectbox("Subject:", options=available_subs, index=0, key="live_sub_sel")
    with col_c2:
        selected_run = st.selectbox("Run Recording:", options=available_runs, index=0, key="live_run_sel")
    with col_c3:
        selected_ch = st.selectbox(
            "Waveform Channel:",
            options=info["ch_names"] if "ch_names" in info else ["C3.."],
            index=0,
            key="live_ch_sel"
        )
    with col_c4:
        use_smoothed = st.checkbox("Enable 3-Window Causal Smoothing", value=False, key="live_smoothing_chk")
        debug_mode = st.checkbox("🛠️ Developer Debug Mode", value=False, key="live_debug_chk")

    # Reset playback indices when subject or run changes
    state_key_sub_run = f"last_sub_run_{selected_sub}_{selected_run}"
    if "current_sub_run_key" not in st.session_state or st.session_state.current_sub_run_key != state_key_sub_run:
        st.session_state.current_sub_run_key = state_key_sub_run
        st.session_state.live_win_idx = 0
        st.session_state.live_is_playing = False
        st.session_state.prediction_history = []
        st.session_state.latency_history = []

    # Model Integrity Verifier
    verifier = ModelIntegrityVerifier(models_dir="models", subject_id=selected_sub)
    integ_res = verifier.verify_integrity()
    if not integ_res["is_valid"]:
        st.error(f"🚨 Model Integrity Error: {integ_res.get('message', 'Incompatible artifacts')}")
        st.stop()

    # Pipeline Initialization
    try:
        pipeline = LiveBCIPipeline(
            subject_id=selected_sub,
            run_id=selected_run,
            data_dir="data",
            models_dir="models",
            model_type="STEP 17B"
        )
        total_windows = pipeline.get_total_windows()
    except Exception as err:
        st.markdown(f"""
        <div style="background:#450A0A; border:1px solid #EF4444; border-radius:8px; padding:16px; margin:12px 0;">
            <div style="color:#EF4444; font-weight:bold; font-size:1.0rem;">⚠️ Dataset Loading Error</div>
            <div style="color:#F8FAFC; margin-top:4px;">Could not initialize EEG pipeline for {selected_sub} {selected_run}: {err}</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ------------------------------------------------------------------
    # 3. PLAYBACK CONTROLS BAR (Requirement 5, 15)
    # ------------------------------------------------------------------
    if "live_win_idx" not in st.session_state:
        st.session_state.live_win_idx = 0
    if "live_is_playing" not in st.session_state:
        st.session_state.live_is_playing = False
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "latency_history" not in st.session_state:
        st.session_state.latency_history = []

    c_p1, c_p2, c_p3, c_p4, c_p5 = st.columns([3.5, 1, 1, 1, 1])

    with c_p1:
        st.session_state.live_win_idx = st.slider(
            "Playback Window Step:",
            min_value=0,
            max_value=max(0, total_windows - 1),
            value=min(st.session_state.live_win_idx, max(0, total_windows - 1)),
            step=1,
            key="live_slider_step"
        )

    with c_p2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.live_is_playing:
            if st.button("⏸ PAUSE", use_container_width=True, key="live_btn_pause"):
                st.session_state.live_is_playing = False
                st.toast("PLAYBACK PAUSED")
                st.rerun()
        else:
            if st.button("▶ PLAY", use_container_width=True, key="live_btn_play"):
                st.session_state.live_is_playing = True
                st.toast("PLAYBACK STARTED")
                st.rerun()

    with c_p3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⏮ PREV", use_container_width=True, key="live_btn_prev"):
            st.session_state.live_win_idx = (st.session_state.live_win_idx - 1) % max(1, total_windows)
            st.toast(f"Step {st.session_state.live_win_idx + 1}")
            st.rerun()

    with c_p4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⏭ NEXT", use_container_width=True, key="live_btn_next"):
            st.session_state.live_win_idx = (st.session_state.live_win_idx + 1) % max(1, total_windows)
            st.toast(f"Step {st.session_state.live_win_idx + 1}")
            st.rerun()

    with c_p5:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⟳ RESET", use_container_width=True, key="live_btn_reset"):
            st.session_state.live_win_idx = 0
            st.session_state.live_is_playing = False
            st.session_state.prediction_history = []
            st.session_state.latency_history = []
            st.toast("RESETTING PLAYBACK")
            st.rerun()

    # Predict current window
    res = pipeline.predict_window(
        st.session_state.live_win_idx,
        past_raw_history=[p["raw_pred"] for p in st.session_state.prediction_history]
    )

    if "error" in res:
        st.error(f"Prediction Error: {res['status_message']}")
        st.stop()

    raw_pred = res["raw_prediction"]
    smoothed_pred = res["smoothed_prediction"]
    active_pred = smoothed_pred if use_smoothed else raw_pred
    act_label = res["actual_label"]
    probs = res["probabilities"]
    latency_info = res["latency"]
    is_correct = res["is_correct"]

    # History Entry
    rec_entry = {
        "win_idx": st.session_state.live_win_idx,
        "raw_pred": raw_pred,
        "smoothed_pred": smoothed_pred,
        "actual": act_label,
        "is_correct": is_correct,
        "latency_ms": latency_info["total_computation_ms"]
    }
    if not any(e["win_idx"] == st.session_state.live_win_idx for e in st.session_state.prediction_history):
        st.session_state.prediction_history.append(rec_entry)
        st.session_state.latency_history.append(latency_info["total_computation_ms"])

    # Playback Banner
    res_color = "#10B981" if is_correct else "#EF4444"
    res_str = "✓ CORRECT" if is_correct else "✗ INCORRECT"

    st.markdown(f"""
    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:10px 16px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center; font-size:0.9rem;">
        <div><strong>SUBJECT:</strong> <span style="color:#38BDF8;">{selected_sub}</span></div>
        <div><strong>RUN:</strong> <span style="color:#38BDF8;">{selected_run}</span></div>
        <div><strong>WINDOW:</strong> <span style="color:#F8FAFC;">{res['window_idx'] + 1} / {total_windows}</span></div>
        <div><strong>TIME:</strong> <span style="color:#F8FAFC;">{res['window_start']}–{res['window_end']} sec</span></div>
        <div><strong>GROUND TRUTH:</strong> <span style="color:#F8FAFC;">{act_label}</span></div>
        <div><strong>MODEL:</strong> <span style="color:#00F2FE;">{active_pred}</span></div>
        <div><strong>RESULT:</strong> <span style="color:{res_color}; font-weight:bold;">{res_str}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 4. MAIN GRID: Waveform | 3D Avatar | Prediction HUD (Requirement 4, 13)
    # ------------------------------------------------------------------
    col_wave, col_avatar, col_hud = st.columns([1.3, 1.2, 1.1])

    with col_wave:
        st.markdown("### EEG Waveform")
        sig_data = res["single_channel_uv"]
        t_arr = np.linspace(res["window_start"], res["window_end"], len(sig_data))

        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t_arr, y=sig_data, mode='lines', line=dict(color='#00F2FE', width=2), name=selected_ch))
        fig_wave.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            height=370,
            margin=dict(l=35, r=20, t=30, b=35),
            xaxis=dict(title="Time (sec)", gridcolor="#334155"),
            yaxis=dict(title="Amplitude (µV)", gridcolor="#334155")
        )
        st.plotly_chart(fig_wave, use_container_width=True)

    with col_avatar:
        st.markdown("### 3D BCI Avatar")
        try:
            avatar_html = render_3d_avatar_html(
                predicted_class=active_pred,
                actual_class=act_label,
                confidence=res["confidence"],
                height=370,
                active_channel=selected_ch
            )
            st.components.v1.html(avatar_html, height=375, scrolling=False)
        except Exception as avatar_err:
            st.markdown("""
            <div style="background:#450A0A; border:1px solid #EF4444; border-radius:10px; padding:20px; text-align:center;">
                <div style="color:#EF4444; font-weight:bold;">⚠️ Avatar Configuration Warning</div>
                <div style="color:#94A3B8; font-size:0.85rem; margin-top:6px;">3D Avatar component rendering interrupted. Model predictions remain active.</div>
            </div>
            """, unsafe_allow_html=True)

    with col_hud:
        st.markdown("### Live Prediction HUD")
        color_map = {"REST": "#38BDF8", "LEFT_FIST": "#10B981", "RIGHT_FIST": "#F59E0B"}
        pred_color = color_map.get(active_pred, "#00F2FE")

        st.markdown(f"""
        <div style="background:#0F172A; border:2px solid {pred_color}; border-radius:12px; padding:16px; text-align:center; margin-bottom:12px;">
            <div style="font-size:0.75rem; font-weight:700; color:#94A3B8;">CURRENT PREDICTION</div>
            <div style="font-size:1.8rem; font-weight:800; color:{pred_color}; font-family:'Space Grotesk',sans-serif; margin:4px 0;">{active_pred}</div>
            <div style="font-size:0.8rem; color:#94A3B8;">Raw: <span style="color:#F8FAFC;">{raw_pred}</span> | Smoothed: <span style="color:#F8FAFC;">{smoothed_pred}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Class Probabilities (model.predict_proba)")
        for cls_name in EXPECTED_CLASSES:
            p_val = probs.get(cls_name, 0.333)
            p_color = color_map.get(cls_name, "#00F2FE")
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#F8FAFC;">
                    <span>{cls_name}</span>
                    <span style="font-weight:700; color:{p_color};">{p_val*100:.1f}%</span>
                </div>
                <div style="background:#1E293B; border-radius:4px; height:8px; width:100%; overflow:hidden;">
                    <div style="background:{p_color}; width:{p_val*100}%; height:100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # Smart Appliance Control State (AC Remote Controller)
        # -------------------------------------------------------------
        if "ac_power_state" not in st.session_state:
            st.session_state.ac_power_state = "OFF"
        if "appliance_log" not in st.session_state:
            st.session_state.appliance_log = []

        st.markdown("##### 📡 IR Dispatch Hardware")
        col_ir1, col_ir2 = st.columns([1.6, 1])
        with col_ir1:
            ir_dispatch_mode = st.radio(
                "IR Dispatch Mode:",
                options=["🖥️ SIMULATION", "📱 REDMI NOTE 14 PRO", "🔌 USB ADB CABLE"],
                index=1,
                key="ir_mode_select",
                label_visibility="collapsed"
            )
        with col_ir2:
            phone_ip_input = st.text_input("Target IP:", value="192.168.1.102", key="ir_phone_ip_input")

        mode_key = "REDMI_PHONE" if "REDMI" in ir_dispatch_mode else ("USB_ADB" if "USB" in ir_dispatch_mode else "SIMULATION")
        ir_controller = ACIRController(mode=mode_key, target_ip=phone_ip_input)

        if active_pred == "RIGHT_FIST" and st.session_state.ac_power_state != "ON":
            st.session_state.ac_power_state = "ON"
            res_ir = ir_controller.send_command("ON")
            st.session_state.appliance_log.append(f"🟢 [RIGHT FIST] AC Power ON -> {res_ir.get('log', res_ir.get('message', 'Sent'))}")
        elif active_pred == "LEFT_FIST" and st.session_state.ac_power_state != "OFF":
            st.session_state.ac_power_state = "OFF"
            res_ir = ir_controller.send_command("OFF")
            st.session_state.appliance_log.append(f"🔴 [LEFT FIST] AC Power OFF -> {res_ir.get('log', res_ir.get('message', 'Sent'))}")

        ac_is_on = st.session_state.ac_power_state == "ON"
        ac_bg = "#064E3B" if ac_is_on else "#1E293B"
        ac_border = "#10B981" if ac_is_on else "#64748B"
        ac_status_text = "POWER ON (22°C)" if ac_is_on else "POWER OFF"
        ac_status_color = "#34D399" if ac_is_on else "#94A3B8"

        st.markdown(f"""
        <div style="background:{ac_bg}; border:2px solid {ac_border}; border-radius:10px; padding:14px; margin-top:14px; text-align:center;">
            <div style="font-size:0.75rem; font-weight:700; color:#CBD5E1; letter-spacing:1px;">📡 REDMI NOTE 14 PRO IR BLASTER</div>
            <div style="font-size:1.4rem; font-weight:800; color:{ac_status_color}; margin:6px 0;">{ac_status_text}</div>
            <div style="font-size:0.75rem; color:#94A3B8;">
                Target IP: <span style="color:#38BDF8;">{phone_ip_input}</span> | Mode: <span style="color:#00F2FE;">{mode_key}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # Manual Test Buttons (3 Options: LEFT_FIST, REST, RIGHT_FIST)
        # -------------------------------------------------------------
        st.markdown("##### 🧪 MANUAL HARDWARE TESTER")
        c_m1, c_m2, c_m3 = st.columns([1, 1, 1])
        with c_m1:
            if st.button("🔴 LEFT (OFF)", use_container_width=True, key="btn_test_left"):
                st.session_state.ac_power_state = "OFF"
                res_ir = ir_controller.send_command("OFF")
                st.session_state.appliance_log.append(f"🔴 [MANUAL TEST] Sent AC OFF -> {res_ir.get('log', 'Command sent')}")
                st.toast("🔴 Sent IR OFF Signal!")
                st.rerun()
        with c_m2:
            if st.button("⏸️ REST (IDLE)", use_container_width=True, key="btn_test_rest"):
                st.session_state.appliance_log.append("⏸️ [MANUAL TEST] REST Baseline -> System Standby")
                st.toast("⏸️ REST Baseline")
                st.rerun()
        with c_m3:
            if st.button("🟢 RIGHT (ON)", use_container_width=True, key="btn_test_right"):
                st.session_state.ac_power_state = "ON"
                res_ir = ir_controller.send_command("ON")
                st.session_state.appliance_log.append(f"🟢 [MANUAL TEST] Sent AC ON -> {res_ir.get('log', 'Command sent')}")
                st.toast("🟢 Sent IR ON Signal!")
                st.rerun()





    # ------------------------------------------------------------------
    # 5. LIVE PERFORMANCE & LATENCY PROFILING (Requirement 20)
    # ------------------------------------------------------------------
    st.markdown("<hr style='margin:16px 0; border-color:#334155;'>", unsafe_allow_html=True)
    c_lat1, c_lat2 = st.columns([1, 1])

    with c_lat1:
        st.markdown("### Execution Latency Profile")
        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:12px; font-size:0.85rem;">
            <div>Total Computation: <strong style="color:#10B981;">{latency_info['total_computation_ms']} ms</strong></div>
            <div>Filter Bank + CSP: <span>{latency_info['filter_csp_ms']} ms</span></div>
            <div>SelectKBest Feature Selection: <span>{latency_info['selection_ms']} ms</span></div>
            <div>StandardScaler Transform: <span>{latency_info['scaling_ms']} ms</span></div>
            <div>Classifier Inference: <span>{latency_info['inference_ms']} ms</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c_lat2:
        st.markdown("### Cumulative Performance Metrics")
        if st.session_state.prediction_history:
            df_hist = pd.DataFrame(st.session_state.prediction_history)
            run_acc = float(df_hist["is_correct"].mean() * 100.0)
            st.metric("Playback Running Accuracy", f"{run_acc:.2f}%")
        else:
            st.info("Step through playback windows to compute running accuracy.")

    if debug_mode:
        st.markdown("<hr style='margin:16px 0; border-color:#EF4444;'>", unsafe_allow_html=True)
        st.markdown("### 🛠️ Developer Diagnostic Debug Mode")
        st.json(integ_res)

    render_scientific_disclosure_footer()
