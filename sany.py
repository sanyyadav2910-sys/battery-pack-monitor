import json
import math
import os
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

# ---------------- Config ----------------
load_dotenv("ayush.env")
BROKER = os.getenv("MQTT_BROKER")
PORT = int(os.getenv("MQTT_PORT", 8883))
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")
TOPIC = os.getenv("MQTT_TOPIC")

st.set_page_config(page_title="Battery Pack Monitor", page_icon="🔋", layout="wide")
st.title("🔋 Battery Pack Monitor")

# Simple cell model: open-circuit voltage vs state of charge
OCV_SOC = [0, 10, 20, 50, 80, 90, 100]
OCV_V = [3.00, 3.35, 3.55, 3.75, 3.95, 4.08, 4.20]

# Operating modes -> C-rate (+ discharge, - charge). None = automatic drive cycle
MODES = {
    "😴 Idle": 0.02,
    "🚗 Cruise": 0.5,
    "🏎️ Heavy load": 1.5,
    "🔌 Charge": -0.5,
    "⚡ Fast charge": -1.5,
    "🛣️ Drive cycle": None,
}

GREEN, ORANGE, RED, BLUE = "#2ecc71", "#f39c12", "#e74c3c", "#2f80ed"


# ---------------- MQTT (connected once, reused across reruns) ----------------
@st.cache_resource
def get_mqtt_client():
    if not all([BROKER, USERNAME, PASSWORD, TOPIC]):
        raise ValueError("Missing MQTT_* values in ayush.env")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    return client


# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    pack_id = st.text_input("Pack ID", "PACK_01")
    num_cells = st.slider("Number of cells (series)", 2, 12, 4)
    capacity_ah = st.number_input("Pack capacity (Ah)", 5.0, 200.0, 20.0, 1.0)
    start_soc = st.slider("Starting SOC (%) - applied on reset", 10, 100, 85)
    soh = st.slider("State of health (%)", 60, 100, 96)
    interval_sec = st.slider("Update interval (sec)", 1, 10, 2)
    sim_speed = st.slider("Simulation speed (x)", 1, 120, 30,
                          help="1 real second = this many simulated seconds. Higher = SOC and temperature move faster.")
    max_points = st.slider("Points kept in memory", 50, 1000, 300, 50)
    window = st.slider("Chart window (samples)", 20, 1000, 150, 10)

    st.subheader("Environment")
    ambient = st.slider("Ambient temperature (°C)", -10, 45, 28)

    st.subheader("Imbalance simulation")
    imbalance_mode = st.toggle("Imbalance mode", value=True)
    weak_cell = st.selectbox("Weak cell", list(range(1, num_cells + 1)), index=num_cells - 1)
    drop_rate = st.number_input("Drop per step (V)", 0.0, 0.01, 0.0005, 0.0001, format="%.4f")

    st.subheader("Alert thresholds")
    warn_th = st.number_input("V_delta warning (V)", 0.005, 0.2, 0.020, 0.005, format="%.3f")
    crit_th = st.number_input("V_delta critical (V)", 0.01, 0.5, 0.050, 0.005, format="%.3f")
    warn_temp = st.slider("Temp warning (°C)", 30, 70, 45)
    crit_temp = st.slider("Temp critical (°C)", 40, 90, 60)
    max_c = st.slider("Max current (C-rate)", 1.0, 5.0, 2.5, 0.1)

    st.subheader("Control")
    publish_mqtt = st.toggle("Publish to HiveMQ", value=True)
    running = st.toggle("Running", value=True)
    protect = st.toggle("BMS protection", value=True,
                        help="Opens the contactor on over-temperature or over-current.")
    balancing = st.toggle("Passive cell balancing", value=False)
    step_clicked = st.button("⏭️ Step once", use_container_width=True)
    reset_clicked = st.button("🗑️ Reset data", use_container_width=True)


# ---------------- Session state ----------------
def init_state():
    ss = st.session_state
    ss.history = []
    ss.weak_drop = 0.0
    ss.soc = float(start_soc)
    ss.temp = float(ambient)
    ss.cell_off = np.random.normal(0, 0.004, num_cells)
    ss.events = []
    ss.tick = 0
    ss.e_out = 0.0
    ss.e_in = 0.0
    ss.ah_thru = 0.0
    ss.open = False
    ss.surge = 0
    ss.last_status = "ok"
    ss.step_once = False
    ss.n_cells = num_cells


def log_event(level, msg):
    ss = st.session_state
    ss.events.append({"time": datetime.now(timezone.utc).strftime("%H:%M:%S"), "level": level, "event": msg})
    ss.events = ss.events[-200:]


if "history" not in st.session_state or st.session_state.get("n_cells") != num_cells:
    init_state()
if reset_clicked:
    init_state()
    st.rerun()
if step_clicked:
    st.session_state.step_once = True

mqtt_client = None
if publish_mqtt:
    try:
        mqtt_client = get_mqtt_client()
        st.sidebar.success("Connected to HiveMQ")
    except Exception as e:
        st.sidebar.error(f"MQTT error: {e}")

cell_cols = [f"Cell {i + 1}" for i in range(num_cells)]

# ---------------- Control deck (live interaction) ----------------
st.markdown("#### 🎛️ Control deck")
d1, d2, d3 = st.columns([3, 2, 2])
with d1:
    mode = st.radio("Operating mode", list(MODES), index=1, horizontal=True)
with d2:
    load_scale = st.slider("Load multiplier (discharge)", 0.5, 2.0, 1.0, 0.1)
with d3:
    fan = st.slider("Cooling fan (%)", 0, 100, 30, 10)

f1, f2, f3, f4 = st.columns(4)
if f1.button("🔥 Overheat spike (+18°C)", use_container_width=True):
    st.session_state.temp += 18
    log_event("warning", "Fault injected: overheat spike +18°C")
if f2.button("⚡ Overcurrent surge", use_container_width=True):
    st.session_state.surge = 5
    log_event("warning", "Fault injected: overcurrent surge (5 steps)")
if f3.button("💥 Cell fault", use_container_width=True):
    st.session_state.weak_drop += 0.15
    log_event("warning", f"Fault injected: voltage collapse on Cell {weak_cell}")
if f4.button("🧯 Clear faults", use_container_width=True):
    st.session_state.surge = 0
    st.session_state.weak_drop = 0.0
    log_event("info", "Faults cleared")


# ---------------- Helpers ----------------
def health_status(v_delta):
    if v_delta >= crit_th:
        return "critical"
    if v_delta >= warn_th:
        return "warning"
    return "ok"


def temp_status(t):
    if t >= crit_temp:
        return "critical"
    if t >= warn_temp:
        return "warning"
    return "ok"


def worst(*states):
    for s in ("critical", "warning"):
        if s in states:
            return s
    return "ok"


def health_score(v_delta, temp, soc, c_rate):
    s = 100.0
    s -= min(40.0, v_delta / crit_th * 30)
    s -= min(35.0, max(0.0, temp - 35) / max(crit_temp - 35, 1) * 30)
    s -= max(0.0, 20 - soc) * 0.75
    s -= (100 - soh) * 0.5
    s -= max(0.0, abs(c_rate) - 1.5) * 10
    return max(0.0, min(100.0, s))


def fmt_hours(h):
    if h is None or h > 99:
        return "-"
    hh = int(h)
    return f"{hh}h {int((h - hh) * 60):02d}m"


def style(fig, h=300):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.15), hovermode="x unified")
    return fig


def show(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key)


def make_gauge(value, title, lo, hi, suffix, steps, bar=BLUE, fmt=".1f"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": suffix, "valueformat": fmt},
        title={"text": title, "font": {"size": 15}},
        gauge={"axis": {"range": [lo, hi]}, "bar": {"color": bar},
               "steps": [{"range": [a, b], "color": c} for a, b, c in steps]}))
    fig.update_layout(height=210, margin=dict(l=20, r=20, t=50, b=5))
    return fig


def cell_cards_html(cells):
    top = cells.max()
    html = '<div style="display:flex;gap:10px;flex-wrap:wrap">'
    for name, v in cells.items():
        dev = top - v
        color = RED if dev >= crit_th else ORANGE if dev >= warn_th else GREEN
        pct = max(0.0, min(100.0, (v - 3.0) / 1.2 * 100))
        html += f"""
        <div style="flex:1;min-width:72px;text-align:center">
          <div style="font-size:12px;opacity:.7">{name}</div>
          <div style="height:90px;border:2px solid {color};border-radius:6px;position:relative;overflow:hidden;margin:4px 0">
            <div style="position:absolute;bottom:0;left:0;right:0;height:{pct:.0f}%;background:{color};opacity:.75"></div>
          </div>
          <div style="font-weight:600">{v:.3f} V</div>
          <div style="font-size:11px;opacity:.7">&Delta; -{dev * 1000:.0f} mV</div>
        </div>"""
    return html + "</div>"


# ---------------- Data generation ----------------
def generate_reading():
    ss = st.session_state
    ss.tick += 1
    dt = interval_sec * sim_speed                      # simulated seconds per step
    cap_eff = capacity_ah * soh / 100

    # --- requested current (A): + discharge, - charge
    c_req = MODES[mode]
    if c_req is None:                                  # automatic drive cycle with regen
        c_req = 0.7 + 0.6 * math.sin(ss.tick / 6) + 0.3 * math.sin(ss.tick / 2.3) + random.gauss(0, 0.08)
        c_req = max(c_req, -0.6)
    else:
        c_req += random.gauss(0, 0.02)
    if c_req > 0:
        c_req *= load_scale
    if ss.surge > 0:
        c_req = 3.5
        ss.surge -= 1
    if c_req < 0:                                      # CC-CV: taper near full
        c_req *= min(1.0, max(0.05, (100 - ss.soc) / 15))
        if ss.soc >= 100:
            c_req = 0.0
    if c_req > 0 and ss.soc <= 0:
        c_req = 0.0
    current = c_req * cap_eff

    # --- BMS protection
    if protect:
        reason = None
        if ss.temp >= crit_temp:
            reason = "over-temperature"
        elif abs(current) > max_c * cap_eff:
            reason = "over-current"
        if reason and not ss.open:
            ss.open = True
            log_event("critical", f"BMS opened contactor: {reason}")
            st.toast(f"BMS tripped: {reason}", icon="⛔")
        elif ss.open and not reason and ss.temp < warn_temp:
            ss.open = False
            log_event("info", "BMS re-closed contactor")
    else:
        ss.open = False
    if ss.open:
        current = 0.0

    # --- imbalance + balancing
    if imbalance_mode:
        ss.weak_drop += drop_rate
    if balancing:
        ss.cell_off += -(ss.cell_off - ss.cell_off.mean()) * 0.25
        ss.weak_drop = max(0.0, ss.weak_drop - 0.0015)

    # --- cell voltages: OCV(SOC) + offsets - I*R + noise
    ocv = float(np.interp(ss.soc, OCV_SOC, OCV_V))
    temp_factor = 1 + max(0.0, 25 - ss.temp) * 0.02
    r_cell = 0.003 * (1 + 0.8 * (1 - soh / 100)) * temp_factor * (20 / capacity_ah)
    r_cells = np.full(num_cells, r_cell)
    idx = weak_cell - 1
    if imbalance_mode:
        r_cells[idx] *= 1.4
    cells = ocv + ss.cell_off - current * r_cells + np.random.normal(0, 0.002, num_cells)
    if imbalance_mode or ss.weak_drop > 0:
        cells[idx] -= ss.weak_drop
    cells = np.clip(cells, 2.5, 4.3).round(3)

    # --- thermal model (exponential approach to steady state)
    p_loss = current ** 2 * float(r_cells.sum()) * 6
    h = 1.2 + 0.05 * fan
    t_ss = ambient + p_loss / h
    ss.temp += (t_ss - ss.temp) * (1 - math.exp(-h * dt / 150))

    # --- SOC and energy
    d_ah = current * dt / 3600
    ss.soc = min(100.0, max(0.0, ss.soc - d_ah / cap_eff * 100))
    pack_v = float(cells.sum())
    power = pack_v * current
    wh = power * dt / 3600
    if wh >= 0:
        ss.e_out += wh
    else:
        ss.e_in += -wh
    ss.ah_thru += abs(d_ah)

    v_delta = round(float(cells.max() - cells.min()), 3)
    status = worst(health_status(v_delta), temp_status(ss.temp))
    if status != ss.last_status:
        lvl = {"ok": "info", "warning": "warning", "critical": "critical"}[status]
        log_event(lvl, f"Status changed to {status.upper()} (V_delta {v_delta * 1000:.0f} mV, {ss.temp:.1f}°C)")
        if status == "critical":
            st.toast("CRITICAL status reached", icon="🚨")
        ss.last_status = status

    now = datetime.now(timezone.utc)
    payload = {
        "pack_id": pack_id,
        "timestamp": now.isoformat(),
        "cell_v": cells.tolist(),
        "pack_current": round(current, 2),
        "temp_c": round(ss.temp, 2),
        "soc": round(ss.soc, 2),
        "power_w": round(power, 1),
        "soh": soh,
        "mode": mode,
        "contactor_closed": not ss.open,
        "fan_pct": fan,
        "status": status,
    }
    if mqtt_client is not None:
        mqtt_client.publish(TOPIC, json.dumps(payload))

    row = {
        "time": now, "pack_v": round(pack_v, 3), "v_delta": v_delta,
        "current": round(current, 2), "power_w": round(power, 1),
        "soc": round(ss.soc, 2), "temp_c": round(ss.temp, 2),
        "c_rate": round(current / cap_eff, 2),
        "r_int_mohm": round(float(r_cells.sum()) * 1000, 2),
        "fan": fan, "contactor": 0 if ss.open else 1,
    }
    row.update(dict(zip(cell_cols, cells.tolist())))
    ss.history.append(row)
    ss.history = ss.history[-max_points:]


# ---------------- Live dashboard (auto-refreshes without blocking) ----------------
@st.fragment(run_every=interval_sec if running else None)
def live_dashboard():
    ss = st.session_state
    if running or ss.step_once:
        generate_reading()
        ss.step_once = False

    if not ss.history:
        st.info("Waiting for data... turn on 'Running' in the sidebar or press 'Step once'.")
        return

    df = pd.DataFrame(ss.history)
    view = df.tail(window)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    latest_cells = last[cell_cols].astype(float)
    weakest = latest_cells.idxmin()
    cap_eff = capacity_ah * soh / 100
    status_v = health_status(last["v_delta"])
    status_t = temp_status(last["temp_c"])
    score = health_score(last["v_delta"], last["temp_c"], last["soc"], last["c_rate"])

    # --- Metrics ---
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Pack voltage", f"{last['pack_v']:.3f} V", f"{last['pack_v'] - prev['pack_v']:.3f} V")
    m2.metric("Pack current", f"{last['current']:.1f} A", f"{last['c_rate']:.2f} C", delta_color="off")
    m3.metric("Power", f"{last['power_w']:.0f} W",
              "Charging" if last["current"] < -0.5 else "Discharging" if last["current"] > 0.5 else "Idle",
              delta_color="off")
    m4.metric("State of charge", f"{last['soc']:.1f} %", f"{last['soc'] - prev['soc']:.2f} %")
    m5.metric("Pack temp", f"{last['temp_c']:.1f} °C", f"{last['temp_c'] - prev['temp_c']:.2f} °C",
              delta_color="inverse")

    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric("V_delta", f"{last['v_delta'] * 1000:.1f} mV",
              delta=f"{(last['v_delta'] - prev['v_delta']) * 1000:.1f} mV" if len(df) > 1 else None,
              delta_color="inverse")
    n2.metric("Max cell", f"{latest_cells.max():.3f} V", latest_cells.idxmax(), delta_color="off")
    n3.metric("Min cell", f"{latest_cells.min():.3f} V", weakest, delta_color="off")
    if last["current"] > 0.5:
        eta_label, eta = "Runtime left", fmt_hours(last["soc"] / 100 * cap_eff / last["current"])
    elif last["current"] < -0.5:
        eta_label, eta = "Time to full", fmt_hours((100 - last["soc"]) / 100 * cap_eff / -last["current"])
    else:
        eta_label, eta = "Runtime left", "-"
    n4.metric(eta_label, eta)
    n5.metric("Equivalent cycles", f"{ss.ah_thru / (2 * cap_eff):.3f}", f"{len(df)} samples", delta_color="off")

    # --- Alert banner ---
    if ss.open:
        st.error("⛔ BMS contactor OPEN - pack isolated from the load until the fault clears.")
    if status_v == "critical":
        st.error(f"🚨 CRITICAL imbalance: {weakest} is {last['v_delta'] * 1000:.0f} mV below the strongest cell.")
    elif status_v == "warning":
        st.warning(f"⚠️ Warning: imbalance rising, weakest cell is {weakest}.")
    if status_t == "critical":
        st.error(f"🔥 CRITICAL temperature: {last['temp_c']:.1f} °C. Raise the fan or reduce the load.")
    elif status_t == "warning":
        st.warning(f"🌡️ Temperature warning: {last['temp_c']:.1f} °C.")
    if status_v == "ok" and status_t == "ok" and not ss.open:
        st.success("✅ Pack is balanced and within thermal limits.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🧭 Overview", "📈 Live charts", "🌡️ Thermal & power", "🔍 Analysis", "📜 Events", "🗂️ Raw data"])

    # --- Tab 1: overview gauges + cell cards ---
    with tab1:
        g1, g2, g3, g4, g5 = st.columns(5)
        with g1:
            show(make_gauge(last["soc"], "SOC", 0, 100, " %",
                            [(0, 15, "rgba(231,76,60,.35)"), (15, 30, "rgba(243,156,18,.35)"),
                             (30, 100, "rgba(46,204,113,.3)")], GREEN), "g_soc")
        with g2:
            show(make_gauge(last["pack_v"], "Pack voltage", num_cells * 3.0, num_cells * 4.2, " V",
                            [(num_cells * 3.0, num_cells * 3.3, "rgba(231,76,60,.35)"),
                             (num_cells * 3.3, num_cells * 4.15, "rgba(46,204,113,.3)"),
                             (num_cells * 4.15, num_cells * 4.2, "rgba(243,156,18,.35)")], fmt=".2f"), "g_volt")
        with g3:
            lim = max_c * cap_eff
            show(make_gauge(last["current"], "Current (+ discharge)", -lim * 1.4, lim * 1.4, " A",
                            [(-lim * 1.4, -lim, "rgba(231,76,60,.35)"), (-lim, lim, "rgba(46,204,113,.3)"),
                             (lim, lim * 1.4, "rgba(231,76,60,.35)")]), "g_cur")
        with g4:
            show(make_gauge(last["temp_c"], "Temperature", min(0, ambient), crit_temp + 20, " °C",
                            [(min(0, ambient), warn_temp, "rgba(46,204,113,.3)"),
                             (warn_temp, crit_temp, "rgba(243,156,18,.35)"),
                             (crit_temp, crit_temp + 20, "rgba(231,76,60,.35)")],
                            RED if status_t == "critical" else ORANGE if status_t == "warning" else BLUE), "g_temp")
        with g5:
            show(make_gauge(score, "Health score", 0, 100, "",
                            [(0, 40, "rgba(231,76,60,.35)"), (40, 70, "rgba(243,156,18,.35)"),
                             (70, 100, "rgba(46,204,113,.3)")], GREEN if score >= 70 else ORANGE if score >= 40 else RED,
                            fmt=".0f"), "g_score")

        st.subheader("Cell status")
        st.markdown(cell_cards_html(latest_cells), unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Cell radar (zoomed)")
            names = cell_cols + [cell_cols[0]]
            vals = latest_cells.tolist() + [latest_cells.iloc[0]]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=vals, theta=names, fill="toself", name="Cell voltage",
                                          line_color=BLUE))
            fig.add_trace(go.Scatterpolar(r=[latest_cells.mean()] * len(names), theta=names, name="Average",
                                          line=dict(dash="dash", color=ORANGE)))
            fig.update_layout(polar=dict(radialaxis=dict(range=[latest_cells.min() - 0.03,
                                                                latest_cells.max() + 0.03])),
                              height=330, margin=dict(l=30, r=30, t=20, b=20),
                              legend=dict(orientation="h", y=-0.1))
            show(fig, "radar")
        with right:
            st.subheader("Imbalance heatmap (mV vs pack mean)")
            dev_hist = view[cell_cols].sub(view[cell_cols].mean(axis=1), axis=0) * 1000
            fig = go.Figure(go.Heatmap(z=dev_hist.T.values, x=view["time"], y=cell_cols,
                                       colorscale="RdYlGn", zmid=0, colorbar=dict(title="mV")))
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
            show(fig, "heat")

    # --- Tab 2: live charts ---
    with tab2:
        st.subheader("Cell voltages over time")
        fig = go.Figure()
        for c in cell_cols:
            fig.add_trace(go.Scatter(x=view["time"], y=view[c], name=c, mode="lines",
                                     line=dict(width=4 if c == weakest else 1.5)))
        show(style(fig, 340), "cells_line")

        left, right = st.columns(2)
        with left:
            st.subheader("V_delta with alert zones (mV)")
            fig = go.Figure(go.Scatter(x=view["time"], y=view["v_delta"] * 1000, fill="tozeroy",
                                       line_color=BLUE, name="V_delta"))
            fig.add_hrect(y0=warn_th * 1000, y1=crit_th * 1000, fillcolor=ORANGE, opacity=0.15, line_width=0)
            fig.add_hrect(y0=crit_th * 1000, y1=max(crit_th * 2000, view["v_delta"].max() * 1100),
                          fillcolor=RED, opacity=0.15, line_width=0)
            show(style(fig, 280), "vdelta")
        with right:
            st.subheader("Cell deviation from pack average (mV)")
            dev = (latest_cells - latest_cells.mean()) * 1000
            fig = go.Figure(go.Bar(x=cell_cols, y=dev.round(1),
                                   marker_color=[RED if d < -warn_th * 1000 else GREEN for d in dev]))
            show(style(fig, 280), "dev_bar")

        left, right = st.columns(2)
        with left:
            st.subheader("State of charge")
            fig = go.Figure(go.Scatter(x=view["time"], y=view["soc"], fill="tozeroy", line_color=GREEN))
            fig.update_yaxes(range=[0, 100])
            show(style(fig, 260), "soc_line")
        with right:
            st.subheader("Discharge curve (pack V vs SOC)")
            fig = go.Figure(go.Scatter(x=view["soc"], y=view["pack_v"], mode="markers",
                                       marker=dict(color=view["current"], colorscale="RdYlGn_r", size=7,
                                                   colorbar=dict(title="A"))))
            fig.update_layout(xaxis_title="SOC (%)", yaxis_title="Pack V", height=260,
                              margin=dict(l=10, r=10, t=30, b=10))
            show(fig, "curve")

    # --- Tab 3: thermal & power ---
    with tab3:
        left, right = st.columns(2)
        with left:
            st.subheader("Temperature and cooling")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=view["time"], y=view["temp_c"], name="Temp °C", line_color=RED), secondary_y=False)
            fig.add_trace(go.Scatter(x=view["time"], y=view["fan"], name="Fan %", line=dict(color=BLUE, dash="dot"),
                                     line_shape="hv"), secondary_y=True)
            fig.add_hline(y=warn_temp, line_dash="dash", line_color=ORANGE)
            fig.add_hline(y=crit_temp, line_dash="dash", line_color=RED)
            show(style(fig, 300), "temp")
        with right:
            st.subheader("Current and power")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=view["time"], y=view["current"], name="Current A",
                                 marker_color=[ORANGE if i >= 0 else GREEN for i in view["current"]]),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=view["time"], y=view["power_w"], name="Power W", line_color=BLUE),
                          secondary_y=True)
            show(style(fig, 300), "power")

        left, right = st.columns(2)
        with left:
            st.subheader("Estimated cell temperatures")
            x = np.linspace(-1, 1, num_cells)
            amp = 0.15 * (last["temp_c"] - ambient) + 0.3
            cell_t = last["temp_c"] + amp * (np.exp(-2 * x ** 2) - 0.5)
            fig = go.Figure(go.Heatmap(z=[cell_t], x=cell_cols, y=["Temp"], colorscale="YlOrRd",
                                       zmin=ambient, zmax=crit_temp,
                                       text=[[f"{t:.1f}°" for t in cell_t]], texttemplate="%{text}",
                                       colorbar=dict(title="°C")))
            fig.update_layout(height=170, margin=dict(l=10, r=10, t=20, b=10))
            show(fig, "cell_temp")
        with right:
            st.subheader("Energy balance (Wh)")
            e1, e2, e3 = st.columns(3)
            e1.metric("Discharged", f"{ss.e_out:.1f}")
            e2.metric("Charged", f"{ss.e_in:.1f}")
            e3.metric("Net", f"{ss.e_in - ss.e_out:+.1f}", delta_color="off")
            fig = go.Figure(go.Bar(x=["Discharged", "Charged"], y=[ss.e_out, ss.e_in],
                                   marker_color=[ORANGE, GREEN]))
            fig.update_layout(height=170, margin=dict(l=10, r=10, t=10, b=10))
            show(fig, "energy")

    # --- Tab 4: analysis ---
    with tab4:
        st.subheader("🔬 Cell inspector")
        pick = st.selectbox("Choose a cell", cell_cols, index=cell_cols.index(weakest), key=f"inspect_{num_cells}")
        ser = view[pick]
        avg = view[cell_cols].mean(axis=1)
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Latest", f"{ser.iloc[-1]:.3f} V")
        i2.metric("vs pack avg", f"{(ser.iloc[-1] - avg.iloc[-1]) * 1000:.1f} mV", delta_color="off")
        i3.metric("Std dev", f"{ser.std() * 1000:.2f} mV" if len(ser) > 1 else "-")
        slope = np.polyfit(np.arange(len(ser)), ser, 1)[0] if len(ser) >= 5 else 0.0
        i4.metric("Trend", f"{slope * 1000:.2f} mV/sample", delta_color="off")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view["time"], y=ser, name=pick, line_color=BLUE))
        fig.add_trace(go.Scatter(x=view["time"], y=ser.rolling(5, min_periods=1).mean(), name="5-pt mean",
                                 line=dict(dash="dot", color=ORANGE)))
        fig.add_trace(go.Scatter(x=view["time"], y=avg, name="Pack average", line=dict(color="gray")))
        show(style(fig, 280), "inspector")

        left, right = st.columns(2)
        with left:
            st.subheader("Per-cell statistics (V)")
            stats = df[cell_cols].agg(["mean", "min", "max", "std"]).T
            stats["latest"] = latest_cells
            stats["vs avg (mV)"] = (latest_cells - latest_cells.mean()) * 1000
            st.dataframe(stats.round(4), use_container_width=True)
        with right:
            st.subheader("Voltage spread per cell")
            fig = go.Figure()
            for c in cell_cols:
                fig.add_trace(go.Box(y=view[c], name=c, boxpoints=False))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            show(fig, "box")

        st.subheader("Trend & prediction")
        recent = df.tail(30)
        if len(recent) >= 5:
            x = np.arange(len(recent))
            slope_delta = np.polyfit(x, recent["v_delta"], 1)[0]
            slope_weak = np.polyfit(x, recent[weakest], 1)[0]
            slope_temp = np.polyfit(x, recent["temp_c"], 1)[0]

            t1, t2, t3 = st.columns(3)
            t1.metric(f"{weakest} trend", f"{slope_weak * 1000:.2f} mV/sample")
            t2.metric("V_delta trend", f"{slope_delta * 1000:.2f} mV/sample")
            t3.metric("Temp trend", f"{slope_temp:.2f} °C/sample")

            if slope_delta > 1e-6 and last["v_delta"] < crit_th:
                steps = (crit_th - last["v_delta"]) / slope_delta
                st.info(f"At the current trend, V_delta reaches the critical limit in about "
                        f"**{int(steps)} samples (~{int(steps * interval_sec)} sec)**.")
            elif last["v_delta"] >= crit_th:
                st.error("Critical V_delta limit already crossed.")
            else:
                st.success("V_delta is stable or improving.")

            if slope_temp > 0.01 and last["temp_c"] < crit_temp:
                steps_t = (crit_temp - last["temp_c"]) / slope_temp
                st.info(f"At the current trend, temperature reaches the critical limit in about "
                        f"**{int(steps_t)} samples (~{int(steps_t * interval_sec)} sec)**.")
        else:
            st.caption("Collecting more samples for trend analysis...")

        st.subheader("🧮 Load planner (what-if)")
        plan_c = st.slider("Planned discharge rate (C)", 0.1, 3.0, 1.0, 0.1, key="plan_c")
        r_pack = last["r_int_mohm"] / 1000
        h_cool = 1.2 + 0.05 * fan
        c_axis = np.linspace(0.1, 3.0, 30)
        runtime = last["soc"] / 100 / c_axis                       # hours (SOC share / C-rate)
        t_steady = ambient + (c_axis * cap_eff) ** 2 * r_pack * 6 / h_cool
        p1, p2, p3 = st.columns(3)
        p1.metric("Runtime at this load", fmt_hours(last["soc"] / 100 / plan_c))
        steady = ambient + (plan_c * cap_eff) ** 2 * r_pack * 6 / h_cool
        p2.metric("Steady-state temp", f"{steady:.1f} °C", "over limit" if steady >= crit_temp else "ok",
                  delta_color="inverse" if steady >= crit_temp else "off")
        p3.metric("Current drawn", f"{plan_c * cap_eff:.1f} A")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=c_axis, y=np.minimum(runtime, 20), name="Runtime (h)", line_color=GREEN),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=c_axis, y=t_steady, name="Steady temp °C", line_color=RED), secondary_y=True)
        fig.add_vline(x=plan_c, line_dash="dash", line_color="gray")
        fig.update_xaxes(title_text="C-rate")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.15))
        show(fig, "planner")

        left, right = st.columns(2)
        with left:
            st.subheader("Factor correlations")
            cols = ["pack_v", "v_delta", "current", "power_w", "soc", "temp_c"]
            corr = view[cols].corr().fillna(0)
            fig = go.Figure(go.Heatmap(z=corr.values, x=cols, y=cols, zmin=-1, zmax=1, colorscale="RdBu",
                                       text=np.round(corr.values, 2), texttemplate="%{text}"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            show(fig, "corr")
        with right:
            st.subheader("Time spent in each state")
            states = df.apply(lambda r: worst(health_status(r["v_delta"]), temp_status(r["temp_c"])),
                              axis=1).value_counts()
            fig = go.Figure(go.Pie(labels=states.index, values=states.values, hole=0.55,
                                   marker=dict(colors=[{"ok": GREEN, "warning": ORANGE, "critical": RED}[s]
                                                       for s in states.index])))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            show(fig, "donut")

    # --- Tab 5: events ---
    with tab5:
        if ss.events:
            ev = pd.DataFrame(ss.events[::-1])
            st.dataframe(ev, use_container_width=True, height=350)
            st.download_button("⬇️ Download event log", ev.to_csv(index=False).encode(),
                               file_name="battery_events.csv", mime="text/csv", key="dl_events")
        else:
            st.caption("No events yet. Try the fault buttons in the control deck.")

    # --- Tab 6: raw data ---
    with tab6:
        st.dataframe(df.sort_values("time", ascending=False), use_container_width=True, height=350)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(),
                           file_name="battery_data.csv", mime="text/csv", key="dl_data")


live_dashboard()