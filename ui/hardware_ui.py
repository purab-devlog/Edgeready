# ui/hardware_ui.py
import streamlit as st
import plotly.graph_objects as go

CHIP_INFO = {
    "STM32H7 (Cortex-M7)": {
        "description": "High-performance MCU used in demanding embedded vision "
                       "and audio applications. The same M7 core used with the "
                       "Genx320 event camera at IISc NeuRonICS Lab.",
        "colour": "#3498db"
    },
    "STM32F4 (Cortex-M4)": {
        "description": "Workhorse of embedded ML. Has an FPU and is widely "
                       "supported by TFLite Micro and X-CUBE-AI. "
                       "Good balance of cost and capability.",
        "colour": "#2ecc71"
    },
    "ESP32-S3 (Xtensa LX7)": {
        "description": "The RISC-V adjacent chip you used at IISc for "
                       "benchmarking. Has vector extensions that help with ML "
                       "inference. Popular in IoT ML applications.",
        "colour": "#f39c12"
    },
    "ESP32 (Xtensa LX6)": {
        "description": "No FPU on base ESP32 — float ops are slow. "
                       "INT8 quantization is strongly recommended before "
                       "deploying any model here.",
        "colour": "#e67e22"
    },
    "Arduino Uno (ATmega328P)": {
        "description": "Almost no ML capability. Listed for reference only. "
                       "Avoid deploying any neural network here — "
                       "even the smallest models won't fit.",
        "colour": "#e74c3c"
    }
}


def render_hardware_selector() -> str:
    st.subheader("🖥️ Target Hardware Estimator")
    st.caption(
        "Select your target chip. Numbers are theoretical estimates based on "
        "published chip specs — actual performance varies with firmware "
        "optimisation and memory layout."
    )

    chip = st.selectbox(
        "Target chip",
        list(CHIP_INFO.keys())
    )

    info = CHIP_INFO[chip]
    st.markdown(f"""
    <div style='border-left: 4px solid {info["colour"]}; 
                padding: 10px 14px; border-radius: 4px;
                background: rgba(0,0,0,0.03); margin-bottom: 12px;'>
        {info["description"]}
    </div>
    """, unsafe_allow_html=True)

    return chip


def render_hardware_estimate(estimate: dict, compressed_size_kb: float = None):
    chip = estimate['chip']
    info = CHIP_INFO.get(chip, {})
    colour = info.get('colour', '#888888')

    # Flash and RAM gauges
    col1, col2, col3 = st.columns(3)

    flash_pct = estimate['flash_usage_pct']
    ram_pct = estimate['ram_usage_pct']
    latency = estimate['estimated_latency_ms']

    col1.metric(
        "Flash Usage",
        f"{flash_pct}%",
        delta="Fits ✅" if estimate['fits_flash'] else "Too large ❌",
        delta_color="off"
    )
    col2.metric(
        "Peak RAM Usage",
        f"{ram_pct}%",
        delta="Fits ✅" if estimate['fits_ram'] else "Too large ❌",
        delta_color="off"
    )
    col3.metric(
        "Est. Inference Latency",
        f"{latency} ms"
    )

    # Gauge charts for flash and RAM
    fig = go.Figure()

    for label, value, threshold in [
        ("Flash Usage %", flash_pct, 80),
        ("RAM Usage %", ram_pct, 80)
    ]:
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": label, "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": colour},
                "steps": [
                    {"range": [0, 60], "color": "#d5f5e3"},
                    {"range": [60, 80], "color": "#fef9e7"},
                    {"range": [80, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": threshold
                }
            },
            domain={"x": [0, 0.45] if label == "Flash Usage %" else [0.55, 1],
                    "y": [0, 1]}
        ))

    fig.update_layout(height=250, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Recommendation
    rec = estimate.get('recommendation', '')
    if estimate['fits_flash'] and estimate['fits_ram']:
        st.success(f"✅ {rec}")
    else:
        st.error(f"❌ {rec}")

    # Compressed comparison
    if compressed_size_kb:
        st.divider()
        st.caption("**After compression:**")
        compressed_flash_pct = round(
            compressed_size_kb /
            (estimate['flash_usage_pct'] / 100 *
             (estimate.get('chip_flash_kb', compressed_size_kb / (estimate['flash_usage_pct'] / 100)))) * 100, 1
        )
        if compressed_flash_pct < flash_pct:
            improvement = round(flash_pct - compressed_flash_pct, 1)
            st.success(
                f"Compression reduces flash usage by ~{improvement}% on this chip. "
                f"New estimated flash usage: {compressed_flash_pct}%"
            )
