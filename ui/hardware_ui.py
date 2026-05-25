# ui/hardware_ui.py
import streamlit as st
import plotly.graph_objects as go

CHIP_INFO = {
    "STM32H7 (Cortex-M7 @ 480MHz)": {
        "description": "High-performance MCU used in demanding embedded vision and audio applications. The same M7 core used with the Genx320 event camera at IISc NeuRonICS Lab.",
        "colour": "#3498db", "family": "STM32"
    },
    "STM32F7 (Cortex-M7 @ 216MHz)": {
        "description": "Mid-range Cortex-M7. Good balance of performance and power for audio and vision ML tasks.",
        "colour": "#2980b9", "family": "STM32"
    },
    "STM32F4 (Cortex-M4 @ 168MHz)": {
        "description": "Workhorse of embedded ML. Has an FPU and is widely supported by TFLite Micro and X-CUBE-AI. Good balance of cost and capability.",
        "colour": "#1abc9c", "family": "STM32"
    },
    "STM32L4 (Cortex-M4 @ 80MHz)": {
        "description": "Ultra-low power M4. Designed for battery-powered applications. Good for periodic ML inference on sensor data.",
        "colour": "#16a085", "family": "STM32"
    },
    "STM32G4 (Cortex-M4 @ 170MHz)": {
        "description": "Newer M4 with DSP and FPU extensions. Suited for signal processing pipelines combined with ML.",
        "colour": "#27ae60", "family": "STM32"
    },
    "STM32F1 (Cortex-M3 @ 72MHz)": {
        "description": "Very constrained MCU with no FPU. One of the most common STM32 boards but poorly suited for ML.",
        "colour": "#e67e22", "family": "STM32"
    },
    "ESP32-S3 (Xtensa LX7 @ 240MHz)": {
        "description": "Has vector extensions that accelerate ML inference. Used at IISc for RISC-V benchmarking work. Popular in IoT ML projects.",
        "colour": "#8e44ad", "family": "ESP32"
    },
    "ESP32 (Xtensa LX6 @ 240MHz)": {
        "description": "No FPU — float ops are software-emulated and slow. INT8 quantization is strongly recommended before deploying any model here.",
        "colour": "#9b59b6", "family": "ESP32"
    },
    "ESP32-C3 (RISC-V @ 160MHz)": {
        "description": "RISC-V core, lower power than ESP32. Limited ML capability without hardware FPU support.",
        "colour": "#a569bd", "family": "ESP32"
    },
    "Raspberry Pi Pico (Cortex-M0+ @ 133MHz)": {
        "description": "Popular, cheap, and well-documented. No FPU on M0+. Suitable for simple TFLite Micro models.",
        "colour": "#e74c3c", "family": "Other"
    },
    "Nordic nRF52840 (Cortex-M4 @ 64MHz)": {
        "description": "BLE-focused MCU with M4F core. Widely used in wearable and health monitoring ML applications.",
        "colour": "#c0392b", "family": "Other"
    },
    "Nordic nRF9160 (Cortex-M33 @ 64MHz)": {
        "description": "Integrated LTE-M/NB-IoT modem with M33 core. Good for cellular IoT nodes with on-device inference.",
        "colour": "#e74c3c", "family": "Other"
    },
    "Silicon Labs EFM32GG (Cortex-M4 @ 48MHz)": {
        "description": "Ultra-low energy MCU targeted by Simplicity Studio. Good for energy-harvesting sensor ML nodes.",
        "colour": "#d35400", "family": "Other"
    },
    "Arduino Nano 33 BLE (Cortex-M4 @ 64MHz)": {
        "description": "Popular TinyML development board. Officially supported by TFLite Micro. Good entry-level platform.",
        "colour": "#f39c12", "family": "Other"
    },
    "Arduino Uno (ATmega328P @ 16MHz)": {
        "description": "Almost no ML capability. Listed for reference — avoid deploying any neural network here.",
        "colour": "#95a5a6", "family": "Other"
    }
}


def render_hardware_selector() -> str:
    st.subheader("🖥️ Target Hardware Estimator")
    st.caption(
        "Select your target chip. Numbers are theoretical estimates based on "
        "published chip specs — actual performance varies with firmware "
        "optimisation and memory layout."
    )

    # Group chips by family
    families = {}
    for chip, info in CHIP_INFO.items():
        fam = info['family']
        families.setdefault(fam, []).append(chip)

    col1, col2 = st.columns([1, 3])
    with col1:
        family = st.selectbox("Family", list(families.keys()))
    with col2:
        chip = st.selectbox("Board", families[family])

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

    col1, col2, col3 = st.columns(3)
    flash_pct = estimate['flash_usage_pct']
    ram_pct = estimate['ram_usage_pct']
    latency = estimate['estimated_latency_ms']

    col1.metric(
        "Flash Usage", f"{flash_pct}%",
        delta="Fits ✅" if estimate['fits_flash'] else "Too large ❌",
        delta_color="off"
    )
    col2.metric(
        "Peak RAM Usage", f"{ram_pct}%",
        delta="Fits ✅" if estimate['fits_ram'] else "Too large ❌",
        delta_color="off"
    )
    col3.metric("Est. Inference Latency", f"{latency} ms")

    # Gauge charts
    fig = go.Figure()
    for label, value, domain in [
        ("Flash Usage %", flash_pct, [0, 0.45]),
        ("RAM Usage %", ram_pct, [0.55, 1])
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
                    "value": 80
                }
            },
            domain={"x": domain, "y": [0, 1]}
        ))

    fig.update_layout(height=250, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    rec = estimate.get('recommendation', '')
    if estimate['fits_flash'] and estimate['fits_ram']:
        st.success(f"✅ {rec}")
    else:
        st.error(f"❌ {rec}")

    st.caption("⚠️ These are theoretical estimates based on chip specs. Always verify with actual on-device testing.")
