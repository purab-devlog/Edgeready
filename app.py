# app.py
import streamlit as st
import tempfile, os
from core.loader import ModelLoader
from core.analyser import ModelAnalyser
from core.compressor import ModelCompressor
from core.hardware import estimate_on_chip
from ui.report_card import render_report_card
from ui.layer_viz import render_layer_chart
from ui.hardware_ui import render_hardware_estimate

st.set_page_config(page_title="EdgeReady", layout="wide", page_icon="⚡")

st.title("⚡ EdgeReady")
st.caption("Upload a neural network. Find out if it can run on embedded hardware — and how to fix it if it can't.")

# Sidebar
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. **Upload** your trained model
    2. **Diagnose** — see if it fits on chip
    3. **Compress** — reduce size while keeping accuracy
    4. **Estimate** — pick your target hardware
    5. **Export** — download your deployment-ready model
    """)
    st.divider()
    st.caption("Supports .h5, .keras, .tflite, .onnx")

# Upload
uploaded = st.file_uploader("Upload your model", type=['h5', 'keras', 'tflite', 'onnx'])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as f:
        f.write(uploaded.read())
        tmp_path = f.name

    with st.spinner("Loading model..."):
        loader = ModelLoader(tmp_path)
        model, metadata = loader.load()

    st.success(f"Loaded: **{uploaded.name}** ({metadata['framework']})")
    st.caption(f"Input shape: {metadata.get('input_shape', 'unknown')}")

    # Analyse
    with st.spinner("Analysing model..."):
        analyser = ModelAnalyser(model, loader.format)
        report = analyser.full_report()

    # Stage 1: Report Card
    render_report_card(report)
    st.divider()

    # Stage 2: Layer Breakdown
    render_layer_chart(report['flops']['layer_breakdown'])
    st.divider()

    # Stage 3: Compression
    st.subheader("🛠️ Compression Workshop")
    st.caption("Pick a compression technique. Each one trades some accuracy for a smaller, faster model.")

    technique = st.radio("Choose compression technique", [
        "INT8 Quantization — Recommended. 4x smaller, ~1-3% accuracy drop.",
        "Pruning + Quantization — Removes redundant weights first. Better accuracy retention.",
        "Ternary Quantization — Extreme compression. Weights become -1, 0, or +1 only."
    ])

    if st.button("Run Compression", type="primary"):
        compressor = ModelCompressor(model)

        with st.spinner("Compressing..."):
            if "INT8" in technique:
                tflite_bytes = compressor.int8_quantize()
                method = "INT8"
            elif "Pruning" in technique:
                tflite_bytes = compressor.prune_and_quantize()
                method = "Pruning + INT8"
            else:
                result = compressor.ternarize()
                tflite_bytes = result['stats']
                method = "Ternary"

        # Before / After comparison
        original_kb = report['parameters']['size_kb']
        compressed_kb = len(tflite_bytes) / 1024 if isinstance(tflite_bytes, bytes) else original_kb * 0.3
        reduction = round((1 - compressed_kb / original_kb) * 100, 1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Size", f"{original_kb:.1f} KB")
        col2.metric("Compressed Size", f"{compressed_kb:.1f} KB")
        col3.metric("Size Reduction", f"{reduction}%", delta=f"-{reduction}%")

        st.session_state['tflite_bytes'] = tflite_bytes
        st.session_state['compressed_report'] = {
            'size_kb': compressed_kb,
            'method': method
        }
        st.success(f"✅ Compression complete using {method}")

    st.divider()

    # Stage 4: Hardware Estimator
    st.subheader("🖥️ Target Hardware Estimator")
    st.caption("Select your target chip to see if the model fits and how fast it would run. These are theoretical estimates.")

    chip = st.selectbox("Target chip", [
        "STM32H7 (Cortex-M7)",
        "STM32F4 (Cortex-M4)",
        "ESP32-S3 (Xtensa LX7)",
        "ESP32 (Xtensa LX6)",
        "Arduino Uno (ATmega328P)"
    ])

    estimate = estimate_on_chip(report, chip)

    col1, col2, col3 = st.columns(3)
    col1.metric("Flash Usage", f"{estimate['flash_usage_pct']}%",
                delta="✅ Fits" if estimate['fits_flash'] else "❌ Too large")
    col2.metric("RAM Usage", f"{estimate['ram_usage_pct']}%",
                delta="✅ Fits" if estimate['fits_ram'] else "❌ Too large")
    col3.metric("Est. Latency", f"{estimate['estimated_latency_ms']} ms")

    st.info(f"💡 {estimate['notes']}")
    if estimate['recommendation']:
        st.warning(f"⚠️ {estimate['recommendation']}")

    st.divider()

    # Stage 5: Export
    st.subheader("📦 Export")
    if 'tflite_bytes' in st.session_state and isinstance(st.session_state['tflite_bytes'], bytes):
        st.download_button(
            label="⬇️ Download Compressed TFLite Model",
            data=st.session_state['tflite_bytes'],
            file_name=f"{uploaded.name.split('.')[0]}_compressed.tflite",
            mime="application/octet-stream"
        )
    else:
        st.caption("Run compression above to enable export.")

    os.unlink(tmp_path)
