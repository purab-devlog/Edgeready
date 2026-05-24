# app.py
import streamlit as st
import tempfile, os
from core.loader import ModelLoader
from core.analyser import ModelAnalyser
from core.compressor import ModelCompressor
from core.hardware import estimate_on_chip
from core.exporter import ModelExporter
from ui.report_card import render_report_card
from ui.layer_viz import render_layer_chart
from ui.compression_ui import render_technique_selector, render_compression_results
from ui.hardware_ui import render_hardware_selector, render_hardware_estimate

st.set_page_config(page_title="EdgeReady", layout="wide", page_icon="⚡")

st.title("⚡ EdgeReady")
st.caption("Upload a neural network. Find out if it can run on embedded hardware — and how to fix it if it can't.")

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

uploaded = st.file_uploader("Upload your model", type=['h5', 'keras', 'tflite', 'onnx'])

if uploaded:
    raw_bytes = uploaded.read()

    suffix = uploaded.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as f:
        f.write(raw_bytes)
        tmp_path = f.name

    # Store raw bytes for compressor use
    st.session_state['uploaded_raw_bytes'] = raw_bytes
    st.session_state['uploaded_format'] = suffix

    with st.spinner("Loading model..."):
        try:
            loader = ModelLoader(tmp_path)
            model, metadata = loader.load()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            os.unlink(tmp_path)
            st.stop()

    st.success(f"Loaded: **{uploaded.name}** ({metadata['framework']})")
    st.caption(f"Input shape: {metadata.get('input_shape', 'unknown')}")

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
    technique = render_technique_selector()

    if st.button("Run Compression", type="primary"):
        compressor = ModelCompressor(model, loader.format)

        with st.spinner("Compressing..."):
            tflite_bytes = None
            ternary_stats = None

            try:
                if "INT8" in technique:
                    tflite_bytes = compressor.int8_quantize()
                    method = "INT8"
                elif "Pruning" in technique:
                    tflite_bytes = compressor.prune_and_quantize()
                    method = "Pruning + INT8"
                else:
                    result = compressor.ternarize()
                    tflite_bytes = result['tflite']
                    ternary_stats = result['stats']
                    method = "Ternary"
            except Exception as e:
                st.error(f"Compression failed: {e}")
                st.stop()

        if tflite_bytes:
            compressed_kb = len(tflite_bytes) / 1024
            exporter = ModelExporter(tflite_bytes, uploaded.name)
            summary = exporter.get_model_summary()

            render_compression_results(report, compressed_kb, method, summary)

            if ternary_stats:
                st.info(
                    f"🔬 Ternary stats — Layers ternarized: {ternary_stats['layers_ternarized']} | "
                    f"Weights zeroed: {ternary_stats['weights_zeroed']:,} / {ternary_stats['total_weights']:,} "
                    f"({ternary_stats['sparsity_pct']}% sparsity)"
                )

            st.session_state['tflite_bytes'] = tflite_bytes
            st.session_state['compressed_kb'] = compressed_kb
            st.session_state['export_filename'] = exporter.get_filename(method)

    st.divider()

    # Stage 4: Hardware Estimator
    chip = render_hardware_selector()
    estimate = estimate_on_chip(report, chip)
    compressed_kb = st.session_state.get('compressed_kb', None)
    render_hardware_estimate(estimate, compressed_kb)

    st.divider()

    # Stage 5: Export
    st.subheader("📦 Export")
    if 'tflite_bytes' in st.session_state and st.session_state['tflite_bytes']:
        st.download_button(
            label="⬇️ Download Compressed TFLite Model",
            data=st.session_state['tflite_bytes'],
            file_name=st.session_state.get('export_filename', 'model_compressed.tflite'),
            mime="application/octet-stream"
        )
    else:
        st.caption("Run compression above to enable export.")

    os.unlink(tmp_path)
