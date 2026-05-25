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
st.caption(
    "Upload any neural network — Keras, ONNX, or TFLite. "
    "Find out if it can run on embedded hardware, compress it, and download it deployment-ready."
)

with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. **Upload** your trained model
    2. **Auto-convert** to TFLite internally
    3. **Diagnose** — traffic light readiness report
    4. **Compress** — shrink it for your target chip
    5. **Estimate** — pick target hardware
    6. **Export** — download the optimised model
    """)
    st.divider()
    st.markdown("**Supported formats**")
    st.markdown("`.h5` `.keras` `.tflite` `.onnx`")
    st.divider()
    st.caption(
        "All formats are converted to TFLite internally — "
        "the actual deployment format for embedded systems."
    )

uploaded = st.file_uploader(
    "Upload your model",
    type=['h5', 'keras', 'tflite', 'onnx'],
    help="Supports Keras (.h5, .keras), TFLite (.tflite), and ONNX (.onnx)"
)

if uploaded:
    raw_bytes = uploaded.read()
    suffix = uploaded.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as f:
        f.write(raw_bytes)
        tmp_path = f.name

    # ── Load & convert ────────────────────────────────────────────────────────
    with st.spinner(f"Loading {suffix.upper()} model and converting to TFLite..."):
        try:
            loader = ModelLoader(tmp_path)
            model, metadata = loader.load()
            tflite_bytes = loader.tflite_bytes
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            os.unlink(tmp_path)
            st.stop()

    # Show conversion banner if format was converted
    original_fmt = metadata.get('original_format', suffix)
    if original_fmt != 'tflite':
        st.success(
            f"✅ **{uploaded.name}** loaded and converted to TFLite internally. "
            f"({metadata['framework']})"
        )
    else:
        st.success(f"✅ **{uploaded.name}** loaded. ({metadata['framework']})")

    col1, col2, col3 = st.columns(3)
    col1.caption(f"**Input shape:** `{metadata.get('input_shape', 'unknown')}`")
    col2.caption(f"**Input dtype:** `{metadata.get('input_dtype', 'unknown')}`")
    col3.caption(f"**Output shape:** `{metadata.get('output_shape', 'unknown')}`")

    # ── Analyse ───────────────────────────────────────────────────────────────
    with st.spinner("Analysing model..."):
        analyser = ModelAnalyser(model, 'tflite')
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
        compressor = ModelCompressor(tflite_bytes)

        with st.spinner("Compressing..."):
            tflite_out = None
            ternary_stats = None

            try:
                if "INT8" in technique:
                    tflite_out = compressor.int8_quantize()
                    method = "INT8"
                elif "Pruning" in technique:
                    tflite_out = compressor.prune_and_quantize()
                    method = "Pruning + INT8"
                else:
                    result = compressor.ternarize()
                    tflite_out = result['tflite']
                    ternary_stats = result['stats']
                    method = "Ternary"
            except Exception as e:
                st.error(f"❌ Compression failed: {e}")
                st.stop()

        if tflite_out:
            compressed_kb = len(tflite_out) / 1024
            exporter = ModelExporter(tflite_out, uploaded.name)
            summary = exporter.get_model_summary()

            render_compression_results(report, compressed_kb, method, summary)

            if ternary_stats:
                note = ternary_stats.get('note', '')
                st.info(
                    f"🔬 **Ternary stats** — "
                    f"Tensors analysed: {ternary_stats['layers_ternarized']} | "
                    f"Near-zero weights: {ternary_stats['weights_zeroed']:,} / "
                    f"{ternary_stats['total_weights']:,} "
                    f"({ternary_stats['sparsity_pct']}% sparsity)"
                    + (f"\n\n_{note}_" if note else "")
                )

            st.session_state['tflite_out'] = tflite_out
            st.session_state['compressed_kb'] = compressed_kb
            st.session_state['export_filename'] = exporter.get_filename(method)
            st.session_state['compressed_report'] = {
                'parameters': {'size_kb': compressed_kb},
                'flops': report['flops'],
                'ram': report['ram']
            }

    st.divider()

    # Stage 4: Hardware Estimator
    chip = render_hardware_selector()
    estimate = estimate_on_chip(report, chip)
    compressed_kb = st.session_state.get('compressed_kb', None)
    render_hardware_estimate(estimate, compressed_kb)

    # Show compressed model estimate if available
    if compressed_kb and 'compressed_report' in st.session_state:
        with st.expander("📊 See hardware estimate after compression"):
            compressed_estimate = estimate_on_chip(
                st.session_state['compressed_report'], chip
            )
            render_hardware_estimate(compressed_estimate, None)

    st.divider()

    # Stage 5: Export
    st.subheader("📦 Export")

    if 'tflite_out' in st.session_state and st.session_state['tflite_out']:
        st.caption(
            "Your model has been compressed and is ready to download as TFLite — "
            "the standard format for embedded ML deployment."
        )
        st.download_button(
            label="⬇️ Download Compressed TFLite Model",
            data=st.session_state['tflite_out'],
            file_name=st.session_state.get('export_filename', 'model_compressed.tflite'),
            mime="application/octet-stream",
            type="primary"
        )
        st.caption(
            f"Size: {st.session_state['compressed_kb']:.1f} KB  |  "
            f"Format: TFLite (ready to flash with TFLite Micro)"
        )
    else:
        st.caption("Run compression above to enable export.")

    os.unlink(tmp_path)
