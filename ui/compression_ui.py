# ui/compression_ui.py
import streamlit as st
import plotly.graph_objects as go

TECHNIQUE_EXPLANATIONS = {
    "INT8 Quantization": {
        "what": "Converts 32-bit decimal weights into 8-bit integers.",
        "impact": "Typically 4x smaller model size with only 1–3% accuracy drop.",
        "best_for": "Most embedded deployments. The safest and most widely used technique.",
        "colour": "#3498db"
    },
    "Pruning + Quantization": {
        "what": "First removes weights close to zero that contribute little, "
                "then applies INT8 quantization on top.",
        "impact": "20–40% additional size reduction on top of quantization "
                  "with minimal accuracy impact if done carefully.",
        "best_for": "Models with redundant neurons or layers. "
                    "Gives better accuracy retention than quantization alone.",
        "colour": "#2ecc71"
    },
    "Ternary Quantization": {
        "what": "Forces all weights to become either -1, 0, or +1. "
                "Multiplications become additions — trivial on any processor.",
        "impact": "Extreme size reduction. Significant accuracy drop "
                  "especially on complex tasks.",
        "best_for": "Very constrained hardware like Cortex-M0 or ATmega. "
                    "Simple classification tasks. Based on IISc RISC-V research.",
        "colour": "#e74c3c"
    }
}


def render_technique_selector() -> str:
    st.subheader("🛠️ Compression Workshop")
    st.caption(
        "Pick a compression technique. Each trades some accuracy "
        "for a smaller, faster model. Read the explanation before choosing."
    )

    technique = st.radio(
        "Choose compression technique",
        list(TECHNIQUE_EXPLANATIONS.keys()),
        label_visibility="collapsed"
    )

    info = TECHNIQUE_EXPLANATIONS[technique]
    st.markdown(f"""
    <div style='border-left: 4px solid {info["colour"]}; 
                padding: 12px 16px; border-radius: 4px;
                background: rgba(0,0,0,0.03); margin-bottom: 12px;'>
        <b>What it does:</b> {info["what"]}<br><br>
        <b>Expected impact:</b> {info["impact"]}<br><br>
        <b>Best for:</b> {info["best_for"]}
    </div>
    """, unsafe_allow_html=True)

    return technique


def render_compression_results(
    original_report: dict,
    compressed_size_kb: float,
    method: str,
    tflite_summary: dict
):
    st.subheader("📉 Compression Results")

    original_kb = original_report['parameters']['size_kb']
    reduction_pct = round((1 - compressed_size_kb / original_kb) * 100, 1)
    original_params = original_report['parameters']['total']
    original_mflops = original_report['flops']['total_mflops']

    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Original Size",
        f"{original_kb:.1f} KB",
    )
    col2.metric(
        "Compressed Size",
        f"{compressed_size_kb:.1f} KB",
        delta=f"-{reduction_pct}%",
        delta_color="inverse"
    )
    col3.metric(
        "Size Reduction",
        f"{reduction_pct}%",
    )

    # Before / After bar chart
    fig = go.Figure()

    categories = ['Model Size (KB)', 'Parameters (K)', 'Est. MFLOPs']
    original_vals = [
        original_kb,
        round(original_params / 1000, 1),
        original_mflops
    ]

    # Estimate compressed equivalents
    ratio = compressed_size_kb / original_kb if original_kb > 0 else 1
    compressed_vals = [
        compressed_size_kb,
        round((original_params * ratio) / 1000, 1),
        round(original_mflops * ratio, 3)
    ]

    fig.add_trace(go.Bar(
        name='Original',
        x=categories,
        y=original_vals,
        marker_color='#e74c3c',
        opacity=0.85
    ))
    fig.add_trace(go.Bar(
        name=f'Compressed ({method})',
        x=categories,
        y=compressed_vals,
        marker_color='#2ecc71',
        opacity=0.85
    ))

    fig.update_layout(
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=350,
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # TFLite model details
    if 'error' not in tflite_summary:
        with st.expander("📦 Exported TFLite Model Details"):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Input shape:** `{tflite_summary.get('input_shape')}`")
            c1.markdown(f"**Input dtype:** `{tflite_summary.get('input_dtype')}`")
            c2.markdown(f"**Output shape:** `{tflite_summary.get('output_shape')}`")
            c2.markdown(f"**Output dtype:** `{tflite_summary.get('output_dtype')}`")
            st.caption(
                "Input dtype `int8` confirms full integer quantization was applied. "
                "This is what embedded runtimes like TFLite Micro expect."
            )
