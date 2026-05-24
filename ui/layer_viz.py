# ui/layer_viz.py
import plotly.graph_objects as go
import streamlit as st

def render_layer_chart(layer_breakdown: list):
    st.subheader("🔬 Layer-by-Layer Breakdown")
    st.caption("Each bar shows the computational cost of one layer. Tall bars are your bottlenecks.")

    names = [l['name'] for l in layer_breakdown if l['flops'] > 0]
    flops = [l['flops'] / 1e3 for l in layer_breakdown if l['flops'] > 0]
    types = [l['type'] for l in layer_breakdown if l['flops'] > 0]
    params = [l['params'] for l in layer_breakdown if l['flops'] > 0]

    color_map = {
        'Conv2D': '#3498db',
        'Dense': '#e74c3c',
        'DepthwiseConv2D': '#2ecc71',
        'BatchNormalization': '#f39c12',
        'LSTM': '#9b59b6',
    }
    colors = [color_map.get(t, '#95a5a6') for t in types]

    fig = go.Figure(go.Bar(
        x=names,
        y=flops,
        marker_color=colors,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "FLOPs: %{y:.1f}K<br>"
            "Params: %{customdata:,}<br>"
            "<extra></extra>"
        ),
        customdata=params
    ))

    fig.update_layout(
        xaxis_title="Layer",
        yaxis_title="FLOPs (thousands)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highlight top bottleneck
    if flops:
        worst_idx = flops.index(max(flops))
        st.info(f"💡 **Biggest bottleneck:** `{names[worst_idx]}` ({types[worst_idx]}) — "
                f"accounts for {max(flops)/sum(flops)*100:.1f}% of total compute. "
                f"Consider replacing with a depthwise separable convolution to reduce this significantly.")
