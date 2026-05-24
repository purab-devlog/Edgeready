# ui/report_card.py
import streamlit as st

THRESHOLDS = {
    'params': {
        'green': 100_000,    # < 100K params
        'yellow': 500_000,   # 100K - 500K
        # red = above 500K
    },
    'size_kb': {
        'green': 256,        # < 256KB (fits STM32F4 flash comfortably)
        'yellow': 512,
    },
    'peak_ram_kb': {
        'green': 64,         # < 64KB
        'yellow': 128,
    },
    'mflops': {
        'green': 50,         # < 50 MFLOPs
        'yellow': 200,
    }
}

def get_status(value, metric):
    t = THRESHOLDS[metric]
    if value <= t['green']:
        return '🟢', 'Good', '#2ecc71'
    elif value <= t['yellow']:
        return '🟡', 'Borderline', '#f39c12'
    else:
        return '🔴', 'Too Large', '#e74c3c'

EXPLANATIONS = {
    'params': "Parameters are the numbers your model has learned. More parameters = smarter model but larger file. A typical STM32F4 has 1MB flash storage.",
    'size_kb': "This is how much flash storage your model needs. Flash is where the model lives permanently on chip, like a hard drive.",
    'peak_ram_kb': "RAM is used during inference — when the model is actually thinking. This is the peak memory needed at the heaviest computation point. RAM is much scarcer than flash on embedded chips.",
    'mflops': "FLOPs (Floating Point Operations) measure computational work per inference. More FLOPs = slower inference on constrained hardware running at 80-480MHz."
}

def render_report_card(report: dict):
    st.subheader("📋 Deployment Readiness Report")

    params = report['parameters']
    flops = report['flops']
    ram = report['ram']

    metrics = [
        ('Parameters', params['total'], 'params', f"{params['total']:,}"),
        ('Model Size', params['size_kb'], 'size_kb', f"{params['size_kb']} KB"),
        ('Peak RAM', ram['peak_ram_kb'], 'peak_ram_kb', f"{ram['peak_ram_kb']} KB"),
        ('Compute', flops['total_mflops'], 'mflops', f"{flops['total_mflops']} MFLOPs"),
    ]

    cols = st.columns(4)
    for col, (label, value, metric, display) in zip(cols, metrics):
        icon, status, color = get_status(value, metric)
        with col:
            st.markdown(f"""
            <div style='border: 1px solid {color}; border-radius: 8px; 
                        padding: 16px; text-align: center;'>
                <div style='font-size: 2rem'>{icon}</div>
                <div style='font-weight: bold; font-size: 1.1rem'>{label}</div>
                <div style='font-size: 1.4rem; color: {color}'>{display}</div>
                <div style='color: gray; font-size: 0.85rem'>{status}</div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("What is this?"):
                st.write(EXPLANATIONS[metric])

    # Overall verdict
    statuses = [get_status(v, m)[1] for _, v, m, _ in metrics]
    if all(s == 'Good' for s in statuses):
        st.success("✅ This model is ready for embedded deployment.")
    elif 'Too Large' in statuses:
        st.error("❌ This model needs compression before it can be deployed on most embedded targets.")
    else:
        st.warning("⚠️ This model may work on higher-end embedded targets but needs optimisation for constrained chips.")
