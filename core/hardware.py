# core/hardware.py
import json

def estimate_on_chip(model_report: dict, chip_name: str) -> dict:
    with open('data/chip_specs.json') as f:
        specs = json.load(f)

    chip = specs[chip_name]
    model_size_kb = model_report['parameters']['size_kb']
    peak_ram_kb = model_report['ram']['peak_ram_kb']
    total_mflops = model_report['flops']['total_mflops']

    # Theoretical inference time
    chip_mflops = chip['clock_mhz'] * chip['mflops_per_mhz']
    latency_ms = round((total_mflops / chip_mflops) * 1000, 2)

    return {
        'chip': chip_name,
        'fits_flash': model_size_kb <= chip['flash_kb'],
        'fits_ram': peak_ram_kb <= chip['ram_kb'],
        'flash_usage_pct': round(model_size_kb / chip['flash_kb'] * 100, 1),
        'ram_usage_pct': round(peak_ram_kb / chip['ram_kb'] * 100, 1),
        'estimated_latency_ms': latency_ms,
        'notes': chip['notes'],
        'recommendation': _get_recommendation(model_size_kb, peak_ram_kb, chip)
    }

def _get_recommendation(size_kb, ram_kb, chip) -> str:
    if size_kb > chip['flash_kb']:
        return f"Model is {size_kb - chip['flash_kb']:.0f}KB too large for flash. Apply INT8 quantization first."
    if ram_kb > chip['ram_kb']:
        return f"Peak RAM exceeds chip capacity by {ram_kb - chip['ram_kb']:.0f}KB. Try pruning or reducing layer width."
    return "Model fits this chip. Verify with actual on-device testing."
