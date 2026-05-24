# core/hardware.py
import json
import os


def estimate_on_chip(model_report: dict, chip_name: str) -> dict:
    # Load chip specs — handle both local and deployed paths
    possible_paths = [
        'data/chip_specs.json',
        '/mount/src/edgeready/data/chip_specs.json',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'chip_specs.json')
    ]

    specs = None
    for path in possible_paths:
        try:
            with open(path) as f:
                specs = json.load(f)
            break
        except FileNotFoundError:
            continue

    if specs is None:
        # Fallback inline specs if file not found
        specs = {
            "STM32H7 (Cortex-M7)": {"clock_mhz": 480, "flash_kb": 2048, "ram_kb": 1024, "mflops_per_mhz": 2.0, "notes": "High performance MCU."},
            "STM32F4 (Cortex-M4)": {"clock_mhz": 168, "flash_kb": 1024, "ram_kb": 192, "mflops_per_mhz": 1.25, "notes": "Has FPU. Widely used for embedded ML."},
            "ESP32-S3 (Xtensa LX7)": {"clock_mhz": 240, "flash_kb": 8192, "ram_kb": 512, "mflops_per_mhz": 1.0, "notes": "Has vector extensions."},
            "ESP32 (Xtensa LX6)": {"clock_mhz": 240, "flash_kb": 4096, "ram_kb": 520, "mflops_per_mhz": 0.5, "notes": "No FPU. INT8 strongly recommended."},
            "Arduino Uno (ATmega328P)": {"clock_mhz": 16, "flash_kb": 32, "ram_kb": 2, "mflops_per_mhz": 0.1, "notes": "Almost no ML capability."},
        }

    chip = specs.get(chip_name, list(specs.values())[0])
    model_size_kb = model_report['parameters']['size_kb']
    peak_ram_kb = model_report['ram']['peak_ram_kb']
    total_mflops = model_report['flops']['total_mflops']

    chip_mflops = chip['clock_mhz'] * chip['mflops_per_mhz']
    latency_ms = round((total_mflops / chip_mflops) * 1000, 2) if chip_mflops > 0 else 0

    fits_flash = model_size_kb <= chip['flash_kb']
    fits_ram = peak_ram_kb <= chip['ram_kb']

    return {
        'chip': chip_name,
        'fits_flash': fits_flash,
        'fits_ram': fits_ram,
        'flash_usage_pct': round(model_size_kb / chip['flash_kb'] * 100, 1),
        'ram_usage_pct': round(peak_ram_kb / chip['ram_kb'] * 100, 1),
        'chip_flash_kb': chip['flash_kb'],
        'chip_ram_kb': chip['ram_kb'],
        'estimated_latency_ms': latency_ms,
        'notes': chip['notes'],
        'recommendation': _get_recommendation(model_size_kb, peak_ram_kb, chip)
    }


def _get_recommendation(size_kb: float, ram_kb: float, chip: dict) -> str:
    if size_kb > chip['flash_kb']:
        excess = size_kb - chip['flash_kb']
        return (f"Model is {excess:.0f}KB too large for flash. "
                f"Apply INT8 quantization to reduce size by ~4x.")
    if ram_kb > chip['ram_kb']:
        excess = ram_kb - chip['ram_kb']
        return (f"Peak RAM exceeds chip capacity by {excess:.0f}KB. "
                f"Try pruning or reducing layer width.")
    return "Model fits this chip. Verify with actual on-device testing."
