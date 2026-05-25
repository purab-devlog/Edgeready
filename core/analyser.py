# core/analyser.py
# All analysis runs on TFLite interpreter — format is always 'tflite'
import tensorflow as tf
import numpy as np


class ModelAnalyser:
    def __init__(self, model, format: str = 'tflite'):
        self.model = model  # always a TFLite interpreter
        self.model.allocate_tensors()

    def count_parameters(self) -> dict:
        weight_params = 0
        for detail in self.model.get_tensor_details():
            try:
                tensor = self.model.get_tensor(detail['index'])
                weight_params += tensor.size
            except Exception:
                continue

        size_bytes = weight_params * 4
        return {
            'total': int(weight_params),
            'trainable': int(weight_params),
            'non_trainable': 0,
            'size_bytes': int(size_bytes),
            'size_kb': round(size_bytes / 1024, 2),
            'size_mb': round(size_bytes / 1024 / 1024, 3)
        }

    def compute_flops(self) -> dict:
        total_flops = 0
        layer_flops = []

        for detail in self.model.get_tensor_details():
            try:
                tensor = self.model.get_tensor(detail['index'])
                shape = tensor.shape
                name = detail.get('name', '') or f"tensor_{detail['index']}"

                if len(shape) == 4:
                    flops = int(np.prod(shape) * shape[-1] * 2)
                elif len(shape) == 2:
                    flops = int(shape[0] * shape[1] * 2)
                else:
                    flops = 0

                total_flops += flops
                if flops > 0:
                    layer_flops.append({
                        'name': name,
                        'type': f"{len(shape)}D Tensor",
                        'flops': flops,
                        'params': int(tensor.size)
                    })
            except Exception:
                continue

        return {
            'total_flops': total_flops,
            'total_mflops': round(total_flops / 1e6, 3),
            'layer_breakdown': layer_flops
        }

    def estimate_ram(self) -> dict:
        peak_bytes = 0
        layer_activations = []

        for detail in self.model.get_tensor_details():
            try:
                tensor = self.model.get_tensor(detail['index'])
                bytes_needed = tensor.size * 4
                peak_bytes = max(peak_bytes, bytes_needed)
                name = detail.get('name', '') or f"tensor_{detail['index']}"
                if tensor.size > 0:
                    layer_activations.append({
                        'name': name,
                        'output_shape': str(tensor.shape),
                        'activation_kb': round(bytes_needed / 1024, 2)
                    })
            except Exception:
                continue

        return {
            'peak_ram_bytes': peak_bytes,
            'peak_ram_kb': round(peak_bytes / 1024, 2),
            'peak_ram_mb': round(peak_bytes / 1024 / 1024, 3),
            'layer_activations': layer_activations
        }

    def full_report(self) -> dict:
        return {
            'parameters': self.count_parameters(),
            'flops': self.compute_flops(),
            'ram': self.estimate_ram()
        }
