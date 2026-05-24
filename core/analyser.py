# core/analyser.py
import tensorflow as tf
import numpy as np


class ModelAnalyser:
    def __init__(self, model, format: str):
        self.model = model
        self.format = format

        # Allocate tensors immediately for TFLite
        if self.format == 'tflite':
            self.model.allocate_tensors()

    def count_parameters(self) -> dict:
        if self.format == 'keras':
            total = self.model.count_params()
            trainable = sum([
                np.prod(v.shape)
                for v in self.model.trainable_weights
            ])
            return {
                'total': int(total),
                'trainable': int(trainable),
                'non_trainable': int(total - trainable),
                'size_bytes': int(total * 4),
                'size_kb': round(total * 4 / 1024, 2),
                'size_mb': round(total * 4 / 1024 / 1024, 3)
            }

        elif self.format == 'tflite':
            interpreter = self.model
            weight_params = 0

            for detail in interpreter.get_tensor_details():
                try:
                    tensor = interpreter.get_tensor(detail['index'])
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

        return {
            'total': 0, 'trainable': 0, 'non_trainable': 0,
            'size_bytes': 0, 'size_kb': 0.0, 'size_mb': 0.0
        }

    def compute_flops(self) -> dict:
        if self.format == 'tflite':
            interpreter = self.model
            total_flops = 0
            layer_flops = []

            for detail in interpreter.get_tensor_details():
                try:
                    tensor = interpreter.get_tensor(detail['index'])
                    shape = tensor.shape
                    if len(shape) == 4:
                        flops = int(np.prod(shape) * shape[-1] * 2)
                    elif len(shape) == 2:
                        flops = int(shape[0] * shape[1] * 2)
                    else:
                        flops = 0

                    total_flops += flops
                    if flops > 0:
                        layer_flops.append({
                            'name': detail['name'] or f"tensor_{detail['index']}",
                            'type': f"Tensor({len(shape)}D)",
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

        # Keras model
        total_flops = 0
        layer_flops = []

        for layer in self.model.layers:
            flops = 0
            try:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    kh, kw = layer.kernel_size
                    cin = layer.input_shape[-1]
                    cout = layer.filters
                    out_h = layer.output_shape[1]
                    out_w = layer.output_shape[2]
                    flops = 2 * kh * kw * cin * cout * out_h * out_w

                elif isinstance(layer, tf.keras.layers.Dense):
                    flops = 2 * layer.input_shape[-1] * layer.units

                elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                    kh, kw = layer.kernel_size
                    cin = layer.input_shape[-1]
                    out_h = layer.output_shape[1]
                    out_w = layer.output_shape[2]
                    flops = 2 * kh * kw * cin * out_h * out_w
            except Exception:
                flops = 0

            total_flops += flops
            layer_flops.append({
                'name': layer.name,
                'type': layer.__class__.__name__,
                'flops': int(flops),
                'params': int(layer.count_params())
            })

        return {
            'total_flops': total_flops,
            'total_mflops': round(total_flops / 1e6, 3),
            'layer_breakdown': layer_flops
        }

    def estimate_ram(self) -> dict:
        if self.format == 'tflite':
            interpreter = self.model
            peak_bytes = 0
            layer_activations = []

            for detail in interpreter.get_tensor_details():
                try:
                    tensor = interpreter.get_tensor(detail['index'])
                    bytes_needed = tensor.size * 4
                    peak_bytes = max(peak_bytes, bytes_needed)
                    if tensor.size > 0:
                        layer_activations.append({
                            'name': detail['name'] or f"tensor_{detail['index']}",
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

        # Keras model
        peak_activation_bytes = 0
        layer_activations = []

        for layer in self.model.layers:
            try:
                output_shape = layer.output_shape
                if isinstance(output_shape, list):
                    output_shape = output_shape[0]
                elements = np.prod([d for d in output_shape if d is not None])
                bytes_needed = int(elements * 4)
                peak_activation_bytes = max(peak_activation_bytes, bytes_needed)
                layer_activations.append({
                    'name': layer.name,
                    'output_shape': str(output_shape),
                    'activation_kb': round(bytes_needed / 1024, 2)
                })
            except Exception:
                continue

        return {
            'peak_ram_bytes': peak_activation_bytes,
            'peak_ram_kb': round(peak_activation_bytes / 1024, 2),
            'peak_ram_mb': round(peak_activation_bytes / 1024 / 1024, 3),
            'layer_activations': layer_activations
        }

    def full_report(self) -> dict:
        params = self.count_parameters()
        flops = self.compute_flops()
        ram = self.estimate_ram()
        return {
            'parameters': params,
            'flops': flops,
            'ram': ram
        }
