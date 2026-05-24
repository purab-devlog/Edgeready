# core/analyser.py
import tensorflow as tf
import numpy as np

class ModelAnalyser:
    def __init__(self, model, format: str):
        self.model = model
        self.format = format

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
                'size_bytes': int(total * 4),  # float32 = 4 bytes
                'size_kb': round(total * 4 / 1024, 2),
                'size_mb': round(total * 4 / 1024 / 1024, 3)
            }

    def compute_flops(self) -> dict:
        """
        Estimates FLOPs per inference.
        For Conv2D: FLOPs = 2 * Kh * Kw * Cin * Cout * Hout * Wout
        For Dense:  FLOPs = 2 * input_size * output_size
        """
        total_flops = 0
        layer_flops = []

        if self.format == 'keras':
            for layer in self.model.layers:
                flops = 0
                config = layer.get_config()

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
        """
        Peak RAM = largest intermediate activation tensor during forward pass
        """
        if self.format != 'keras':
            return {}

        peak_activation_bytes = 0
        layer_activations = []

        for layer in self.model.layers:
            try:
                output_shape = layer.output_shape
                if isinstance(output_shape, list):
                    output_shape = output_shape[0]
                elements = np.prod([d for d in output_shape if d is not None])
                bytes_needed = int(elements * 4)  # float32
                peak_activation_bytes = max(peak_activation_bytes, bytes_needed)
                layer_activations.append({
                    'name': layer.name,
                    'output_shape': str(output_shape),
                    'activation_kb': round(bytes_needed / 1024, 2)
                })
            except:
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
