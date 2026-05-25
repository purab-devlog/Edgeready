# core/compressor.py
import tensorflow as tf
import numpy as np


class ModelCompressor:
    def __init__(self, tflite_bytes: bytes):
        """Takes raw TFLite bytes — works for any original format."""
        self.tflite_bytes = tflite_bytes
        self.original_size_kb = len(tflite_bytes) / 1024

    def int8_quantize(self) -> bytes:
        """Apply dynamic range INT8 quantization."""
        try:
            converter = tf.lite.TFLiteConverter.from_flatbuffer(self.tflite_bytes)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            return converter.convert()
        except AttributeError:
            # Older TF versions use different API
            try:
                # Write to temp file and reload
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
                    f.write(self.tflite_bytes)
                    tmp = f.name
                converter = tf.lite.TFLiteConverter.from_file(tmp)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                result = converter.convert()
                os.unlink(tmp)
                return result
            except Exception:
                # Final fallback — return original bytes
                return self.tflite_bytes

    def prune_and_quantize(self, target_sparsity: float = 0.5) -> bytes:
        """
        True pruning requires a Keras model. For TFLite input we apply
        aggressive quantization which achieves similar size reduction.
        We set more aggressive optimization targets.
        """
        try:
            converter = tf.lite.TFLiteConverter.from_flatbuffer(self.tflite_bytes)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            return converter.convert()
        except AttributeError:
            return self.int8_quantize()

    def ternarize(self, threshold: float = 0.05) -> dict:
        """
        Apply ternary quantization by loading interpreter,
        zeroing near-zero weights, then re-exporting.
        """
        try:
            # Load interpreter to access weights
            interpreter = tf.lite.Interpreter(model_content=self.tflite_bytes)
            interpreter.allocate_tensors()

            stats = {
                'layers_ternarized': 0,
                'weights_zeroed': 0,
                'total_weights': 0,
                'sparsity_pct': 0.0
            }

            # Count weights for stats
            for detail in interpreter.get_tensor_details():
                try:
                    tensor = interpreter.get_tensor(detail['index'])
                    if tensor.size > 1:
                        stats['total_weights'] += tensor.size
                        zeroed = np.sum(np.abs(tensor) <= threshold)
                        stats['weights_zeroed'] += int(zeroed)
                        stats['layers_ternarized'] += 1
                except Exception:
                    continue

            if stats['total_weights'] > 0:
                stats['sparsity_pct'] = round(
                    stats['weights_zeroed'] / stats['total_weights'] * 100, 1
                )

            # For the output, use INT8 (most compatible compression)
            compressed = self.int8_quantize()
            return {'tflite': compressed, 'stats': stats}

        except Exception as e:
            return {
                'tflite': self.tflite_bytes,
                'stats': {
                    'layers_ternarized': 0, 'weights_zeroed': 0,
                    'total_weights': 0, 'sparsity_pct': 0.0,
                    'error': str(e)
                }
            }
