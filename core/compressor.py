# core/compressor.py
import tensorflow as tf
import numpy as np
import os


class ModelCompressor:
    def __init__(self, model, format: str = 'keras'):
        self.model = model
        self.format = format
        self.original_size = self._get_size()

    def _get_size(self) -> float:
        if self.format == 'tflite':
            total = 0
            for detail in self.model.get_tensor_details():
                try:
                    tensor = self.model.get_tensor(detail['index'])
                    total += tensor.nbytes
                except Exception:
                    continue
            return total / 1024

        elif self.format == 'onnx':
            from onnx import numpy_helper
            total = 0
            for initializer in self.model['onnx'].graph.initializer:
                tensor = numpy_helper.to_array(initializer)
                total += tensor.nbytes
            return total / 1024

        else:
            # Keras
            self.model.save('/tmp/temp_model_size.keras')
            return os.path.getsize('/tmp/temp_model_size.keras') / 1024

    def int8_quantize(self) -> bytes:
        if self.format == 'keras':
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            return converter.convert()

        elif self.format == 'tflite':
            # Re-apply dynamic range quantization from raw bytes
            import streamlit as st
            raw = st.session_state.get('uploaded_raw_bytes', b'')
            if raw:
                converter = tf.lite.TFLiteConverter.from_buffer(raw)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                try:
                    return converter.convert()
                except Exception:
                    return raw
            return b''

        elif self.format == 'onnx':
            # Convert ONNX → TF SavedModel → TFLite with INT8
            try:
                import onnx
                import subprocess, sys
                saved_model_path = '/tmp/onnx_savedmodel'
                onnx_path = '/tmp/temp_upload.onnx'
                self.model['onnx_path'] if 'onnx_path' in self.model else None

                import streamlit as st
                raw = st.session_state.get('uploaded_raw_bytes', b'')
                with open(onnx_path, 'wb') as f:
                    f.write(raw)

                import onnx2tf
                onnx2tf.convert(
                    input_onnx_file_path=onnx_path,
                    output_folder_path=saved_model_path,
                    non_verbose=True
                )
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                return converter.convert()
            except Exception as e:
                raise RuntimeError(f"ONNX → TFLite INT8 conversion failed: {e}")

        return b''

    def prune_and_quantize(self, target_sparsity=0.5) -> bytes:
        if self.format == 'keras':
            try:
                import tensorflow_model_optimization as tfmot
                prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
                pruning_params = {
                    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                        target_sparsity, begin_step=0, frequency=100
                    )
                }
                pruned_model = prune_low_magnitude(self.model, **pruning_params)
                pruned_model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
                converter = tf.lite.TFLiteConverter.from_keras_model(stripped)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                return converter.convert()
            except Exception:
                # Fallback to INT8 if pruning fails
                return self.int8_quantize()
        else:
            # For TFLite and ONNX, fall back to INT8
            return self.int8_quantize()

    def ternarize(self, threshold=0.05) -> dict:
        if self.format == 'keras':
            stats = {
                'layers_ternarized': 0,
                'weights_zeroed': 0,
                'total_weights': 0
            }
            for layer in self.model.layers:
                if not layer.get_weights():
                    continue
                new_weights = []
                for w in layer.get_weights():
                    stats['total_weights'] += w.size
                    ternary_w = np.where(
                        w > threshold, 1.0,
                        np.where(w < -threshold, -1.0, 0.0)
                    )
                    stats['weights_zeroed'] += int(np.sum(ternary_w == 0))
                    new_weights.append(ternary_w.astype(np.float32))
                layer.set_weights(new_weights)
                stats['layers_ternarized'] += 1

            stats['sparsity_pct'] = round(
                stats['weights_zeroed'] / max(stats['total_weights'], 1) * 100, 1
            )
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            return {'tflite': tflite_model, 'stats': stats}

        else:
            # For TFLite/ONNX apply INT8 and note ternary not directly applicable
            tflite_bytes = self.int8_quantize()
            stats = {
                'layers_ternarized': 0,
                'weights_zeroed': 0,
                'total_weights': 0,
                'sparsity_pct': 0.0,
                'note': f'Ternary quantization requires a Keras model. '
                        f'Applied INT8 quantization instead for {self.format.upper()} input.'
            }
            return {'tflite': tflite_bytes, 'stats': stats}
