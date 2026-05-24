# core/compressor.py
import tensorflow as tf
import numpy as np

class ModelCompressor:
    def __init__(self, model):
        self.model = model
        self.original_size = self._get_size()

    def _get_size(self) -> float:
        self.model.save('/tmp/temp_model.h5')
        import os
        return os.path.getsize('/tmp/temp_model.h5') / 1024

    def int8_quantize(self, representative_data_gen=None) -> bytes:
        """Full integer quantization — most practical for deployment"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_data_gen:
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        return converter.convert()

    def prune_and_quantize(self, target_sparsity=0.5) -> bytes:
        """Remove near-zero weights then quantize"""
        import tensorflow_model_optimization as tfmot

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity, begin_step=0, frequency=100
            )
        }
        pruned_model = prune_low_magnitude(self.model, **pruning_params)
        pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])

        # Strip pruning wrappers
        stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)

        # Now quantize
        converter = tf.lite.TFLiteConverter.from_keras_model(stripped)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()

    def ternarize(self, threshold=0.05) -> dict:
        """
        Replace all weights with -1, 0, or +1 based on threshold.
        Returns modified model and stats — your IISc angle.
        """
        stats = {'layers_ternarized': 0, 'weights_zeroed': 0, 'total_weights': 0}

        for layer in self.model.layers:
            if not layer.get_weights():
                continue
            new_weights = []
            for w in layer.get_weights():
                stats['total_weights'] += w.size
                ternary_w = np.where(w > threshold, 1.0,
                            np.where(w < -threshold, -1.0, 0.0))
                zeroed = np.sum(ternary_w == 0)
                stats['weights_zeroed'] += int(zeroed)
                new_weights.append(ternary_w.astype(np.float32))
            layer.set_weights(new_weights)
            stats['layers_ternarized'] += 1

        stats['sparsity_pct'] = round(stats['weights_zeroed'] / stats['total_weights'] * 100, 1)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        return {'tflite': tflite_model, 'stats': stats}
