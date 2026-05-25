# core/loader.py
import tensorflow as tf
import numpy as np
import os
from pathlib import Path


class ModelLoader:
    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.original_format = self._detect_format()
        self.format = 'tflite'  # everything becomes tflite internally
        self.model = None
        self.metadata = {}
        self.tflite_bytes = None

    def _detect_format(self) -> str:
        suffix = self.path.suffix.lower()
        if suffix in ['.h5', '.keras']:
            return 'keras'
        elif suffix == '.tflite':
            return 'tflite'
        elif suffix == '.onnx':
            return 'onnx'
        raise ValueError(f"Unsupported format: {suffix}. Supported: .h5, .keras, .tflite, .onnx")

    def load(self):
        """Load model and immediately convert to TFLite internally."""
        if self.original_format == 'keras':
            self._load_and_convert_keras()
        elif self.original_format == 'tflite':
            self._load_tflite_direct()
        elif self.original_format == 'onnx':
            self._load_and_convert_onnx()

        return self.model, self.metadata

    def _load_and_convert_keras(self):
        """Load Keras model and convert to TFLite."""
        try:
            keras_model = tf.keras.models.load_model(str(self.path), compile=False)
        except Exception as e:
            raise RuntimeError(
                f"Could not load Keras model. "
                f"Make sure it was saved with TensorFlow 2.x. Error: {e}"
            )

        # Convert to TFLite
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            converter.optimizations = []  # no compression at load time
            self.tflite_bytes = converter.convert()
        except Exception as e:
            raise RuntimeError(f"Could not convert Keras model to TFLite: {e}")

        self._build_interpreter()
        self.metadata['framework'] = f'Keras → TFLite'
        self.metadata['original_format'] = self.original_format

    def _load_tflite_direct(self):
        """Load TFLite model directly."""
        with open(str(self.path), 'rb') as f:
            self.tflite_bytes = f.read()
        self._build_interpreter()
        self.metadata['framework'] = 'TFLite'
        self.metadata['original_format'] = 'tflite'

    def _load_and_convert_onnx(self):
        """Load ONNX and convert to TFLite via onnx2tf."""
        try:
            import onnx
            onnx_model = onnx.load(str(self.path))
            onnx.checker.check_model(onnx_model)
        except ImportError:
            raise RuntimeError("onnx package required. Add 'onnx' to requirements.txt.")
        except Exception as e:
            raise RuntimeError(f"Could not load ONNX model: {e}")

        try:
            import onnx2tf
            saved_model_path = '/tmp/onnx_to_savedmodel'
            onnx2tf.convert(
                input_onnx_file_path=str(self.path),
                output_folder_path=saved_model_path,
                non_verbose=True
            )
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            self.tflite_bytes = converter.convert()
        except Exception as e:
            raise RuntimeError(f"Could not convert ONNX to TFLite: {e}")

        self._build_interpreter()
        self.metadata['framework'] = 'ONNX → TFLite'
        self.metadata['original_format'] = 'onnx'

    def _build_interpreter(self):
        """Build TFLite interpreter from bytes."""
        self.model = tf.lite.Interpreter(model_content=self.tflite_bytes)
        self.model.allocate_tensors()
        input_details = self.model.get_input_details()
        self.metadata['input_shape'] = input_details[0]['shape'].tolist()
        self.metadata['input_dtype'] = str(input_details[0]['dtype'])
        output_details = self.model.get_output_details()
        self.metadata['output_shape'] = output_details[0]['shape'].tolist()
