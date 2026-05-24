# core/loader.py
import tensorflow as tf
import numpy as np
from pathlib import Path


class ModelLoader:
    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.format = self._detect_format()
        self.model = None
        self.metadata = {}

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
        if self.format == 'keras':
            self._load_keras()
        elif self.format == 'tflite':
            self._load_tflite()
        elif self.format == 'onnx':
            self._load_onnx()
        return self.model, self.metadata

    def _load_keras(self):
        try:
            self.model = tf.keras.models.load_model(str(self.path), compile=False)
        except Exception as e:
            raise RuntimeError(
                f"Could not load Keras model. This is usually a version mismatch — "
                f"the model was saved with a different Keras version. "
                f"Try converting to .tflite before uploading. Error: {e}"
            )
        self.metadata['framework'] = 'TensorFlow/Keras'
        try:
            self.metadata['input_shape'] = self.model.input_shape
        except Exception:
            self.metadata['input_shape'] = 'unknown'

    def _load_tflite(self):
        try:
            interpreter = tf.lite.Interpreter(model_path=str(self.path))
            interpreter.allocate_tensors()
            self.model = interpreter
        except Exception as e:
            raise RuntimeError(f"Could not load TFLite model: {e}")
        self.metadata['framework'] = 'TensorFlow Lite'
        try:
            input_details = interpreter.get_input_details()
            self.metadata['input_shape'] = input_details[0]['shape'].tolist()
            self.metadata['input_dtype'] = str(input_details[0]['dtype'])
        except Exception:
            self.metadata['input_shape'] = 'unknown'

    def _load_onnx(self):
        try:
            import onnx
            import onnxruntime as ort
            onnx_model = onnx.load(str(self.path))
            onnx.checker.check_model(onnx_model)
            session = ort.InferenceSession(str(self.path))
            # Store both — onnx model for analysis, session for inference
            self.model = {'onnx': onnx_model, 'session': session}
        except ImportError:
            raise RuntimeError("onnx and onnxruntime packages are required for ONNX support.")
        except Exception as e:
            raise RuntimeError(f"Could not load ONNX model: {e}")
        self.metadata['framework'] = 'ONNX'
        try:
            session = self.model['session']
            input_info = session.get_inputs()[0]
            self.metadata['input_shape'] = input_info.shape
            self.metadata['input_dtype'] = input_info.type
        except Exception:
            self.metadata['input_shape'] = 'unknown'
