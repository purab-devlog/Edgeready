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
        raise ValueError(f"Unsupported format: {suffix}")

    def load(self):
        if self.format == 'keras':
            self.model = tf.keras.models.load_model(self.path)
            self.metadata['framework'] = 'TensorFlow/Keras'
            self.metadata['input_shape'] = self.model.input_shape

        elif self.format == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=str(self.path))
            interpreter.allocate_tensors()
            self.model = interpreter
            self.metadata['framework'] = 'TFLite'
            self.metadata['input_shape'] = interpreter.get_input_details()[0]['shape'].tolist()

        elif self.format == 'onnx':
            import onnx
            self.model = onnx.load(str(self.path))
            self.metadata['framework'] = 'ONNX'

        return self.model, self.metadata
