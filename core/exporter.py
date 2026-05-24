# core/exporter.py
import os
import tempfile
import tensorflow as tf


class ModelExporter:
    def __init__(self, tflite_bytes: bytes, original_filename: str):
        self.tflite_bytes = tflite_bytes
        self.original_filename = original_filename.rsplit('.', 1)[0]

    def get_download_bytes(self) -> bytes:
        return self.tflite_bytes

    def get_filename(self, method: str) -> str:
        method_slug = method.lower().replace(" ", "_").replace("+", "and")
        return f"{self.original_filename}_{method_slug}.tflite"

    def get_size_kb(self) -> float:
        return round(len(self.tflite_bytes) / 1024, 2)

    def get_model_summary(self) -> dict:
        """
        Load the TFLite model and extract basic metadata
        for display after export.
        """
        try:
            interpreter = tf.lite.Interpreter(model_content=self.tflite_bytes)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            return {
                "size_kb": self.get_size_kb(),
                "input_shape": input_details[0]["shape"].tolist(),
                "input_dtype": str(input_details[0]["dtype"]),
                "output_shape": output_details[0]["shape"].tolist(),
                "output_dtype": str(output_details[0]["dtype"]),
                "num_inputs": len(input_details),
                "num_outputs": len(output_details),
            }
        except Exception as e:
            return {
                "size_kb": self.get_size_kb(),
                "error": str(e)
            }
