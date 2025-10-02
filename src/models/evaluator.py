"""TensorFlow Lite model evaluation utilities."""

import os
import time
import tempfile


def evaluate_tflite(model, x_test_sample, verbose=False):
    """Evaluate model size and inference latency as TFLite model.
    
    Args:
        model: Trained Keras model
        x_test_sample: Sample input for latency measurement
        verbose: Whether to show conversion logs
        
    Returns:
        tuple: (size_kb, latency_ms)
    """
    

    import tensorflow as tf
    
    # Create temporary file for TFLite model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tflite") as tmpfile:
        # Convert to TFLite with suppressed logging
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Additional converter optimizations and logging suppression
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if not verbose:
            # Comprehensive log suppression for TFLite conversion
            stderr_backup = os.dup(2)
            stdout_backup = os.dup(1)
            
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 2)  # Redirect stderr
                os.dup2(devnull.fileno(), 1)  # Redirect stdout
                
                try:
                    tflite_model = converter.convert()
                finally:
                    # Restore original stderr and stdout
                    os.dup2(stderr_backup, 2)
                    os.dup2(stdout_backup, 1)
                    os.close(stderr_backup)
                    os.close(stdout_backup)
        else:
            tflite_model = converter.convert()
            
        # Save TFLite model
        with open(tmpfile.name, "wb") as f:
            f.write(tflite_model)
        
        # Get model size
        size_kb = os.path.getsize(tmpfile.name) / 1024.0
        
        # Measure inference latency with suppressed logs
        if not verbose:
            stderr_backup = os.dup(2)
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 2)
                try:
                    interpreter = tf.lite.Interpreter(model_path=tmpfile.name)
                    interpreter.allocate_tensors()
                finally:
                    os.dup2(stderr_backup, 2)
                    os.close(stderr_backup)
        else:
            interpreter = tf.lite.Interpreter(model_path=tmpfile.name)
            interpreter.allocate_tensors()
            
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Use a sample for latency measurement
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], x_test_sample)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        latency_ms = (time.time() - start) * 1000
        
        # Clean up
        os.remove(tmpfile.name)
        
        return size_kb, latency_ms