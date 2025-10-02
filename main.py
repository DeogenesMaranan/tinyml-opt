import os
import time
import tempfile
import argparse
import numpy as np
import optuna
import warnings
from contextlib import redirect_stderr, redirect_stdout
import io

# -------------------------------
# Configuration and Setup
# -------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Neural Architecture Search for TinyML")
    parser.add_argument("--verbose", action="store_true", help="Show raw TensorFlow logs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU only execution")
    parser.add_argument("--trials", type=int, default=5, help="Number of optimization trials")
    return parser.parse_args()

def setup_environment(args):
    """Configure environment variables and logging based on arguments"""
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.filterwarnings("ignore", category=UserWarning)
    
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def setup_tensorflow_logging():
    """Suppress TensorFlow logging if not verbose"""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    
    # Suppress absl logging
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

# -------------------------------
# Data Management
# -------------------------------
def load_data():
    """Load and prepare MNIST dataset"""
    import tensorflow as tf
    from tensorflow import keras
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Reduce dataset size for faster experimentation
    x_train, y_train = x_train[:5000], y_train[:5000]
    x_test, y_test = x_test[:1000], y_test[:1000]
    
    # Convert labels to categorical
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return (x_train, y_train), (x_test, y_test), num_classes

# -------------------------------
# Model Management
# -------------------------------
def build_model(trial, input_shape, num_classes):
    """Build a CNN model with hyperparameters from Optuna trial"""
    import tensorflow as tf
    from tensorflow.keras import layers
    
    # Hyperparameter search space
    num_filters = trial.suggest_categorical("num_filters", [8, 16, 32])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    dense_units = trial.suggest_categorical("dense_units", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    # Model architecture
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(num_filters, kernel_size, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax"),
    ])
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# -------------------------------
# TFLite Evaluation
# -------------------------------
def evaluate_tflite(model, x_test_sample, verbose=False):
    """Evaluate model size and inference latency as TFLite model"""
    import tensorflow as tf
    
    # Create temporary file for TFLite model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tflite") as tmpfile:
        # Convert to TFLite with suppressed logging
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if not verbose:
            # Suppress conversion logs
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                tflite_model = converter.convert()
        else:
            tflite_model = converter.convert()
            
        # Save TFLite model
        with open(tmpfile.name, "wb") as f:
            f.write(tflite_model)
        
        # Get model size
        size_kb = os.path.getsize(tmpfile.name) / 1024.0
        
        # Measure inference latency
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

# -------------------------------
# Optimization Objective
# -------------------------------
def create_objective(x_train, y_train, x_test, y_test, num_classes, args):
    """Create the objective function for Optuna optimization"""
    
    def objective(trial):
        import tensorflow as tf
        
        # Suggest batch size
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        
        # Build and train model
        model = build_model(trial, (28, 28, 1), num_classes)
        
        # Train with optional verbose output
        model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1 if args.verbose else 0
        )
        
        # Evaluate model
        _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        # Evaluate TFLite performance
        test_sample = np.expand_dims(x_test[0], axis=0).astype(np.float32)
        size_kb, latency_ms = evaluate_tflite(model, test_sample, args.verbose)
        
        # Print trial results
        print(f"Trial {trial.number:2d}: acc={test_acc:.4f}, "
              f"size={size_kb:6.1f}KB, latency={latency_ms:5.2f}ms")
        
        return test_acc, size_kb, latency_ms
    
    return objective

# -------------------------------
# Results Management
# -------------------------------
def sort_trials(study):
    """Sort trials by weighted score (prioritizing accuracy, then size, then latency)"""
    scored_trials = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            acc, size, latency = trial.values
            # Weighted score: higher accuracy is better, lower size/latency is better
            score = acc * 100 - size * 0.01 - latency * 0.1
            scored_trials.append((score, trial))
    
    # Sort by score (descending)
    scored_trials.sort(key=lambda x: x[0], reverse=True)
    return [trial for _, trial in scored_trials]

def print_best_trials(study, top_k=3):
    """Print the best trials in a clean format"""
    print("\n" + "="*60)
    print("BEST TRIALS (sorted by weighted score)")
    print("="*60)
    
    sorted_trials = sort_trials(study)
    
    for i, trial in enumerate(sorted_trials[:top_k]):
        acc, size, latency = trial.values
        print(f"\n#{i+1}: acc={acc:.4f}, size={size:.1f}KB, latency={latency:.2f}ms")
        print("Parameters:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")

# -------------------------------
# Main Execution
# -------------------------------
def main():
    # Parse arguments and setup environment
    args = parse_arguments()
    setup_environment(args)
    
    # Import TensorFlow after environment setup
    setup_tensorflow_logging()
    
    # Load data
    (x_train, y_train), (x_test, y_test), num_classes = load_data()
    
    print(f"Starting optimization with {args.trials} trials...")
    if args.cpu:
        print("CPU-only mode enabled")
    if not args.verbose:
        print("Logging suppressed (use --verbose for detailed logs)")
    print("-" * 50)
    
    # Create study and optimize
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name="tinyml_nas"
    )
    
    objective_func = create_objective(x_train, y_train, x_test, y_test, num_classes, args)
    study.optimize(objective_func, n_trials=args.trials)
    
    # Display results
    print_best_trials(study)

if __name__ == "__main__":
    main()