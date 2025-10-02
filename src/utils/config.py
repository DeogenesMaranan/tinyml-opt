"""Configuration and environment setup utilities."""

import os
import argparse
import warnings
import optuna


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Neural Architecture Search for TinyML")
    parser.add_argument("--verbose", action="store_true", help="Show raw TensorFlow logs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU only execution")
    parser.add_argument("--trials", type=int, default=5, help="Number of optimization trials")
    return parser.parse_args()


def setup_environment(args):
    """Configure environment variables and logging based on arguments."""
    if not args.verbose:
        # Comprehensive TensorFlow logging suppression
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["TFLITE_LOG_LEVEL"] = "3"
        
        # Suppress various TensorFlow warnings and info messages
        os.environ["AUTOGRAPH_VERBOSITY"] = "0"
        os.environ["TF_DISABLE_MKL"] = "1"
        
        # Suppress Python warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def setup_tensorflow_logging():
    """Suppress TensorFlow logging if not verbose."""
    import tensorflow as tf
    
    # Suppress TensorFlow Python logging
    tf.get_logger().setLevel("ERROR")
    
    # Suppress absl logging (C++ level)
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    
    # Additional TensorFlow C++ logging suppression
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # Suppress TensorFlow Lite logging
    os.environ["TFLITE_LOG_LEVEL"] = "3"
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.ERROR)