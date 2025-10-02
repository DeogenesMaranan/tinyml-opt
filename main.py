"""TinyML Optimization - Main Entry Point"""

import os
import sys


def main():
    """Main execution function."""
    # Parse arguments first (before any imports that might trigger TensorFlow)
    import argparse
    parser = argparse.ArgumentParser(description="Neural Architecture Search for TinyML")
    parser.add_argument("--verbose", action="store_true", help="Show raw TensorFlow logs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU only execution")
    parser.add_argument("--trials", type=int, default=5, help="Number of optimization trials")
    args = parser.parse_args()
    
    # Setup environment BEFORE any TensorFlow imports
    if not args.verbose:
        # Comprehensive TensorFlow logging suppression
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["TFLITE_LOG_LEVEL"] = "3"
        os.environ["AUTOGRAPH_VERBOSITY"] = "0"
        os.environ["TF_DISABLE_MKL"] = "1"
        
        # Suppress Python warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Show loading indicator during initial setup
    from src.utils.loading import LoadingIndicator
    
    setup_loader = LoadingIndicator("Initializing TensorFlow and dependencies")
    setup_loader.start()
    
    try:
        # Now import modules (TensorFlow logging should be suppressed)
        from src.utils.config import setup_tensorflow_logging
        from src.data.loader import load_data
        from src.optimization.optimizer import run_optimization
        
        # Additional TensorFlow logging setup
        setup_tensorflow_logging()
    finally:
        setup_loader.stop()
    
    # Load data
    (x_train, y_train), (x_test, y_test), num_classes = load_data()
    
    print(f"\n🚀 Starting optimization with {args.trials} trials...")
    if args.cpu:
        print("💻 CPU-only mode enabled")
    if not args.verbose:
        print("🔇 Logging suppressed (use --verbose for detailed logs)")
    print("-" * 60)
    
    # Run optimization
    study = run_optimization(x_train, y_train, x_test, y_test, num_classes, args)

if __name__ == "__main__":
    main()