"""Optuna-based neural architecture search optimization."""

import time
import numpy as np
import optuna
from src.utils.loading import LoadingIndicator
from src.models.builder import build_model
from src.models.evaluator import evaluate_tflite


def create_objective(x_train, y_train, x_test, y_test, num_classes, args):
    """Create the objective function for Optuna optimization.
    
    Args:
        x_train, y_train: Training data
        x_test, y_test: Test data
        num_classes: Number of output classes
        args: Command line arguments
        
    Returns:
        callable: Objective function for Optuna
    """
    
    def objective(trial):
        

        import tensorflow as tf
        
        trial_num = trial.number
        print(f"\nTrial {trial_num:2d}: Building and training model...")
        
        # Suggest batch size
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        
        # Build model
        build_loader = LoadingIndicator(f"Trial {trial_num} - Building model")
        build_loader.start()
        model = build_model(trial, (28, 28, 1), num_classes)
        build_loader.stop()
        
        # Train model
        train_loader = LoadingIndicator(f"Trial {trial_num} - Training model")
        train_loader.start()
        try:
            model.fit(
                x_train, y_train,
                epochs=1,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0  # Always suppress training output for cleaner display
            )
        finally:
            train_loader.stop()
        
        # Evaluate model
        eval_loader = LoadingIndicator(f"Trial {trial_num} - Evaluating accuracy")
        eval_loader.start()
        try:
            _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        finally:
            eval_loader.stop()
        
        # Evaluate TFLite performance
        tflite_loader = LoadingIndicator(f"Trial {trial_num} - Converting to TFLite")
        tflite_loader.start()
        try:
            test_sample = np.expand_dims(x_test[0], axis=0).astype(np.float32)
            size_kb, latency_ms = evaluate_tflite(model, test_sample, args.verbose)
        finally:
            tflite_loader.stop()
        
        # Print trial results
        print(f"Trial {trial_num:2d}: ✓ acc={test_acc:.4f}, "
              f"size={size_kb:6.1f}KB, latency={latency_ms:5.2f}ms")
        
        return test_acc, size_kb, latency_ms
    
    return objective


def sort_trials(study):
    """Sort trials by weighted score (prioritizing accuracy, then size, then latency).
    
    Args:
        study: Optuna study object
        
    Returns:
        list: Sorted list of trials
    """
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
    """Print the best trials in a clean format.
    
    Args:
        study: Optuna study object
        top_k: Number of top trials to display
    """
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


def run_optimization(x_train, y_train, x_test, y_test, num_classes, args):
    """Run the neural architecture search optimization.
    
    Args:
        x_train, y_train: Training data
        x_test, y_test: Test data
        num_classes: Number of output classes
        args: Command line arguments
        
    Returns:
        optuna.Study: Completed optimization study
    """
    # Create study with loading indicator
    study_loader = LoadingIndicator("Initializing optimization study")
    study_loader.start()
    try:
        study = optuna.create_study(
            directions=["maximize", "minimize", "minimize"],
            study_name="tinyml_nas"
        )
        objective_func = create_objective(x_train, y_train, x_test, y_test, num_classes, args)
    finally:
        study_loader.stop()
    
    print("✓ Optimization study initialized")
    print(f"📊 Progress: Running {args.trials} trials...")
    
    # Run optimization
    study.optimize(objective_func, n_trials=args.trials)
    
    # Display results with loading indicator
    results_loader = LoadingIndicator("Analyzing results")
    results_loader.start()
    try:
        time.sleep(0.5)  # Brief pause for effect
    finally:
        results_loader.stop()
    
    print("\n✓ Optimization completed!")
    print_best_trials(study)
    
    return study