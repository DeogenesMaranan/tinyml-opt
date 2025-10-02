"""Data loading and preprocessing utilities."""

import numpy as np
from src.utils.loading import LoadingIndicator


def load_data():
    """Load and prepare MNIST dataset for training and testing.
    
    Returns:
        tuple: ((x_train, y_train), (x_test, y_test), num_classes)
    """
    
    import tensorflow as tf
    from tensorflow import keras
    
    loader = LoadingIndicator("Loading MNIST dataset")
    loader.start()
    
    try:
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
        
    finally:
        loader.stop()
    
    print("✓ Dataset loaded and preprocessed")
    return (x_train, y_train), (x_test, y_test), num_classes