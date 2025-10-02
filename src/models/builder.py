"""Neural network model building utilities."""


def build_model(trial, input_shape, num_classes):
    """Build a CNN model with hyperparameters from Optuna trial.
    
    Args:
        trial: Optuna trial object for hyperparameter suggestions
        input_shape: Shape of input data (e.g., (28, 28, 1))
        num_classes: Number of output classes
        
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    

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