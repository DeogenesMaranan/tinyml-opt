# TinyML Neural Architecture Search

A hyperparameter optimization tool for finding efficient neural network architectures suitable for TinyML applications. This project uses Optuna to automatically search for optimal CNN architectures that balance accuracy, model size, and inference latency.

## Features

- **Multi-objective optimization**: Simultaneously optimizes for accuracy, model size (KB), and inference latency (ms)
- **TFLite integration**: Automatically converts models to TensorFlow Lite format for realistic size and latency measurements
- **Configurable search space**: Optimizes network depth, filter sizes, dense units, dropout, learning rate, and batch size
- **CPU/GPU flexibility**: Can force CPU-only execution for consistent benchmarking

## Requirements

**Python Version**: This project requires Python 3.11 or 3.12. Other versions are not supported.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow
- optuna
- numpy

## Usage

### Basic Usage

Run with default settings (5 trials):

```bash
python main.py
```

### Command Line Options

```bash
python main.py [OPTIONS]
```

**Available Options:**

- `--trials N`: Number of optimization trials to run (default: 5)
- `--verbose`: Show detailed TensorFlow logs and conversion output
- `--cpu`: Force CPU-only execution (useful for consistent benchmarking)

### Examples

**Run 20 trials with verbose output:**
```bash
python main.py --trials 20 --verbose
```

**Run on CPU only for consistent benchmarking:**
```bash
python main.py --cpu --trials 10
```

**Quick test with minimal output:**
```bash
python main.py --trials 3
```

## How It Works

### Search Space

The optimizer explores the following hyperparameters:

| Parameter | Options | Description |
|-----------|---------|-------------|
| `num_filters` | [8, 16, 32] | Number of filters in Conv2D layer |
| `kernel_size` | [3, 5] | Convolution kernel size |
| `dense_units` | [32, 64, 128] | Units in dense layer |
| `dropout` | [0.0, 0.3] | Dropout rate (continuous) |
| `learning_rate` | [1e-4, 1e-2] | Adam optimizer learning rate (log scale) |
| `batch_size` | [16, 32] | Training batch size |

### Model Architecture

Each trial creates a CNN with the following structure:
1. Conv2D layer (parameterized filters and kernel size)
2. MaxPooling2D (2x2)
3. Flatten
4. Dense layer (parameterized units)
5. Dropout (parameterized rate)
6. Output Dense layer (10 classes for MNIST)

### Evaluation Metrics

For each trial, the system measures:

1. **Accuracy**: Test accuracy on MNIST dataset
2. **Model Size**: TFLite model size in KB
3. **Inference Latency**: Single inference time in milliseconds

### Dataset

- **Dataset**: MNIST handwritten digits
- **Training samples**: 5,000 (reduced for faster experimentation)
- **Test samples**: 1,000
- **Training epochs**: 1 (for quick iteration)

*Note: The reduced dataset size and single epoch are designed for rapid prototyping. *
## Output Format

The tool provides real-time progress updates and a final summary:

### During Execution
```
Trial  0: acc=0.9240, size=  15.2KB, latency= 2.45ms
Trial  1: acc=0.9180, size=  28.7KB, latency= 3.12ms
Trial  2: acc=0.9350, size=  12.1KB, latency= 1.98ms
```

### Final Results
```
============================================================
BEST TRIALS (sorted by weighted score)
============================================================

#1: acc=0.9350, size=12.1KB, latency=1.98ms
Parameters:
  num_filters: 16
  kernel_size: 3
  dense_units: 64
  dropout: 0.1
  lr: 0.001
  batch_size: 32

#2: acc=0.9240, size=15.2KB, latency=2.45ms
Parameters:
  num_filters: 8
  kernel_size: 5
  dense_units: 32
  dropout: 0.2
  lr: 0.0015
  batch_size: 16
```

## Scoring System

The tool uses a weighted scoring system to rank trials:

```
Score = Accuracy × 100 - Size × 0.01 - Latency × 0.1
```

This prioritizes:
1. **Accuracy** (highest weight)
2. **Model size** (smaller is better)
3. **Inference latency** (faster is better)

You can modify the weights in the `sort_trials()` function to adjust priorities for your specific use case.

## Customization

### Modify Search Space

Edit the `build_model()` function to change hyperparameter ranges:

```python
# Example: Add more filter options
num_filters = trial.suggest_categorical("num_filters", [8, 16, 32, 64])

# Example: Expand learning rate range
learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
```

### Change Dataset

Replace the `load_data()` function to use your own dataset:

```python
def load_data():
    # Load your custom dataset here
    # Return: (x_train, y_train), (x_test, y_test), num_classes
    pass
```

### Adjust Model Architecture

Modify the model structure in `build_model()`:

```python
# Example: Add more layers
model = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape),
    layers.Conv2D(num_filters, kernel_size, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(num_filters*2, kernel_size, activation="relu"),  # Additional layer
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(dense_units, activation="relu"),
    layers.Dropout(dropout),
    layers.Dense(num_classes, activation="softmax"),
])
```

### Dependencies

If you encounter import errors, ensure all packages are installed:

```bash
pip install --upgrade tensorflow optuna numpy
```

**Note**: Make sure you're using Python 3.11 or 3.12. You can check your Python version with:

```bash
python --version
```

## Contributing

To extend this project:

1. Fork the repository
2. Modify the search space, model architecture, or evaluation metrics
3. Test with your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.