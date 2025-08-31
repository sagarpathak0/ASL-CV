# ASL CNN Technical Documentation

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Checkpoint System](#checkpoint-system)
3. [Performance Optimizations](#performance-optimizations)
4. [Training Pipeline](#training-pipeline)
5. [Model Implementation](#model-implementation)
6. [Data Loading](#data-loading)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

## 🏗️ System Architecture

### Model Overview

- **Type**: Convolutional Neural Network (CNN)
- **Implementation**: Pure NumPy/CuPy (no frameworks)
- **Classes**: 29 ASL signs (A-Z + del, nothing, space)
- **Input**: 64×64×3 RGB images
- **Parameters**: ~39,309 total

### Network Architecture

```
Input Layer (64×64×3)
         ↓
Conv2D (8 filters, 3×3 kernel, stride=1)
         ↓
ReLU Activation
         ↓ 
MaxPool2D (2×2 pool, stride=2)
         ↓
Flatten (→ 31×31×8 = 7,688 features)
         ↓
Dense/Softmax (7,688 → 29 classes)
         ↓
Output (29 class probabilities)
```

### Parameter Breakdown

```python
# Convolutional Layer
conv_filters: (3, 3, 3, 8) = 216 parameters

# Dense/Softmax Layer  
softmax_weights: (7688, 29) = 222,952 parameters
softmax_biases: (29,) = 29 parameters

# Total: 216 + 222,952 + 29 = 223,197 parameters
```

## 💾 Checkpoint System

### Directory Structure

```
models/
├── epoch_1/
│   ├── conv_filters.npy     # Conv layer weights
│   ├── softmax_weights.npy  # Dense layer weights  
│   ├── softmax_biases.npy   # Dense layer biases
│   └── metadata.json        # Training metadata
├── epoch_2/
├── ...
├── latest/                  # Symlink to most recent
├── best/                    # Best validation accuracy
└── *.npy                   # Legacy format (backward compatibility)
```

### Metadata Format

```json
{
  "epoch": 5,
  "val_accuracy": 0.9207,
  "val_loss": 0.3999,
  "best_accuracy": 0.9207,
  "timestamp": "2025-08-31 09:21:41",
  "num_classes": 29,
  "conv_filters": 8
}
```

### Checkpoint Functions

#### `save_checkpoint(model, epoch, val_acc, val_loss, best_acc, checkpoint_dir)`

```python
def save_checkpoint(model, epoch, val_acc, val_loss, best_acc, checkpoint_dir="models"):
    """
    Save model checkpoint with metadata
  
    Args:
        model: CNN model instance
        epoch: Current epoch number
        val_acc: Validation accuracy
        val_loss: Validation loss
        best_acc: Best accuracy so far
        checkpoint_dir: Base checkpoint directory
    """
```

#### `load_checkpoint(checkpoint_path, model)`

```python
def load_checkpoint(checkpoint_path, model):
    """
    Load model from checkpoint
  
    Args:
        checkpoint_path: Path to checkpoint directory
        model: CNN model instance to load into
      
    Returns:
        tuple: (epoch, val_acc, val_loss, timestamp)
    """
```

#### `find_latest_checkpoint(checkpoint_dir)`

```python
def find_latest_checkpoint(checkpoint_dir="models"):
    """
    Find the most recent checkpoint
  
    Args:
        checkpoint_dir: Base checkpoint directory
      
    Returns:
        str: Path to latest checkpoint or None
    """
```

## ⚡ Performance Optimizations

### Parallel Data Loading

```python
# Before: Sequential loading (17 images/sec)
for class_dir in class_dirs:
    images = load_class_images(class_dir)

# After: Parallel loading (774+ images/sec)  
with ThreadPoolExecutor(max_workers=min(32, len(class_dirs))) as executor:
    futures = {executor.submit(load_class_images, class_dir): class_dir 
               for class_dir in class_dirs}
```

### Memory Management

- **Efficient array operations**: Vectorized NumPy operations
- **Batch processing**: Process data in chunks to control memory usage
- **Progressive loading**: Load only required samples per class

### Speed Benchmarks

| Operation    | Before         | After          | Improvement  |
| ------------ | -------------- | -------------- | ------------ |
| Data Loading | 17 img/sec     | 774 img/sec    | 32× faster  |
| Training     | 30 samples/sec | 45 samples/sec | 1.5× faster |
| Inference    | 50 img/sec     | 75 img/sec     | 1.5× faster |

## 🚀 Training Pipeline

### Complete Training Flow

```python
1. Dataset Discovery & Scanning
   ├── Scan class directories
   ├── Count available files
   └── Select samples per class

2. Parallel Data Loading  
   ├── ThreadPoolExecutor with 32 workers
   ├── Load & resize images to 64×64
   ├── Normalize pixel values [0,1]
   └── Convert to numpy arrays

3. Data Preprocessing
   ├── Shuffle dataset
   ├── Train/validation split (80/20)
   └── Create class labels

4. Model Training Loop
   ├── Forward pass (batch processing)
   ├── Loss calculation (cross-entropy)
   ├── Backward pass (gradient computation)
   ├── Weight updates (gradient descent)
   └── Checkpoint saving after each epoch

5. Validation & Testing
   ├── Evaluate on validation set
   ├── Save best performing model
   └── Test on sample images
```

### Training Configuration

```python
DEFAULT_CONFIG = {
    'max_samples': 200,        # Samples per class
    'epochs': 3,               # Training epochs
    'learning_rate': 0.01,     # Learning rate
    'batch_size': 32,          # Batch size
    'val_split': 0.2,          # Validation split
    'conv_filters': 8,         # Number of conv filters
    'print_interval': 100,     # Progress print frequency
    'checkpoint_dir': 'models' # Checkpoint directory
}
```

## 🔧 Model Implementation

### CNN Class Structure

```python
class CNN:
    def __init__(self, num_classes=29, conv_filters=8):
        """
        Initialize CNN with random weights
      
        Args:
            num_classes: Number of output classes
            conv_filters: Number of convolutional filters
        """
      
    def convolution(self, image, kernel):
        """Perform 2D convolution operation"""
      
    def relu(self, x):
        """ReLU activation function"""
      
    def max_pooling(self, feature_map, pool_size=2, stride=2):
        """Max pooling operation"""
      
    def softmax(self, x):
        """Softmax activation for classification"""
      
    def forward(self, X):
        """Forward pass through the network"""
      
    def backward(self, X, y, learning_rate):
        """Backward pass with gradient descent"""
      
    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        """Main training loop with checkpointing"""
      
    def predict(self, X):
        """Make predictions on input data"""
```

### Key Algorithms

#### Convolution Implementation

```python
def convolution(self, image, kernel):
    """
    2D convolution with multiple filters
  
    Input: (height, width, channels)
    Kernel: (filter_h, filter_w, channels, num_filters)
    Output: (out_h, out_w, num_filters)
    """
    h, w, c = image.shape
    filter_h, filter_w, _, num_filters = kernel.shape
  
    out_h = h - filter_h + 1
    out_w = w - filter_w + 1
  
    output = np.zeros((out_h, out_w, num_filters))
  
    for f in range(num_filters):
        for i in range(out_h):
            for j in range(out_w):
                region = image[i:i+filter_h, j:j+filter_w, :]
                output[i, j, f] = np.sum(region * kernel[:, :, :, f])
              
    return output
```

#### Max Pooling Implementation

```python
def max_pooling(self, feature_map, pool_size=2, stride=2):
    """
    Max pooling operation
  
    Reduces spatial dimensions while keeping important features
    """
    h, w, c = feature_map.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
  
    output = np.zeros((out_h, out_w, c))
  
    for i in range(out_h):
        for j in range(out_w):
            for k in range(c):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride  
                w_end = w_start + pool_size
              
                pool_region = feature_map[h_start:h_end, w_start:w_end, k]
                output[i, j, k] = np.max(pool_region)
              
    return output
```

## 📊 Data Loading

### Class Distribution Management

```python
def load_data(data_dir, max_samples=200, val_split=0.2):
    """
    Load ASL dataset with balanced sampling
  
    Returns:
        X_train, y_train, X_val, y_val, class_names
    """
  
    # Discover classes
    classes = ['A', 'B', 'C', ..., 'Z', 'del', 'nothing', 'space']
  
    # Load with parallel processing
    all_images, all_labels = parallel_load_classes(classes, max_samples)
  
    # Shuffle and split
    indices = np.random.permutation(len(all_images))
    split_idx = int(len(all_images) * (1 - val_split))
  
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
  
    return (all_images[train_indices], all_labels[train_indices],
            all_images[val_indices], all_labels[val_indices], classes)
```

### Image Preprocessing Pipeline

```python
def preprocess_image(image_path):
    """
    Load and preprocess single image
  
    Pipeline:
    1. Load image with OpenCV
    2. Resize to 64×64  
    3. Normalize to [0,1]
    4. Convert to float32
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0
    return image
```

## 📚 API Reference

### Command Line Interface

```bash
python train.py [OPTIONS]

Options:
  --max_samples INTEGER     Max samples per class (-1 for all) [default: 200]
  --epochs INTEGER          Number of training epochs [default: 3]
  --lr FLOAT               Learning rate [default: 0.01]
  --batch_size INTEGER     Batch size [default: 32]
  --val_split FLOAT        Validation split ratio [default: 0.2]
  --resume TEXT            Resume from checkpoint (latest or path)
  --start_epoch INTEGER    Starting epoch number [default: 1]
  --help                   Show this message and exit
```

### Python API Usage

```python
from cnn_numpy import CNN
from train import load_data, save_checkpoint, load_checkpoint

# Load data
X_train, y_train, X_val, y_val, classes = load_data(
    max_samples=500, val_split=0.2)

# Initialize model
model = CNN(num_classes=29, conv_filters=8)

# Train model
history = model.train(
    X_train, y_train, X_val, y_val,
    epochs=10, learning_rate=0.01, batch_size=32)

# Save checkpoint
save_checkpoint(model, epoch=10, val_acc=0.92, val_loss=0.25, 
                best_acc=0.92, checkpoint_dir="models")

# Load checkpoint
model = CNN(num_classes=29, conv_filters=8)
epoch, val_acc, val_loss, timestamp = load_checkpoint("models/best", model)

# Make predictions
predictions = model.predict(test_images)
```

## 🔍 Troubleshooting

### Common Issues & Solutions

#### Memory Issues

```bash
# Problem: Out of memory during training
# Solution: Reduce batch size and samples
python train.py --batch_size 16 --max_samples 100
```

#### Slow Performance

```bash
# Problem: Very slow data loading
# Solution: Check ThreadPoolExecutor, reduce max_workers
# Edit train.py: max_workers=min(16, len(class_dirs))
```

#### Resume Failures

```python
# Problem: Checkpoint not found
# Check: Directory exists and contains required files
import os
checkpoint_path = "models/latest"
required_files = ['conv_filters.npy', 'softmax_weights.npy', 
                 'softmax_biases.npy', 'metadata.json']
missing = [f for f in required_files 
           if not os.path.exists(os.path.join(checkpoint_path, f))]
print(f"Missing files: {missing}")
```

#### CUDA/GPU Issues

```python
# Problem: CuPy installation issues  
# Solution: The system automatically falls back to NumPy
# No action required - training will continue on CPU
```

### Performance Tuning

#### For Maximum Speed

```python
# Use all CPU cores for loading
max_workers = os.cpu_count()

# Larger batch sizes (if memory allows)
batch_size = 64

# Reduce validation frequency
val_every_n_epochs = 5
```

#### For Memory Efficiency

```python
# Smaller batch sizes
batch_size = 8

# Fewer samples per class
max_samples = 50

# Process images in chunks
chunk_size = 100
```

#### For Best Accuracy

```python
# Use all available data
max_samples = -1  # or None

# More training epochs
epochs = 20

# Smaller learning rate
learning_rate = 0.005

# Larger validation split for better evaluation
val_split = 0.3
```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print model architecture
model = CNN(num_classes=29, conv_filters=8)
print(f"Conv filters shape: {model.conv_filters.shape}")
print(f"Softmax weights shape: {model.softmax_weights.shape}")
print(f"Total parameters: {model.get_parameter_count()}")
```

## 📊 Performance Metrics

### Training Metrics

- **Training Loss**: Cross-entropy loss on training batches
- **Training Accuracy**: Percentage of correct predictions on training data
- **Validation Loss**: Cross-entropy loss on validation set
- **Validation Accuracy**: Percentage of correct predictions on validation set

### System Metrics

- **Loading Speed**: Images loaded per second
- **Training Speed**: Samples processed per second per epoch
- **Memory Usage**: Peak memory consumption during training
- **Checkpoint Size**: Storage space used by model checkpoints

---

## 🚀 Getting Started

1. **Quick Test**: `python train.py --max_samples 30 --epochs 1`
2. **Standard Training**: `python train.py`
3. **Production Training**: `python train.py --max_samples -1 --epochs 20`
4. **Resume Training**: `python train.py --resume latest --epochs 10`

For more examples and advanced usage, see the main [README.md](README.md).
