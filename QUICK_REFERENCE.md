# ASL CNN Quick Reference

## 🚀 Essential Commands

### Start Training
```bash
# Quick test (30 seconds)
python train.py --max_samples 30 --epochs 1

# Standard training (2-3 minutes)  
python train.py

# High performance (10-15 minutes)
python train.py --max_samples 1000 --epochs 5

# Full dataset (1-2 hours)
python train.py --max_samples -1 --epochs 20
```

### Resume Training
```bash
# Resume from latest checkpoint
python train.py --resume latest --epochs 10

# Resume from specific epoch
python train.py --resume models/epoch_5 --epochs 15

# Resume with different settings
python train.py --resume latest --lr 0.005 --epochs 25
```

## 📁 File Structure
```
CNN model/
├── train.py              # Main training script
├── cnn_numpy.py           # CNN implementation  
├── README.md              # Quick start guide
├── DOCUMENTATION.md       # Technical documentation
├── QUICK_REFERENCE.md     # This file
├── models/                # Saved checkpoints
│   ├── epoch_1/          # Epoch-specific saves
│   ├── epoch_2/          
│   ├── latest/           # Most recent checkpoint
│   ├── best/            # Best performing model
│   └── *.npy           # Legacy format
└── ASL_Alphabet_Dataset/ # Training data
    ├── asl_alphabet_train/
    └── asl_alphabet_test/
```

## ⚙️ Parameter Quick Reference

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--max_samples` | 200 | 30, 100, 500, 1000, -1 | Samples per class |
| `--epochs` | 3 | 1, 5, 10, 20 | Training epochs |
| `--lr` | 0.01 | 0.001, 0.005, 0.01, 0.05 | Learning rate |
| `--batch_size` | 32 | 8, 16, 32, 64 | Batch size |
| `--resume` | None | latest, models/epoch_N | Resume checkpoint |

## 🎯 Use Case Scenarios

### Development & Testing
```bash
# Fast iteration during development
python train.py --max_samples 30 --epochs 1 --lr 0.05

# Quick validation of changes
python train.py --max_samples 100 --epochs 2
```

### Production Training
```bash
# Standard production model
python train.py --max_samples 1000 --epochs 10 --lr 0.005

# Maximum accuracy model
python train.py --max_samples -1 --epochs 20 --lr 0.001
```

### Long Training Sessions
```bash
# Start long training
python train.py --max_samples -1 --epochs 50

# Resume if interrupted
python train.py --resume latest --epochs 50

# Continue with different LR
python train.py --resume latest --lr 0.001 --epochs 70
```

## 📊 Expected Performance

| Configuration | Time | Accuracy | Use Case |
|---------------|------|----------|----------|
| `--max_samples 30 --epochs 1` | 30s | 50-60% | Quick test |
| `python train.py` (default) | 2-3min | 70-80% | Standard |
| `--max_samples 500 --epochs 5` | 8-12min | 75-85% | Validation |
| `--max_samples 1000 --epochs 10` | 25-35min | 80-90% | Production |
| `--max_samples -1 --epochs 20` | 1-2hrs | 85-95% | Maximum |

## 🔧 Troubleshooting Quick Fixes

### Common Problems
```bash
# Too slow? Reduce samples
python train.py --max_samples 100

# Out of memory? Reduce batch size  
python train.py --batch_size 16

# Low accuracy? More data + epochs
python train.py --max_samples 1000 --epochs 10

# Training interrupted? Resume
python train.py --resume latest

# Dataset not found? Check folder structure
ls ASL_Alphabet_Dataset/asl_alphabet_train/
```

### Performance Issues
```bash
# GPU not detected (normal on CPU systems)
CuPy not available, using CPU with NumPy

# Slow loading (reduce thread count if system struggles)
# Edit train.py: max_workers=min(8, len(class_dirs))

# Memory warnings (reduce batch size)
python train.py --batch_size 8 --max_samples 200
```

## 📈 Training Output Interpretation

### Key Metrics to Watch
```
📊 Batch Loss: Should decrease over time
📊 Train Acc: Should increase towards 1.0 (100%)  
📊 Val Accuracy: Should increase (target: >85%)
⏱️ Loading Speed: Should be >500 images/sec
💾 Checkpoint: Saved after each epoch
```

### Good Training Signs
- Loss decreasing steadily
- Training accuracy increasing
- Validation accuracy >80%
- Loading speed >500 img/sec
- No memory errors

### Warning Signs  
- Loss not decreasing after 2-3 epochs
- Validation accuracy stuck <70%
- Very slow loading (<100 img/sec)
- Out of memory errors

## 💾 Checkpoint Management

### Checkpoint Locations
```bash
models/latest/     # Always most recent
models/best/       # Best validation accuracy
models/epoch_N/    # Specific epoch saves
```

### Checkpoint Commands
```bash
# View checkpoints
ls models/

# Check latest metadata
cat models/latest/metadata.json

# Resume from best model
python train.py --resume models/best --epochs 10

# Remove old checkpoints (save space)
rm -rf models/epoch_1 models/epoch_2
```

## 🧪 Testing & Validation

### Quick Model Test
```python
from cnn_numpy import CNN
from train import load_data

# Load test data
X_train, y_train, X_val, y_val, classes = load_data(max_samples=50)

# Test model loading
model = CNN(num_classes=29, conv_filters=8)
model.load_weights("models/best/")

# Check predictions
predictions = model.predict(X_val[:5])
print(f"Sample predictions: {predictions}")
```

### Performance Validation
```bash
# Test on different sample sizes
python train.py --max_samples 100 --epochs 3
python train.py --max_samples 200 --epochs 3  
python train.py --max_samples 500 --epochs 3

# Compare validation accuracies
grep "Val Accuracy" output.log
```

---

## 🎉 Quick Start Checklist

1. ✅ **Install dependencies**: `pip install numpy opencv-python`
2. ✅ **Download ASL dataset** to `ASL_Alphabet_Dataset/`
3. ✅ **Test installation**: `python train.py --max_samples 30 --epochs 1`
4. ✅ **Standard training**: `python train.py`
5. ✅ **Check results**: Look for validation accuracy >70%
6. ✅ **Resume training**: `python train.py --resume latest --epochs 10`

**Need help?** Check [README.md](README.md) for detailed guide or [DOCUMENTATION.md](DOCUMENTATION.md) for technical details.
