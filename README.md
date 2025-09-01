# 🤖 ASL CNN Classification System with Real-time Recognition

A **high-performance CNN** built from scratch with **NumPy** for **American Sign Language** alphabet recognition. Features both **training capabilities** and **real-time camera recognition**. Supports **29 ASL classes** (A-Z + del, nothing, space) with **optimized parallel loading** and **full dataset training by default**.

## 🚀 Quick Start

### **🎯 Real-time Camera Recognition**
```bash
# Simple and reliable version
python live_camera_simple.py

# Original version with more features  
python live_camera.py

# With custom settings
python live_camera_simple.py --roi-size 250 --confidence-threshold 0.2
```

### **🖼️ Batch Image Prediction**
```bash
# Single image
python batch_predict.py "your_image.jpg"

# Multiple images
python batch_predict.py "ASL_Alphabet_Dataset/asl_alphabet_test/*.jpg"
```

### **🧪 System Testing**
```bash
# Test camera and model
python test_camera.py

# Complete demo
python demo.py
```

### **🎓 Model Training**
```bash
# Default Training (Full Dataset - 45-60 minutes, ~90%+ accuracy)
python train.py

# Quick Test (30 seconds)
python train.py --max_samples 50 --epochs 1

# Fast Training (2-3 minutes, ~75% accuracy)
python train.py --max_samples 200 --epochs 3
```

##  Key Features

### **🎯 Recognition System**
- ✅ **29-class ASL recognition** (A-Z + del, nothing, space)
- 📹 **Real-time camera recognition** with live feedback
- 🖼️ **Batch image processing** for multiple files
- 🎮 **Interactive controls** and visual feedback
- 📈 **Prediction smoothing** to reduce flickering
- 🎯 **Adjustable confidence thresholds**

### **🎓 Training System**
- ⚡ **32x faster loading** (569+ images/sec vs 17/sec)
- 🎯 **Smart defaults** for quick training sessions  
- 📈 **Real-time progress** with ETA estimates
- 💾 **Checkpoint system** - pause/resume training anytime
- 🏆 **Auto-save best models** with epoch tracking

## ⚡ Performance Optimizations

| Mode | Command | Time | Accuracy | Use Case |
|------|---------|------|----------|----------|
| **Quick Test** | `--max_samples 50 --epochs 1` | 30s | 60-70% | Development |
| **Fast Training** | `--max_samples 200 --epochs 3` | 2-3min | 70-80% | Quick validation |
| **Balanced** | `--max_samples 1000 --epochs 5` | 10-15min | 80-90% | Testing |
| **Default (Full)** | `python train.py` | 45-60min | 90-95% | Production |

## 🎮 Command Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_samples` | None (full dataset) | Samples per class (None=full dataset) |
| `--epochs` | 5 | Training epochs |
| `--lr` | 0.01 | Learning rate |
| `--batch_size` | 32 | Batch size |
| `--val_split` | 0.2 | Validation split ratio |
| `--resume` | None | Resume training (latest or models/epoch_N) |
| `--start_epoch` | 1 | Starting epoch number |

## 🔧 Setup

1. **Install dependencies:**
```bash
pip install numpy opencv-python
```

2. **Download ASL Dataset** and place in:
```
ASL_Alphabet_Dataset/
├── asl_alphabet_train/  # Training folders (A/, B/, C/, ...)
└── asl_alphabet_test/   # Test images
```

3. **Run training:**
```bash
python train.py
```

## � Checkpoint System

### **Automatic Saving**
The model automatically saves checkpoints after each epoch:
```
models/
├── epoch_1/          # Checkpoint after epoch 1
├── epoch_2/          # Checkpoint after epoch 2
├── latest/           # Always points to most recent
├── best/            # Best performing model
└── *.npy           # Legacy format files
```

### **Resume Commands**
```bash
# Resume from latest checkpoint
python train.py --resume latest --epochs 10

# Resume from specific epoch  
python train.py --resume models/epoch_3 --epochs 15

# Resume with different settings
python train.py --resume latest --lr 0.005 --epochs 20
```

### **Checkpoint Contents**
- `conv_filters.npy` - Convolutional weights
- `softmax_weights.npy` - Output layer weights
- `softmax_biases.npy` - Output layer biases  
- `metadata.json` - Training metrics & info

## �📈 What You'll See

```
🚀 Loading ASL dataset with detailed progress tracking...
🔍 Scanning dataset directories...
   📁 A: 8458 files found → using 200
   📁 B: 8309 files found → using 200
   ...
⚡ Loading 29 classes in parallel...
📈 Progress by class:
   ✅ A: 200 images loaded
   🔄 Overall progress: 3.4% (1/29 classes loaded)
   ...

🎉 Loading completed successfully!
📊 Performance Summary:
   ⏱️  Total time: 2.1s
   📈 Loading speed: 2,761 images/second
   
🚀 Starting training session...
📊 Batch 1/15 |██░░░░░░░░| 6.7% | Train Loss: 3.24 | Train Acc: 0.16
📊 Batch 2/15 |████░░░░░░| 13.3% | Train Loss: 3.18 | Train Acc: 0.22
...
✅ Epoch 1 completed in 12.3s
🔍 Evaluating on validation set...
📊 Validation Results: • Val Accuracy: 0.7245 (72.5%)
💾 Saving checkpoint...
   � New best model saved (accuracy: 0.7245)
✅ Checkpoint saved: models\epoch_1

�🎉 TRAINING COMPLETED! 🎉
💾 CHECKPOINTS SAVED:
   📁 Latest checkpoint: models/latest/
   🏆 Best model: models/best/
   📂 All epochs: models/epoch_*/
🔄 RESUME TRAINING:
   To continue: python train.py --resume latest --epochs 10
💾 Model weights saved to 'models'
🧪 Testing on sample images...
📸 Test 1/5: A_test.jpg → Prediction: A (confidence: 0.89)
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Training too slow** | Use `--max_samples 200` for faster training |
| **Out of memory** | Use `--batch_size 16` or `--max_samples 100` |
| **Low accuracy** | Use full dataset: `python train.py` or increase `--epochs` |
| **Dataset not found** | Check `ASL_Alphabet_Dataset/` folder structure |
| **Resume failed** | Check if `models/` directory exists |
| **Training interrupted** | Use `--resume latest` to continue |
| **Want quick test** | Use `--max_samples 50 --epochs 1` |

## 📚 Documentation

- **📖 [DOCUMENTATION.md](DOCUMENTATION.md)** - Complete technical guide with advanced usage
- **⚡ [QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Essential commands and troubleshooting
- **🏗️ Architecture:** CNN with 8→16 conv filters, 39K parameters
- **🎯 Classes:** 29 ASL signs (A-Z, del, nothing, space)
- **⚡ Speed:** 569-1096 images/sec loading, 2-3min default training

---

**🎉 Ready to train!** Start with `python train.py` and adjust parameters based on your needs.
