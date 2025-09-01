# 🎯 ASL Recognition System - Clean & Simple

## ✅ **Working Scripts**

### 📹 **Live Camera Recognition**
- **`live_camera_simple.py`** ⭐ **RECOMMENDED** - Simple, reliable, no dependencies
- **`live_camera.py`** - Original version with more features

### 🖼️ **Batch Image Processing**  
- **`batch_predict.py`** - Process single or multiple images

### 🧪 **Testing**
- **`test_camera.py`** - Verify camera, model, and OpenCV

## 🚀 **Quick Start**

### **Get Started Immediately**
```bash
# Test your system
python test_camera.py

# Start live recognition (recommended)
python live_camera_simple.py

# Test with an image
python batch_predict.py "ASL_Alphabet_Dataset/asl_alphabet_test/A_test.jpg"
```

### **Customization Options**
```bash
# Larger detection area
python live_camera_simple.py --roi-size 250

# Higher confidence threshold (more strict)
python live_camera_simple.py --confidence-threshold 0.3

# More prediction smoothing (less flickering)
python live_camera_simple.py --prediction-smoothing 5

# Combine options
python live_camera_simple.py --roi-size 200 --confidence-threshold 0.2 --prediction-smoothing 4
```

## 🎮 **Interactive Controls**

While using live camera:
- **Q** - Quit application
- **S** - Save current prediction as image
- **C** - Clear prediction history
- **R** - Reset recent predictions (if stuck)
- **H** - Show help menu

## 🎯 **Features**

### **Live Recognition**
- ✅ Real-time ASL sign detection
- 📊 Confidence scores and top 3 predictions
- 🎯 Green ROI rectangle with crosshairs for alignment
- 📈 FPS monitoring
- 🔄 Prediction smoothing to reduce flickering
- 💾 Save predictions as images

### **Batch Processing**
- 🖼️ Process single images or entire directories
- 📊 Accuracy calculation for test images
- 💾 Save results to JSON file
- 👁️ Optional image display during processing

### **Model Performance**
- 🏆 **78.8% accuracy** on validation set
- 🎯 **29 ASL classes**: A-Z + del, nothing, space
- ⚡ **Fast inference**: Real-time performance
- 📐 **28x28 input**: Optimized for speed

## 🔧 **Troubleshooting**

### **If predictions seem stuck on F and L:**
```bash
# Press 'R' to reset recent predictions
# Or restart with higher confidence threshold
python live_camera_simple.py --confidence-threshold 0.3
```

### **If accuracy is low:**
```bash
# Try different lighting conditions
# Use larger ROI area
python live_camera_simple.py --roi-size 300

# Increase prediction smoothing
python live_camera_simple.py --prediction-smoothing 5
```

### **Camera issues:**
```bash
# Test camera first
python test_camera.py

# Try different camera index
python live_camera_simple.py --camera 1
```

## 💡 **Tips for Best Results**

### **Lighting & Setup**
- 🔆 **Good lighting**: Bright, even illumination
- 🎯 **Plain background**: Solid color, contrasting with hand
- 📏 **Distance**: About arm's length from camera
- 👁️ **Position**: Face camera directly

### **Sign Recognition**
- ✋ **Clear signs**: Make distinct, well-formed ASL signs
- ⏱️ **Hold steady**: Keep sign stable for 1-2 seconds
- 🎯 **Center in ROI**: Keep hand in green rectangle
- 📐 **Fill area**: Hand should fill ~70% of ROI

### **Improving Accuracy**
- 📊 **Check confidence**: Aim for >50% confidence scores
- 🔄 **Use smoothing**: Default 3-frame smoothing reduces noise
- 🎯 **Adjust threshold**: Higher threshold = more strict predictions
- 📏 **ROI size**: Smaller ROI = more focused detection

## 📊 **Supported Signs**

```
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
del (delete) | nothing (no sign) | space
```

## 🎓 **Training Your Own Model**

```bash
# Quick training (2-3 minutes)
python train.py --max_samples 200 --epochs 3

# Full training (45-60 minutes, best accuracy)
python train.py

# Resume training
python train.py --resume latest --epochs 10
```

## 📁 **File Structure**

```
live_camera_simple.py    # Simple live recognition (recommended)
live_camera.py          # Feature-rich live recognition  
batch_predict.py        # Batch image processing
test_camera.py          # System testing
train.py               # Model training
cnn_numpy.py           # CNN implementation
models/best/           # Best trained model
ASL_Alphabet_Dataset/  # Training data
```

## 🎉 **Success!**

Your ASL recognition system is now clean, simple, and reliable! 

**Recommended workflow:**
1. `python test_camera.py` - Verify system
2. `python live_camera_simple.py` - Start recognizing
3. Adjust settings as needed for your environment

Happy signing! 🤟

---

**Note**: MediaPipe has been removed due to prediction issues. The current system uses a reliable ROI-based approach that works consistently across different environments.
