# ASL Real-time Recognition Demo

## Quick Start

Run the camera script with default settings:
```bash
python live_camera.py
```

## Features

### 🎯 Real-time Prediction
- Live camera feed with ASL sign detection
- Shows prediction confidence and top 3 possibilities
- Mirror mode for natural interaction

### 📊 Performance Info
- Real-time FPS display
- Confidence threshold filtering
- Prediction history tracking

### 🎮 Interactive Controls
- **Q**: Quit application
- **S**: Save current prediction image
- **C**: Clear prediction history
- **H**: Show help menu

### ⚙️ Customization Options

```bash
# Use a different model
python live_camera.py --model models/latest

# Use external webcam (camera index 1)
python live_camera.py --camera 1

# Smaller detection area (good for close-up)
python live_camera.py --roi-size 250

# Higher confidence threshold (more strict)
python live_camera.py --confidence-threshold 0.3

# Combine multiple options
python live_camera.py --model models/latest --roi-size 200 --confidence-threshold 0.25
```

## Supported ASL Signs

The model recognizes 29 different signs:
- **Letters**: A-Z (26 letters)
- **Special**: del (delete), nothing (no sign), space

## Tips for Better Recognition

1. **Lighting**: Use good, even lighting
2. **Background**: Plain, contrasting background works best
3. **Position**: Keep hand centered in the blue rectangle
4. **Distance**: About arm's length from camera
5. **Clarity**: Make clear, distinct signs
6. **Stability**: Hold the sign steady for better recognition

## Troubleshooting

### Camera Issues
```bash
# Test camera functionality
python test_camera.py

# Try different camera index
python live_camera.py --camera 1
```

### Low Accuracy
- Increase confidence threshold: `--confidence-threshold 0.2`
- Try different lighting conditions
- Use the best model: `--model models/best`
- Make sure signs are clear and centered

### Performance Issues
- Reduce ROI size: `--roi-size 200`
- Close other applications using the camera
- Check camera resolution settings

## Model Information

Current best model performance:
- **Accuracy**: 78.8%
- **Classes**: 29 ASL signs
- **Architecture**: Custom CNN with NumPy
- **Training**: Trained on ASL Alphabet Dataset

## Files Created
- `live_camera.py`: Main camera application
- `test_camera.py`: System testing utility
- `prediction_*.jpg`: Saved prediction images (when pressing 'S')

## Example Session

```
🎯 ASL Real-time Recognition System
==================================================
Loading model from: models/best
Model info: 29 classes, 8 conv filters
Best accuracy: 0.7881
✅ Model loaded successfully!
Initializing camera 0...
✅ Camera initialized successfully!

📸 Starting live prediction...
Show ASL signs in the blue rectangle area
Press 'q' to quit
```

Happy signing! 🤟
