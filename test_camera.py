#!/usr/bin/env python3
"""
ASL Camera Test Script
Quick test to verify camera and model functionality
"""

import cv2
import numpy as np
from pathlib import Path
import json

def test_camera(camera_index=0):
    """Test if camera is accessible"""
    print(f"Testing camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Camera {camera_index} not accessible")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Cannot read from camera {camera_index}")
        cap.release()
        return False
    
    print(f"✅ Camera {camera_index} working - Resolution: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()
    return True

def test_model_files(model_path="models/best"):
    """Test if model files exist and are valid"""
    print(f"Testing model at: {model_path}")
    model_dir = Path(model_path)
    
    required_files = ["metadata.json", "conv_filters.npy", "softmax_weights.npy", "softmax_biases.npy"]
    
    for file in required_files:
        file_path = model_dir / file
        if not file_path.exists():
            print(f"❌ Missing file: {file_path}")
            return False
        print(f"✅ Found: {file}")
    
    # Test metadata
    try:
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        print(f"✅ Model metadata: {metadata.get('num_classes', 'Unknown')} classes, "
              f"accuracy: {metadata.get('best_accuracy', 'Unknown')}")
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return False
    
    return True

def test_opencv():
    """Test OpenCV installation"""
    print("Testing OpenCV...")
    try:
        print(f"✅ OpenCV version: {cv2.__version__}")
        return True
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
        return False

def main():
    print("🧪 ASL Camera System Test")
    print("=" * 40)
    
    all_good = True
    
    # Test OpenCV
    if not test_opencv():
        all_good = False
    
    print()
    
    # Test model files
    if not test_model_files():
        all_good = False
    
    print()
    
    # Test camera
    if not test_camera():
        all_good = False
    
    print()
    print("=" * 40)
    
    if all_good:
        print("🎉 All tests passed! You can run the live camera script:")
        print("   python live_camera.py")
        print("\nOptional parameters:")
        print("   --model models/latest    (use latest model)")
        print("   --camera 1               (use camera 1 instead of 0)")
        print("   --roi-size 250           (smaller detection area)")
        print("   --confidence-threshold 0.2  (higher confidence threshold)")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    exit(main())
