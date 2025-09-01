#!/usr/bin/env python3
"""
ASL Recognition System Demo
Comprehensive demonstration of all available features
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_header(title):
    """Print a nice header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a step"""
    print(f"\n🎯 Step {step_num}: {description}")
    print("-" * 40)

def run_command(command, description):
    """Run a command with description"""
    print(f"Running: {description}")
    print(f"Command: {command}")
    print()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ Error!")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print_header("🤖 ASL Recognition System - Complete Demo")
    
    python_exe = "\"C:/Users/sagar/OneDrive/desktop/CNN model/.venv/Scripts/python.exe\""
    
    print("""
This demo will showcase all the features of the ASL Recognition System:

1. System Testing
2. MediaPipe Functionality Test  
3. Batch Prediction (Original)
4. Batch Prediction (MediaPipe Enhanced)
5. Live Camera Options

Press Enter to continue or Ctrl+C to exit...
    """)
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return 0
    
    # Step 1: System Testing
    print_step(1, "System Testing")
    run_command(f"{python_exe} test_camera.py", "Testing camera and model files")
    
    # Step 2: MediaPipe Testing
    print_step(2, "MediaPipe Functionality Test")
    print("This will open a 10-second hand tracking test...")
    time.sleep(2)
    run_command(f"{python_exe} test_mediapipe.py", "Testing MediaPipe hand tracking")
    
    # Step 3: Batch Prediction (Original)
    print_step(3, "Batch Prediction - Original Method")
    run_command(f"{python_exe} batch_predict.py \"ASL_Alphabet_Dataset/asl_alphabet_test/A_test.jpg\" \"ASL_Alphabet_Dataset/asl_alphabet_test/B_test.jpg\" --top 3", 
                "Testing original batch prediction")
    
    # Step 4: Batch Prediction (MediaPipe)
    print_step(4, "Batch Prediction - MediaPipe Enhanced")
    run_command(f"{python_exe} batch_predict_mp.py \"ASL_Alphabet_Dataset/asl_alphabet_test/A_test.jpg\" \"ASL_Alphabet_Dataset/asl_alphabet_test/B_test.jpg\" --fallback --top 3", 
                "Testing MediaPipe batch prediction with fallback")
    
    # Step 5: Live Camera Information
    print_step(5, "Live Camera Options")
    print("""
📹 Live Camera Recognition Options:

1. Original Version:
   python live_camera.py
   • Fixed ROI rectangle
   • Manual positioning required
   
2. MediaPipe Enhanced Version:
   python live_camera_mp.py
   • Automatic hand detection
   • Hand landmark tracking
   • Better accuracy
   
3. MediaPipe with Hand Landmarks:
   python live_camera_mp.py --show-landmarks
   • Shows skeletal hand overlay
   • Great for debugging
   
4. Customized Settings:
   python live_camera_mp.py --hand-confidence 0.8 --confidence-threshold 0.2
   • Higher detection thresholds
   • More accurate predictions

🎮 Interactive Controls (for both versions):
   Q - Quit application
   S - Save current prediction
   C - Clear prediction history  
   H - Show help menu
    """)
    
    print_header("📊 Demo Summary")
    print("""
✅ Available Scripts:

🧪 Testing:
   • test_camera.py        - Camera and model verification
   • test_mediapipe.py     - MediaPipe functionality test

📸 Live Recognition:
   • live_camera.py        - Original version with fixed ROI
   • live_camera_mp.py     - MediaPipe enhanced version

🖼️  Batch Processing:
   • batch_predict.py      - Original batch prediction
   • batch_predict_mp.py   - MediaPipe enhanced batch prediction

📚 Documentation:
   • USAGE_GUIDE.md        - Comprehensive usage guide
   • CAMERA_DEMO.md        - Camera demo instructions
   • MEDIAPIPE_COMPARISON.md - Feature comparison

🎯 Supported ASL Signs:
   A-Z (26 letters) + del, nothing, space (29 total)

🚀 Recommended Usage:
   python live_camera_mp.py --show-landmarks
   
   This provides the best experience with:
   • Automatic hand detection
   • Visual feedback with landmarks
   • No positioning constraints
   • Better accuracy
    """)
    
    print_header("🎉 Demo Complete!")
    print("""
Thank you for trying the ASL Recognition System!

Next steps:
1. Try the live camera: python live_camera_mp.py
2. Test with your own images: python batch_predict_mp.py your_image.jpg
3. Explore different settings and options
4. Check the documentation files for detailed guides

Happy signing! 🤟
    """)
    
    return 0

if __name__ == "__main__":
    exit(main())
