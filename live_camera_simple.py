import cv2
import numpy as np
import os
import time
from pathlib import Path
import argparse
import json

from cnn_numpy import CNN

def get_class_mapping():
    """Returns the mapping of class names to indices and vice versa"""
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'del', 'nothing', 'space']
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}
    
    return classes, class_to_idx, idx_to_class

def load_model(model_path="models/best"):
    """Load the trained CNN model from saved weights"""
    print(f"Loading model from: {model_path}")
    
    # Load metadata
    metadata_path = Path(model_path) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_classes = metadata.get('num_classes', 29)
    conv_filters = metadata.get('conv_filters', 8)
    
    print(f"Model info: {num_classes} classes, {conv_filters} conv filters")
    print(f"Best accuracy: {metadata.get('best_accuracy', 'Unknown'):.4f}")
    
    # Initialize CNN with same architecture
    cnn = CNN(num_classes=num_classes)
    
    # Load saved weights
    conv_filters_path = Path(model_path) / "conv_filters.npy"
    softmax_weights_path = Path(model_path) / "softmax_weights.npy"
    softmax_biases_path = Path(model_path) / "softmax_biases.npy"
    
    if not all([conv_filters_path.exists(), softmax_weights_path.exists(), softmax_biases_path.exists()]):
        raise FileNotFoundError("Model weight files not found!")
    
    # Load and set weights
    cnn.conv.filters = np.load(conv_filters_path)
    cnn.softmax.weights = np.load(softmax_weights_path)
    cnn.softmax.biases = np.load(softmax_biases_path)
    
    print("✅ Model loaded successfully!")
    return cnn

def preprocess_image(image, target_size=(28, 28)):
    """Preprocess image for CNN prediction"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    image = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add channel dimension for CNN (height, width, channels)
    image = np.expand_dims(image, axis=-1)
    
    return image

def draw_prediction_info(frame, prediction, confidence, top_3_predictions, fps):
    """Draw prediction information on the frame"""
    height, width = frame.shape[:2]
    
    # Create overlay for better text visibility
    overlay = frame.copy()
    
    # Draw prediction box
    cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
    
    # Main prediction
    cv2.putText(frame, f"Prediction: {prediction}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Top 3 predictions
    cv2.putText(frame, "Top 3:", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    for i, (cls, conf) in enumerate(top_3_predictions):
        y_pos = 120 + i * 20
        cv2.putText(frame, f"{i+1}. {cls}: {conf:.2%}", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 100, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Instructions
    instructions = [
        "Press 'q' to quit",
        "Press 's' to save prediction",
        "Press 'c' to clear prediction history",
        "Press 'r' to reset ROI position"
    ]
    
    for i, instruction in enumerate(instructions):
        y_pos = height - 80 + i * 20
        cv2.putText(frame, instruction, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def draw_roi(frame, roi_x, roi_y, roi_size):
    """Draw region of interest rectangle"""
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 255, 0), 2)
    cv2.putText(frame, "ASL Sign Area", (roi_x, roi_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add crosshairs for better alignment
    center_x = roi_x + roi_size // 2
    center_y = roi_y + roi_size // 2
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)

def main():
    parser = argparse.ArgumentParser(description="Real-time ASL Sign Language Recognition")
    parser.add_argument("--model", "-m", default="models/best", 
                        help="Path to model directory (default: models/best)")
    parser.add_argument("--camera", "-c", type=int, default=0, 
                        help="Camera index (default: 0)")
    parser.add_argument("--roi-size", type=int, default=200, 
                        help="Size of ROI square (default: 200)")
    parser.add_argument("--confidence-threshold", type=float, default=0.1, 
                        help="Minimum confidence to show prediction (default: 0.1)")
    parser.add_argument("--prediction-smoothing", type=int, default=3,
                        help="Number of frames to smooth predictions over (default: 3)")
    parser.add_argument("--target-fps", type=int, default=15,
                        help="Target FPS for camera feed (default: 15)")
    
    args = parser.parse_args()
    
    print("🎯 ASL Real-time Recognition System (Simplified)")
    print("=" * 50)
    
    try:
        # Load the trained model
        cnn = load_model(args.model)
        classes, _, idx_to_class = get_class_mapping()
        
        # Initialize camera
        print(f"Initializing camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, args.target_fps)  # Use configurable FPS
        
        print("✅ Camera initialized successfully!")
        print("\n📸 Starting live prediction...")
        print("Show ASL signs in the green rectangle area")
        print("Press 'q' to quit\n")
        
        # Initialize variables
        roi_size = args.roi_size
        prediction_history = []
        recent_predictions = []
        frame_count = 0
        start_time = time.time()
        target_fps = args.target_fps
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Calculate ROI position (center of frame)
            roi_x = (width - roi_size) // 2
            roi_y = (height - roi_size) // 2
            
            # Extract ROI
            roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
            
            # Preprocess ROI for prediction
            processed_roi = preprocess_image(roi)
            
            # Make prediction
            try:
                prediction_probs = cnn.forward_inference(processed_roi)
                
                # Get top prediction
                predicted_idx = np.argmax(prediction_probs)
                predicted_class = idx_to_class[predicted_idx]
                confidence = prediction_probs[predicted_idx]
                
                # Get top 3 predictions
                top_3_idx = np.argsort(prediction_probs)[-3:][::-1]
                top_3_predictions = [(idx_to_class[i], prediction_probs[i]) for i in top_3_idx]
                
                # Add to recent predictions for smoothing
                recent_predictions.append((predicted_class, confidence))
                if len(recent_predictions) > args.prediction_smoothing:
                    recent_predictions.pop(0)
                
                # Simple smoothing: use most common prediction in recent frames
                if len(recent_predictions) >= args.prediction_smoothing:
                    class_counts = {}
                    total_confidence = 0
                    for pred_class, pred_conf in recent_predictions:
                        if pred_conf >= args.confidence_threshold:
                            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                            total_confidence += pred_conf
                    
                    if class_counts:
                        # Use most frequent prediction
                        smoothed_class = max(class_counts, key=class_counts.get)
                        smoothed_confidence = total_confidence / len(recent_predictions)
                        
                        if class_counts[smoothed_class] >= 2:  # At least 2 out of 3 frames
                            predicted_class = smoothed_class
                            confidence = smoothed_confidence
                
                # Only show prediction if confidence is above threshold
                if confidence < args.confidence_threshold:
                    predicted_class = "Unclear"
                    confidence = 0.0
                
                # Add to history
                prediction_history.append({
                    'class': predicted_class,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
                
                # Keep only last 20 predictions
                if len(prediction_history) > 20:
                    prediction_history.pop(0)
                
            except Exception as e:
                predicted_class = "Error"
                confidence = 0.0
                top_3_predictions = [("Error", 0.0), ("", 0.0), ("", 0.0)]
                print(f"Prediction error: {e}")
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            
            # Draw UI elements
            draw_roi(frame, roi_x, roi_y, roi_size)
            draw_prediction_info(frame, predicted_class, confidence, top_3_predictions, fps)
            
            # Show frame
            cv2.imshow('ASL Real-time Recognition', frame)
            
            # Frame rate limiting
            current_time = time.time()
            elapsed_time = current_time - last_frame_time
            sleep_time = frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.time()
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and predicted_class not in ["Unclear", "Error"]:
                # Save current prediction
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"prediction_{predicted_class}_{timestamp}.jpg"
                cv2.imwrite(filename, roi)
                print(f"Saved prediction: {filename}")
            elif key == ord('c'):
                # Clear prediction history
                prediction_history.clear()
                recent_predictions.clear()
                print("Prediction history cleared")
            elif key == ord('r'):
                # Reset recent predictions (useful if getting stuck predictions)
                recent_predictions.clear()
                print("Recent predictions reset")
            elif key == ord('h'):
                # Show help
                print("\n" + "="*50)
                print("KEYBOARD SHORTCUTS:")
                print("q - Quit application")
                print("s - Save current prediction image")
                print("c - Clear prediction history")
                print("r - Reset recent predictions")
                print("h - Show this help")
                print("="*50 + "\n")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        
        # Show summary
        if 'prediction_history' in locals() and prediction_history:
            print(f"\n📊 Session Summary:")
            print(f"Total predictions: {len(prediction_history)}")
            print(f"Average FPS: {fps:.1f}")
            
            # Show most common predictions
            class_counts = {}
            for pred in prediction_history:
                if pred['class'] not in ['Unclear', 'Error']:
                    class_counts[pred['class']] = class_counts.get(pred['class'], 0) + 1
            
            if class_counts:
                print("Most detected signs:")
                for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  • {cls}: {count} times")
    
    print("👋 Thanks for using ASL Recognition System!")
    return 0

if __name__ == "__main__":
    exit(main())
