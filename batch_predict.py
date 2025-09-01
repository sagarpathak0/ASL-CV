#!/usr/bin/env python3
"""
ASL Batch Image Prediction
Test ASL signs from static images
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import json
import glob

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
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_classes = metadata.get('num_classes', 29)
    
    # Initialize CNN with same architecture
    cnn = CNN(num_classes=num_classes)
    
    # Load saved weights
    cnn.conv.filters = np.load(Path(model_path) / "conv_filters.npy")
    cnn.softmax.weights = np.load(Path(model_path) / "softmax_weights.npy")
    cnn.softmax.biases = np.load(Path(model_path) / "softmax_biases.npy")
    
    return cnn

def preprocess_image(image_path, target_size=(28, 28)):
    """Load and preprocess image for CNN prediction"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    image = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add channel dimension for CNN
    image = np.expand_dims(image, axis=-1)
    
    return image

def predict_image(cnn, image_path, idx_to_class, show_top_n=3):
    """Predict ASL sign for a single image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        # Make prediction
        prediction_probs = cnn.forward_inference(processed_image)
        
        # Get top predictions
        top_n_idx = np.argsort(prediction_probs)[-show_top_n:][::-1]
        top_n_predictions = [(idx_to_class[i], prediction_probs[i]) for i in top_n_idx]
        
        return top_n_predictions, True
    
    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return [], False

def main():
    parser = argparse.ArgumentParser(description="Batch ASL Sign Language Prediction")
    parser.add_argument("input", nargs="+", 
                        help="Image file(s) or directory to predict")
    parser.add_argument("--model", "-m", default="models/best", 
                        help="Path to model directory (default: models/best)")
    parser.add_argument("--top", "-t", type=int, default=3, 
                        help="Show top N predictions (default: 3)")
    parser.add_argument("--save-results", "-s", action="store_true",
                        help="Save results to a file")
    parser.add_argument("--show-images", action="store_true",
                        help="Display images during prediction")
    
    args = parser.parse_args()
    
    print("🖼️  ASL Batch Image Prediction")
    print("=" * 50)
    
    try:
        # Load model
        cnn = load_model(args.model)
        classes, _, idx_to_class = get_class_mapping()
        
        # Collect image paths
        image_paths = []
        for input_path in args.input:
            path = Path(input_path)
            if path.is_file():
                if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.append(path)
            elif path.is_dir():
                # Find all image files in directory
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_paths.extend(path.glob(ext))
                    image_paths.extend(path.glob(ext.upper()))
        
        if not image_paths:
            print("❌ No valid image files found!")
            return 1
        
        print(f"Found {len(image_paths)} image(s) to process")
        print()
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
            
            # Make prediction
            predictions, success = predict_image(cnn, image_path, idx_to_class, args.top)
            
            if success and predictions:
                predicted_class = predictions[0][0]
                confidence = predictions[0][1]
                
                print(f"  🎯 Prediction: {predicted_class} ({confidence:.2%})")
                
                # Show top predictions
                if args.top > 1:
                    print(f"  📊 Top {args.top}:")
                    for j, (cls, conf) in enumerate(predictions, 1):
                        print(f"     {j}. {cls}: {conf:.2%}")
                
                # Check if prediction matches filename (for test images)
                actual_class = None
                if "_test.jpg" in image_path.name:
                    actual_class = image_path.name.replace("_test.jpg", "").upper()
                elif image_path.parent.name in classes:
                    actual_class = image_path.parent.name
                
                if actual_class:
                    total_predictions += 1
                    if predicted_class.upper() == actual_class.upper():
                        correct_predictions += 1
                        print(f"  ✅ Correct! (Expected: {actual_class})")
                    else:
                        print(f"  ❌ Wrong (Expected: {actual_class})")
                
                results.append({
                    'image': str(image_path),
                    'predicted': predicted_class,
                    'confidence': confidence,
                    'actual': actual_class,
                    'correct': predicted_class.upper() == (actual_class.upper() if actual_class else None),
                    'top_predictions': predictions
                })
                
                # Show image if requested
                if args.show_images:
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        # Resize for display
                        height, width = img.shape[:2]
                        if height > 500 or width > 500:
                            scale = min(500/height, 500/width)
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            img = cv2.resize(img, (new_width, new_height))
                        
                        # Add prediction text
                        cv2.putText(img, f"Prediction: {predicted_class}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(img, f"Confidence: {confidence:.2%}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.imshow(f"Prediction: {predicted_class}", img)
                        cv2.waitKey(1000)  # Show for 1 second
                        cv2.destroyAllWindows()
            
            else:
                print(f"  ❌ Failed to process image")
            
            print()
        
        # Summary
        print("=" * 50)
        print("📊 Summary:")
        print(f"   Total images processed: {len(image_paths)}")
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f"   Images with known labels: {total_predictions}")
            print(f"   Correct predictions: {correct_predictions}")
            print(f"   Accuracy: {accuracy:.2%}")
        
        # Save results if requested
        if args.save_results:
            import json
            output_file = "batch_prediction_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"   Results saved to: {output_file}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
