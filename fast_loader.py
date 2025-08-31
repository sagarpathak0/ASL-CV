import os
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def get_class_mapping():
    """Returns the mapping of class names to indices and vice versa"""
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'del', 'nothing', 'space']
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}
    
    return classes, class_to_idx, idx_to_class

def load_single_image(img_path, img_size=28):
    """Load and preprocess a single image"""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        return img.reshape(img_size, img_size, 1)
    except:
        return None

def count_images_per_class(data_dir, max_samples_per_class=None):
    """Count total images to pre-allocate arrays"""
    classes, class_to_idx, _ = get_class_mapping()
    total_count = 0
    class_counts = {}
    
    for cls in classes:
        cls_folder = Path(data_dir) / cls
        if not cls_folder.is_dir():
            class_counts[cls] = 0
            continue
            
        files = [f for f in cls_folder.iterdir() 
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
        
        count = len(files)
        if max_samples_per_class:
            count = min(count, max_samples_per_class)
        
        class_counts[cls] = count
        total_count += count
    
    return total_count, class_counts

def load_class_batch(args):
    """Load all images for a single class"""
    cls, cls_folder, label_idx, img_size, max_samples = args
    
    cls_path = Path(cls_folder)
    if not cls_path.is_dir():
        return [], []
    
    # Get all image files
    files = [f for f in cls_path.iterdir() 
             if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
    
    if max_samples:
        files = files[:max_samples]
    
    # Load images
    images = []
    labels = []
    
    for img_path in files:
        img = load_single_image(img_path, img_size)
        if img is not None:
            images.append(img)
            labels.append(label_idx)
    
    return images, labels

def load_asl_dataset_fast(data_dir, img_size=28, val_split=0.2, seed=42, max_samples_per_class=None):
    """Ultra-fast ASL dataset loading"""
    print("🚀 Fast loading ASL dataset...")
    start_time = time.time()
    
    classes, class_to_idx, idx_to_class = get_class_mapping()
    
    # Count total images for pre-allocation
    print("📊 Counting images...")
    total_count, class_counts = count_images_per_class(data_dir, max_samples_per_class)
    print(f"📁 Found {total_count} images across {len(classes)} classes")
    
    if total_count == 0:
        raise ValueError("No images found! Check your data directory path.")
    
    # Prepare class loading arguments
    class_args = []
    for cls in classes:
        if class_counts[cls] > 0:
            cls_folder = os.path.join(data_dir, cls)
            class_args.append((cls, cls_folder, class_to_idx[cls], img_size, max_samples_per_class))
    
    # Load all classes in parallel
    print(f"⚡ Loading {len(class_args)} classes in parallel...")
    
    all_images = []
    all_labels = []
    
    with ThreadPoolExecutor(max_workers=min(8, len(class_args))) as executor:
        results = list(executor.map(load_class_batch, class_args))
    
    for i, (images, labels) in enumerate(results):
        if images:
            all_images.extend(images)
            all_labels.extend(labels)
            cls = class_args[i][0]
            print(f"✅ {cls}: {len(images)} images")
    
    # Convert to numpy arrays
    print("🔄 Converting to numpy arrays...")
    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    loading_time = time.time() - start_time
    print(f"\n🎉 Loading completed in {loading_time:.2f} seconds!")
    print(f"📈 Loading speed: {len(X)/loading_time:.1f} images/second")
    print(f"💾 Memory usage: {X.nbytes / 1024 / 1024:.1f} MB")
    
    # Shuffle data
    print("🔀 Shuffling data...")
    rng = np.random.RandomState(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split into train/validation
    n_val = int(len(X) * val_split)
    if n_val == 0:
        n_val = max(1, len(X) // 10)
    
    X_train = X[n_val:]
    y_train = y[n_val:]
    X_val = X[:n_val]
    y_val = y[:n_val]
    
    print(f"📚 Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    return (X_train, y_train), (X_val, y_val)

if __name__ == "__main__":
    # Test the fast loader
    data_dir = "ASL_Alphabet_Dataset/asl_alphabet_train"
    (X_train, y_train), (X_val, y_val) = load_asl_dataset_fast(
        data_dir, max_samples_per_class=100
    )
    print(f"Final shapes: X_train={X_train.shape}, y_train={y_train.shape}")
