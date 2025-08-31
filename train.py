import os, cv2, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import sys

# Try CuPy first, fallback to NumPy
try:
    import numpy as np
    GPU_AVAILABLE = False
    print("CuPy not available, using CPU with NumPy")
except ImportError:
    import numpy as np
    GPU_AVAILABLE = False
    print("CuPy not available, using CPU with NumPy")

from cnn_numpy import CNN

def get_class_mapping():
    """Returns the mapping of class names to indices and vice versa"""
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'del', 'nothing', 'space']
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}
    
    return classes, class_to_idx, idx_to_class

def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', length=50, fill='█'):
    """Print a detailed progress bar with percentage and ETA"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()  # New line when complete

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m {int(seconds%60)}s"

def print_training_header():
    """Print a nice training header"""
    print("\n" + "="*80)
    print("🤖 CNN TRAINING SESSION STARTED 🤖".center(80))
    print("="*80)

def print_epoch_header(epoch, total_epochs):
    """Print epoch header with progress"""
    print(f"\n{'='*20} EPOCH {epoch}/{total_epochs} {'='*20}")
    print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")

def print_batch_progress(batch_num, total_batches, loss, acc, elapsed_time, eta):
    """Print detailed batch progress"""
    progress = batch_num / total_batches * 100
    bar_length = 30
    filled = int(bar_length * batch_num // total_batches)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\r📊 Batch {batch_num:4d}/{total_batches:4d} |{bar}| "
          f"{progress:5.1f}% | Train Loss: {loss:.4f} | Train Acc: {acc:.4f} | "
          f"⏱️ {format_time(elapsed_time)} | ETA: {format_time(eta)}", 
          end='', flush=True)

def load_single_image(img_path, img_size=28):
    """Load and preprocess a single image efficiently"""
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
    """Count total images to pre-allocate arrays with detailed output"""
    classes, class_to_idx, _ = get_class_mapping()
    total_count = 0
    class_counts = {}
    
    print("🔍 Scanning dataset directories...")
    
    for cls in classes:
        cls_folder = Path(data_dir) / cls
        if not cls_folder.is_dir():
            print(f"   ⚠️  {cls}: Directory not found")
            class_counts[cls] = 0
            continue
            
        files = [f for f in cls_folder.iterdir() 
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
        
        count = len(files)
        if max_samples_per_class:
            actual_count = min(count, max_samples_per_class)
            print(f"   📁 {cls}: {count} files found → using {actual_count}")
        else:
            actual_count = count
            print(f"   📁 {cls}: {count} files found")
        
        class_counts[cls] = actual_count
        total_count += actual_count
    
    return total_count, class_counts

def load_class_batch(args):
    """Load all images for a single class efficiently with progress tracking"""
    cls, cls_folder, label_idx, img_size, max_samples = args
    
    cls_path = Path(cls_folder)
    if not cls_path.is_dir():
        print(f"⚠️  {cls}: Directory not found")
        return [], []
    
    # Get all image files
    files = [f for f in cls_path.iterdir() 
             if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
    
    total_files = len(files)
    if max_samples:
        files = files[:max_samples]
        print(f"📂 {cls}: Found {total_files} files, using {len(files)}")
    else:
        print(f"📂 {cls}: Found {total_files} files, loading all...")
    
    # Load images with progress tracking
    images = []
    labels = []
    
    # Show progress for large classes
    show_progress = len(files) > 50
    if show_progress:
        print(f"   🔄 Loading {cls}...", end='', flush=True)
    
    for i, img_path in enumerate(files):
        img = load_single_image(img_path, img_size)
        if img is not None:
            images.append(img)
            labels.append(label_idx)
        
        # Show progress for large classes
        if show_progress and (i + 1) % max(1, len(files) // 10) == 0:
            progress = (i + 1) / len(files) * 100
            print(f"\r   🔄 Loading {cls}: {progress:.0f}% ({i+1}/{len(files)})", end='', flush=True)
    
    if show_progress:
        print(f"\r   ✅ {cls}: {len(images)} images loaded successfully")
    else:
        print(f"   ✅ {cls}: {len(images)} images loaded")
    
    return images, labels

#Data Load and Split

def load_asl_dataset(data_dir, img_size=28, val_split=0.2, seed=42, max_samples_per_class=None):
    """Ultra-fast ASL dataset loading with detailed progress tracking"""
    print("🚀 Loading ASL dataset with detailed progress tracking...")
    start_time = time.time()
    
    classes, class_to_idx, idx_to_class = get_class_mapping()
    
    # Count total images for estimation with detailed output
    count_start = time.time()
    total_count, class_counts = count_images_per_class(data_dir, max_samples_per_class)
    count_time = time.time() - count_start
    
    print(f"\n� Dataset Summary (scan took {count_time:.1f}s):")
    print(f"   📁 Total images to load: {total_count:,}")
    print(f"   📚 Classes with data: {sum(1 for c in class_counts.values() if c > 0)}/29")
    
    if total_count == 0:
        raise ValueError("❌ No images found! Check your data directory path.")
    
    if total_count > 5000:
        print(f"⚠️  Large dataset detected ({total_count:,} images)")
        print(f"   💡 Consider using --max_samples for faster loading during development")
        print(f"   ⏱️  Estimated loading time: {total_count/500:.1f}-{total_count/200:.1f} minutes")
    
    # Prepare class loading arguments
    class_args = []
    for cls in classes:
        if class_counts[cls] > 0:
            cls_folder = os.path.join(data_dir, cls)
            class_args.append((cls, cls_folder, class_to_idx[cls], img_size, max_samples_per_class))
    
    # Load all classes with detailed progress
    print(f"\n⚡ Loading {len(class_args)} classes in parallel...")
    print("📈 Progress by class:")
    
    all_images = []
    all_labels = []
    
    # Use ThreadPoolExecutor but with progress tracking
    with ThreadPoolExecutor(max_workers=min(4, len(class_args))) as executor:
        # Submit all tasks
        future_to_class = {executor.submit(load_class_batch, args): args[0] for args in class_args}
        
        # Process results as they complete
        completed = 0
        
        for future in as_completed(future_to_class):
            cls_name = future_to_class[future]
            try:
                images, labels = future.result()
                if images:
                    all_images.extend(images)
                    all_labels.extend(labels)
                
                completed += 1
                progress = completed / len(class_args) * 100
                print(f"🔄 Overall progress: {progress:.1f}% ({completed}/{len(class_args)} classes loaded)")
                
            except Exception as exc:
                print(f"❌ {cls_name} generated an exception: {exc}")
    
    # Convert to numpy arrays
    print(f"\n🔄 Converting {len(all_images)} images to numpy arrays...")
    array_start = time.time()
    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    array_time = time.time() - array_start
    
    loading_time = time.time() - start_time
    print(f"\n🎉 Loading completed successfully!")
    print(f"📊 Performance Summary:")
    print(f"   ⏱️  Total time: {format_time(loading_time)}")
    print(f"   📈 Loading speed: {len(X)/loading_time:.1f} images/second")
    print(f"   💾 Memory usage: {X.nbytes / 1024 / 1024:.1f} MB")
    print(f"   🔄 Array conversion: {format_time(array_time)}")

    # Convert to GPU if available
    if GPU_AVAILABLE:
        print("🚀 Converting to GPU arrays...")
        X = np.asarray(X)  # Ensure it's on GPU
        y = np.asarray(y)

    # Shuffle data
    print("🔀 Shuffling data...")
    shuffle_start = time.time()
    rng = np.random.RandomState(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X = X[indices]
    y = y[indices]
    shuffle_time = time.time() - shuffle_start

    # Split into train/validation
    print("✂️  Splitting into train/validation sets...")
    n_val = int(len(X) * val_split)
    if n_val == 0:
        n_val = max(1, len(X) // 10)

    X_train = X[n_val:]
    y_train = y[n_val:]
    X_val = X[:n_val]
    y_val = y[:n_val]

    print(f"📚 Dataset split completed:")
    print(f"   🎯 Training samples: {len(X_train):,}")
    print(f"   🔍 Validation samples: {len(X_val):,}")
    print(f"   📊 Validation split: {len(X_val)/len(X)*100:.1f}%")
    
    return (X_train, y_train), (X_val, y_val)

def predict_image(cnn, image_path, img_size=28):
    """Predict the class of a single image"""
    classes, _, idx_to_class = get_class_mapping()
    
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, "Could not load image"
    
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(img_size, img_size, 1)
    
    # Forward pass (we use label 0 as dummy, only need the output)
    out, _, _ = cnn.forward(img, 0)
    
    # Get prediction
    predicted_idx = np.argmax(out)
    confidence = out[predicted_idx]
    predicted_class = idx_to_class[predicted_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(out)[-3:][::-1]
    top_3 = [(idx_to_class[i], out[i]) for i in top_3_idx]
    
    return predicted_class, confidence, top_3


def save_checkpoint(cnn, epoch, val_acc, val_loss, out_dir, best_acc=None, is_best=False):
    """Save model checkpoint after each epoch"""
    os.makedirs(out_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(out_dir, f"epoch_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model weights
    if GPU_AVAILABLE:
        import numpy as numpy_lib
        conv_filters = np.asnumpy(cnn.conv.filters)
        softmax_weights = np.asnumpy(cnn.softmax.weights) 
        softmax_biases = np.asnumpy(cnn.softmax.biases)
        numpy_lib.save(os.path.join(checkpoint_dir, "conv_filters.npy"), conv_filters)
        numpy_lib.save(os.path.join(checkpoint_dir, "softmax_weights.npy"), softmax_weights)
        numpy_lib.save(os.path.join(checkpoint_dir, "softmax_biases.npy"), softmax_biases)
    else:
        np.save(os.path.join(checkpoint_dir, "conv_filters.npy"), cnn.conv.filters)
        np.save(os.path.join(checkpoint_dir, "softmax_weights.npy"), cnn.softmax.weights)
        np.save(os.path.join(checkpoint_dir, "softmax_biases.npy"), cnn.softmax.biases)
    
    # Save training metadata
    metadata = {
        'epoch': epoch,
        'val_accuracy': float(val_acc),
        'val_loss': float(val_loss),
        'best_accuracy': float(best_acc) if best_acc else float(val_acc),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_classes': cnn.num_classes,
        'conv_filters': cnn.conv.num_filters
    }
    
    import json
    with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save as "latest" checkpoint for easy resuming
    latest_dir = os.path.join(out_dir, "latest")
    os.makedirs(latest_dir, exist_ok=True)
    
    if GPU_AVAILABLE:
        numpy_lib.save(os.path.join(latest_dir, "conv_filters.npy"), conv_filters)
        numpy_lib.save(os.path.join(latest_dir, "softmax_weights.npy"), softmax_weights)
        numpy_lib.save(os.path.join(latest_dir, "softmax_biases.npy"), softmax_biases)
    else:
        np.save(os.path.join(latest_dir, "conv_filters.npy"), cnn.conv.filters)
        np.save(os.path.join(latest_dir, "softmax_weights.npy"), cnn.softmax.weights)
        np.save(os.path.join(latest_dir, "softmax_biases.npy"), cnn.softmax.biases)
    
    with open(os.path.join(latest_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save best model separately
    if is_best:
        best_dir = os.path.join(out_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        
        if GPU_AVAILABLE:
            numpy_lib.save(os.path.join(best_dir, "conv_filters.npy"), conv_filters)
            numpy_lib.save(os.path.join(best_dir, "softmax_weights.npy"), softmax_weights)
            numpy_lib.save(os.path.join(best_dir, "softmax_biases.npy"), softmax_biases)
        else:
            np.save(os.path.join(best_dir, "conv_filters.npy"), cnn.conv.filters)
            np.save(os.path.join(best_dir, "softmax_weights.npy"), cnn.softmax.weights)
            np.save(os.path.join(best_dir, "softmax_biases.npy"), cnn.softmax.biases)
        
        with open(os.path.join(best_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   🏆 New best model saved (accuracy: {val_acc:.4f})")
    
    return checkpoint_dir

def load_checkpoint(cnn, checkpoint_path):
    """Load model checkpoint to resume training"""
    import json
    
    # Load metadata
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model weights
    conv_filters_path = os.path.join(checkpoint_path, "conv_filters.npy")
    softmax_weights_path = os.path.join(checkpoint_path, "softmax_weights.npy")
    softmax_biases_path = os.path.join(checkpoint_path, "softmax_biases.npy")
    
    if not all(os.path.exists(p) for p in [conv_filters_path, softmax_weights_path, softmax_biases_path]):
        raise FileNotFoundError("One or more model weight files not found")
    
    # Load weights
    conv_filters = np.load(conv_filters_path)
    softmax_weights = np.load(softmax_weights_path)
    softmax_biases = np.load(softmax_biases_path)
    
    # Convert to GPU arrays if using CuPy
    if GPU_AVAILABLE:
        conv_filters = np.asarray(conv_filters)
        softmax_weights = np.asarray(softmax_weights)
        softmax_biases = np.asarray(softmax_biases)
    
    # Set weights
    cnn.conv.filters = conv_filters
    cnn.softmax.weights = softmax_weights
    cnn.softmax.biases = softmax_biases
    
    print(f"✅ Loaded checkpoint from epoch {metadata['epoch']}")
    print(f"   📊 Previous validation accuracy: {metadata['val_accuracy']:.4f}")
    print(f"   📊 Previous validation loss: {metadata['val_loss']:.4f}")
    print(f"   📅 Saved at: {metadata['timestamp']}")
    
    return metadata['epoch'], metadata['val_accuracy'], metadata['val_loss']

def find_latest_checkpoint(out_dir):
    """Find the latest checkpoint to resume from"""
    latest_path = os.path.join(out_dir, "latest")
    if os.path.exists(latest_path) and os.path.exists(os.path.join(latest_path, "metadata.json")):
        return latest_path
    
    # If no latest, find highest numbered epoch
    epoch_dirs = []
    if os.path.exists(out_dir):
        for item in os.listdir(out_dir):
            if item.startswith("epoch_") and item[6:].isdigit():
                epoch_dirs.append((int(item[6:]), os.path.join(out_dir, item)))
    
    if epoch_dirs:
        epoch_dirs.sort(reverse=True)  # Sort by epoch number, descending
        return epoch_dirs[0][1]  # Return path of highest epoch
    
    return None


#Training Loop
def train_loop(cnn, X_train, y_train, X_val, y_val, epochs=3, lr=0.005, print_every=100, batch_size=32, 
               out_dir="models", resume_from=None, start_epoch=1):
    """
    Enhanced training loop with epoch-wise checkpointing and resume capability
    
    Args:
        cnn: CNN model instance
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        epochs: Total number of epochs to train
        lr: Learning rate
        print_every: Print progress every N batches
        batch_size: Batch size for training
        out_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        start_epoch: Epoch number to start from (for resuming)
    """
    n = len(X_train)
    total_batches = (n + batch_size - 1) // batch_size  # Ceiling division
    
    # Initialize tracking variables
    best_val_acc = 0.0
    
    # Load checkpoint if resuming
    if resume_from:
        try:
            prev_epoch, prev_val_acc, prev_val_loss = load_checkpoint(cnn, resume_from)
            best_val_acc = prev_val_acc
            print(f"🔄 Resuming training from epoch {prev_epoch + 1}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("🚀 Starting fresh training...")
            start_epoch = 1
    
    print_training_header()
    print(f"📋 Training Configuration:")
    print(f"   • Samples: {n:,}")
    print(f"   • Epochs: {start_epoch}-{epochs}")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Total Batches per Epoch: {total_batches}")
    print(f"   • Learning Rate: {lr}")
    print(f"   • Device: {'GPU' if GPU_AVAILABLE else 'CPU'}")
    print(f"   • Checkpoint directory: {out_dir}")
    print(f"   • Auto-save after each epoch: ✅")
    
    epoch_start_time = time.time()
    
    for epoch in range(start_epoch, epochs+1):
        print_epoch_header(epoch, epochs)
        epoch_time = time.time()
        
        # Shuffle training data
        perm = np.random.permutation(n)
        X_train = X_train[perm]
        y_train = y_train[perm]

        running_loss = 0.0
        running_acc = 0
        batch_count = 0
        
        # Process in mini-batches for better memory management
        for batch_idx, batch_start in enumerate(range(0, n, batch_size), 1):
            batch_time = time.time()
            batch_end = min(batch_start + batch_size, n)
            batch_x = X_train[batch_start:batch_end]
            batch_y = y_train[batch_start:batch_end]
            
            batch_loss = 0.0
            batch_acc = 0
            
            # Process each sample in the batch
            for sample_idx, (img, lbl) in enumerate(zip(batch_x, batch_y)):
                loss, acc = cnn.train(img, int(lbl), lr)
                batch_loss += loss
                batch_acc += acc
            
            # Update running statistics
            current_batch_size = len(batch_x)
            running_loss += batch_loss
            running_acc += batch_acc
            batch_count += current_batch_size
            
            # Calculate averages
            avg_loss = running_loss / batch_count
            avg_acc = running_acc / batch_count
            
            # Calculate timing
            elapsed_batch_time = time.time() - batch_time
            elapsed_total_time = time.time() - epoch_time
            
            # Estimate remaining time
            if batch_idx > 0:
                avg_batch_time = elapsed_total_time / batch_idx
                eta = avg_batch_time * (total_batches - batch_idx)
            else:
                eta = 0
            
            # Print progress every few batches or at the end
            if batch_idx % max(1, total_batches // 20) == 0 or batch_idx == total_batches:
                print_batch_progress(batch_idx, total_batches, avg_loss, avg_acc, 
                                   elapsed_total_time, eta)
            
            # Print detailed stats at specified intervals
            if batch_idx % print_every == 0 or batch_idx == total_batches:
                print(f"\n   📈 Training Stats: Samples processed: {batch_end:,}/{n:,} | "
                      f"Batch Train Loss: {batch_loss/current_batch_size:.4f} | "
                      f"Batch Train Acc: {batch_acc/current_batch_size:.4f} | "
                      f"Speed: {current_batch_size/elapsed_batch_time:.1f} samples/sec")
                
                # Reset running averages for next interval
                running_loss = 0.0
                running_acc = 0
                batch_count = 0
        
        # Epoch completion
        epoch_duration = time.time() - epoch_time
        print(f"\n✅ Epoch {epoch} completed in {format_time(epoch_duration)}")
        
        # Evaluate on validation set after each epoch
        print("🔍 Evaluating on validation set...")
        val_start_time = time.time()
        val_loss, val_acc, per_class_acc = evaluate_detailed(cnn, X_val, y_val)
        val_duration = time.time() - val_start_time
        
        print(f"📊 Validation Results (took {format_time(val_duration)}):")
        print(f"   • Val Loss: {val_loss:.4f}")
        print(f"   • Val Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
        
        # Save checkpoint after each epoch
        print("💾 Saving checkpoint...")
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        checkpoint_path = save_checkpoint(cnn, epoch, val_acc, val_loss, out_dir, best_val_acc, is_best)
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # Show per-class accuracy for first and last epoch
        if epoch == 1 or epoch == epochs:
            print("📋 Per-class validation accuracy:")
            classes, _, _ = get_class_mapping()
            for i in range(0, len(classes), 10):
                class_group = classes[i:i+10]
                acc_line = "   "
                for cls in class_group:
                    acc_line += f"{cls}: {per_class_acc[cls]:.3f}  "
                print(acc_line)
        
        # Show progress towards completion
        total_elapsed = time.time() - epoch_start_time
        if epoch < epochs:
            avg_epoch_time = total_elapsed / (epoch - start_epoch + 1)
            eta_total = avg_epoch_time * (epochs - epoch)
            print(f"⏰ Total elapsed: {format_time(total_elapsed)} | "
                  f"ETA for completion: {format_time(eta_total)}")
        
        print("=" * 80)
    
    print(f"\n🏆 Training completed! Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc

def evaluate(cnn, X_val, y_val):
    total_loss = 0.0
    total_acc = 0
    n = len(X_val)
    predictions = []
    
    for img, lbl in zip(X_val, y_val):
        out, loss, acc = cnn.forward(img, int(lbl))
        total_loss += loss
        total_acc += acc
        predictions.append(np.argmax(out))
    
    if n == 0:
        return 0, 0
    
    return total_loss / n, total_acc / n

def evaluate_detailed(cnn, X_val, y_val):
    """Evaluate with per-class accuracy and progress tracking"""
    classes, class_to_idx, idx_to_class = get_class_mapping()
    n = len(X_val)
    if n == 0:
        return 0, 0, {}
    
    total_loss = 0.0
    predictions = []
    true_labels = []
    
    print(f"   🔄 Processing {n} validation samples...", end='', flush=True)
    
    # Process in chunks to show progress
    chunk_size = max(1, n // 10)  # Show progress every 10%
    
    for i, (img, lbl) in enumerate(zip(X_val, y_val)):
        out, loss, _ = cnn.forward(img, int(lbl))
        total_loss += loss
        predictions.append(np.argmax(out))
        true_labels.append(int(lbl))
        
        # Show progress dots
        if (i + 1) % chunk_size == 0 or i == n - 1:
            progress = (i + 1) / n * 100
            print(f"\r   🔄 Processing validation samples... {progress:.0f}% ({i+1}/{n})", 
                  end='', flush=True)
    
    print()  # New line after progress
    
    # Calculate overall accuracy
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    overall_acc = correct / n
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for i, cls in enumerate(classes):
        class_true = [t for t in true_labels if t == i]
        class_correct = [p for p, t in zip(predictions, true_labels) if t == i and p == t]
        if len(class_true) > 0:
            per_class_acc[cls] = len(class_correct) / len(class_true)
        else:
            per_class_acc[cls] = 0.0
    
    return total_loss / n, overall_acc, per_class_acc

#main and Saving
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="ASL_Alphabet_Dataset/asl_alphabet_train", help="Path to the ASL alphabet dataset directory")
    parser.add_argument("--img_size", type=int, default=28, help="Image Size to resize to 28")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples per class (for faster training/testing)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--out_dir", type=str, default="models", help="where to save .npy weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (or 'latest' to auto-find)")
    parser.add_argument("--start_epoch", type=int, default=1, help="Epoch to start training from (for manual resume)")

    args = parser.parse_args()

    # Handle resume logic
    resume_from = None
    start_epoch = args.start_epoch
    
    if args.resume:
        if args.resume.lower() == 'latest':
            # Auto-find latest checkpoint
            resume_from = find_latest_checkpoint(args.out_dir)
            if resume_from:
                print(f"🔍 Found latest checkpoint: {resume_from}")
            else:
                print("⚠️  No checkpoints found, starting fresh training")
        else:
            # Use specific checkpoint path
            if os.path.exists(args.resume):
                resume_from = args.resume
                print(f"🔍 Using specified checkpoint: {resume_from}")
            else:
                print(f"❌ Checkpoint not found: {args.resume}")
                print("🚀 Starting fresh training...")

    #load Data (all ASL classes)
    (x_train, y_train), (x_val, y_val) = load_asl_dataset(
        args.data_dir,
        img_size=args.img_size,
        val_split=args.val_split,
        max_samples_per_class=args.max_samples
    )

    cnn = CNN()
    print(f"\n🧠 CNN Model Initialized:")
    print(f"   • Convolutional filters: {cnn.conv.num_filters}")
    print(f"   • Output classes: {cnn.num_classes}")
    print(f"   • Total parameters: ~{cnn.conv.num_filters * 3 * 3 + 13*13*8*cnn.num_classes + cnn.num_classes:,}")

    #train
    print(f"\n🚀 Starting training session...")
    print(f"   • Learning rate: {args.lr}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Print interval: every {100} batches")
    if resume_from:
        print(f"   • Resuming from: {resume_from}")
    
    training_start_time = time.time()
    best_val_acc = train_loop(cnn, x_train, y_train, x_val, y_val, epochs=args.epochs, lr=args.lr, 
                             print_every=100, batch_size=args.batch_size, out_dir=args.out_dir,
                             resume_from=resume_from, start_epoch=start_epoch)
    
    total_training_time = time.time() - training_start_time
    print(f"\n🎉 TRAINING COMPLETED! 🎉")
    print(f"📊 Final Training Summary:")
    print(f"   • Total training time: {format_time(total_training_time)}")
    print(f"   • Average time per epoch: {format_time(total_training_time / args.epochs)}")
    print(f"   • Best validation accuracy: {best_val_acc:.4f}")
    print(f"   • Samples processed: {len(x_train) * args.epochs:,}")
    print(f"   • Training speed: {(len(x_train) * args.epochs) / total_training_time:.1f} samples/sec")
    
    print(f"\n💾 CHECKPOINTS SAVED:")
    print(f"   📁 Latest checkpoint: {args.out_dir}/latest/")
    print(f"   🏆 Best model: {args.out_dir}/best/")
    print(f"   📂 All epochs: {args.out_dir}/epoch_*/")
    
    print(f"\n🔄 RESUME TRAINING:")
    print(f"   To continue training: python train.py --resume latest --epochs {args.epochs + 5}")
    print(f"   To resume from specific epoch: python train.py --resume {args.out_dir}/epoch_N")
    
    print(f"\n💾 LEGACY MODEL SAVE...")
    
    #Saved learn parameters (for backward compatibility)
    os.makedirs(args.out_dir, exist_ok=True)
    
    save_start_time = time.time()
    
    # Convert GPU arrays back to CPU for saving if needed
    if GPU_AVAILABLE:
        import numpy as numpy_lib  # Import regular numpy for saving
        conv_filters = np.asnumpy(cnn.conv.filters)
        softmax_weights = np.asnumpy(cnn.softmax.weights)
        softmax_biases = np.asnumpy(cnn.softmax.biases)
        numpy_lib.save(os.path.join(args.out_dir, "conv_filters.npy"), conv_filters)
        numpy_lib.save(os.path.join(args.out_dir, "softmax_weights.npy"), softmax_weights)
        numpy_lib.save(os.path.join(args.out_dir, "softmax_biases.npy"), softmax_biases)
    else:
        np.save(os.path.join(args.out_dir, "conv_filters.npy"), cnn.conv.filters)
        np.save(os.path.join(args.out_dir, "softmax_weights.npy"), cnn.softmax.weights)
        np.save(os.path.join(args.out_dir, "softmax_biases.npy"), cnn.softmax.biases)

    save_time = time.time() - save_start_time
    print(f"✅ Legacy model weights saved to '{args.out_dir}' (took {format_time(save_time)})")
    
    # Test on a few sample images if they exist
    test_dir = "ASL_Alphabet_Dataset/asl_alphabet_test"
    if os.path.exists(test_dir):
        print(f"\n🧪 TESTING ON SAMPLE IMAGES...")
        print(f"📂 Test directory: {test_dir}")
        
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if test_files:
            print(f"🔍 Found {len(test_files)} test images, testing first 5...")
            
            for i, test_file in enumerate(test_files[:5], 1):
                test_path = os.path.join(test_dir, test_file)
                print(f"\n📸 Test {i}/5: {test_file}")
                
                pred_class, confidence, top_3 = predict_image(cnn, test_path, args.img_size)
                
                if pred_class:
                    print(f"   🎯 Prediction: {pred_class} (confidence: {confidence:.3f})")
                    print(f"   🏆 Top 3: {[(cls, f'{conf:.3f}') for cls, conf in top_3]}")
                else:
                    print(f"   ❌ Error loading image")
        else:
            print("⚠️  No test images found in test directory")
    else:
        print("⚠️  Test directory not found")
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎊 TRAINING SESSION COMPLETE! 🎊".center(80))
    print(f"{'='*80}")
    print(f"📈 Model Performance Summary:")
    print(f"   • Architecture: CNN with {cnn.conv.num_filters} conv filters")
    print(f"   • Classes supported: {len(get_class_mapping()[0])} ASL signs")
    print(f"   • Training samples: {len(x_train):,}")
    print(f"   • Validation samples: {len(x_val):,}")
    print(f"   • Total training time: {format_time(total_training_time)}")
    print(f"   • Model weights saved to: {args.out_dir}")
    print(f"\n🚀 Ready for inference! Use the saved model for ASL recognition.")
    print(f"📝 Supported classes: {', '.join(get_class_mapping()[0])}")
    print("="*80)
