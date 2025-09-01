#!/usr/bin/env python3
"""
Test FPS timing functionality for camera scripts
"""
import time
import argparse

def test_fps_timing(target_fps, duration=5):
    """Test FPS timing logic"""
    print(f"Testing FPS timing for {target_fps} FPS over {duration} seconds...")
    
    frame_time = 1.0 / target_fps
    start_time = time.time()
    last_frame_time = start_time
    frame_count = 0
    
    while time.time() - start_time < duration:
        # Simulate some processing time
        time.sleep(0.01)  # 10ms processing simulation
        
        # Frame rate limiting (same logic as in camera scripts)
        current_time = time.time()
        elapsed_time = current_time - last_frame_time
        sleep_time = frame_time - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_frame_time = time.time()
        
        frame_count += 1
    
    total_time = time.time() - start_time
    actual_fps = frame_count / total_time
    
    print(f"Target FPS: {target_fps}")
    print(f"Actual FPS: {actual_fps:.2f}")
    print(f"Frame time target: {frame_time*1000:.1f}ms")
    print(f"Actual frame time: {total_time/frame_count*1000:.1f}ms")
    print(f"FPS accuracy: {(actual_fps/target_fps)*100:.1f}%")
    
    return actual_fps

def main():
    parser = argparse.ArgumentParser(description="Test FPS timing functionality")
    parser.add_argument("--target-fps", type=int, default=15,
                        help="Target FPS to test (default: 15)")
    parser.add_argument("--duration", type=int, default=3,
                        help="Test duration in seconds (default: 3)")
    
    args = parser.parse_args()
    
    print("🎯 FPS Timing Test")
    print("=" * 30)
    
    # Test different FPS values
    fps_values = [10, 15, 20, 30] if args.target_fps == 15 else [args.target_fps]
    
    for fps in fps_values:
        test_fps_timing(fps, args.duration)
        print("-" * 30)
    
    print("✅ FPS timing test completed!")

if __name__ == "__main__":
    main()
