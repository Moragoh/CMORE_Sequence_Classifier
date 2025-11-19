import os
import shutil
import random
import argparse
import sys

def create_test_split(target_dir, ratio=0.1):
    """
    Moves a percentage of files from positive_clips/negative_clips to 
    a new test_clips directory structure within target_dir.
    """
    
    # Define source folders
    class_folders = ["positive_clips", "negative_clips"]
    
    # Define destination root
    test_root = os.path.join(target_dir, "test_clips")
    
    print(f"--- Processing Directory: {target_dir} ---")
    print(f"Target Split Ratio: {ratio * 100}%")

    for cls in class_folders:
        src_path = os.path.join(target_dir, cls)
        
        # Check if source exists
        if not os.path.exists(src_path):
            print(f"[WARN] Source folder not found: {src_path}")
            continue
            
        # Create corresponding destination folder
        # e.g., target_dir/test_clips/positive_clips
        dest_path = os.path.join(test_root, cls)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
            
        # List all video files
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        files = [f for f in os.listdir(src_path) if f.lower().endswith(valid_extensions)]
        
        total_files = len(files)
        num_to_move = int(total_files * ratio)
        
        if num_to_move == 0:
            print(f"[INFO] {cls}: Not enough files to split (Total: {total_files}). Skipping.")
            continue

        # Randomly select files
        files_to_move = random.sample(files, num_to_move)
        
        print(f"[MOVE] {cls}: Moving {num_to_move} of {total_files} files to test set...")
        
        # Move files
        for f in files_to_move:
            src_file = os.path.join(src_path, f)
            dest_file = os.path.join(dest_path, f)
            try:
                shutil.move(src_file, dest_file)
            except Exception as e:
                print(f"Error moving {f}: {e}")

    print(f"--- Done. Test set created at {test_root} ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into Train (remaining) and Test (moved).")
    parser.add_argument("--dir", required=True, help="Path to the directory containing positive_clips and negative_clips")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio of files to move to test set (default: 0.1 for 10%)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist.")
        sys.exit(1)
        
    create_test_split(args.dir, args.ratio)