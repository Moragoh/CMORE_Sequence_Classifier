import cv2
import pandas as pd
import numpy as np
import os
import argparse
from keypoint_detector import BoxDetector

# --- CONFIGURATION ---
DEFAULT_CLIP_LEN = 30  # Number of frames per clip
DEFAULT_STRIDE = 1    # Default sliding window stride
MODEL_PATH = "keypoint_detector.pt" 
DEFAULT_OUT_PATH = "/media/oeste/BeaGL2/CMORE/Sequence_Classifier/Sequence_Data_30Frames" 
SHRINK_FACTOR = 0.9    # Factor from sequence_annotator.py

def get_negative_intervals(positive_intervals, total_frames):
    """
    Calculates the gaps between positive intervals (Negative samples).
    """
    negatives = []
    current_frame = 0
    positive_intervals.sort(key=lambda x: x[0])
    
    for start, end in positive_intervals:
        if start > current_frame:
            negatives.append([current_frame, start - 1])
        current_frame = end + 1
        
    if current_frame < total_frames:
        negatives.append([current_frame, total_frames - 1])
        
    return negatives

def pad_frames(frames, target_len):
    """
    Loop-pads a list of frames to reach target_len.
    """
    current_len = len(frames)
    if current_len >= target_len:
        return frames[:target_len]
    
    needed = target_len - current_len
    repeats = (needed // current_len) + 1
    padded = list(frames) # Make a copy
    for _ in range(repeats):
        padded.extend(frames)
        
    return padded[:target_len]

def shrink_centroid(points, factor: float):
    """
    Shrinks the polygon defined by points towards its centroid.
    """
    centroid = np.mean(points, axis=0)
    vectors = points - centroid
    new_points = centroid + vectors * factor
    return new_points.astype(np.int32)

def get_contour_points(keypoints, handedness):
    """
    Extracts specific keypoints to define the contour of the object 
    based on handedness logic.
    """
    if keypoints is None: return None
    try:
        # Linear Interpolation for Back Wall logic
        denom = (keypoints['Back top left'][0] - keypoints['Back top right'][0])
        if denom == 0: denom = 0.001 
        
        m = (keypoints['Back top left'][1] - keypoints['Back top right'][1]) / denom
        c = keypoints['Back top left'][1] - m * keypoints['Back top left'][0]
        back_top_middle_y = m * keypoints['Back divider top'][0] + c
        
        # Select points based on Handedness
        if handedness == 'Left':
            points = np.array([
                keypoints['Back top left'],
                [min(keypoints['Back divider top'][0], keypoints['Front divider top'][0]), back_top_middle_y],
                keypoints['Front top middle'],
                keypoints['Front top left']
            ], dtype=np.int32)
        else: # Right
            points = np.array([
                [max(keypoints['Back divider top'][0], keypoints['Front divider top'][0]), back_top_middle_y],
                keypoints['Back top right'],
                keypoints['Front top right'],
                keypoints['Front top middle']
            ], dtype=np.int32)

        return shrink_centroid(points, SHRINK_FACTOR)
    except Exception as e:
        return None

def get_crop_coordinates(contour_points, img_w, img_h):
    """
    Calculates the bounding box (x, y, w, h) from contour points.
    Adds a small margin for safety.
    """
    x, y, w, h = cv2.boundingRect(contour_points)
    
    # Optional: Add margin
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_w - x, w + (margin * 2))
    h = min(img_h - y, h + (margin * 2))
    
    return x, y, w, h

def process_intervals(video_path, intervals, output_dir, file_prefix, clip_len, stride, detector, handedness):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip_count = 0

    print(f"Processing into {output_dir}...")

    for start_f, end_f in intervals:
        start_f = max(0, start_f)
        end_f = min(total_frames_video - 1, end_f)
        duration_frames = end_f - start_f + 1
        
        if duration_frames <= 0:
            continue

        # Define sliding windows
        if duration_frames < clip_len:
            window_starts = [start_f]
        else:
            window_starts = range(start_f, end_f - clip_len + 1, stride)

        for ws in window_starts:
            # 1. Extract Frames
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, ws)
            
            frames_to_read = clip_len if duration_frames >= clip_len else duration_frames

            for _ in range(frames_to_read):
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)

            if not frames: continue

            # 2. Pad if necessary
            if len(frames) < clip_len:
                frames = pad_frames(frames, clip_len)

            # 3. Detect Crop Region (Run on middle frame)
            mid_idx = len(frames) // 2
            ref_frame = frames[mid_idx]
            
            success, result = detector.detect(ref_frame)

            if not success:
                continue

            # 4. Calculate Crop & Contour
            contour_points = get_contour_points(result, handedness)
            
            if contour_points is None or len(contour_points) == 0:
                continue

            # Get the bounding rect of the contour
            x, y, w, h = get_crop_coordinates(contour_points, width_orig, height_orig)

            # Ensure valid crop dimensions
            if w <= 0 or h <= 0:
                continue

            # --- PREPARE CONTOUR MASK ---
            # Shift contour points to be relative to the top-left of the crop (0,0)
            # This effectively moves the polygon into the coordinate space of the cropped image.
            mask_contour = contour_points - np.array([x, y])

            # 5. Save Clip
            save_name = f"{file_prefix}_{start_f}_{ws}.mp4"
            save_path = os.path.join(output_dir, save_name)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

            for f in frames:
                # A. Crop the frame first
                cropped_frame = f[y:y+h, x:x+w]
                
                # Resize check (defensive coding if slice hits edge)
                if cropped_frame.shape[:2] != (h, w):
                    cropped_frame = cv2.resize(cropped_frame, (w, h))
                
                # B. Create Black Mask
                # Initialize black image of same size as crop
                mask = np.zeros_like(cropped_frame)
                
                # C. Draw White Polygon on Mask
                # fillPoly expects a list of arrays
                cv2.fillPoly(mask, [mask_contour], (255, 255, 255))
                
                # D. Apply Mask (Bitwise AND)
                # Pixels inside white polygon are kept; others become black.
                masked_frame = cv2.bitwise_and(cropped_frame, mask)
                
                out.write(masked_frame)
            
            out.release()
            clip_count += 1
            print(f"Saved {save_name}", end='\r')

    cap.release()
    print(f"\nFinished. Total clips saved to {output_dir}: {clip_count}")

def main():
    parser = argparse.ArgumentParser(description="Generate cropped positive/negative clips.")
    parser.add_argument("--video", type=str, required=True, help="Path to the source video")
    parser.add_argument("--csv", type=str, required=True, help="Path to the annotations CSV")
    parser.add_argument("--clip_len", type=int, default=DEFAULT_CLIP_LEN, help="Length of clips in frames")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Sliding window stride")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to YOLO model")
    parser.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH, help="Root directory to save positive/negative folders")
    
    # --- Handedness Flag Group (Required) ---
    hand_group = parser.add_mutually_exclusive_group(required=True)
    hand_group.add_argument("--R", "--right", action="store_true", help="Set handedness to Right")
    hand_group.add_argument("--L", "--left", action="store_true", help="Set handedness to Left")
    
    args = parser.parse_args()

    if args.stride <= 0:
        print(f"Warning: Stride cannot be 0. Setting stride to 1.")
        args.stride = 1

    # Determine Handedness
    handedness = "Right" if args.R else "Left"

    # --- LOGIC FOR VIDEO-SPECIFIC FOLDER ---
    video_basename = os.path.basename(args.video)
    video_name_no_ext = os.path.splitext(video_basename)[0]

    video_out_dir = os.path.join(args.out_path, video_name_no_ext)
    pos_dir = os.path.join(video_out_dir, "positive_clips")
    neg_dir = os.path.join(video_out_dir, "negative_clips")

    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)
    if not os.path.exists(neg_dir):
        os.makedirs(neg_dir)

    print("Loading Keypoint Detector...")
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    detector = BoxDetector(args.model)

    df = pd.read_csv(args.csv)
    positive_intervals = df[['Start Frame', 'End Frame']].values.tolist()
    
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    negative_intervals = get_negative_intervals(positive_intervals, total_frames)

    print(f"Found {len(positive_intervals)} positive intervals.")
    print(f"Found {len(negative_intervals)} negative intervals.")
    print(f"Processing with Handedness: {handedness}")
    print(f"Saving output to: {video_out_dir}")

    # Process Positives
    process_intervals(args.video, positive_intervals, pos_dir, "positive_clips", args.clip_len, args.stride, detector, handedness)
    
    # Process Negatives
    process_intervals(args.video, negative_intervals, neg_dir, "negative_clips", args.clip_len, args.stride, detector, handedness)

if __name__ == "__main__":
    main()