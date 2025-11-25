import os
import glob
import torch
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

# R3D Standard Input
INPUT_SIZE = 112

# Kinetics-400 Normalization Stats
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

def load_video_cv2(video_path):
    """
    Loads a video using OpenCV.
    Returns: Tensor of shape (Time, Height, Width, Channels)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV reads in BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
    
    if len(frames) == 0:
        return torch.empty(0)
    
    # Convert list of numpy arrays to a single tensor
    return torch.from_numpy(np.stack(frames))

class BinaryVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames, train=True):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.train = train 
        
        if not os.path.exists(root_dir):
            raise ValueError(f"The directory {root_dir} does not exist.")

        self.pos_files = []
        self.neg_files = []

        print(f"Scanning {root_dir}...")
        
        # Walk through directories to find positive/negative folders
        for dirpath, dirnames, filenames in os.walk(root_dir):
            folder_name = os.path.basename(dirpath).lower()
            
            if "positive" in folder_name:
                for f in filenames:
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        self.pos_files.append(os.path.join(dirpath, f))
            
            elif "negative" in folder_name:
                for f in filenames:
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        self.neg_files.append(os.path.join(dirpath, f))

        self.files = self.pos_files + self.neg_files
        self.labels = [1] * len(self.pos_files) + [0] * len(self.neg_files)
        
        print(f"Found {len(self.pos_files)} positive and {len(self.neg_files)} negative clips.")
        
        if len(self.files) == 0:
            raise ValueError("No video files found. Please check your folder naming.")

        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
        self.normalize = T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD)

    def __len__(self):
        return len(self.files)
    
    def get_class_weights(self):
        num_pos = len(self.pos_files)
        num_neg = len(self.neg_files)
        if num_pos == 0: return torch.tensor(1.0)
        return torch.tensor(num_neg / num_pos)

    def __getitem__(self, idx):
        video_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            # Use OpenCV helper
            frames = load_video_cv2(video_path)
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return torch.zeros((3, self.num_frames, INPUT_SIZE, INPUT_SIZE)), torch.tensor(label, dtype=torch.float)

        total_frames = frames.shape[0]
        
        if total_frames == 0:
            return torch.zeros((3, self.num_frames, INPUT_SIZE, INPUT_SIZE)), torch.tensor(label, dtype=torch.float)

        # --- 1. TEMPORAL SAMPLING ---
        if total_frames > self.num_frames:
            print("MORE THAN")
            if self.train:
                max_start = total_frames - self.num_frames
                start = random.randint(0, max_start)
            else:
                start = (total_frames - self.num_frames) // 2
            frames = frames[start : start + self.num_frames]
        elif total_frames < self.num_frames:
            print("LESS THAN")
            repeats = (self.num_frames // total_frames) + 1
            frames = frames.repeat(repeats, 1, 1, 1)[:self.num_frames]

        # OpenCV loads as uint8 (0-255), convert to float (0.0-1.0)
        frames = frames.float() / 255.0
        
        # --- 2. SPATIAL AUGMENTATION ---
        if self.train:
            rotation_angle = random.uniform(-10, 10)
            do_flip = random.random() > 0.5
            augmented_frames = []
            
            # OpenCV tensor is (Time, Height, Width, Channels) -> (T, H, W, C)
            # Convert to (T, C, H, W) for transforms
            frames = frames.permute(0, 3, 1, 2)
            
            for frame in frames:
                frame = F.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                frame = F.rotate(frame, rotation_angle)
                if do_flip: 
                    frame = F.hflip(frame)
                frame = self.color_jitter(frame)
                frame = self.normalize(frame)
                augmented_frames.append(frame)
        else:
            augmented_frames = []
            frames = frames.permute(0, 3, 1, 2)
            for frame in frames:
                frame = F.resize(frame, (INPUT_SIZE, INPUT_SIZE)) 
                frame = self.normalize(frame)
                augmented_frames.append(frame)
        
        frames = torch.stack(augmented_frames)
        
        # Final Output Requirement: (Channels, Time, Height, Width)
        frames = frames.permute(1, 0, 2, 3) 

        return frames, torch.tensor(label, dtype=torch.float)