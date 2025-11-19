import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import torchvision.models.video as models
import torch.nn as nn
from dataset import BinaryVideoDataset

# ==========================================
#               CONFIGURATION
# ==========================================
# Path to the NEW, unseen dataset
TEST_DATA_ROOT = "/media/oeste/BeaGL2/CMORE/Sequence_Classifier_Data/Subj_02_RightHand_HELD_OUT" 

# Path to your trained model checkpoint
MODEL_PATH = "best_sequence_classifier.pt"

# Where to create the visualization folders
OUT_PATH = "evaluation_results"

# Model settings (Must match training)
NUM_FRAMES = 10
BATCH_SIZE = 8 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

def get_model():
    """
    Re-instantiates the model architecture to load weights into.
    """
    # Load R3D-18 architecture
    # weights=None because we are overwriting them with our own checkpoint anyway
    model = models.r3d_18(weights=None) 
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    return model

def setup_directories(base_path):
    """
    Creates the output directory structure.
    Structure:
        base/
        ├── correct/
        │   ├── predicted_positive/
        │   └── predicted_negative/
        └── wrong/
            ├── predicted_positive/
            └── predicted_negative/
    """
    if os.path.exists(base_path):
        print(f"Warning: Output path '{base_path}' exists. Merging results...")
    
    paths = [
        os.path.join(base_path, "correct", "predicted_positive"),
        os.path.join(base_path, "correct", "predicted_negative"),
        os.path.join(base_path, "wrong", "predicted_positive"),
        os.path.join(base_path, "wrong", "predicted_negative"),
    ]
    
    for p in paths:
        os.makedirs(p, exist_ok=True)

def evaluate():
    print(f"--- Starting Evaluation on {DEVICE} ---")
    
    # 1. Load Dataset (train=False for deterministic resizing)
    try:
        dataset = BinaryVideoDataset(root_dir=TEST_DATA_ROOT, num_frames=NUM_FRAMES, train=False)
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        return

    # IMPORTANT: shuffle=False ensures we can map predictions back to file paths by index
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Loaded {len(dataset)} clips from {TEST_DATA_ROOT}")
    
    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 3. Setup Output Dirs
    setup_directories(OUT_PATH)
    
    # 4. Inference Loop
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running Inference and Sorting Clips...")
    
    # We track the global index to find the original file path in dataset.files
    global_idx = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE) # (Batch_Size)
            
            outputs = model(inputs) # (Batch_Size, 1)
            outputs = outputs.squeeze(1) # (Batch_Size)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # --- Per-Clip Processing for sorting ---
            batch_size = inputs.size(0)
            for i in range(batch_size):
                file_path = dataset.files[global_idx]
                true_label = labels[i].item()
                pred_label = preds[i].item()
                confidence = probs[i].item()
                
                # Determine destination
                is_correct = (true_label == pred_label)
                pred_class_str = "predicted_positive" if pred_label == 1 else "predicted_negative"
                correctness_str = "correct" if is_correct else "wrong"
                
                # Define annotated filename
                # Format: [CONFIDENCE]_ORIGINAL_NAME.mp4
                filename = os.path.basename(file_path)
                new_filename = f"[Conf_{confidence:.4f}]_{filename}"
                
                dest_dir = os.path.join(OUT_PATH, correctness_str, pred_class_str)
                dest_path = os.path.join(dest_dir, new_filename)
                
                # Copy the file
                try:
                    shutil.copy2(file_path, dest_path)
                except Exception as e:
                    print(f"\nError copying {filename}: {e}")

                global_idx += 1
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 5. Metrics & Reporting
    print("\n" + "="*30)
    print("       EVALUATION REPORT")
    print("="*30)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")
    
    # Detailed Report (Precision, Recall, F1)
    print("\nClassification Report:")
    target_names = ['Negative (0)', 'Positive (1)']
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Plotting Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Neg', 'Pred Pos'], 
                yticklabels=['Actual Neg', 'Actual Pos'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    save_plot_path = os.path.join(OUT_PATH, 'confusion_matrix.png')
    plt.savefig(save_plot_path)
    print(f"Confusion matrix saved to: {save_plot_path}")
    print(f"Sorted video clips saved to: {OUT_PATH}")

if __name__ == "__main__":
    evaluate()