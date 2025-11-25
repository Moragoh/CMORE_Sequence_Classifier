import os
import torch
import shutil
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend to prevent "Qt/xcb" errors on headless servers
matplotlib.use('Agg') 
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
TEST_DATA_ROOT = "/media/oeste/BeaGL2/CMORE/Sequence_Classifier/Sequence_Data_30Frames/Subj_05_LeftHand"

# Path to your trained model checkpoint
MODEL_PATH = "/home/oeste/Desktop/Jun/KingLabs/StrokeResearch/CMORE_Sequence_Classifier/train_model/models/best_sequence_classifier_30frames.pt"
# Where to create the visualization folders
OUT_PATH = "/media/oeste/BeaGL2/CMORE/Sequence_Classifier/model_test_results"

# Model settings (Must match training)
NUM_FRAMES = 30
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

def get_model():
    """
    Re-instantiates the model architecture to load weights into.
    """
    # Load R3D-18 architecture
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

def plot_classification_report(y_true, y_pred, target_names, save_path):
    """
    Generates a classification report, converts it to a dataframe,
    and plots it as a heatmap.
    """
    # Get report as a dictionary
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Convert to DataFrame and transpose
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Separate 'support' (count) from metrics for cleaner color scaling
    # We plot metrics (0-1 range) in the heatmap
    metrics_df = report_df.drop(columns=['support'])
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df, annot=True, cmap='viridis', fmt='.4f', vmin=0.0, vmax=1.0)
    plt.title('Classification Report Metrics')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    print(f"Classification report saved to: {save_path}")

def evaluate():
    print(f"--- Starting Evaluation on {DEVICE} ---")
    
    # 1. Load Dataset
    try:
        dataset = BinaryVideoDataset(root_dir=TEST_DATA_ROOT, num_frames=NUM_FRAMES, train=False)
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Loaded {len(dataset)} clips from {TEST_DATA_ROOT}")
    
    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    model = get_model()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Error loading state dict. Ensure architecture matches training. \n{e}")
        return
        
    model.to(DEVICE)
    model.eval()
    
    # 3. Setup Output Dirs
    setup_directories(OUT_PATH)
    
    # 4. Inference Loop
    all_preds = []
    all_labels = []
    
    # Track global index to find file paths
    global_idx = 0
    
    print("Running Inference and Sorting Clips...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # --- Sort Clips ---
            batch_size = inputs.size(0)
            for i in range(batch_size):
                file_path = dataset.files[global_idx]
                true_label = labels[i].item()
                pred_label = preds[i].item()
                confidence = probs[i].item()
                
                is_correct = (true_label == pred_label)
                pred_class_str = "predicted_positive" if pred_label == 1 else "predicted_negative"
                correctness_str = "correct" if is_correct else "wrong"
                
                filename = os.path.basename(file_path)
                new_filename = f"[Conf_{confidence:.4f}]_{filename}"
                
                dest_dir = os.path.join(OUT_PATH, correctness_str, pred_class_str)
                dest_path = os.path.join(dest_dir, new_filename)
                
                try:
                    shutil.copy2(file_path, dest_path)
                except Exception as e:
                    print(f"\nError copying {filename}: {e}")

                global_idx += 1
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Metrics Visualization
    print("\n" + "="*30)
    print("       GENERATING PLOTS")
    print("="*30)
    
    # A. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Neg', 'Pred Pos'], 
                yticklabels=['Actual Neg', 'Actual Pos'])
    plt.title(f'Confusion Matrix\nTN:{tn} FP:{fp} FN:{fn} TP:{tp}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_save_path = os.path.join(OUT_PATH, 'confusion_matrix.png')
    plt.savefig(cm_save_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_save_path}")
    
    # B. Classification Report Heatmap
    cr_save_path = os.path.join(OUT_PATH, 'classification_report.png')
    plot_classification_report(all_labels, all_preds, 
                             target_names=['Negative (0)', 'Positive (1)'], 
                             save_path=cr_save_path)

    print(f"Evaluation complete. Results saved to: {OUT_PATH}")

if __name__ == "__main__":
    evaluate()