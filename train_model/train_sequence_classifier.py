import os
import warnings
# Suppress torchvision video deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models.video as models
from dataset import BinaryVideoDataset

# =============================================================================
#                               CONFIGURATION
# =============================================================================
# Path to the TRAINING set (e.g., Subject 1, Subject 3, Subject 4)
TRAIN_ROOT = "/media/oeste/BeaGL2/CMORE/Sequence_Classifier/train"

# Path to the HELD-OUT VALIDATION set (e.g., Subject 2 only)
# The model will NEVER see these clips during backpropagation.
VAL_ROOT = "/media/oeste/BeaGL2/CMORE/Sequence_Classifier/Subj_03_LeftHand"

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 150
NUM_FRAMES = 10
# =============================================================================

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

def get_model():
    """
    Loads R3D_18 (ResNet3D 18 layers) pre-trained on Kinetics-400.
    Modifies the output layer for binary classification.
    """
    model = models.r3d_18(weights=models.R3D_18_Weights.KINETICS400_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    return model

def plot_history(train_acc, val_acc, train_loss, val_loss):
    """
    Plots training and validation accuracy and loss, then saves to a file.
    """
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b-', label='Training Acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation Acc (Held Out)')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss (Held Out)')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    print("Plot saved as 'training_results.png'")

def train():
    print("Initializing Datasets...")
    
    # 1. Setup Datasets (Directly from separate folders)
    # Train Set: Uses Data Augmentation (Rotation, Flip, Jitter)
    train_dataset = BinaryVideoDataset(root_dir=TRAIN_ROOT, num_frames=NUM_FRAMES, train=True)
    
    # Val Set: Deterministic (No Augmentation) - strictly for evaluation
    val_dataset = BinaryVideoDataset(root_dir=VAL_ROOT, num_frames=NUM_FRAMES, train=False)
    
    print(f"Training Data:   {len(train_dataset)} clips from {TRAIN_ROOT}")
    print(f"Validation Data: {len(val_dataset)} clips from {VAL_ROOT}")

    # Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=4, pin_memory=True, drop_last=False)
    
    # 2. Setup Loss with Class Weights
    # We calculate weights based on the TRAINING set distribution only
    pos_weight = train_dataset.get_class_weights().to(device)
    print(f"Using positive class weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 3. Setup Model and Optimizer
    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # We track BEST LOSS for checkpointing (safest for imbalance)
    best_val_loss = float('inf') 

    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        train_corrects = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            train_corrects += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            train_bar.set_postfix(loss=loss.item())
                
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = train_corrects / train_total
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f"Train | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        val_bar = tqdm(val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_corrects += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_corrects / val_total
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f"Val   | Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.4f}")
        
        # --- CHECKPOINTING (Lowest Loss) ---
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), "best_sequence_classifier.pt")
            print(f"--> New Best Model Saved! (Low Loss: {best_val_loss:.4f})")

    # Save final model
    torch.save(model.state_dict(), "final_sequence_classifier.pt")
    print("\nTraining complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    print("Generating plots...")
    plot_history(history['train_acc'], history['val_acc'], 
                 history['train_loss'], history['val_loss'])

if __name__ == "__main__":
    # Basic check to see if paths exist
    if os.path.exists(TRAIN_ROOT) and os.path.exists(VAL_ROOT):
        train()
    else:
        print("Error: Please check your TRAIN_ROOT and VAL_ROOT paths.")
        if not os.path.exists(TRAIN_ROOT): print(f"Missing: {TRAIN_ROOT}")
        if not os.path.exists(VAL_ROOT): print(f"Missing: {VAL_ROOT}")