import os
import warnings
# Suppress torchvision video deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg') # <--- THIS FIXES THE CRASH
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models.video as models
from dataset import BinaryVideoDataset

# =============================================================================
#                               CONFIGURATION
# =============================================================================
# Path to the TRAINING set (e.g., Subject 1, Subject 3, Subject 4)
TRAIN_ROOT = "/media/oeste/BeaGL2/CMORE/Sequence_Classifier/CurrTrain/Train"

# Path to the HELD-OUT VALIDATION set (e.g., Subject 2 only)
VAL_ROOT = "/media/oeste/BeaGL2/CMORE/Sequence_Classifier/CurrTrain/Val"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NUM_FRAMES = 30

# --- NEW HYPERPARAMETERS ---
WEIGHT_DECAY = 1e-4           # Regularization strength
LR_PATIENCE = 10              # Epochs to wait before lowering LR
LR_FACTOR = 0.1               # Factor to reduce LR by (LR * 0.1)
EARLY_STOPPING_PATIENCE = 15  # Epochs to wait before stopping completely
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
    
    # 1. Setup Datasets
    train_dataset = BinaryVideoDataset(root_dir=TRAIN_ROOT, num_frames=NUM_FRAMES, train=True)
    val_dataset = BinaryVideoDataset(root_dir=VAL_ROOT, num_frames=NUM_FRAMES, train=False)
    
    print(f"Training Data:   {len(train_dataset)} clips from {TRAIN_ROOT}")
    print(f"Validation Data: {len(val_dataset)} clips from {VAL_ROOT}")

    # Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=4, pin_memory=True, drop_last=False)
    
    # 2. Setup Loss with Class Weights
    pos_weight = train_dataset.get_class_weights().to(device)
    print(f"Using positive class weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 3. Setup Model, Optimizer, Scheduler
    model = get_model().to(device)
    
    # --- ADDED WEIGHT DECAY ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # --- ADDED SCHEDULER ---
    # Reduces LR when val_loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE
    )
    
    best_val_loss = float('inf') 
    
    # --- ADDED EARLY STOPPING COUNTER ---
    early_stopping_counter = 0

    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | Current LR: {current_lr:.2e}")
        
        
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
        
        # --- STEP SCHEDULER ---
        # Updates Learning Rate based on Val Loss
        scheduler.step(val_epoch_loss)
        
        # --- CHECKPOINTING & EARLY STOPPING ---
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), "./models/best_sequence_classifier_30frames.pt")
            print(f"--> New Best Model Saved! (Low Loss: {best_val_loss:.4f})")
            # Reset counter since we found a better model
            early_stopping_counter = 0
        else:
            # Increment counter if no improvement
            early_stopping_counter += 1
            print(f"No improvement. Early Stopping Counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")
            
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print("\n!!! EARLY STOPPING TRIGGERED !!!")
                print(f"Validation loss hasn't improved for {EARLY_STOPPING_PATIENCE} epochs.")
                break

    # Save final model
    torch.save(model.state_dict(), "./models/final_sequence_classifier_30frames.pt")
    print("\nTraining complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    print("Generating plots...")
    plot_history(history['train_acc'], history['val_acc'], 
                 history['train_loss'], history['val_loss'])

if __name__ == "__main__":
    if os.path.exists(TRAIN_ROOT) and os.path.exists(VAL_ROOT):
        train()
    else:
        print("Error: Please check your TRAIN_ROOT and VAL_ROOT paths.")
        if not os.path.exists(TRAIN_ROOT): print(f"Missing: {TRAIN_ROOT}")
        if not os.path.exists(VAL_ROOT): print(f"Missing: {VAL_ROOT}")