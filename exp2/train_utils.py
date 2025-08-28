import time
import torch
import random
import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

# Set random seeds for deterministic behavior
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def train_model(model, train_loader, val_loader, epochs, lr, device, pos_weight, thr=0.7):
    print(f"model to device {device}")
    model = model.to(device)
    print("optimizer and criterion defined")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    best_precision = 0.0
    best_model_state = None
    best_epoch = 0
    for epoch in range(epochs):
        # print("model in train mode")
        model.train()
        total_train_loss = 0.0
        # print("before train_loader iteration")
        start = time.time()
        for idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Check for NaNs in X_batch and y_batch
            if torch.isnan(X_batch).any():
                print(f"NaN detected in X_batch at batch {idx}")
                nan_indices = torch.isnan(X_batch).nonzero(as_tuple=False)  # Shape: [num_nans, 3]
                for b, s, f in nan_indices:
                    print(f"  → Sample {b.item()}, Sequence Step {s.item()}, Feature {f.item()}: NaN")
            if torch.isnan(y_batch).any():
                print(f"NaN detected in y_batch at batch {idx}")
            
            # print("X and y batches to device done")
            optimizer.zero_grad()
            logits = model(X_batch)
            
            if torch.isnan(logits).any():
                print(f"NaN detected in logits at batch {idx}")
        
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            if (idx - 1) % 2000 == 0:
                delta = time.time() - start
                tmp_train_loss = total_train_loss / (idx)
                print(f"Batch {idx}/{len(train_loader)} - Loss: {tmp_train_loss:.4f}")
                print(f"time of 2000 batches: {delta:.3f}", )
                start = time.time()
            # if idx >= 100:
            #     break


        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        y_true, y_pred, y_probs = [], [], []
        avg_val_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    total_val_loss += loss.item()

                    probs = torch.sigmoid(logits)
                    preds = (probs >= thr).int()
                    y_true.extend(y_batch.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_probs.extend(probs.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
        y_true_binary = (np.array(y_true) >= 0.5).astype(int)
        precision = precision_score(y_true_binary, y_pred, pos_label=1, zero_division=0.0)
        recall = recall_score(y_true_binary, y_pred, pos_label=1, zero_division=0.0)
        f1 = f1_score(y_true_binary, y_pred, pos_label=1, zero_division=0.0)

        # Save the best model state
        if precision > best_precision:
            best_epoch = epoch
            best_precision = precision
            best_model_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1}/{epochs} -> ", end="")
        print(f"train loss: {avg_train_loss:.4f}", end="; ")
        print(f"val loss: {avg_val_loss:.4f}", end="; ")
        print(f"p@1: {precision:.4f}", end="; ")
        print(f"r@1: {recall:.4f}", end="; ")
        print(f"f1@1: {f1:.4f}")
        print("-" * 40)

    cleanup_memory(device)

    return best_model_state, best_precision, best_epoch, model.lstm.hidden_size, model.lstm.num_layers, model.lstm.dropout

def cleanup_memory(device):
    # Clear PyTorch cache
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()