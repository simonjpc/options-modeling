import gc
import torch
import random
import numpy as np
from exp2.model import LSTMClassifier
from exp2.train_utils import train_model, cleanup_memory
from exp2.data import preprocess_to_tensors, Wrapper
from torch.utils.data import DataLoader, TensorDataset
from exp2.constants import BATCH_SIZE
# Set random seeds for deterministic behavior
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def time_series_real_eval(df, features, month, label_col='label', sequence_length=16):
    print("preparing dataset for sequence model compatibility...")
    df_orig_bid_ask = df[["strike", "expire_date", "strike_orig", "bid_orig", "ask_orig"]].copy()
    df = df.drop(columns=["strike_orig", "bid_orig", "ask_orig"])
    df = df.sort_index()
    n = len(df)
    train_val_cutoff = int(n * 0.7)
    val_test_cutoff = int(n * 0.85)
    df_train = df.iloc[:train_val_cutoff]
    df_valid = df.iloc[train_val_cutoff:val_test_cutoff]
    df_test = df.iloc[val_test_cutoff:]
    
    X_train_tensor, y_train_tensor, y_train_indices = preprocess_to_tensors(df_train, month, "train", sequence_length=sequence_length, feature_cols=features, label_col=label_col)
    # print(f"X_train_tensor shape: {X_train_tensor.shape}")
    X_val_tensor, y_val_tensor, y_val_indices = preprocess_to_tensors(df_valid, month, "val", sequence_length=sequence_length, feature_cols=features, label_col=label_col)
    X_test_tensor, y_test_tensor, y_test_indices = preprocess_to_tensors(df_test, month, "test", sequence_length=sequence_length, feature_cols=features, label_col=label_col)
    train_set = TensorDataset(X_train_tensor, y_train_tensor)
    val_set = TensorDataset(X_val_tensor, y_val_tensor)
    test_set = TensorDataset(X_test_tensor, y_test_tensor)
    train_set = Wrapper(train_set, y_train_indices, df_train)
    # print(f"train_set df shape: {train_set.df.shape}")
    val_set = Wrapper(val_set, y_val_indices, df_valid)
    test_set = Wrapper(test_set, y_test_indices, df_test)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0#, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0#, prefetch_factor=4
    )

    def objective(trial):
        hidden_size = trial.suggest_int('hidden_size', 32, 128)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        input_dim = len(features)
        model = LSTMClassifier(input_size=input_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout)
        print("model defined")
        y_train = df_train["label"].copy().astype(np.float32).values
        # y_train = [y for _, y in train_set]
        print("y_train list comprehension")
        pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)
        print(f"pos_weights computed: {pos_weight}")
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
        print(f"device {device} defined")
        best_model_state, best_precision, best_epoch, hidden_size, num_layers, dropout = train_model(
            model, train_loader, val_loader, epochs=8, lr=lr, device=device, pos_weight=pos_weight
        )
        trial.set_user_attr("best_model_state", best_model_state)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_precision", best_precision)
        trial.set_user_attr("hidden_size", hidden_size)
        trial.set_user_attr("num_layers", num_layers)
        trial.set_user_attr("dropout", dropout)
        gc.collect()
        
        return best_precision
    
    cleanup_memory(device)

    return objective, train_set, val_set, test_set, df_orig_bid_ask
