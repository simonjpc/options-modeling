import gc
import torch
import random
import numpy as np
import pandas as pd
from collections import deque
from datetime import timedelta
from exp2.model import LSTMClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score
from exp2.constants import BATCH_SIZE

# Set random seeds for deterministic behavior
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Re-train model on full training set with best Optuna parameters

def money_evaluation(optuna_study, features, train_set, val_set, test_set, df_orig_bid_ask, starting_capital=1000, nb_contracts=1):

    # best_params = optuna_study.best_params
    best_trial = optuna_study.best_trial
    best_model_state = best_trial.user_attrs["best_model_state"]
    best_hidden_size = best_trial.user_attrs['hidden_size']
    best_num_layers = best_trial.user_attrs['num_layers']
    best_dropout = best_trial.user_attrs['dropout']

    model = LSTMClassifier(
        input_size=len(features),
        hidden_size=best_hidden_size, #best_params['hidden_size'],
        num_layers=best_num_layers, #best_params['num_layers'],
        dropout=best_dropout, #best_params['dropout']
    )

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    model = model.to(device)
    # Full train loader
    # train_loader = DataLoader(
    #     train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0#, prefetch_factor=4#, collate_fn=collate
    # )
    # val_loader = DataLoader(
    #     val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0#, prefetch_factor=4#, collate_fn=collate
    # )

    # Retrieve the original DataFrame rows for the training set
    indices = pd.DataFrame(train_set.indices, columns=["datetime", "strike", "expire_date"])
    indices["merge_order"] = np.arange(len(indices))

    train_set.df[["strike", "expire_date"]] = train_set.df[["strike", "expire_date"]].astype(object)
    indices[["strike", "expire_date"]] = indices[["strike", "expire_date"]].astype(object)
    df_train_copy = train_set.df.reset_index().merge(
        indices, on=["datetime", "strike", "expire_date"], how="inner"
    )

    df_train_copy = df_train_copy.sort_values("merge_order").drop(columns="merge_order")
    df_train_copy = df_train_copy.set_index("datetime").copy()

    
    model.load_state_dict(best_model_state)

    gc.collect()
    # Get predicted probabilities on training set
    train_probs = []
    train_labels = []

    # Set decision threshold
    best_val_threshold = 0.8

    print("inside money_evaluation. moving model to eval mode...")
    model.eval()
    print("inside money_evaluation. model moved to eval mode")
    
    positive_flat_seqs = []
    positive_probs = []
    start_idx = 0
    train_loader_no_shuffle = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0#, prefetch_factor=4#, collate_fn=collate
    )
    with torch.no_grad():
        for X_batch, y_batch in train_loader_no_shuffle:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            for j in range(len(X_batch)):
                row = df_train_copy.iloc[start_idx + j]
                if (
                    row['label'] > 0.5 and
                    row['time_to_expiry'] < 1500 and
                    row['moneyness'] < 1.11 and
                    probs[j] >= best_val_threshold  # or another cutoff
                ):
                    flattened = X_batch[j].cpu().reshape(-1).numpy()  # shape: (sequence_length * num_features)
                    positive_flat_seqs.append(flattened)
                    positive_probs.append(probs[j].cpu().numpy())
            start_idx += len(X_batch)

            train_probs.extend(probs.cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())
    
    # Create DataFrame and attach prediction probabilities
    X_train_pos_reference = pd.DataFrame(positive_flat_seqs)
    X_train_pos_reference["pred_proba"] = positive_probs

    # Sort by probability and keep top 1000
    X_train_pos_reference = X_train_pos_reference.sort_values("pred_proba", ascending=False).head(1000)

    # Optionally drop the probability column for quantile filtering
    X_train_pos_reference = X_train_pos_reference.drop(columns=["pred_proba"])

    del train_loader_no_shuffle
    gc.collect()
    # Add back metadata to locate top positives

    print("debugging prints inside money_evaluation")
    
    print(f"all labels match: {all([lab_batch == df_train_copy['label'].iloc[idx] for idx, lab_batch in enumerate(train_labels)])}")
    df_train_copy["pred_proba"] = train_probs


    # Predict on test set and apply threshold
    # Rebuild test dataset
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0#, prefetch_factor=4#, collate_fn=collate
    )

    y_test_probs = []
    y_test_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            y_test_probs.extend(probs.cpu().numpy())
            y_test_true.extend(y_batch.cpu().numpy())
    gc.collect()
    # Apply test-time filters (time_to_expiry >= 1500 or moneyness >= 1.11 → prob = 0)
    indices = pd.DataFrame(test_set.indices, columns=["datetime", "strike", "expire_date"])
    indices["merge_order"] = np.arange(len(indices))
    df_test_copy = test_set.df.reset_index().merge(
        indices, on=["datetime", "strike", "expire_date"], how="inner"
    )
    df_test_copy = df_test_copy.sort_values("merge_order").drop(columns="merge_order")
    df_test_copy = df_test_copy.set_index("datetime").copy()
    
    print(f"all test labels match: {all([lab_batch == df_test_copy['label'].iloc[idx] for idx, lab_batch in enumerate(y_test_true)])}")
    
    invalid_mask = (df_test_copy['time_to_expiry'] >= 1500) | (df_test_copy['moneyness'] >= 1.11)
    y_test_probs = np.array(y_test_probs)
    y_test_probs[invalid_mask.values] = 0.0

    # Set decision threshold
    best_val_threshold = 0.8

    y_test_preds = (y_test_probs >= best_val_threshold).astype(int)

    # Final evaluation
    y_true_binary = (np.array(y_test_true) >= 0.5).astype(int)
    best_test_precision = precision_score(y_true_binary, y_test_preds, pos_label=1, zero_division=0.0)

    # Strategy simulation
    final_capital, capital_history, trade_log_df, df = simulate_strategy_corrected(
        df_test_copy,
        y_test_probs,
        model,
        df_orig_bid_ask,
        X_train_pos_reference=X_train_pos_reference if len(X_train_pos_reference) else None,
        best_threshold=best_val_threshold,
        starting_capital=starting_capital,
        nb_contracts=nb_contracts,
        features=[
            'delta', #5
            'gamma', #6
            'vega', #7
            'theta', #8
            'rho', #9
            'intra_bar_return', #15
            'intra_bar_volatility', #16
            'relative_close_position', #17
            'up_down_gap', #18
        ]
    )

    # Train-set precision for reference
    train_preds = (np.array(train_probs) >= best_val_threshold).astype(int)
    train_labels_binary = (np.array(train_labels) >= 0.5).astype(int)
    best_train_precision = precision_score(train_labels_binary, train_preds, pos_label=1, zero_division=0.0)

    # ---------------------
    # TMP val set precision
    # ---------------------

    # y_val_probs = []
    # y_val_true = []

    # with torch.no_grad():
    #     for X_batch, y_batch in val_loader:
    #         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #         logits = model(X_batch)
    #         probs = torch.sigmoid(logits)
    #         y_val_probs.extend(probs.cpu().numpy())
    #         y_val_true.extend(y_batch.cpu().numpy())

    # indices = pd.DataFrame(val_set.indices, columns=["datetime", "strike", "expire_date"])
    # indices["merge_order"] = np.arange(len(indices))
    # df_val_copy = val_set.df.reset_index().merge(indices, on=["datetime", "strike", "expire_date"], how="inner")
    # df_val_copy = df_val_copy.sort_values("merge_order").drop(columns="merge_order")
    # df_val_copy = df_val_copy.set_index("datetime").copy()
    
    # y_val_probs = np.array(y_val_probs)
    # y_val_preds = (y_val_probs >= best_val_threshold).astype(int)
    # best_val_precision = precision_score(y_val_true, y_val_preds, pos_label=1, zero_division=0.0)
    
    # print(f"all test labels match: {all([lab_batch == df_test_copy['label'].iloc[idx] for idx, lab_batch in enumerate(y_test_true)])}")
    
    # ---------------------
    # TMP val set precision
    # ---------------------


    # Print results
    print(f"best_threshold: {best_val_threshold}")
    print(f"best_test_precision: {best_test_precision:.4f}")
    print(f"best_train_precision: {best_train_precision:.4f}")
    print(f"best_precision from optuna (val set): {best_trial.user_attrs['best_precision']:.4f}")
    print(f" ===> Final capital (on unseen test set): ${final_capital:.2f} <===")

    if best_train_precision <= 0.8:
        final_capital = starting_capital
        capital_history = None
    return final_capital, capital_history, trade_log_df, df, best_val_threshold, best_test_precision

def simulate_strategy_corrected(
        df_test,          # Original test DataFrame
        y_test_proba,           # Prediction probs
        model,                 # Trained PyTorch model
        df_orig_bid_ask,
        X_train_pos_reference=None,
        q_low=0.05,
        q_high=0.95,
        best_threshold=0.5,
        starting_capital=1000,
        nb_contracts=1,
        features=None,         # List of features used for model input
        sequence_length=16,
        device='mps',
    ):

    gc.collect()
    df_orig_bid_ask = df_orig_bid_ask.reset_index()
    
    model.eval()
    df = df_test.copy().reset_index()
    df_orig_bid_ask["strike"] = df_orig_bid_ask["strike"].astype(object)
    df = df.merge(df_orig_bid_ask, on=["datetime", "strike", "expire_date"], how="left")
    df['datetime'] = pd.to_datetime(df['datetime'])

    print("debugging prints inside simulate_strategy_corrected")
    print(f"Length of test df: {len(df)}")
    print(f"Length of y_test_proba: {len(y_test_proba)}")

    df['pred_proba'] = y_test_proba
    df['prediction'] = (y_test_proba >= best_threshold).astype(int)
    print("proba and prediction columns assigned")
    df = df.sort_index()

    # 3️⃣ Initialize capital and trade logs
    capital = starting_capital
    capital_history = []
    trade_log = []
    invested_contracts = set()
    pending_returns = deque()

    feature_idxs = []
    print("for j in range(sequence_length):")
    for j in range(sequence_length):
        cols = [5*(j+1), 6*(j+1), 7*(j+1), 8*(j+1), 9*(j+1), 15*(j+1), 16*(j+1), 17*(j+1), 18*(j+1)]
        feature_idxs.extend(cols)
    
    print("for current_time, sub_df in df.groupby('datetime'):")
    df_test = df_test.reset_index().sort_values("datetime")

    for current_time, sub_df in df.groupby('datetime'):

        # Optional drift filtering
        if X_train_pos_reference is not None and features is not None:
            # low_pos = X_train_pos_reference[features].quantile(q_low)
            # high_pos = X_train_pos_reference[features].quantile(q_high)

            # outside_pos = ((sub_df[features] < low_pos) | (sub_df[features] > high_pos)).any(axis=1)
            # sub_df = sub_df[~outside_pos]
            # print("feature_idxs")
            # print(feature_idxs)
            low_quantile = X_train_pos_reference[feature_idxs[int(len(feature_idxs) // 2):]].quantile(q_low)
            high_quantile = X_train_pos_reference[feature_idxs[int(len(feature_idxs) // 2):]].quantile(q_high)
            keep_indices = []

            grouped_test = dict(tuple(df_test.groupby(['strike', 'expire_date'])))
            # Use itertuples for faster iteration
            for row in sub_df.itertuples(index=True):
                key = (row.strike, row.expire_date)
                if key not in grouped_test:
                    continue

                option_df = grouped_test[key]

                past_window = option_df[option_df['datetime'] <= row.datetime].tail(sequence_length // 2)
                if len(past_window) < sequence_length // 2:
                    continue

                seq = past_window[features].values  # shape: (16, F)
                flattened = seq.reshape(-1)         # shape: (16 * F)

                flat_series = pd.Series(flattened)
                flat_series.index = low_quantile.index
                outside = (flat_series < low_quantile) | (flat_series > high_quantile)

                if not outside.any():
                    keep_indices.append(row.Index)

            sub_df = sub_df.loc[keep_indices]

            if len(sub_df) == 0:
                capital_history.append((current_time, capital))
                continue
        
        print("for current_time, sub_df in df.groupby('datetime'): iteration finished")
        gc.collect()
        # Release matured returns
        while pending_returns and pending_returns[0][0] <= current_time:
            _, cap_delta = pending_returns.popleft()
            capital += cap_delta

        capital_history.append((current_time, capital))

        # 4️⃣ Select top-3 predicted positive options at this time
        candidates = sub_df[sub_df['prediction'] == 1]
        candidates = candidates.sort_values('pred_proba', ascending=False).head(3)

        print("for _, row in candidates.iterrows():")
        for _, row in candidates.iterrows():
            option_id = (row['strike'], row['expire_date'])
            if option_id in invested_contracts:
                continue

            option_price = row["orig_ask"] * 100 * nb_contracts
            if capital < option_price:
                continue

            invested_contracts.add(option_id)
            capital -= option_price

            if row['label'] > 0.5:#if row['label'] == 1:
                gain = min(row['percent_increase'], 2.0)  # CAP return at +200%
                capital_return = option_price * (1 + gain)
                result = 'win'
            else:
                capital_return = 0
                result = 'loss'

            release_time = row['datetime'] + timedelta(hours=row['hours_to_max'])
            pending_returns.append((release_time, capital_return))

            trade_log.append({
                'datetime': row['datetime'],
                'release_time': release_time,
                'strike': row['strike'],
                'expire_date': row['expire_date'],
                'result': result,
                'locked_capital': option_price,
                'expected_return': capital_return,
                'capital_available_after': release_time,
                'percent_increase': row['percent_increase'],
                'hours_to_max': row['hours_to_max']
            })
        print("for _, row in candidates.iterrows(): iteration finished")
    
    # Finalize any remaining returns
    last_time = df['datetime'].max()
    while pending_returns:
        release_time, cap_delta = pending_returns.popleft()
        if release_time > last_time:
            capital_history.append((release_time, capital + cap_delta))
        capital += cap_delta

    capital_history = pd.DataFrame(capital_history, columns=['datetime', 'capital'])
    trade_log_df = pd.DataFrame(trade_log)

    gc.collect()
    return capital, capital_history, trade_log_df, df