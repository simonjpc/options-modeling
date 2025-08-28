import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from exp1.preprocessing import prepare_labels, build_features_short
from exp1.constants import STOCK_PRICE_ABS_PATH, OPTIONS_CHAIN_ABS_PATH_MAP
from exp2.constants import FEATURES_TO_SCALE

def load_stock_data():
    stock_price_data = pd.read_csv(STOCK_PRICE_ABS_PATH)
    return stock_price_data

def load_options_data(month):
    with open(OPTIONS_CHAIN_ABS_PATH_MAP[month], "r") as f:
        options_chain_data = f.readlines()
    return options_chain_data
    
def preprocess_data(stock_data, options_data, window, month):
    df_labeled = prepare_labels(stock_data, options_data)
    df_labeled = build_features_short(df_labeled, window=window)
    df_labeled = suffix_columns(df_labeled)
    df_labeled = handle_scale_variant_feats(df_labeled)
    df_labeled = scale_features(df_labeled)
    return df_labeled

def handle_scale_variant_feats(df):
    # intra bar return: close/open
    df['intra_bar_return'] = 0.0
    mask = df['open'] > 0
    df.loc[mask, 'intra_bar_return'] = df.loc[mask, 'close'] / df.loc[mask, 'open']
    # intra bar volatility: (high - low)/open
    df['intra_bar_volatility'] = 0.0
    mask = df['open'] > 0
    df.loc[mask, 'intra_bar_volatility'] = (df.loc[mask, 'high'] - df.loc[mask, 'low']) / df.loc[mask, 'open']
    # Relative close position in bar (0 = low, 1 = high): (close - low)/(high - low)
    df['relative_close_position'] = 0.0
    mask = df['high'] - df['low'] != 0
    df.loc[mask, 'relative_close_position'] = (df.loc[mask, 'close'] - df.loc[mask, 'low']) / (df.loc[mask, 'high'] - df.loc[mask, 'low'])
    # Gap up/down: (open - prev_close)/prev_close
    df = df.groupby(['expire_date', 'strike'], group_keys=False).apply(get_up_down_gap)
    return df
    
def get_up_down_gap(group):
    prev_close = group["close"].shift(1).fillna(0)
    group['up_down_gap'] = 0.0
    mask = prev_close > 0
    group.loc[mask, 'up_down_gap'] = (group.loc[mask, "open"] - prev_close.loc[mask]) / prev_close.loc[mask]
    return group

def load_and_preprocess_df(month, window=16, exp_round=3):
    path = f"/Volumes/T7/backup/Documents/perso/repos_perso/options-modeling/data/lgbm-train/exp_round{exp_round}/n{window}/df_{month}.csv"
    if os.path.exists(path):
        df_labeled = pd.read_csv(path)
        df_labeled = df_labeled.set_index("datetime")
        df_labeled.index = pd.to_datetime(df_labeled.index)
    else:
        stock_price_data = load_stock_data()
        option_chain_data = load_options_data(month=month)
        df_labeled = preprocess_data(stock_price_data, option_chain_data, window=window, month=month)
        # df_labeled.to_csv(path)
        # print("dtypes of cols when building df:")
        # print(df_labeled.index.dtype)
        # print(df_labeled.dtypes)

    print(f"dataset completed for month {month}")
    return df_labeled


def scale_features(df, train_percent_thr=0.7):
    df = df.sort_index()
    train_last_idx = int(len(df) * train_percent_thr)
    df_train = df.iloc[:train_last_idx].copy()
    df_rest = df.iloc[train_last_idx:].copy()

    for col in ['strike', 'bid', 'ask', 'underlying_volume']: # 'delta_volume' has negative values
        df_train[col] = np.log1p(df_train[col])
        df_rest[col] = np.log1p(df_rest[col])

    for col in FEATURES_TO_SCALE:
        scaler = StandardScaler()
        df_train.loc[:, col] = scaler.fit_transform(df_train[[col]])
        df_rest.loc[:, col] = scaler.transform(df_rest[[col]])  # only transform here

    df = pd.concat([df_train, df_rest], axis=0)
    df = df.sort_index()
    return df

class OptionSequenceDataset(Dataset):
    def __init__(self, df, sequence_length=16, feature_cols=None, label_col='label'):
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols or [
            col for col in df.columns if col not in [
                'label', 'expire_date', 'strike', 'percent_increase', 'hours_to_max'
            ]
        ]
        self.label_col = label_col
        self.df = df.sort_index()

        self.sequence_starts = []
        self.indices = []

        # Store meta information
        self.meta = df[['strike', 'expire_date']].values
        self.features_np = df[self.feature_cols].values.astype(np.float32)
        self.labels_np = df[self.label_col].values.astype(np.float32)

        # NEW: Integer row lookup for fast access
        self.index_to_position = {
            (idx, row['strike'], row['expire_date']): i
            for i, (idx, row) in enumerate(df.iterrows())
        }

        grouped = df.groupby(['strike', 'expire_date'])
        for (strike, expire_date), group in grouped:
            group = group.sort_index()
            if len(group) < sequence_length:
                continue
            indices = group.index
            for i in range(sequence_length - 1, len(group)):
                self.sequence_starts.append((indices[i-sequence_length+1:i+1], (indices[i], strike, expire_date)))
                self.indices.append((indices[i], strike, expire_date))

        self.sequence_starts = np.array(self.sequence_starts, dtype=object)
        self.indices = np.array(self.indices, dtype=object)

    def __len__(self):
        return len(self.sequence_starts)

    def __getitem__(self, idx):
        seq_indices, (target_idx, strike, expire_date) = self.sequence_starts[idx]

        # Fast lookup via precomputed position mapping
        seq_rows = [self.index_to_position[(ix, strike, expire_date)] for ix in seq_indices]
        target_row = self.index_to_position[(target_idx, strike, expire_date)]

        X = self.features_np[seq_rows]
        y = self.labels_np[target_row]

        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32).squeeze()

def preprocess_to_tensors(df, month, task_type, sequence_length=16, feature_cols=None, label_col='label', exp_round=3, window=16):
    
    pt_path = f"/Volumes/T7/backup/Documents/perso/repos_perso/options-modeling/data/lgbm-train/exp_round{exp_round}/n{window}/{task_type}_tensors_{month}.pt"
    
    if os.path.exists(pt_path):
        X_tensor, y_tensor, indices_tensor = torch.load(pt_path, weights_only=False)
        return (X_tensor, y_tensor, indices_tensor)
    
    feature_cols = feature_cols or [
        col for col in df.columns if col not in [
            'label', 'expire_date', 'strike', 'percent_increase', 'hours_to_max'
        ]
    ]
    df = df.sort_index()

    grouped = df.groupby(['strike', 'expire_date'])

    X_list = []
    y_list = []
    indices = []

    index_to_position = {
        (idx, row['strike'], row['expire_date']): i
        for i, (idx, row) in enumerate(df.iterrows())
    }

    features_np = df[feature_cols].values.astype(np.float32)
    labels_np = df[label_col].values.astype(np.float32)

    for (strike, expire_date), group in grouped:
        group = group.sort_index()
        if len(group) < sequence_length:
            continue
        indices_in_group = group.index
        for i in range(sequence_length - 1, len(group)):
            seq_indices = indices_in_group[i-sequence_length+1:i+1]
            target_idx = indices_in_group[i]

            try:
                seq_rows = [index_to_position[(ix, strike, expire_date)] for ix in seq_indices]
                target_row = index_to_position[(target_idx, strike, expire_date)]
            except KeyError:
                continue  # Skip if any key is missing

            X_seq = features_np[seq_rows]
            y_val = labels_np[target_row]

            X_list.append(X_seq)
            y_list.append(y_val)
            indices.append((target_idx, strike, expire_date))

    X_tensor = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)
    indices_tensor = np.array(indices, dtype=object)

    torch.save((X_tensor, y_tensor, indices_tensor), pt_path)
    print(f"Saved tensors with shape: X={X_tensor.shape}, y={y_tensor.shape}")
    return (X_tensor, y_tensor, indices_tensor)


def save_experiment_results(results, filepath, save_mode="wb") -> None:
    with open(filepath, save_mode) as f:
        pickle.dump(results, f)

class Wrapper(Dataset):
    def __init__(self, tensor_dataset, indices, df):
        self.dataset = tensor_dataset
        self.indices = indices
        self.df = df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def suffix_columns(df, columns=["strike", "ask", "bid"], suffix="orig"):
    if columns is not None:
        for col in columns:
            if col in df.columns:
                df[f"{col}_{suffix}"] = df[col]
    return df