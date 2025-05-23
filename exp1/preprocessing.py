import os
import gc
import numpy as np
import pandas as pd
import time
from numba import njit
from scipy.stats import norm
from exp1.utils import build_dict, isolate_option_types
from joblib import Parallel, delayed
from exp1.constants import CUMBERSOME_COLS


def prepare_labels(stock_price_data, options_chain_data):
    raw_keys = options_chain_data[0]
    options_hist_data = [
        item
        for raw_values in options_chain_data[1:]
        for item in isolate_option_types(build_dict(raw_keys, raw_values))
    ]
    options_chain_data_hist = pd.DataFrame(options_hist_data)

    # options data manip
    options_chain_data_hist_reduced = options_chain_data_hist.rename(columns={"quote_readtime": "datetime"})
    call_chain_data_hist_reduced = options_chain_data_hist_reduced.loc[
        options_chain_data_hist_reduced["type"] == "call",
        ["datetime", "strike", "iv", "expire_date", "bid", "ask", "volume", "delta", "gamma", "vega", "theta", "rho"]
    ]
    put_chain_data_hist_reduced = options_chain_data_hist_reduced.loc[
        options_chain_data_hist_reduced["type"] == "put",
        ["datetime", "strike", "iv", "expire_date", "bid", "ask", "volume", "delta", "gamma", "vega", "theta", "rho"]
    ]

    # stocks data manip
    stock_price_data_15min = stock_price_data.loc[stock_price_data["date"].apply(
        lambda x:(":30" in x[:-3]) or (":15" in x[:-3]) or (":00" in x[:-3]) or (":45" in x[:-3])
    ), ["date", "open", "close", "low", "high", "volume"]]
    stock_price_data_15min = stock_price_data_15min.rename(columns={"date": "datetime", "volume": "underlying_volume"})
    stock_price_data_15min["datetime"] = stock_price_data_15min["datetime"].apply(lambda x: x[:-3])

    # reset indexes
    call_chain_data_hist_reduced = call_chain_data_hist_reduced.set_index("datetime")
    put_chain_data_hist_reduced = put_chain_data_hist_reduced.set_index("datetime")
    stock_price_data_15min = stock_price_data_15min.set_index("datetime")
    # concat_options_chain_prices = call_chain_data_hist_reduced.join(stock_price_data_15min, how="left") # call chain
    concat_options_chain_prices = put_chain_data_hist_reduced.join(stock_price_data_15min, how="left") # put chain
    concat_options_chain_prices = concat_options_chain_prices[concat_options_chain_prices.index.map(lambda x: " 16:00" not in x)]
    df = concat_options_chain_prices.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Group the DataFrame
    grouped = df.groupby(['strike', 'expire_date'])

    # Process in parallel
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(tag_returns_process_group)(strike, expire_date, group)
        for (strike, expire_date), group in grouped
    )

    # Combine results
    df_labeled = pd.concat(results).sort_index()

    # compute timestep delta from cumulative delta
    df_labeled['date'] = df_labeled.index.date
    df_labeled['delta_volume'] = (
        df_labeled.groupby(['date', 'strike', 'expire_date'])['volume']
        .diff()
        .fillna(0)
    )

    # drop rows with an index (datetime) with less than 1 week to max index datetime and with a label of 0
    df_labeled = clean_tail_edge_cases(df_labeled)
    df_labeled = df_labeled.drop(columns=['date', 'volume'])
    return df_labeled

def clean_tail_edge_cases(df_labeled):
    cleaned_dfs = []

    grouped = df_labeled.groupby(['strike', 'expire_date'])

    for _, group in grouped:
        group = group.sort_index()
        max_time = group.index.max()
        cutoff = max_time - pd.Timedelta(weeks=1)

        final_week = group[group.index > cutoff]

        if final_week['label'].any():
            last_positive_time = final_week[final_week['label'] == 1].index.max()
            cleaned_group = group.loc[:last_positive_time]
        else:
            cleaned_group = group[group.index <= cutoff]

        cleaned_dfs.append(cleaned_group)

    return pd.concat(cleaned_dfs).sort_index()


def tag_returns_process_group(strike, expire_date, group):
    group = group.sort_index()
    times = group.index
    asks = group['ask'].values
    bids = group['bid'].values
    labels = np.full(len(group), 0.1, dtype=np.float16)#labels = np.zeros(len(group), dtype=np.int8)
    percent_increase = np.zeros(len(group), dtype=np.float16)
    hours_to_max = np.zeros(len(group), dtype=np.float16)

    for i in range(len(group)):
        ask_t = asks[i]
        if ask_t == 0:
            continue

        time_t = times[i]
        window_end = min(time_t + pd.Timedelta(weeks=1), pd.to_datetime(expire_date))
        mask = (times > time_t) & (times <= window_end)

        future_bids = bids[mask]
        future_times = times[mask]
        returns = (future_bids - ask_t) / ask_t if len(future_bids) > 0 else np.array([])

        if len(returns) > 0:
            max_idx = np.argmax(returns)
            max_return = returns[max_idx]
            max_time = future_times[max_idx]

            if max_return >= 1: # greater or equals to 100%
                labels[i] = 0.9#labels[i] = 1
                percent_increase[i] = (np.max(bids[mask]) - ask_t) / ask_t
                hours_to_max[i] = (max_time - time_t).total_seconds() / 3600.0  # in hours

    group_result = group.copy()
    group_result['label'] = labels
    group_result['percent_increase'] = percent_increase
    group_result['hours_to_max'] = hours_to_max

    return group_result


# def create_option_dataset_full(df, n=16, label_column='label'):
#     df = df.copy()
#     df.index = pd.to_datetime(df.index)
#     df = df.sort_index()

#     expiration_times = pd.to_datetime(df['expire_date']) + pd.Timedelta(hours=16)
#     df['time_to_expiry'] = (expiration_times - df.index).dt.total_seconds() / 3600

#     time_series_cols = ['open', 'iv', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho', 'delta_volume']
#     static_cols = [
#         'strike', 'expire_date', 'time_to_expiry', 'underlying_volume', 'dvol', 'baspread', 'rel_baspread', 
#         'open_return', 'option_return', 'embedded_leverage', 'idiosyncratic_vol', 'm_degree', 'midpoint', 
#         'optspread'
#     ]
#     label_cols = [label_column, 'percent_increase', 'hours_to_max']

#     X_rows = []
#     y_rows = []

#     grouped = df.groupby(['strike', 'expire_date'], sort=False)

#     for _, group in grouped:
#         if len(group) < n:
#             continue

#         group = group.sort_index()

#         ts = group[time_series_cols].values
#         static = group[static_cols].values
#         labels = group[label_cols].values
#         datetimes = group.index.values

#         T = len(group)
#         num_windows = T - (n-1)

#         idx = np.arange(n)[None, :] + np.arange(num_windows)[:, None]  # t0 to t-15
#         ts_windows = ts[idx]  # (num_windows, 16, F)

#         # Special treatment for 'open': all t0 to t-15
#         open_windows = ts_windows[:, :, 0]  # open is feature 0, shape (num_windows, 16)

#         # For others: keep t0 to t-5, t-10, t-15
#         if n == 16:
#             selected_idx = [0,1,2,3,4,5,10,15]
#         elif n == 24:
#             selected_idx = [0,1,2,3,4,5,10,15,20,23]
#         other_features_windows = ts_windows[:, selected_idx, 1:]  # exclude 'open'

#         open_flat = open_windows  # no reshape

#         # Compute open_change_tN = (open_t-(N-1) - open_t-N) / open_t-N
#         open_change = (open_flat[:, :-1] - open_flat[:, 1:]) / open_flat[:, 1:]
#         # Rename columns: open_change_t0 corresponds to change from t0 to t-1
#         open_change_flat = open_change  # shape (num_windows, 15)

#         other_flat = other_features_windows.reshape(num_windows, -1)

#         # ts_flat = np.concatenate([open_flat, other_flat], axis=1)
#         ts_flat = np.concatenate([open_flat, open_change_flat, other_flat], axis=1)

#         static_final = static[idx[:, 0]]
#         labels_final = labels[idx[:, 0]]
#         dates_final = datetimes[idx[:, 0]]

#         X_group = np.column_stack([dates_final, static_final, ts_flat])
#         y_group = labels_final

#         X_rows.append(X_group)
#         y_rows.append(y_group)

#     X_final = np.vstack(X_rows)
#     y_final = np.vstack(y_rows)

#     # Build column names
#     # open_col_names = [f'open_t{-i}' for i in range(0, 16)]  # open_t0 to open_t-15
#     # feature_times = ['t0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-10', 't-15']
#     # other_col_names = [f'{col}_{t}' for t in feature_times for col in time_series_cols[1:]]
#     open_col_names = [f'open_t{-i}' for i in range(0, n)]
#     open_change_col_names = [f'open_change_t{-i}' for i in range(0, (n-1))]
#     if n == 16:
#         feature_times = ['t0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-10', 't-15']
#     elif n == 24:
#         feature_times = ['t0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-10', 't-15', 't-20', 't-23']
#     other_col_names = [f'{col}_{t}' for t in feature_times for col in time_series_cols[1:]]

#     # col_names = ['datetime', 'strike', 'expire_date', 'time_to_expiry'] + open_col_names + other_col_names
#     col_names = [
#         'datetime', 'strike', 'expire_date', 'time_to_expiry',
#         'underlying_volume', 'dvol', 'baspread', 'rel_baspread', 
#         'open_return', 'option_return', 'embedded_leverage',
#         'idiosyncratic_vol', 'm_degree', 'midpoint', 'optspread'
#     ] + open_col_names + open_change_col_names + other_col_names

#     X = pd.DataFrame(X_final, columns=col_names)
#     X['datetime'] = pd.to_datetime(X['datetime'])

#     float_cols = [col for col in col_names if ("datetime" not in col) and ("expire_date" not in col)]
#     X[float_cols] = X[float_cols].astype(np.float32)

#     y = pd.DataFrame(y_final, columns=["target", "percent_increase", "hours_to_max"])

#     return X, y

def create_option_dataset_full_helper(group, window=16, label_column="label"):
    group = group.sort_index()

    if window == 16:
        selected_idx = [0, 1, 2, 3, 4, 5, 10, 15]
    elif window == 24:
        selected_idx = [0, 1, 2, 3, 4, 5, 10, 15, 20, 23]

    time_series_cols = ['open', 'iv', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho', 'delta_volume']
    
    # Precompute percent changes for all columns at once
    pct_changes = group[time_series_cols].pct_change().fillna(0)

    # for column 'open'
    for s_idx in range(window):
        suffix = f"t-{s_idx}" if s_idx > 0 else f"t{s_idx}"
        shifted_open = group[["open"]].shift(s_idx)
        shifted_open_pct = pct_changes[["open"]].shift(s_idx)

        shifted_open.columns = [f"open_{suffix}"]
        shifted_open_pct.columns = [f"open_change_{suffix}"]

        group = pd.concat([group, shifted_open, shifted_open_pct], axis=1)

    # for all other columns
    for s_idx in selected_idx:
        suffix = f"t-{s_idx}" if s_idx > 0 else f"t{s_idx}"
        
        # Use group shifting across all columns at once
        shifted_cols = group[time_series_cols[1:]].shift(s_idx)
        shifted_pct = pct_changes[time_series_cols[1:]].shift(s_idx)

        shifted_cols.columns = [f"{col}_{suffix}" for col in time_series_cols[1:]]
        shifted_pct.columns = [f"{col}_change_{suffix}" for col in time_series_cols[1:]]

        group = pd.concat([group, shifted_cols, shifted_pct], axis=1)

    group = group.drop(columns=time_series_cols[1:])
    return group

def create_option_dataset_full_new(df, window=16):

    df = df.groupby(['strike', 'expire_date'], group_keys=False).apply(create_option_dataset_full_helper, window)
    df = df.reset_index()

    y_cols = ['label', 'percent_increase', 'hours_to_max']
    
    X = df[[col for col in df.columns if col not in y_cols]]
    X['datetime'] = pd.to_datetime(X['datetime'])
    float_cols = [col for col in X.columns if ("datetime" not in col) and ("expire_date" not in col)]
    X[float_cols] = X[float_cols].astype(np.float32)
    
    y = df[y_cols]
    y = y.rename(columns={"label":"target"})

    return X, y

def create_option_dataset_full(df, n=16, label_column='label'):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    expiration_times = pd.to_datetime(df['expire_date']) + pd.Timedelta(hours=16)
    df['time_to_expiry'] = (expiration_times - df.index).dt.total_seconds() / 3600

    time_series_cols = ['open', 'iv', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho', 'delta_volume']
    static_cols = [
        'strike', 'expire_date', 'time_to_expiry', 'underlying_volume', 'dvol', 'baspread', 'rel_baspread', 
        'open_return', 'option_return', 'embedded_leverage', 'idiosyncratic_vol', 'm_degree', 'midpoint', 
        'optspread'
    ]
    label_cols = [label_column, 'percent_increase', 'hours_to_max']

    X_rows = []
    y_rows = []

    grouped = df.groupby(['strike', 'expire_date'], sort=False)

    for _, group in grouped:
        if len(group) < n:
            continue

        group = group.sort_index()

        ts = group[time_series_cols].values
        static = group[static_cols].values
        labels = group[label_cols].values
        datetimes = group.index.values

        T = len(group)
        num_windows = T - (n-1)

        idx = np.arange(n)[None, :] + np.arange(num_windows)[:, None]
        ts_windows = ts[idx]  # (num_windows, n, F)

        # Special treatment for 'open': all t0 to t-15
        open_windows = ts_windows[:, :, 0]  # shape (num_windows, n)

        # For others: keep t0 to t-5, t-10, t-15 (or similar for n=24)
        if n == 16:
            selected_idx = [0,1,2,3,4,5,10,15]
        elif n == 24:
            selected_idx = [0,1,2,3,4,5,10,15,20,23]
        other_features_windows = ts_windows[:, selected_idx, 1:]  # exclude 'open'

        open_flat = open_windows  # shape (num_windows, n)

        # Corrected rate of change: (r_t - r_(t-1)) / r_(t-1)
        open_change_raw = (open_flat[:, 1:] - open_flat[:, :-1]) / open_flat[:, :-1]
        pad = np.full((open_change_raw.shape[0], 1), np.nan)
        open_change_flat = np.concatenate([pad, open_change_raw], axis=1)  # shape (num_windows, n - 1)

        other_flat = other_features_windows.reshape(num_windows, -1)

        ts_flat = np.concatenate([open_flat, open_change_flat, other_flat], axis=1)

        static_final = static[idx[:, 0]]
        labels_final = labels[idx[:, 0]]
        dates_final = datetimes[idx[:, 0]]

        X_group = np.column_stack([dates_final, static_final, ts_flat])
        y_group = labels_final

        X_rows.append(X_group)
        y_rows.append(y_group)

    X_final = np.vstack(X_rows)
    y_final = np.vstack(y_rows)

    # Build column names
    open_col_names = [f'open_t{-i}' for i in range(0, n)]
    open_change_col_names = [f'open_change_t{-i}' for i in range(1, n)]  # starts at t-1 (change from t-1 to t0)
    if n == 16:
        feature_times = ['t0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-10', 't-15']
    elif n == 24:
        feature_times = ['t0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-10', 't-15', 't-20', 't-23']
    other_col_names = [f'{col}_{t}' for t in feature_times for col in time_series_cols[1:]]

    col_names = [
        'datetime', 'strike', 'expire_date', 'time_to_expiry',
        'underlying_volume', 'dvol', 'baspread', 'rel_baspread', 
        'open_return', 'option_return', 'embedded_leverage',
        'idiosyncratic_vol', 'm_degree', 'midpoint', 'optspread'
    ] + open_col_names + open_change_col_names + other_col_names

    X = pd.DataFrame(X_final, columns=col_names)
    X['datetime'] = pd.to_datetime(X['datetime'])

    float_cols = [col for col in col_names if ("datetime" not in col) and ("expire_date" not in col)]
    X[float_cols] = X[float_cols].astype(np.float32)

    y = pd.DataFrame(y_final, columns=["target", "percent_increase", "hours_to_max"])

    return X, y


def add_datetime_features(X):
    X = X.copy()
    datetime_features = pd.DataFrame({
        'hour': X['datetime'].dt.hour,
        'minute': X['datetime'].dt.minute,
        'day_of_week': X['datetime'].dt.dayofweek,
    }, index=X.index)
    X = pd.concat([X, datetime_features], axis=1)
    X['day_of_week'] = X['day_of_week'].astype("category")
    return X

def add_advanced_features_new(df, window=16):
    # List of variable base names
    variables = ['open', 'iv', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho', 'delta_volume']

    # Time deltas you're interested in
    deltas = [1, 2, 3, 4, 5, 10, 15]

    # Loop through each variable and each delta to compute slopes
    for var in variables:
        for delta in deltas:
            col_t0 = f'{var}_t0'
            col_t_minus = f'{var}_t-{delta}'
            slope_col = f'{var}_slope_t_t-{delta}'
            if col_t0 in df.columns and col_t_minus in df.columns:
                df[slope_col] = (df[col_t0] - df[col_t_minus]) / delta

    open_cols_to_drop = ["open", "open_return"] + [f'open_t{-i}' for i in range(0, window)]
    df = df.drop(columns=open_cols_to_drop, errors='ignore')
    return df

def add_advanced_features(X: pd.DataFrame, n=6):
    X = X.copy()
    
    # Base features
    base_features = ['open', 'iv', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho', 'delta_volume']
    if n == 16:
        time_steps = ['t0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-10', 't-15']
    elif n == 24:
        time_steps = ['t0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-10', 't-15', 't-20', 't-23']

    # Pre-extract necessary arrays
    data_all = {feature: X[[f'{feature}_{step}' for step in time_steps]].to_numpy(dtype=np.float32)
                for feature in base_features}

    all_new_features = {}

    for feature, data in data_all.items():
        # Slopes between t and (t-1, t-2, t-5, t-10, t-15)
        if n == 16:
            steps = [1, 2, 5, 6, 7]
        elif n == 24:
            steps = [1, 2, 5, 6, 7, 8, 9] # should I put here ?
        for idx, step in enumerate(steps):  # Corresponds to t-1, t-2, t-5, t-10, t-15
            diff = (data[:, 0] - data[:, step])  # value_t - value_t-k
            delta_time = step  # step corresponds to the distance in steps
            slope = diff / delta_time
            if n == 16:
                all_new_features[f'{feature}_slope_t_t-{[1,2,5,10,15][idx]}'] = slope
            elif n == 24:
                all_new_features[f'{feature}_slope_t_t-{[1,2,5,10,15,20,23][idx]}'] = slope

    # Add moneyness (strike relative to open_t0)
    open_t0 = X['open_t0'].astype(np.float32).values
    strike = X['strike'].astype(np.float32).values
    moneyness = strike / open_t0
    all_new_features['moneyness'] = moneyness

    X = X.assign(**all_new_features)

    # Drop raw open_t* columns and keep open_change_t* instead
    open_cols_to_drop = [f'open_t{-i}' for i in range(0, n)]
    X = X.drop(columns=open_cols_to_drop, errors='ignore')

    return X

# def preprocess_dataset(stock_data: pd.DataFrame, options_data: pd.DataFrame, n: int, month: str):
#     Xpath = f"/Volumes/T7/backup/Documents/perso/repos_perso/options-modeling/data/lgbm-train/n{n}/X_{month}.csv"
#     ypath = f"/Volumes/T7/backup/Documents/perso/repos_perso/options-modeling/data/lgbm-train/n{n}/y_{month}.csv"

#     if os.path.exists(Xpath) and os.path.exists(ypath):
#         print("data exists... loading source")
#         X_month = pd.read_csv(Xpath)
#         y_month = pd.read_csv(ypath)
#     else:
#         # 2. Preprocessing: same steps as you did for January
#         df_labeled = prepare_labels(stock_data, options_data)
#         print(f"labels created for month {month}")
#         X_month, y_month = create_option_dataset_full(df_labeled, n=n)
#         X_month = add_datetime_features(X_month)
#         X_month = add_advanced_features(X_month, n=n)
#         X_month.to_csv(Xpath, index=False)
#         y_month.to_csv(ypath, index=False)
#     return X_month, y_month

def preprocess_dataset(stock_data: pd.DataFrame, options_data: pd.DataFrame, n: int, month: str, exp_round=2):
    Xpath = f"/Volumes/T7/backup/Documents/perso/repos_perso/options-modeling/data/lgbm-train/exp_round{exp_round}/n{n}/X_{month}.csv"
    ypath = f"/Volumes/T7/backup/Documents/perso/repos_perso/options-modeling/data/lgbm-train/exp_round{exp_round}/n{n}/y_{month}.csv"

    if os.path.exists(Xpath) and os.path.exists(ypath):
        print("data exists... loading source")
        X_month = pd.read_csv(Xpath)
        y_month = pd.read_csv(ypath)
    else:
        # 2. Preprocessing: same steps as you did for January
        df_labeled = prepare_labels(stock_data, options_data)
        print(f"labels created for month {month}")
        # df_labeled = build_features(df_labeled, window=n)
        # print("function `build_features` executed")
        df_labeled = build_features_short(df_labeled, window=n)
        print("function `build_features_short` executed")
        X_month, y_month = create_option_dataset_full_new(df_labeled, window=n)
        print("function `create_option_dataset_full` executed")
        X_month = add_datetime_features(X_month)
        print("datetime features added")
        X_month = add_advanced_features_new(X_month, window=n)
        print("advanced features added")
        X_month.to_csv(Xpath, index=False)
        y_month.to_csv(ypath, index=False)

    print(f"dataset features shape before dropna: {X_month.shape}")
    for col in CUMBERSOME_COLS:
        if col in X_month.columns:
            X_month = X_month.drop(columns=[col])
    X_month = X_month.dropna(axis=0)
    print(f"dataset features shape after dropna: {X_month.shape}")
    y_month = y_month.loc[X_month.index]
    return X_month, y_month

def build_features_short(df, window=16):
    start = time.time()
    df = df.sort_index()

    # time to expiry
    df["time_to_expiry"] = (
        (pd.to_datetime(df["expire_date"]) + pd.Timedelta(hours=16)) - pd.to_datetime(df.index)
    ).dt.total_seconds() / 3600
    print("\t time_to_expiry computed")
    gc.collect()

    #moneyness
    df["moneyness"] = df["strike"] / df["open"]
    print("\t moneyness computed")
    gc.collect()

    df["orig_ask"] = df["ask"]
    df["orig_bid"] = df["bid"]

    end = time.time()
    tdelta = end - start
    print("----")
    print(f"short features preparation execution time: {tdelta:.3f} seconds")
    print("----")

    return df

def build_features(df, window=16):
    start = time.time()
    df = df.sort_index()
    # dollar volume
    df["dvol"] = df["delta_volume"] * ((df["ask"] + df["bid"]) / 2)
    print("\t dvol computed")
    gc.collect()

    # aboslute illiquidity
    ask_price_diff = df.groupby(["strike", "expire_date"])["ask"].diff().fillna(0)
    stock_price_diff = df.groupby(["strike", "expire_date"])["open"].diff().fillna(0)
    df["ailliq"] = 0.0
    mask = df["dvol"] > 0
    df.loc[mask, "ailliq"] = abs(ask_price_diff.loc[mask] - df.loc[mask, "delta"] * stock_price_diff.loc[mask]) / df.loc[mask, "dvol"]
    print("\t ailliq computed")
    gc.collect()

    # amihud illiquidity
    # stock_return = stock_price_diff / df["open"].shift(1).fillna(0)
    # df["amihudilliq"] = abs(stock_return) / df["dvol"]
    prev_open = df["open"].shift(1).fillna(0)
    dvol = df["dvol"]
    valid_mask = (prev_open > 0) & (dvol > 0)
    df["amihudilliq"] = 0.0
    stock_return = stock_price_diff / prev_open
    amihud_illiq = abs(stock_return) / dvol
    df.loc[valid_mask, "amihudilliq"] = amihud_illiq[valid_mask]
    print("\t amihudilliq computed")
    gc.collect()

    # bid ask spread
    df["baspread"] = df["ask"] - df["bid"]
    print("\t baspread computed")
    gc.collect()
    # relative bid ask spread
    midpoint = (df["bid"] + df["ask"]) / 2
    df["rel_baspread"] = 0.0
    mask = midpoint > 0
    df.loc[mask, "rel_baspread"] = df.loc[mask, "baspread"] / midpoint.loc[mask]
    print("\t rel_baspread computed")
    gc.collect()

    # EXTRA FOR LATER USE
    # stock and option return (1 timestep)
    df = df.groupby(['strike', 'expire_date'], group_keys=False).apply(compute_open_return)
    print("\t open_return computed")
    gc.collect()

    prev_ask = df['ask'].shift(1)
    numerator = df['bid'] - prev_ask
    denominator = prev_ask

    # Avoid division by zero explicitly
    df = df.groupby(['strike', 'expire_date'], group_keys=False).apply(compute_option_return)
    # df['option_return'] = np.where(
    #     denominator != 0,
    #     numerator / denominator,
    #     0  # If denominator is zero, return 0
    # )
    print("\t option_return computed")
    gc.collect()

    # df['contract_id'] = df['strike'].astype(str) + '_' + df['expire_date'].astype(str)
    # df = apply_bear_beta(df, window=window, min_points=3)
    # df = apply_beta(df, window=window, min_points=3)
    # df = df.drop(columns=["contract_id"], axis=1)

    # df = compute_normalized_momentum(df)

    # df = compute_ps_liquidity(df)

    df["embedded_leverage"] = 0.0
    mask = df["ask"] > 0
    df.loc[mask, "embedded_leverage"] = abs(df.loc[mask, "delta"]) * df.loc[mask, "open"] / df.loc[mask, "ask"]
    # df["embedded_leverage"] = abs(df["delta"]) * df["open"] / df["ask"]
    print("\t embedded_leverage computed")
    gc.collect()

    # df = fast_grouped_kurtosis(df)
    # df = fast_grouped_skewness(df)

    # df = compute_historic_volatility(df)

    df = compute_idiosyncratic_vol(df)
    print("\t idiosyncratic_vol computed")
    gc.collect()

    df = df.groupby(['strike', 'expire_date'], group_keys=False).apply(compute_illiq)
    print("\t illiq computed")
    gc.collect()

    df['iv_rank'] = (
        df
        .sort_index()
        .groupby(['strike', 'expire_date'], group_keys=False)
        .apply(apply_rank)
    )
    print("\t iv_rank computed")
    gc.collect()

    # df = df.groupby(['strike', 'expire_date'], group_keys=False).apply(compute_iv_minus_realized_vol, window)

    # df = df.groupby(['strike', 'expire_date'], group_keys=False).apply(compute_iv_minus_realized_vol_ratio, window)

    df = compute_n_degree(df)
    print("\t n_degree computed")
    gc.collect()

    df["midpoint"] = (df["bid"] + df["ask"]) / 2
    print("\t midpoint computed")
    gc.collect()

    df["moneyness"] = df["strike"] / df["open"]
    print("\t moneyness computed")
    gc.collect()

    df["optspread"] = 0.0
    mask = (df["bid"] + df["ask"]) > 0
    df.loc[mask, "optspread"] = 2 * (df.loc[mask, "bid"] - df.loc[mask, "ask"]) / (df.loc[mask, "bid"] + df.loc[mask, "ask"])
    print("\t optspread computed")
    gc.collect()

    # df = compute_pfht(df, window=window)

    # df = compute_pifht(df, window=window)

    # df["pilliq"] = np.where(
    #     (df["dvol"] * df["ask"]) == 0,
    #     0,
    #     abs(ask_price_diff - df["delta"] * stock_price_diff) / (df["dvol"] * df["ask"])
    # )
    # print("\t pilliq computed")
    # gc.collect()

    # df = df.groupby(['strike', 'expire_date'], group_keys=False).apply(compute_piroll, window)

    # df = df.groupby(['strike', 'expire_date'], group_keys=False).apply(compute_retvol, window)

    # df = df.groupby(['strike', 'expire_date'], group_keys=False).apply(compute_std_dolvol, window)

    # df = df.groupby(["strike", "expire_date"], group_keys=False).apply(compute_std_amihud, window)

    df["time_to_expiry"] = (
        (pd.to_datetime(df["expire_date"]) + pd.Timedelta(hours=16)) - pd.to_datetime(df.index)
    ).dt.total_seconds() / 3600
    print("\t time_to_expiry computed")
    gc.collect()

    # df = df.groupby(["strike", "expire_date"], group_keys=False).apply(compute_volga)

    # df = df.groupby(["strike", "expire_date"], group_keys=False).apply(compute_zerotrade, window)

    end = time.time()
    delta = end - start
    print("----")
    print(f"features preparation execution time: {delta:.3f} seconds")
    print("----")
    return df

    # y_columns = ["label", "percent_increase", "hours_to_max"]
    # X = df[[col for col in df.columns if col not in y_columns]]
    # float_cols = [col for col in X.columns if col not in ("datetime", "expire_date")]
    # X[float_cols] = X[float_cols].astype(np.float32)
    # y = df[y_columns]
    # y = y.rename(columns={"label": "target"})
    return X, y

# --------
# --------
# --------

def compute_open_return(group):
    group['open_return'] = group['open'].pct_change().fillna(0)
    return group

def compute_option_return(group):
    prev_ask = group['ask'].shift(1)
    numerator = group['bid'] - prev_ask
    denominator = prev_ask

    # Avoid division by zero explicitly
    group['option_return'] = np.where(
        denominator != 0,
        numerator / denominator,
        0  # If denominator is zero, return 0
    )
    group['option_return'] = group['option_return'].fillna(0)
    return group

@njit
def compute_bear_beta_numba(contract_indices, mid_rets, under_rets, window, min_points):
    n = len(contract_indices)
    bear_beta = np.full(n, np.nan)

    current = contract_indices[0]
    start = 0

    for i in range(1, n):
        if contract_indices[i] != current or i == n - 1:
            end = i if i != n - 1 else n
            for t in range(start + window, end):
                x_win = under_rets[start + t - window:t]
                y_win = mid_rets[start + t - window:t]

                valid = (x_win < 0) & (~np.isnan(y_win))
                if np.sum(valid) >= min_points:
                    x_valid = x_win[valid]
                    y_valid = y_win[valid]
                    x_c = x_valid - x_valid.mean()
                    y_c = y_valid - y_valid.mean()
                    var = np.dot(x_c, x_c)
                    beta = np.dot(x_c, y_c) / var if var > 0 else np.nan
                    bear_beta[start + t] = beta
            current = contract_indices[i]
            start = i

    return bear_beta

def apply_bear_beta(df, window=16, min_points=3):
    df = df.sort_values(['contract_id', df.index.name]).copy()
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    df['mid_ret'] = df.groupby('contract_id')['mid_price'].pct_change()
    under_ret = df.groupby(df.index)['open'].first().pct_change()
    df['under_ret'] = df.index.map(under_ret)

    contract_codes = pd.factorize(df['contract_id'])[0].astype(np.int32)

    bear_beta = compute_bear_beta_numba(
        contract_codes,
        df['mid_ret'].values.astype(np.float64),
        df['under_ret'].values.astype(np.float64),
        window,
        min_points,
    )

    df['bear_beta'] = bear_beta
    df = df.drop(columns=['mid_price', 'mid_ret', 'under_ret'])
    return df

@njit
def compute_beta_numba(contract_indices, mid_rets, under_rets, window, min_points):
    n = len(contract_indices)
    beta = np.full(n, np.nan)

    current = contract_indices[0]
    start = 0

    for i in range(1, n):
        if contract_indices[i] != current or i == n - 1:
            end = i if i != n - 1 else n
            for t in range(start + window, end):
                x_win = under_rets[start + t - window:t]
                y_win = mid_rets[start + t - window:t]

                valid = (x_win > 0) & (~np.isnan(y_win))  # <-- upside moves only
                if np.sum(valid) >= min_points:
                    x_valid = x_win[valid]
                    y_valid = y_win[valid]
                    x_c = x_valid - x_valid.mean()
                    y_c = y_valid - y_valid.mean()
                    var = np.dot(x_c, x_c)
                    beta_val = np.dot(x_c, y_c) / var if var > 0 else np.nan
                    beta[start + t] = beta_val
            current = contract_indices[i]
            start = i

    return beta

def apply_beta(df, window=16, min_points=3):
    df = df.sort_values(['contract_id', df.index.name]).copy()
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    df['mid_ret'] = df.groupby('contract_id')['mid_price'].pct_change()

    under_ret = df.groupby(df.index)['open'].first().pct_change()
    df['under_ret'] = df.index.map(under_ret)

    contract_codes = pd.factorize(df['contract_id'])[0].astype(np.int32)

    beta_vals = compute_beta_numba(
        contract_codes,
        df['mid_ret'].values.astype(np.float64),
        df['under_ret'].values.astype(np.float64),
        window,
        min_points,
    )

    df['beta'] = beta_vals
    df = df.drop(columns=['mid_price', 'mid_ret', 'under_ret'], axis=1)
    return df

def compute_normalized_momentum(df):
    df = df.copy()
    df = df.sort_index()

    # Lags to compute
    lags = [1, 2, 3, 4, 5, 10, 15]

    # Group by unique contract (strike + expire_date)
    grouped = df.groupby(['strike', 'expire_date'])

    # Compute normalized momentum for each lag
    for lag in lags:
        col_name = f'norm_mom_{lag}'
        df[col_name] = grouped['open'].transform(lambda x: (x - x.shift(lag)) / x.shift(lag))

    return df

def compute_ps_liquidity(df: pd.DataFrame, window: int = 16) -> pd.DataFrame:
    df = df.copy()

    # Midquote
    df["midquote"] = (df["bid"] + df["ask"]) / 2

    # Group contracts
    df["contract_id"] = df.groupby(["strike", "expire_date"], sort=False).ngroup()

    # Prepare output array
    psliq = np.full(len(df), np.nan)

    # Work on NumPy arrays for speed
    midquote = df["midquote"].values
    volume = df["delta_volume"].values
    contract_ids = df["contract_id"].values

    # Loop over unique contracts
    for cid in np.unique(contract_ids):
        idx = np.where(contract_ids == cid)[0]

        if len(idx) < window:
            continue

        mid = midquote[idx]
        vol = volume[idx]

        # Compute forward return
        ret = np.full_like(mid, np.nan, dtype=np.float64)
        
        # 'ret[:-1] = (mid[1:] - mid[:-1]) / mid[:-1]' is
        # substituted by the following 3 lines to avoid zero div
        # but it might understate return calculations
        ret[:-1] = 0.0  # initialize safely
        safe = mid[:-1] != 0.0
        ret[:-1][safe] = (mid[1:][safe] - mid[:-1][safe]) / mid[:-1][safe]

        # Compute flow proxy
        flow = np.full_like(mid, np.nan, dtype=np.float64)
        flow[1:] = np.sign(ret[:-1]) * np.sqrt(np.maximum(vol[:-1], 0))

        # Rolling regression
        for i in range(window, len(idx)):
            y = ret[i - window + 1:i + 1]
            x = flow[i - window + 1:i + 1]

            if np.any(np.isnan(y)) or np.any(np.isnan(x)):
                continue

            x_mat = np.column_stack((np.ones_like(x), x))
            beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
            psliq[idx[i]] = beta[1]

    df["psliq"] = psliq
    df.drop(columns=["contract_id", "midquote"], inplace=True)
    return df

@njit
def rolling_kurtosis_numba(arr, window):
    n = len(arr)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        window_slice = arr[i - window + 1:i + 1]
        mean = np.mean(window_slice)
        m2 = np.mean((window_slice - mean) ** 2)
        m4 = np.mean((window_slice - mean) ** 4)
        if m2 != 0:
            result[i] = m4 / (m2 ** 2)
        else:
            result[i] = 0
    return result

def fast_grouped_kurtosis(df, window=16):
    results = []
    for _, group in df.groupby(['strike', 'expire_date']):
        arr = group['open'].to_numpy()
        kurt_vals = rolling_kurtosis_numba(arr, window)
        group = group.copy()
        group['historic_kurtosis'] = kurt_vals
        results.append(group)
    return pd.concat(results)

@njit
def rolling_skew_numba(arr, window):
    n = len(arr)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        window_data = arr[i - window + 1:i + 1]
        mean = np.mean(window_data)
        std = np.std(window_data)
        if std == 0:
            result[i] = 0.0
        else:
            result[i] = np.mean(((window_data - mean) / std) ** 3)
    return result


def fast_grouped_skewness(df, window=16):
    results = []

    for (strike, expiry), group in df.groupby(['strike', 'expire_date']):
        open_vals = group['open'].values.astype(np.float64)
        skew_vals = rolling_skew_numba(open_vals, window)
        result = group.copy()
        result['historic_skewness'] = skew_vals
        results.append(result)
    return pd.concat(results)

def compute_historic_volatility(df, window=16):
    # Ensure datetime index is sorted
    df = df.sort_index()
    
    # Group by option contract (strike, expire_date)
    def compute_group_vol(group):
        # Compute log returns of "open" price
        log_returns = np.log(group['open'] / group['open'].shift(1))
        # Rolling standard deviation over the window
        group['historic_volatility'] = log_returns.rolling(window=window).std()
        return group

    result = df.groupby(['strike', 'expire_date'], group_keys=False).apply(compute_group_vol)
    return result

@njit
def rolling_idio_vol(y, x, window):
    n = len(y)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        y_win = y[i - window + 1:i + 1]
        x_win = x[i - window + 1:i + 1]

        if np.any(np.isnan(y_win)) or np.any(np.isnan(x_win)):
            continue

        # Manually build design matrix with intercept (2D)
        X = np.empty((window, 2))
        X[:, 0] = 1.0           # intercept
        X[:, 1] = x_win         # regressor

        # OLS beta = (X'X)^-1 X'y
        XtX = X.T @ X
        if np.linalg.cond(XtX) > 1 / np.finfo(XtX.dtype).eps:
            continue
        Xty = X.T @ y_win
        beta = np.linalg.solve(XtX, Xty)

        residuals = y_win - X @ beta
        result[i] = np.std(residuals)

    for i in range(n):
        if np.isnan(result[i]):
            result[i] = 0.0

    return result

def compute_idiosyncratic_vol(df, window=16):
    df = df.sort_index()

    out_frames = []

    for _, group in df.groupby(['strike', 'expire_date']):
        y = group['option_return'].values
        x = group['open_return'].values
        vol = rolling_idio_vol(y, x, window)

        group = group.copy()
        group['idiosyncratic_vol'] = vol
        out_frames.append(group)

    return pd.concat(out_frames)

def compute_illiq(group, window=16):
    group = group.sort_index().copy()

    nan_ask_mask = group[pd.isna(group["ask"])].index
    zero_ask_mask = group[group["ask"] == 0].index

    # Safe log transform
    group.loc[group['ask'] == 0, 'ask'] = np.nan
    group['log_ask'] = np.log(group['ask'])
    group.loc[pd.isna(group['log_ask']), 'log_ask'] = 0

    # Δp_t
    group['delta_p'] = group['log_ask'].diff()

    # Δp_{t-1}
    group['delta_p_lag1'] = group['delta_p'].shift(1)

    # Rolling covariance between delta_p_lag1 (t-1) and delta_p (t)
    group['illiq'] = -group['delta_p'].rolling(window=window, min_periods=window)\
                                   .cov(group['delta_p_lag1'])
    
    group = group.drop(columns=["log_ask", "delta_p", "delta_p_lag1"])

    group.loc[nan_ask_mask, "ask"] = np.nan
    group.loc[zero_ask_mask, "ask"] = 0

    return group

def rank_last_normalized(x, window=16):
    return (np.argsort(np.argsort(x))[-1]) / (window - 1)

def apply_rank(group, window=16):
    result = group['iv'].rolling(window=window, min_periods=window).apply(rank_last_normalized, raw=True)
    return result

def compute_iv_minus_realized_vol(group, window=16):
    group = group.sort_values('datetime')
    
    # Calculate log returns of the 'ask' price

    group['log_return'] = np.log(group['ask'].where(group['ask'] > 0)).diff()
    group['log_return'].fillna(0)
    
    # Realized volatility over a 16-sample window (using std dev of log returns)
    group['realized_vol'] = group['log_return'].rolling(window=window).std()

    # Implied minus realized
    group['iv_minus_realized'] = group['realized_vol'] - group['iv']
    
    # drop tmp cols
    group = group.drop(columns=["log_return", "realized_vol"])
    return group

def compute_iv_minus_realized_vol_ratio(group, window=16):
    group = group.sort_values('datetime')
    
    # Calculate log returns of the 'ask' price

    group['log_return'] = np.log(group['ask'].where(group['ask'] > 0)).diff()
    group['log_return'].fillna(0)
    
    # Realized volatility over a 16-sample window (using std dev of log returns)
    group['realized_vol'] = group['log_return'].rolling(window=window).std()

    # Implied minus realized
    group['iv_minus_realized_ratio'] = np.where(
        (group['realized_vol'] == 0),
        0,
        group['iv'] / group['realized_vol']
    )
    
    # drop tmp cols
    group = group.drop(columns=["log_return", "realized_vol"])
    return group

def compute_n_degree(df):

    # Assuming df is your DataFrame
    df = df.copy()

    # Convert datetime index and expire_date to datetime if not already
    df.index = pd.to_datetime(df.index)
    df['expire_datetime'] = pd.to_datetime(df['expire_date'].astype(str) + ' 16:00:00')

    # Step 1: Compute time to expiration (τ) in days
    df['tau'] = (df['expire_datetime'] - df.index).dt.total_seconds() / (24 * 3600)

    # Step 2: Group by datetime and expiration to find ATM IV
    def get_atm_iv(group):
        # Find strike closest to the 'open' price
        atm_option = group.iloc[(group['strike'] - group['open']).abs().argsort()[:1]]
        atm_iv = atm_option['iv'].values[0]
        return pd.Series([atm_iv] * len(group), index=group.index)

    # df['atm_iv'] = df.groupby([df.index, 'expire_date']).apply(get_atm_iv).reset_index(level=[0,1], drop=True)
    df = df.reset_index().rename(columns={'index': 'datetime'})
    df['atm_iv'] = df.groupby(['datetime', 'expire_date']).apply(get_atm_iv).reset_index(drop=True)
    df = df.set_index('datetime')

    # Step 3: Compute standardized moneyness
    df['m_degree'] = np.log(df['strike'] / df['open']) / (np.sqrt(df['tau']) * df['atm_iv'])

    df = df.drop(columns=["expire_datetime", "tau", "atm_iv"])

    return df

def compute_pfht(df, window=16):
    df = df.sort_index()

    # Calculate returns per contract
    df['return'] = df.groupby(['strike', 'expire_date'])['ask'].pct_change()

    results = []

    # Process each contract separately for efficiency
    for _, group in df.groupby(['strike', 'expire_date']):
        group = group.copy()
        group['pzeros'] = (group['return'] == 0).astype(int)

        # Rolling zero return ratio
        pzeros = group['pzeros'].rolling(window=window, min_periods=window).mean()

        # Rolling standard deviation
        sigma = group['return'].rolling(window=window, min_periods=window).std()

        # Vectorized pfht computation
        quantile = norm.ppf((1 + pzeros) / 2)
        pfht = 2 * sigma * quantile

        group['pfht'] = pfht
        results.append(group)

    # Combine all processed groups
    df_result = pd.concat(results).sort_index()

    df_result.drop(columns=["return", "pzeros"])
    return df_result

def compute_pifht(df, window=16):
    df = df.sort_index()

    # Calculate returns per contract
    df['return'] = df.groupby(['strike', 'expire_date'])['ask'].pct_change()

    results = []

    # Process each contract separately for efficiency
    for _, group in df.groupby(['strike', 'expire_date']):
        group = group.copy()
        group['pzeros'] = (group['return'] == 0).astype(int)

        # Rolling zero return ratio
        pzeros = group['pzeros'].rolling(window=window, min_periods=window).mean()

        # Rolling standard deviation
        sigma = group['return'].rolling(window=window, min_periods=window).std()

        # Vectorized pfht computation
        quantile = norm.ppf((1 + pzeros) / 2)
        pfht = 2 * sigma * quantile

        # avg dvol
        q_bar = group['dvol'].rolling(window=window, min_periods=window).mean()
        
        group['pifht'] = np.where((q_bar == 0) & (~pd.isna(pfht)), 0, pfht / q_bar)

        results.append(group)

    # Combine all processed groups
    df_result = pd.concat(results).sort_index()

    df_result.drop(columns=["return"])
    return df_result

def compute_piroll(group, window=16):

    group = group.sort_index().copy()

    # Safe log transform
    group.loc[group['ask'] <= 0, 'ask'] = np.nan
    group['log_ask'] = np.log(group['ask'])
    group.loc[pd.isna(group['log_ask']), 'log_ask'] = 0

    # Δp_t
    group['delta_p'] = group['log_ask'].diff()

    # Δp_{t-1}
    group['delta_p_lag1'] = group['delta_p'].shift(1)

    # Rolling covariance between delta_p_lag1 (t-1) and delta_p (t)
    group['illiq'] = -group['delta_p'].rolling(window=window, min_periods=window).cov(group['delta_p_lag1'])

    group["roll"] = 0.0
    # group["roll"] = group["roll"].astype(float)
    mask = group["illiq"] >= 0
    group.loc[mask, "roll"] = 2 * np.sqrt(group.loc[mask, "illiq"])

    # Q bar
    q_bar = group['dvol'].rolling(window=window, min_periods=window).mean()

    group["piroll"] = np.where(q_bar == 0, 0, group["roll"] / q_bar)

    group = group.drop(columns=['log_ask', 'delta_p', 'delta_p_lag1', 'illiq', 'roll'])

    return group

def compute_retvol(group, window=16):

    group = group.sort_index().copy()

    # Safe log transform
    group.loc[group['ask'] <= 0, 'ask'] = np.nan
    group['log_ask'] = np.log(group['ask'])
    group.loc[pd.isna(group['log_ask']), 'log_ask'] = 0

    # Δp_t
    group['delta_p'] = group['log_ask'].diff()

    group['retvol'] = group['delta_p'].rolling(window=window, min_periods=window).std()
    
    group = group.drop(columns=['log_ask', 'delta_p'])

    return group

def compute_std_dolvol(group, window=16):

    group = group.sort_index().copy()

    group["dolvol"] = group["underlying_volume"] * group["open"]

    group["std_dolvol"] = group["dolvol"].rolling(window=window, min_periods=window).std()

    group = group.drop(columns=['dolvol'])

    return group

def compute_std_amihud(group, window=16):

    group = group.sort_index().copy()

    group["std_amihud"] = group["amihudilliq"].rolling(window=window, min_periods=window).std()

    return group

def compute_volga(group):

    group["vega_shift"] = group["vega"].shift(1)
    group["iv_shift"] = group["iv"].shift(1)
    group["delta_vega"] = group["vega"] - group["vega_shift"]
    group["delta_iv"] = group["iv"] - group["iv_shift"]
    group["volga"] = group["delta_vega"] / group["delta_iv"]
    group["scaled_volga"] = group["volga"] / group["open"]
    
    group = group.drop(columns=["vega_shift", "iv_shift", "delta_vega", "delta_iv", "volga"])
    return group

def compute_zerotrade(group, window=16):

    group["zerotrade"] = (group["volume"] == 0).rolling(window=window, min_periods=window).sum()

    return group