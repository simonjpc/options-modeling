import numpy as np
import pandas as pd
from numba import njit
from utils import build_dict, isolate_option_types
from joblib import Parallel, delayed

def prepare_labels(stock_price_data, options_chain_data):

    # raw_keys = options_chain_data[0]
    # options_hist_data = []
    # for raw_values in options_chain_data[1:]:
    #     element = build_dict(raw_keys, raw_values)
    #     call_element, put_element = isolate_option_types(element)
    #     options_hist_data.append(call_element)
    #     options_hist_data.append(put_element)
    # options_chain_data_hist = pd.DataFrame(options_hist_data)
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
        ["datetime", "strike", "iv", "expire_date", "bid", "ask", "delta", "gamma", "vega", "theta", "rho"]
    ]
    put_chain_data_hist_reduced = options_chain_data_hist_reduced.loc[
        options_chain_data_hist_reduced["type"] == "put",
        ["datetime", "strike", "iv", "expire_date", "bid", "ask", "delta", "gamma", "vega", "theta", "rho"]
    ]

    # stocks data manip
    stock_price_data_15min = stock_price_data.loc[stock_price_data["date"].apply(
        lambda x:(":30" in x[:-3]) or (":15" in x[:-3]) or (":00" in x[:-3]) or (":45" in x[:-3])
    ), ["date", "open"]]
    stock_price_data_15min = stock_price_data_15min.rename(columns={"date": "datetime"})
    stock_price_data_15min["datetime"] = stock_price_data_15min["datetime"].apply(lambda x: x[:-3])

    # reset indexes
    call_chain_data_hist_reduced = call_chain_data_hist_reduced.set_index("datetime")
    put_chain_data_hist_reduced = put_chain_data_hist_reduced.set_index("datetime")
    stock_price_data_15min = stock_price_data_15min.set_index("datetime")
    concat_call_options_chain_prices = call_chain_data_hist_reduced.join(stock_price_data_15min, how="left")
    concat_put_options_chain_prices = put_chain_data_hist_reduced.join(stock_price_data_15min, how="left")
    concat_call_options_chain_prices = concat_call_options_chain_prices[concat_call_options_chain_prices.index.map(lambda x: " 16:00" not in x)]
    df = concat_call_options_chain_prices.copy()
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
    return df_labeled


def tag_returns_process_group(strike, expire_date, group):
    group = group.sort_index()
    times = group.index
    asks = group['ask'].values
    bids = group['bid'].values
    labels = np.zeros(len(group), dtype=np.int8)
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
                labels[i] = 1
                percent_increase[i] = (np.max(bids[mask]) - ask_t) / ask_t
                hours_to_max[i] = (max_time - time_t).total_seconds() / 3600.0  # in hours

    group_result = group.copy()
    group_result['label'] = labels
    group_result['percent_increase'] = percent_increase
    group_result['hours_to_max'] = hours_to_max

    return group_result


def create_option_dataset_full(df, n=8, label_column='label'):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Compute time to expiry once
    expiration_times = pd.to_datetime(df['expire_date']) + pd.Timedelta(hours=16)
    df['time_to_expiry'] = (expiration_times - df.index).dt.total_seconds() / 3600

    time_series_cols = ['open', 'iv', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho']
    static_cols = ['strike', 'expire_date', 'time_to_expiry']
    label_cols = [label_column, 'percent_increase', 'hours_to_max']

    X_rows = []
    y_rows = []

    grouped = df.groupby(['strike', 'expire_date'], sort=False)

    for _, group in grouped:
        if len(group) < n:
            continue

        group = group.sort_index()

        ts = group[time_series_cols].values      # shape (T, F)
        static = group[static_cols].values       # shape (T, 3)
        labels = group[label_cols].values        # shape (T, 3)
        datetimes = group.index.values           # shape (T,)

        T = len(group)
        num_windows = T - n + 1

        # Build indices for windowing
        idx = np.arange(n)[None, :] + np.arange(num_windows)[:, None]  # shape (num_windows, n)
        ts_windows = ts[idx]                                            # shape (num_windows, n, F)
        ts_flat = ts_windows.reshape(num_windows, -1)                   # flatten to (num_windows, n*F)

        static_final = static[idx[:, -1]]                               # (num_windows, 3)
        labels_final = labels[idx[:, -1]]                               # (num_windows, 3)
        dates_final = datetimes[idx[:, -1]]                             # (num_windows,)

        # Stack everything into final rows
        X_group = np.column_stack([dates_final, static_final, ts_flat])
        y_group = labels_final

        X_rows.append(X_group)
        y_rows.append(y_group)

    # Final concatenation
    X_final = np.vstack(X_rows)
    y_final = np.vstack(y_rows)

    # Build column names
    ts_col_names = [f'{col}_t{-n+i+1}' for i in range(n) for col in time_series_cols]
    col_names = ['datetime', 'strike', 'expire_date', 'time_to_expiry'] + ts_col_names
    X = pd.DataFrame(X_final, columns=col_names)
    X['datetime'] = pd.to_datetime(X['datetime'])  # Convert back datetime column
    X["strike"] = X["strike"].astype(np.float32)
    X["time_to_expiry"] = X["time_to_expiry"].astype(np.float32)
    X[ts_col_names] = X[ts_col_names].astype(np.float32)

    y = pd.DataFrame(y_final, columns=["target", "percent_increase", "hours_to_max"])
    return X, y

def add_datetime_features(X):
    X = X.copy()
    
    datetime_features = pd.DataFrame({
        'hour': X['datetime'].dt.hour,
        'minute': X['datetime'].dt.minute,
        'day_of_week': X['datetime'].dt.dayofweek
    }, index=X.index)

    X = pd.concat([X, datetime_features], axis=1)
    X['day_of_week'] = X['day_of_week'].astype("category")
    return X


@njit(fastmath=True)
def compute_slopes(data, x, x_mean, x_var):
    n_samples, n_steps = data.shape
    slopes = np.empty(n_samples, dtype=np.float32)

    x = x.astype(np.float32)
    x_mean = np.float32(x_mean)
    x_var = np.float32(x_var)

    for i in range(n_samples):
        y = data[i]
        y_mean = np.float32(np.mean(y))
        diff_x = x - x_mean
        diff_y = y - y_mean
        slopes[i] = np.dot(diff_x, diff_y) / (n_steps * x_var)

    return slopes


@njit(fastmath=True)
def compute_derived_features(open_data, iv_data, strike):
    open_t0, open_start, open_half, open_end = open_data.T
    iv_t0, iv_start, iv_half, iv_end = iv_data.T

    n_samples = strike.shape[0]
    out = np.empty((10, n_samples), dtype=np.float32)

    for i in range(n_samples):
        s = strike[i] if strike[i] != 0 else 1
        out[0, i] = open_t0[i] / s

        out[1, i] = iv_t0[i] - iv_start[i]
        out[2, i] = iv_t0[i] / iv_start[i] if iv_start[i] != 0 else 0
        out[3, i] = (open_t0[i] - open_start[i]) / open_start[i] if open_start[i] != 0 else 0

        out[4, i] = iv_t0[i] - iv_half[i]
        out[5, i] = iv_t0[i] / iv_half[i] if iv_half[i] != 0 else 0
        out[6, i] = (open_t0[i] - open_half[i]) / open_half[i] if open_half[i] != 0 else 0

        out[7, i] = iv_t0[i] - iv_end[i]
        out[8, i] = iv_t0[i] / iv_end[i] if iv_end[i] != 0 else 0
        out[9, i] = (open_t0[i] - open_end[i]) / open_end[i] if open_end[i] != 0 else 0

    return out


def add_advanced_features(X: pd.DataFrame, n=8):
    X = X.copy()
    base_features = ['open', 'iv', 'bid', 'ask']
    suffixes = [f"_t{-n+i+1}" for i in range(n)]
    cols = {feature: [f"{feature}{s}" for s in suffixes] for feature in base_features}

    all_new_features = {}

    x = np.arange(n, dtype=np.float32)
    x_mean = x.mean()
    x_var = x.var()

    # Pre-extract all needed data at once
    data_all = {feature: X[cols[feature]].to_numpy(dtype=np.float32) for feature in base_features}

    for feature, data in data_all.items():
        all_new_features.update({
            f'{feature}_mean': data.mean(axis=1),
            f'{feature}_std': data.std(axis=1),
            f'{feature}_min': data.min(axis=1),
            f'{feature}_max': data.max(axis=1),
            f'{feature}_change': data[:, -1] - data[:, 0],
            f'{feature}_slope': compute_slopes(data, x, x_mean, x_var)
        })

    open_data = X[[f'open_t0', f'open_t{-n+1}', f'open_t{(-n+1)//2}', f'open_t-1']].to_numpy(dtype=np.float32)
    iv_data = X[[f'iv_t0', f'iv_t{-n+1}', f'iv_t{(-n+1)//2}', f'iv_t-1']].to_numpy(dtype=np.float32)
    strike = X['strike'].to_numpy(dtype=np.float32)

    derived = compute_derived_features(open_data, iv_data, strike)

    derived_feature_names = [
        'moneyness',
        'iv_skew', 'iv_ratio', 'stock_return',
        'iv_skew_half', 'iv_ratio_half', 'stock_return_half',
        'iv_skew_end', 'iv_ratio_end', 'stock_return_end'
    ]

    for name, arr in zip(derived_feature_names, derived):
        all_new_features[name] = arr

    X = X.assign(**all_new_features)

    return X
