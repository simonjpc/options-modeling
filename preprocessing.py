import numpy as np
import pandas as pd
from numba import njit
from utils import build_dict, isolate_option_types
from joblib import Parallel, delayed

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


def create_option_dataset_full(df, n=6, label_column='label'):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    expiration_times = pd.to_datetime(df['expire_date']) + pd.Timedelta(hours=16)
    df['time_to_expiry'] = (expiration_times - df.index).dt.total_seconds() / 3600

    time_series_cols = ['open', 'iv', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho', 'volume']
    static_cols = ['strike', 'expire_date', 'time_to_expiry']
    label_cols = [label_column, 'percent_increase', 'hours_to_max']

    X_rows = []
    y_rows = []

    grouped = df.groupby(['strike', 'expire_date'], sort=False)

    for _, group in grouped:
        if len(group) < 16:
            continue

        group = group.sort_index()

        ts = group[time_series_cols].values
        static = group[static_cols].values
        labels = group[label_cols].values
        datetimes = group.index.values

        T = len(group)
        num_windows = T - 15

        idx = np.arange(16)[None, :] + np.arange(num_windows)[:, None]  # t0 to t-15
        ts_windows = ts[idx]  # (num_windows, 16, F)

        # Special treatment for 'open': all t0 to t-15
        open_windows = ts_windows[:, :, 0]  # open is feature 0, shape (num_windows, 16)

        # For others: keep t0 to t-5, t-10, t-15
        selected_idx = [0,1,2,3,4,5,10,15]
        other_features_windows = ts_windows[:, selected_idx, 1:]  # exclude 'open'

        open_flat = open_windows  # no reshape

        # Compute open_change_tN = (open_t-(N-1) - open_t-N) / open_t-N
        open_change = (open_flat[:, :-1] - open_flat[:, 1:]) / open_flat[:, 1:]
        # Rename columns: open_change_t0 corresponds to change from t0 to t-1
        open_change_flat = open_change  # shape (num_windows, 15)

        other_flat = other_features_windows.reshape(num_windows, -1)

        # ts_flat = np.concatenate([open_flat, other_flat], axis=1)
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
    # open_col_names = [f'open_t{-i}' for i in range(0, 16)]  # open_t0 to open_t-15
    # feature_times = ['t0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-10', 't-15']
    # other_col_names = [f'{col}_{t}' for t in feature_times for col in time_series_cols[1:]]
    open_col_names = [f'open_t{-i}' for i in range(0, 16)]
    open_change_col_names = [f'open_change_t{-i}' for i in range(0, 15)]
    feature_times = ['t0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-10', 't-15']
    other_col_names = [f'{col}_{t}' for t in feature_times for col in time_series_cols[1:]]

    # col_names = ['datetime', 'strike', 'expire_date', 'time_to_expiry'] + open_col_names + other_col_names
    col_names = ['datetime', 'strike', 'expire_date', 'time_to_expiry'] + open_col_names + open_change_col_names + other_col_names

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

def add_advanced_features(X: pd.DataFrame, n=6):
    X = X.copy()
    
    # Base features
    base_features = ['open', 'iv', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho', 'volume']
    time_steps = ['t0', 't-1', 't-2', 't-3', 't-4', 't-5', 't-10', 't-15']

    # Pre-extract necessary arrays
    data_all = {feature: X[[f'{feature}_{step}' for step in time_steps]].to_numpy(dtype=np.float32)
                for feature in base_features}

    all_new_features = {}

    for feature, data in data_all.items():
        # Slopes between t and (t-1, t-2, t-5, t-10, t-15)
        for idx, step in enumerate([1, 2, 5, 6, 7]):  # Corresponds to t-1, t-2, t-5, t-10, t-15
            diff = (data[:, 0] - data[:, step])  # value_t - value_t-k
            delta_time = step  # step corresponds to the distance in steps
            slope = diff / delta_time
            all_new_features[f'{feature}_slope_t_t-{[1,2,5,10,15][idx]}'] = slope

    # Add moneyness (strike relative to open_t0)
    open_t0 = X['open_t0'].astype(np.float32).values
    strike = X['strike'].astype(np.float32).values
    moneyness = strike / open_t0
    all_new_features['moneyness'] = moneyness

    X = X.assign(**all_new_features)

    # Drop raw open_t* columns and keep open_change_t* instead
    open_cols_to_drop = [f'open_t{-i}' for i in range(0, 16)]
    X = X.drop(columns=open_cols_to_drop, errors='ignore')

    return X
