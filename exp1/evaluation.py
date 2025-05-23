import numpy as np
import pandas as pd
from collections import deque
from wrapper import DummyModel
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, classification_report
from statistic_measure import (
    prediction_entropy, filter_far_from_centroid, filter_low_support_confident_samples, train_isolation_forest
)

def simulate_strategy_corrected(
        X_test, y_test, model,
        X_train_pos_reference=None,
        X_train_neg_reference=None,
        q_low=0.01, q_high=0.99,
        threshold=0.5, starting_capital=1000, nb_contracts=1
    ):
    df = X_test.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df[['target', 'percent_increase', 'hours_to_max']] = y_test[['target', 'percent_increase', 'hours_to_max']].values

    df = df.sort_values('datetime')

    feature_cols = [
        col for col in df.columns if col not in [
            'datetime', 'expire_date', 'target',
            'percent_increase', 'hours_to_max',
            'pred_proba', 'prediction', 'day_of_week',
            # below for testing of distance
            'dvol', 'baspread', 'midpoint', 'underlying_volume', 'idiosyncratic_vol',
            'strike', 'time_to_expiry', 'iv', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho', 'volume', 'iv_t0', 'bid_t0', 'ask_t0', 'delta_t0', 'gamma_t0', 'theta_t0', 'vega_t0', 'rho_t0', 'volume_t0', 'iv_t-1', 'bid_t-1', 'ask_t-1', 'delta_t-1', 'gamma_t-1', 'theta_t-1', 'vega_t-1', 'rho_t-1', 'volume_t-1', 'iv_t-2', 'bid_t-2', 'ask_t-2', 'delta_t-2', 'gamma_t-2', 'theta_t-2', 'vega_t-2', 'rho_t-2', 'volume_t-2', 'iv_t-3', 'bid_t-3', 'ask_t-3', 'delta_t-3', 'gamma_t-3', 'theta_t-3', 'vega_t-3', 'rho_t-3', 'volume_t-3', 'iv_t-4', 'bid_t-4', 'ask_t-4', 'delta_t-4', 'gamma_t-4', 'theta_t-4', 'vega_t-4', 'rho_t-4', 'volume_t-4', 'iv_t-5', 'bid_t-5', 'ask_t-5', 'delta_t-5', 'gamma_t-5', 'theta_t-5', 'vega_t-5', 'rho_t-5', 'volume_t-5', 'iv_t-10', 'bid_t-10', 'ask_t-10', 'delta_t-10', 'gamma_t-10', 'theta_t-10', 'vega_t-10', 'rho_t-10', 'volume_t-10', 'iv_t-15', 'bid_t-15', 'ask_t-15', 'delta_t-15', 'gamma_t-15', 'theta_t-15', 'vega_t-15', 'rho_t-15', 'volume_t-15', 'iv_t-20', 'bid_t-20', 'ask_t-20', 'delta_t-20', 'gamma_t-20', 'theta_t-20', 'vega_t-20', 'rho_t-20', 'volume_t-20', 'iv_t-23', 'bid_t-23', 'ask_t-23', 'delta_t-23', 'gamma_t-23', 'theta_t-23', 'vega_t-23', 'rho_t-23', 'volume_t-23',
        ]
    ]
    # if X_train_pos_reference is not None:
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X_train_pos_reference[feature_cols])
    #     X_train_pos_reference_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X_train_pos_reference.index)

    # Predict all
    y_pred_proba = model.predict(df.drop(columns=['strike', 'datetime', 'expire_date', 'target', 'percent_increase', 'hours_to_max']))

    # Mask invalid test points
    # invalid_mask = (df['time_to_expiry'] >= 1500) | (df['strike'] >= 1.11 * df['open_t0'])
    invalid_mask = (df['time_to_expiry'] >= 1500) | (df['moneyness'] >= 1.11)
    y_pred_proba[invalid_mask.values] = 0.0

    df['pred_proba'] = y_pred_proba
    df['prediction'] = (y_pred_proba >= threshold).astype(int)

    capital = starting_capital
    capital_history = []
    trade_log = []
    invested_contracts = set()
    pending_returns = deque()

    for current_time, sub_df in df.groupby('datetime'):

        # 🧠 Optional drift detection before making predictions
        if X_train_pos_reference is not None: # and X_train_neg_reference is not None:

            low_pos = X_train_pos_reference[feature_cols].quantile(q_low)
            high_pos = X_train_pos_reference[feature_cols].quantile(q_high)
            outside_pos = ((sub_df[feature_cols] < low_pos) | (sub_df[feature_cols] > high_pos)).any(axis=1)

            drift_mask = outside_pos
            
            # # Drop samples outside (low, high) quantiles
            # if idx_ == 0:
            sub_df_quantiles = sub_df[~drift_mask].copy()
            sub_df = sub_df_quantiles

            # # Drop samples with high prediction entropy
            # elif idx_ == 1:
            #     sub_df_high_entropy = sub_df[prediction_entropy(sub_df["pred_proba"]) < 0.6].copy()
            #     sub_df = sub_df_high_entropy

            # # Drop samples too far from centroid
            # elif idx_ == 2:
            # sub_df_scaled = pd.DataFrame(scaler.transform(sub_df[feature_cols]), columns=feature_cols, index=sub_df.index)
            # sub_df_ff_centroid = filter_far_from_centroid(
            #     sub_df_scaled, X_train_pos_reference_scaled, feature_cols, distance_thresh=50.0
            # ).copy()
            # sub_df_ = sub_df_ff_centroid

            # # Drop samples with low similarity support from confident positives
            # elif idx_ == 3:
            #     sub_df_low_similarity = filter_low_support_confident_samples(sub_df, X_train_pos_reference, feature_cols).copy()
            #     sub_df = sub_df_low_similarity

            # # Drop outliers based on Isolation Forest
            # elif idx_ == 4:
            #     iso_model = train_isolation_forest(X_train_pos_reference, feature_cols)
            #     preds = iso_model.predict(sub_df[feature_cols])
            #     sub_df_inliers = sub_df[preds == 1].copy()
            #     sub_df = sub_df_inliers

            if len(sub_df) == 0:
                capital_history.append((current_time, capital))
                continue

        # Release matured trades
        while pending_returns and pending_returns[0][0] <= current_time:
            _, cap_delta = pending_returns.popleft()
            capital += cap_delta

        capital_history.append((current_time, capital))

        # Top 3 predictions at this timestamp
        # sub_df = sub_df.loc[sub_df_.index]
        candidates = sub_df[sub_df['prediction'] == 1]
        candidates = candidates.sort_values('pred_proba', ascending=False).head(3)

        for _, row in candidates.iterrows():
            option_id = (row['strike'], row['expire_date'])

            if option_id in invested_contracts:
                continue

            option_price = row["ask_t0"] * 100 * nb_contracts
            if capital < option_price:
                continue

            invested_contracts.add(option_id)
            capital -= option_price

            if row['target'] == 1:
                gain = min(row['percent_increase'], 2.0)  # NEW CAP
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

    # Final cleanup of pending returns
    last_time = df['datetime'].max()
    while pending_returns:
        release_time, cap_delta = pending_returns.popleft()
        if release_time > last_time:
            capital_history.append((release_time, capital + cap_delta))
        capital += cap_delta

    capital_history = pd.DataFrame(capital_history, columns=['datetime', 'capital'])
    trade_log_df = pd.DataFrame(trade_log)

    return capital, capital_history, trade_log_df, df


def simulate_strategy_inliers(X_test, y_test, model, anomaly_scores, anomaly_threshold, threshold=0.5, starting_capital=1000, nb_contracts=1):
    df = X_test.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df[['target', 'percent_increase', 'hours_to_max']] = y_test[['target', 'percent_increase', 'hours_to_max']].values

    df = df.sort_values('datetime')

    # Predict all
    y_pred_proba = model.predict(df.drop(columns=['strike', 'datetime', 'expire_date', 'target', 'percent_increase', 'hours_to_max']))

    # Mask invalid test points
    # invalid_mask = (df['time_to_expiry'] >= 1500) | (df['strike'] >= 1.11 * df['open_t0'])
    invalid_mask = (df['time_to_expiry'] >= 1500) | (df['moneyness'] >= 1.11)
    y_pred_proba[invalid_mask.values] = 0.0

    for i, proba in enumerate(y_pred_proba):
        if proba >= threshold:
            if anomaly_scores[i] < anomaly_threshold:
                y_pred_proba[i] = 0.0  # Reject uncertain sample

    df['pred_proba'] = y_pred_proba
    df['prediction'] = (y_pred_proba >= threshold).astype(int)

    capital = starting_capital
    capital_history = []
    trade_log = []
    invested_contracts = set()
    pending_returns = deque()

    for current_time, sub_df in df.groupby('datetime'):
        # Release matured trades
        while pending_returns and pending_returns[0][0] <= current_time:
            _, cap_delta = pending_returns.popleft()
            capital += cap_delta

        capital_history.append((current_time, capital))

        # Top 3 predictions at this timestamp
        candidates = sub_df[sub_df['prediction'] == 1]
        candidates = candidates.sort_values('pred_proba', ascending=False).head(3)

        for _, row in candidates.iterrows():
            option_id = (row['strike'], row['expire_date'])

            if option_id in invested_contracts:
                continue

            option_price = row["ask_t0"] * 100 * nb_contracts
            if capital < option_price:
                continue

            invested_contracts.add(option_id)
            capital -= option_price

            if row['target'] == 1:
                gain = min(row['percent_increase'], 2.0)  # NEW CAP
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

    # Final cleanup of pending returns
    last_time = df['datetime'].max()
    while pending_returns:
        release_time, cap_delta = pending_returns.popleft()
        if release_time > last_time:
            capital_history.append((release_time, capital + cap_delta))
        capital += cap_delta

    capital_history = pd.DataFrame(capital_history, columns=['datetime', 'capital'])
    trade_log_df = pd.DataFrame(trade_log)

    return capital, capital_history, trade_log_df, df


def evaluate_month_with_existing_models(X_month, y_month, models, n_splits=5, threshold=0.8, starting_capital=1000, nb_contracts=2):
    
    X_month = X_month.sort_values("datetime")
    y_month = y_month.loc[X_month.index]

    if n_splits == 2:
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X_month) * 0.3))
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []
    fold_capitals = []
    all_trades = []
    all_capitals = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_month)):
        print(f"\n🔵 Month fold {fold_idx+1}/{n_splits}")

        X_train, X_test = X_month.iloc[train_idx].copy(), X_month.iloc[test_idx].copy()
        y_train, y_test = y_month.iloc[train_idx].copy(), y_month.iloc[test_idx].copy()

        # Find last training datetime
        last_train_datetime = X_train['datetime'].max()

        # Check if there are any label=1 in training set
        label1_train = y_train[y_train['target'] == 1]

        if not label1_train.empty:
            # Find the latest datetime among label 1
            latest_label1_idx = X_train.loc[label1_train.index]['datetime'].idxmax()
            latest_label1_datetime = X_train.loc[latest_label1_idx, 'datetime']
            latest_label1_hours_to_max = y_train.loc[latest_label1_idx, 'hours_to_max']

            # Required gap: from latest label 1 trade
            required_gap_datetime = max(
                latest_label1_datetime + pd.Timedelta(hours=latest_label1_hours_to_max + 4),
                last_train_datetime + pd.Timedelta(hours=latest_label1_hours_to_max + 4),
            )
        else:
            # No label 1 trades: fallback to last training point (maybe + 1 week) + 4h
            required_gap_datetime = last_train_datetime + pd.Timedelta(hours=4)

        # Now compare natural gap
        first_test_datetime = X_test['datetime'].min()

        if first_test_datetime < required_gap_datetime:
            # Need to filter test set
            X_test = X_test[X_test['datetime'] > required_gap_datetime]
            y_test = y_test.loc[X_test.index]

            if len(X_test) == 0:
                print(f"⚠️ Fold {fold_idx} skipped: no test data after required gap.")
                continue

        # Predict with each January model
        preds_per_model = []
        for model in models:
            if isinstance(model, tuple):
                model, best_threshold = model
            else:
                print(f"No best threshold found... Using default value {threshold}")
                best_threshold = threshold
            pred_proba = model.predict(X_test.drop(columns=['strike', 'datetime', 'expire_date'], errors='ignore'))

            # Mask invalid predictions
            # invalid_mask = (X_test['time_to_expiry'] >= 1500) | (X_test['strike'] >= 1.11 * X_test['open_t0'])
            invalid_mask = (X_test['time_to_expiry'] >= 1500) | (X_test['moneyness'] >= 1.11)
            pred_proba[invalid_mask.values] = 0.0

            preds_per_model.append(pred_proba)

        preds_per_model = np.vstack(preds_per_model)  # shape: (n_models, n_samples)
        avg_preds = preds_per_model.mean(axis=0)  # average across models

        # Binarize predictions
        y_pred = (avg_preds >= best_threshold).astype(int)

        # Metrics
        precision = precision_score(y_test['target'], y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test['target'], y_pred, pos_label=1, zero_division=0)

        # print(classification_report(y_test['target'], y_pred, digits=3))

        # Strategy simulation
        X_test_copy = X_test.copy()
        X_test_copy['datetime'] = pd.to_datetime(X_test_copy['datetime'])
        y_test_copy = y_test.copy()

        final_capital, capital_history, trade_log, _ = simulate_strategy_corrected(
            X_test_copy,
            y_test_copy,
            DummyModel(avg_preds),
            threshold=best_threshold,
            starting_capital=starting_capital,
            nb_contracts=nb_contracts,
        )

        print(f"💵 Final Capital for fold {fold_idx}: ${final_capital:.2f}")

        fold_metrics.append({
            'classification_report': classification_report(y_test['target'], y_pred, digits=3),
            'precision': precision,
            'recall': recall,
            'final_capital': final_capital,
        })
        fold_capitals.append(capital_history)
        trade_log['fold'] = fold_idx
        all_trades.append(trade_log)

        all_capitals.append(final_capital)

    avg_final_capital = np.mean(all_capitals)
    
    print("\n====================")
    print(f"Average Final Capital: ${avg_final_capital:.2f}")
    print("====================")

    if len(all_trades):
        trades_df = pd.concat(all_trades, ignore_index=True)
    else:
        trades_df = pd.DataFrame()
    return fold_metrics, fold_capitals, trades_df
