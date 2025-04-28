from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import lightgbm as lgb
from constants import CATEGORICAL_COLS
from evaluation import simulate_strategy_corrected
from sklearn.metrics import average_precision_score, classification_report, precision_score, recall_score

def time_series_cv(X, y, n_splits=5, threshold=0.5, starting_capital=1000, nb_contracts=1, results={}):
    if n_splits == 2:
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X) * 0.3))
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)

    capital_over_folds = []
    final_capitals = []
    all_trades = []
    inspect_info_over_folds = {}
    models = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n🌀 Fold {fold_idx+1}/{n_splits}")

        # Prepare data
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

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
            required_gap_datetime = latest_label1_datetime + pd.Timedelta(hours=latest_label1_hours_to_max + 4)  # 4h = 16 timesteps
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

        # FILTER TRAINING DATA
        expiry_mask = X_train['time_to_expiry'] < 1500
        strike_mask = X_train['strike'] < 1.11 * X_train['open_t0']
        valid_mask = expiry_mask & strike_mask
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        labels_distribution = y_train["target"].value_counts(normalize=True)

        if labels_distribution[1] < 0.01:
            continue

        # Class imbalance handling
        pos_weight = (y_train['target'] == 0).sum() / (y_train['target'] == 1).sum()

        lgb_train = lgb.Dataset(X_train.drop(columns=['datetime', 'expire_date']), label=y_train['target'], categorical_feature=CATEGORICAL_COLS)
        lgb_test = lgb.Dataset(X_test.drop(columns=['datetime', 'expire_date']), label=y_test['target'], categorical_feature=CATEGORICAL_COLS)

        # Train model
        model = lgb.train(
            {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'scale_pos_weight': pos_weight,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'seed': 42,
            },
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            num_boost_round=1600,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)],
        )
        models.append(model)

        # Predict
        y_pred_proba = model.predict(X_test.drop(columns=['datetime', 'expire_date']))

        # FILTER PREDICTIONS BASED ON CONDITIONS
        invalid_mask = (X_test['time_to_expiry'] >= 1500) | (X_test['strike'] >= 1.11 * X_test['open_t0'])
        y_pred_proba[invalid_mask.values] = 0.0

        avg_prec = average_precision_score(y_test['target'], y_pred_proba)

        y_pred = (y_pred_proba >= threshold).astype(int)

        final_capital, cap_history, trades, df_preds = simulate_strategy_corrected(
            X_test,
            y_test,
            model,
            threshold=threshold,
            starting_capital=starting_capital,
            nb_contracts=nb_contracts,
        )
        print(f"💰 Final capital: ${final_capital:.2f}")

        results[f"01_{fold_idx}"] = {
            "classif_report": classification_report(y_test['target'], y_pred, digits=3),
            "precision": precision_score(y_test['target'], y_pred, pos_label=1),
            "recall": recall_score(y_test['target'], y_pred, pos_label=1),
            "final_capital": final_capital,
            "trade_log": trades,
            "capital_history": cap_history,
            "df_preds": df_preds,
        }

        final_capitals.append(final_capital)
        capital_over_folds.append(cap_history)
        trades['fold'] = fold_idx
        all_trades.append(trades)

        inspect_info_over_folds[fold_idx] = {
            "X_test_cutted": X_test[["datetime", "strike", "expire_date", "time_to_expiry", "open_t0"]],
            "predictions": pd.DataFrame(y_pred_proba, columns=["pred_proba"]),
            "labels": y_test,
        }
    if len(all_trades):
        trades_df = pd.concat(all_trades)
    else:
        trades_df = pd.DataFrame()
    avg_final_capital = np.mean(final_capitals)

    print("\n====================")
    print(f"Average Final Capital: ${avg_final_capital:.2f}")
    print("====================")

    return final_capitals, capital_over_folds, trades_df, inspect_info_over_folds, models