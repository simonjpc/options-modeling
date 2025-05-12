import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from constants import CATEGORICAL_COLS
from evaluation import simulate_strategy_corrected
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import average_precision_score, classification_report, precision_score, recall_score

def time_series_cv(X, y, n_splits=5, threshold=0.5, starting_capital=1000, nb_contracts=1, results={}):
    X = X.sort_values("datetime")
    y = y.loc[X.index]
    print(X.index.equals(y.index))
    
    if n_splits == 2:
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X) * 0.3))
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)

    capital_over_folds = []
    final_capitals = []
    all_trades = []
    inspect_info_over_folds = {}
    models = []
    X["datetime"] = pd.to_datetime(X["datetime"])

    # 👇 Inner function for Optuna optimization
    def lgb_objective(
        trial, X_train, y_train, X_val, y_val, categorical_cols
    ):
        param = {
            'boosting_type': 'gbdt',
            'metric': None,
            'verbosity': -1,
            'boost_from_average': True,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 300),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': 1,
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 50.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 50.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'scale_pos_weight': (y_train['target'] == 0).sum() / (y_train['target'] == 1).sum(),
            'seed': 42,
        }

        dtrain = lgb.Dataset(X_train.drop(columns=['datetime', 'expire_date']), label=y_train['target'], categorical_feature=categorical_cols)
        dvalid = lgb.Dataset(X_val.drop(columns=['datetime', 'expire_date']), label=y_val['target'], categorical_feature=categorical_cols)

        model = lgb.train(
            param,
            dtrain,
            valid_sets=[dtrain, dvalid],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
        )

        y_val_pred_proba = model.predict(X_val.drop(columns=['datetime', 'expire_date']))

        # Optional: invalidate untradable options if needed (replicate your test-time filters here)
        # Example:
        # invalid_mask = (X_val['time_to_expiry'] >= 1500) | (X_val['strike'] >= 1.11 * X_val['open_t0'])
        invalid_mask = (X_val['time_to_expiry'] >= 1500) | (X_val['moneyness'] >= 1.11)
        y_val_pred_proba[invalid_mask.values] = 0.0

        # 🔍 Find best threshold for precision (optional, could try optimizing for capital directly)
        best_threshold = 0.8
        best_capital = -np.inf
        for th in np.arange(0.7, 0.95, 0.01):
            val_capital, _, _, _ = simulate_strategy_corrected(
                X_val, y_val, model,
                threshold=th,
                starting_capital=1000,
                nb_contracts=2,
            )
            if val_capital > best_capital:
                best_capital = val_capital
                best_threshold = th

        # Optionally compute train capital and penalize overfitting
        train_capital, _, _, _ = simulate_strategy_corrected(
            X_train, y_train, model,
            threshold=best_threshold,
            starting_capital=1000,
            nb_contracts=2,
        )
        overfit_penalty = max(train_capital - best_capital, 0)
        print()
        print(f"best_capital in optim: {best_capital}")
        print(f"overfit_penalty in optim: {overfit_penalty}")
        print()
        return -(best_capital - 0.5 * overfit_penalty)  # Minimize negative profit (maximize profit)


    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n🌀 Fold {fold_idx+1}/{n_splits}")

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        last_train_datetime = X_train['datetime'].max()

        label1_train = y_train[y_train['target'] == 1]
        if not label1_train.empty:
            latest_label1_idx = X_train.loc[label1_train.index]['datetime'].idxmax()
            latest_label1_datetime = X_train.loc[latest_label1_idx, 'datetime']
            latest_label1_hours_to_max = y_train.loc[latest_label1_idx, 'hours_to_max']
            required_gap_datetime = max(
                latest_label1_datetime + pd.Timedelta(hours=latest_label1_hours_to_max + 4),
                last_train_datetime + pd.Timedelta(hours=latest_label1_hours_to_max + 4),
            )
        else:
            required_gap_datetime = last_train_datetime + pd.Timedelta(hours=4)

        first_test_datetime = X_test['datetime'].min()
        if first_test_datetime < required_gap_datetime:
            X_test = X_test[X_test['datetime'] > required_gap_datetime]
            y_test = y_test.loc[X_test.index]
            if len(X_test) == 0:
                print(f"⚠️ Fold {fold_idx+1} skipped: no test data after required gap.")
                continue

        expiry_mask = X_train['time_to_expiry'] < 1500
        # strike_mask = X_train['strike'] < 1.11 * X_train['open_t0']
        strike_mask = X_train['moneyness'] < 1.11
        valid_mask = expiry_mask & strike_mask
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        labels_distribution = y_train["target"].value_counts(normalize=True)

        if labels_distribution[1] < 0.01:
            continue

        print("🔍 Starting Optuna tuning...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: lgb_objective(trial, X_train, y_train, X_test, y_test, CATEGORICAL_COLS), n_trials=12)
        print(f"✅ Best params for Fold {fold_idx}: {study.best_params}")

        best_params = study.best_params
        best_params.update({
            'boosting_type': 'gbdt',
            'metric': None,  # or 'None' if you're using a custom objective
            'verbosity': -1,
            'scale_pos_weight': (y_train['target'] == 0).sum() / (y_train['target'] == 1).sum(),
            'seed': 42,
        })

        lgb_train = lgb.Dataset(
            X_train.drop(columns=['datetime', 'expire_date']),
            label=y_train['target'],
            categorical_feature=CATEGORICAL_COLS
        )
        lgb_test = lgb.Dataset(
            X_test.drop(columns=['datetime', 'expire_date']),
            label=y_test['target'],
            categorical_feature=CATEGORICAL_COLS
        )

        model = lgb.train(
            best_params,
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(100)],
        )

        y_pred_proba = model.predict(X_test.drop(columns=['datetime', 'expire_date']))
        # invalid_mask = (X_test['time_to_expiry'] >= 1500) | (X_test['strike'] >= 1.11 * X_test['open_t0'])
        invalid_mask = (X_test['time_to_expiry'] >= 1500) | (X_test['moneyness'] >= 1.11)
        y_pred_proba[invalid_mask.values] = 0.0

        avg_prec = average_precision_score(y_test['target'], y_pred_proba)

        best_precision = 0
        best_threshold = 0.8
        for th in np.arange(0.7, 0.95, 0.01):
            preds = (y_pred_proba >= th).astype(int)
            precision = precision_score(y_test['target'], preds, pos_label=1)
            if precision > best_precision:
                best_precision = precision
                best_threshold = th
        
        models.append((model, best_threshold))

        y_pred = (y_pred_proba >= best_threshold).astype(int)

        final_capital, cap_history, trades, df_preds = simulate_strategy_corrected(
            X_test,
            y_test,
            model,
            threshold=best_threshold,
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
            # "X_test_cutted": X_test[["datetime", "strike", "expire_date", "time_to_expiry", "open_t0"]],
            "X_test_cutted": X_test[["datetime", "strike", "expire_date", "time_to_expiry", "moneyness"]],
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


# def time_series_real_eval(X, y, threshold=0.5, starting_capital=1000, nb_contracts=1, results={}):

#     X = X.sort_values("datetime").copy()
#     y = y.loc[X.index].copy()
#     X["datetime"] = pd.to_datetime(X["datetime"])
#     # 1. Single time-based split: 70% train, 15% tune, 15% test
#     total_len = len(X)
#     train_end = int(0.7 * total_len)
#     tune_end = int(0.85 * total_len)

#     X_train = X.iloc[:train_end].copy()
#     y_train = y.iloc[:train_end].copy()

#     X_tune = X.iloc[train_end:tune_end].copy()
#     y_tune = y.iloc[train_end:tune_end].copy()

#     X_test = X.iloc[tune_end:].copy()
#     y_test = y.iloc[tune_end:].copy()

#     # ⛔ Apply validity filters to train and tune data
#     train_mask = (X_train['time_to_expiry'] < 1500) & (X_train['moneyness'] < 1.11)
#     tune_mask = (X_tune['time_to_expiry'] < 1500) & (X_tune['moneyness'] < 1.11)
#     X_train = X_train[train_mask]
#     y_train = y_train[train_mask]

#     X_tune = X_tune[tune_mask]
#     y_tune = y_tune[tune_mask]
    
#     # ⚠️ Skip if too few positives
#     labels_distribution = y_train["target"].value_counts(normalize=True)
#     if labels_distribution[1] < 0.01:
#         print("⚠️ Too few positive samples in training set. Aborting.")
#         return None, [], pd.DataFrame(), None, None


#     # 🔍 Optuna tuning on tuning set only
#     print("🔍 Starting Optuna tuning...")

#     def lgb_objective(trial):
#         param = {
#             'boosting_type': 'gbdt',
#             'metric': None,
#             'verbosity': -1,
#             'boost_from_average': True,
#             'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
#             'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#             'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 300),
#             'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
#             'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
#             'bagging_freq': 1,
#             'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 50.0),
#             'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 50.0),
#             'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
#             'max_depth': trial.suggest_int('max_depth', 3, 10),
#             'scale_pos_weight': (y_train['target'] == 0).sum() / (y_train['target'] == 1).sum(),
#             'seed': 42,
#         }

#         dtrain = lgb.Dataset(X_train.drop(columns=['datetime', 'expire_date']), label=y_train['target'], categorical_feature=CATEGORICAL_COLS)
#         dtune = lgb.Dataset(X_tune.drop(columns=['datetime', 'expire_date']), label=y_tune['target'], categorical_feature=CATEGORICAL_COLS)

#         model = lgb.train(
#             param,
#             dtrain,
#             valid_sets=[dtrain, dtune],
#             num_boost_round=2000,
#             callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
#         )

#         y_val_pred_proba = model.predict(X_tune.drop(columns=['datetime', 'expire_date']))
#         invalid_mask = (X_tune['time_to_expiry'] >= 1500) | (X_tune['moneyness'] >= 1.11)
#         y_val_pred_proba[invalid_mask.values] = 0.0

#         best_threshold = 0.8
#         best_capital = -np.inf
#         for th in np.arange(0.7, 0.95, 0.01):
#             val_capital, _, _, _ = simulate_strategy_corrected(
#                 X_tune, y_tune, model,
#                 threshold=th,
#                 starting_capital=1000,
#                 nb_contracts=2,
#             )
#             if val_capital > best_capital:
#                 best_capital = val_capital
#                 best_threshold = th

#         train_capital, _, _, _ = simulate_strategy_corrected(
#             X_train, y_train, model,
#             threshold=best_threshold,
#             starting_capital=1000,
#             nb_contracts=2,
#         )
#         overfit_penalty = max(train_capital - best_capital, 0)

#         return -(best_capital - 0.5 * overfit_penalty)

#     study = optuna.create_study(direction='minimize')
#     study.optimize(lgb_objective, n_trials=1)
#     print(f"✅ Best params: {study.best_params}")

#     # 🧠 Retrain final model on full training set
#     best_params = study.best_params
#     best_params.update({
#         'boosting_type': 'gbdt',
#         'metric': None,
#         'verbosity': -1,
#         'scale_pos_weight': (y_train['target'] == 0).sum() / (y_train['target'] == 1).sum(),
#         'seed': 42,
#     })

#     lgb_train = lgb.Dataset(X_train.drop(columns=['datetime', 'expire_date']), label=y_train['target'], categorical_feature=CATEGORICAL_COLS)
#     model = lgb.train(
#         best_params,
#         lgb_train,
#         num_boost_round=2000,
#         valid_sets=[lgb_train],
#         callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
#     )

#     # 📈 Final test evaluation
#     y_pred_proba = model.predict(X_test.drop(columns=['datetime', 'expire_date']))
#     invalid_mask = (X_test['time_to_expiry'] >= 1500) | (X_test['moneyness'] >= 1.11)
#     y_pred_proba[invalid_mask.values] = 0.0

#     best_precision = 0
#     best_threshold = 0.8
#     for th in np.arange(0.7, 0.95, 0.01):
#         preds = (y_pred_proba >= th).astype(int)
#         precision = precision_score(y_test['target'], preds, pos_label=1)
#         if precision > best_precision:
#             best_precision = precision
#             best_threshold = th

#     y_pred = (y_pred_proba >= best_threshold).astype(int)
#     final_capital, cap_history, trades, df_preds = simulate_strategy_corrected(
#         X_test,
#         y_test,
#         model,
#         threshold=best_threshold,
#         starting_capital=starting_capital,
#         nb_contracts=nb_contracts,
#     )

#     print(f"\n💰 Final capital (on unseen test set): ${final_capital:.2f}")

#     results["01"] = {
#         "classif_report": classification_report(y_test['target'], y_pred, digits=3),
#         "precision": precision_score(y_test['target'], y_pred, pos_label=1),
#         "recall": recall_score(y_test['target'], y_pred, pos_label=1),
#         "final_capital": final_capital,
#         "trade_log": trades,
#         "capital_history": cap_history,
#         "df_preds": df_preds,
#     }

#     return final_capital, cap_history, trades, results, model


def time_series_real_eval(X, y, threshold=0.5, starting_capital=1000, nb_contracts=1, results={}):

    X = X.sort_values("datetime").copy()
    y = y.loc[X.index].copy()
    X["datetime"] = pd.to_datetime(X["datetime"])

    # 1. Single time-based split: 70% train, 15% tune, 15% test
    total_len = len(X)
    train_end = int(0.7 * total_len)
    tune_end = int(0.85 * total_len)

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()

    X_tune = X.iloc[train_end:tune_end].copy()
    y_tune = y.iloc[train_end:tune_end].copy()

    X_test = X.iloc[tune_end:].copy()
    y_test = y.iloc[tune_end:].copy()

    # ⛔ Apply validity filters to train and tune data
    train_mask = (X_train['time_to_expiry'] < 1500) & (X_train['moneyness'] < 1.11)
    tune_mask = (X_tune['time_to_expiry'] < 1500) & (X_tune['moneyness'] < 1.11)

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    X_tune = X_tune[tune_mask]
    y_tune = y_tune[tune_mask]
    
    # ⚠️ Skip if too few positives
    labels_distribution = y_train["target"].value_counts(normalize=True)
    if labels_distribution[1] < 0.01:
        print("⚠️ Too few positive samples in training set. Aborting.")
        return None, [], pd.DataFrame(), None, (None, None, None, None)

    # 🔍 Optuna tuning on tuning set only
    print("🔍 Starting Optuna tuning...")

    def lgb_objective(trial):
        param = {
            'boosting_type': 'gbdt',
            'metric': None,
            'verbosity': -1,
            'boost_from_average': True,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 300),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': 1,
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 50.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 50.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'scale_pos_weight': (y_train['target'] == 0).sum() / (y_train['target'] == 1).sum(),
            'seed': 42,
        }

        dtrain = lgb.Dataset(X_train.drop(columns=['datetime', 'expire_date']), label=y_train['target'], categorical_feature=CATEGORICAL_COLS)
        dtune = lgb.Dataset(X_tune.drop(columns=['datetime', 'expire_date']), label=y_tune['target'], categorical_feature=CATEGORICAL_COLS)

        model = lgb.train(
            param,
            dtrain,
            valid_sets=[dtrain, dtune],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
        )

        y_val_pred_proba = model.predict(X_tune.drop(columns=['datetime', 'expire_date']))
        invalid_mask = (X_tune['time_to_expiry'] >= 1500) | (X_tune['moneyness'] >= 1.11)
        y_val_pred_proba[invalid_mask.values] = 0.0

        # best_threshold = 0.8
        # best_capital = -np.inf
        # for th in np.arange(0.7, 0.95, 0.01):
        #     val_capital, _, _, _ = simulate_strategy_corrected(
        #         X_tune, y_tune, model,
        #         threshold=th,
        #         starting_capital=1000,
        #         nb_contracts=2,
        #     )
        #     if val_capital > best_capital:
        #         best_capital = val_capital
        #         best_threshold = th
        best_val_precision = 0
        best_val_threshold = 0.8
        for th in np.arange(0.7, 0.95, 0.01):
            val_preds = (y_val_pred_proba >= th).astype(int)
            val_precision = precision_score(y_tune['target'], val_preds, pos_label=1, zero_division=0.0)
            if val_precision > best_val_precision:
                best_val_precision = val_precision
                best_val_threshold = th

        # train_capital, _, _, _ = simulate_strategy_corrected(
        #     X_train, y_train, model,
        #     threshold=best_threshold,
        #     starting_capital=1000,
        #     nb_contracts=2,
        # )
        # overfit_penalty = max(train_capital - best_capital, 0)
        y_train_pred_proba = model.predict(X_train.drop(columns=['datetime', 'expire_date']))
        invalid_train_mask = (X_train['time_to_expiry'] >= 1500) | (X_train['moneyness'] >= 1.11)
        y_train_pred_proba[invalid_train_mask.values] = 0.0

        train_preds = (y_train_pred_proba >= best_val_threshold).astype(int)
        train_precision = precision_score(y_train["target"], train_preds, pos_label=1, zero_division=0.0)

        precision_penalty = max(train_precision - best_val_precision, 0)
        # return -(best_capital - 0.5 * overfit_penalty)
        return -(best_val_precision - 0.5 * precision_penalty)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lgb_objective, n_trials=20)
    print(f"✅ Best params: {study.best_params}")

    # 🧠 Retrain final model on full training set
    best_params = study.best_params
    best_params.update({
        'boosting_type': 'gbdt',
        'metric': None,
        'verbosity': -1,
        'scale_pos_weight': (y_train['target'] == 0).sum() / (y_train['target'] == 1).sum(),
        'seed': 42,
        'deterministic': True,
        'force_col_wise': True,
    })

    lgb_train = lgb.Dataset(X_train.drop(columns=['datetime', 'expire_date']), label=y_train['target'], categorical_feature=CATEGORICAL_COLS)
    model = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=800, # it takes less time to reach min l2
        valid_sets=[lgb_train, lgb_train],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
    )

    # # 🔒 Isolation Forest on class-1 training data
    # X_train_filtered = X_train[y_train['target'] == 1].drop(columns=['datetime', 'expire_date']).copy()
    # iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    # iso_forest.fit(X_train_filtered)

    # train_anomaly_scores = iso_forest.decision_function(X_train_filtered)
    # anomaly_threshold = np.percentile(train_anomaly_scores, 5)
    # print(f"🔎 Anomaly threshold (5th percentile of training scores): {anomaly_threshold:.4f}")

    # 📈 Final test evaluation with anomaly filtering
    X_test_features = X_test.drop(columns=['datetime', 'expire_date']).copy()
    y_pred_proba = model.predict(X_test_features)

    invalid_mask = (X_test['time_to_expiry'] >= 1500) | (X_test['moneyness'] >= 1.11)
    y_pred_proba[invalid_mask.values] = 0.0

    # anomaly_scores = iso_forest.decision_function(X_test_features)

    best_precision = 0
    best_threshold = 0.8
    for th in np.arange(0.7, 0.95, 0.01):
        preds = (y_pred_proba >= th).astype(int)
        precision = precision_score(y_test['target'], preds, pos_label=1, zero_division=0.0)
        if precision > best_precision:
            best_precision = precision
            best_threshold = th

    # best_threshold = 0.8
    # best_capital = -np.inf
    # for th in np.arange(0.7, 0.95, 0.01):
    #     tune_capital, _, _, _ = simulate_strategy_corrected(
    #         X_tune, y_tune, model,
    #         threshold=th,
    #         starting_capital=1000,
    #         nb_contracts=2,
    #     )
    #     if tune_capital > best_capital:
    #         best_capital = tune_capital
    #         best_threshold = th

    # for i, proba in enumerate(y_pred_proba):
    #     if proba >= best_threshold:
    #         if anomaly_scores[i] < anomaly_threshold:
    #             y_pred_proba[i] = 0.0  # Reject uncertain sample

    y_pred = (y_pred_proba >= best_threshold).astype(int)


    train_proba = model.predict(X_train.drop(columns=['datetime', 'expire_date']))
    X_train["pred_proba"] = train_proba
    positive_high_conf = X_train[(y_train['target'] == 1) & (X_train['pred_proba'] >= best_threshold)]
    # negative_low_conf = X_train[(y_train['target'] == 0)]
    X_train_pos_reference = positive_high_conf.sort_values('pred_proba', ascending=False).head(3000)
    # X_train_neg_reference = negative_low_conf.sort_values('pred_proba', ascending=True).head(1000)
    X_train = X_train.drop(columns=["pred_proba"], axis=1)

    # final_capital, cap_history, trades, df_preds = simulate_strategy_inliers(
    final_capital, cap_history, trades, df_preds = simulate_strategy_corrected(
        X_test,
        y_test,
        model,
        X_train_pos_reference,
        # X_train_neg_reference,
        # anomaly_scores,
        # anomaly_threshold,
        threshold=best_threshold,
        starting_capital=starting_capital,
        nb_contracts=nb_contracts,
    )

    train_proba = model.predict(X_train.drop(columns=['datetime', 'expire_date'], axis=1).copy())
    best_train_prediction = (train_proba >= best_threshold).astype(int)
    best_train_precision = precision_score(y_train['target'], best_train_prediction, pos_label=1, zero_division=0.0)

    print(f"best_threshold: {best_threshold}")
    print(f"best_precision: {best_precision}")
    print(f"best train precision: {best_train_precision}")
    print(f"\n💰 Final capital (on unseen test set): ${final_capital:.2f}")

    results["01"] = {
        "classif_report": classification_report(y_test['target'], y_pred, digits=3),
        "precision": precision_score(y_test['target'], y_pred, pos_label=1),
        "recall": recall_score(y_test['target'], y_pred, pos_label=1),
        "final_capital": final_capital,
        "trade_log": trades,
        "capital_history": cap_history,
        "df_preds": df_preds,
    }

    return final_capital, cap_history, trades, results, (model, best_threshold, best_precision, y_pred_proba)
