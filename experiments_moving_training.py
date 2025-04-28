import gc
import pandas as pd
from evaluation import evaluate_month_with_existing_models
from preprocessing import (
    prepare_labels,
    create_option_dataset_full,
    add_datetime_features,
    add_advanced_features,
)
from constants import (
    MONTHS_NUMBERS,
    MODELS_FOLDER_NAME,
    MODELS_FILENAME_BASE,
    EXPERIMENT_MOVING_RESULTS_PATH,
    OPTIONS_CHAIN_ABS_PATH,
    OPTIONS_CHAIN_ABS_PATH_MAP,
    STOCK_PRICE_ABS_PATH,
)
from training import time_series_cv
from load_and_save import save_experiment_results

results = {}

all_months = ["01"] + MONTHS_NUMBERS
stock_price_data = pd.read_csv(STOCK_PRICE_ABS_PATH)
print("base models loaded.")

for month in all_months:
    print(f"\n🗓 Processing month {month}")

    # 1. Load data for that month
    with open(OPTIONS_CHAIN_ABS_PATH_MAP[month], "r") as f:
        options_chain_data = f.readlines()
    print(f"options chain data loaded for month {month}")

    # 2. Preprocessing: same steps as you did for January
    df_labeled = prepare_labels(stock_price_data, options_chain_data)
    print(f"labels created for month {month}")
    X_month, y_month = create_option_dataset_full(df_labeled, n=16)
    X_month = add_datetime_features(X_month)
    X_month = add_advanced_features(X_month, n=16)
    print(f"dataset completed for month {month}")

    # # 3. Predict with ensemble
    # avg_preds = predict_with_ensemble(models, X_month)

    # # 4. Metrics
    # threshold = 0.8
    # y_pred = (avg_preds >= threshold).astype(int)

    # # 5. Simulate strategy
    # X_month_copy = X_month.copy()
    # X_month_copy['datetime'] = pd.to_datetime(X_month_copy['datetime'])
    # y_month_copy = y_month.copy()
    # final_capital, capital_history, trade_log, _ = simulate_strategy_corrected(
    #     X_month_copy,
    #     y_month_copy,
    #     DummyModel(avg_preds),
    #     threshold=threshold,
    #     starting_capital=1000,
    #     nb_contracts=2,
    # )
    # print(f"💵 Final Capital for {month}: ${final_capital:.2f}")
    fold_metrics, fold_capitals, trades_df, _, _ = time_series_cv(
        X_month,
        y_month,
        n_splits=5,
        threshold=0.8,
        starting_capital=1000,
        nb_contracts=2,
    )

    results[month] = {
        "fold_metrics": fold_metrics,
        "fold_capitals": fold_capitals,
        "trades_df": trades_df,
    }

    print(f"month {month} processed.")

    # save results (save on each iteration to have information before full run)
    save_experiment_results(results, EXPERIMENT_MOVING_RESULTS_PATH)
    print("tmp results saved. \n")

    del df_labeled, X_month, y_month
    gc.collect()
