import gc
import numpy as np
import pandas as pd
from datetime import datetime
from preprocessing import (
    prepare_labels,
    add_datetime_features,
    add_advanced_features,
    create_option_dataset_full,
    preprocess_dataset,
)
from constants import (
    MONTHS_NUMBERS,
    EXPERIMENT_MOVING_COMPOUND_RESULTS_PATH,
    OPTIONS_CHAIN_ABS_PATH_MAP,
    STOCK_PRICE_ABS_PATH,
)
from training import time_series_cv, time_series_real_eval
from load_and_save import save_experiment_results

results = {}

all_months = ["01"] + MONTHS_NUMBERS
stock_price_data = pd.read_csv(STOCK_PRICE_ABS_PATH)
print("base models loaded.")

intial_capital = 1000
current_capital = intial_capital

for month in all_months[11:]:
    print(f"\n🗓 Processing month {month}")

    # 1. Load data for that month
    with open(OPTIONS_CHAIN_ABS_PATH_MAP[month], "r") as f:
        options_chain_data = f.readlines()
    print(f"options chain data loaded for month {month}")

    # 2. Preprocessing: same steps as you did for January
    X_month, y_month = preprocess_dataset(
        stock_price_data,
        options_chain_data,
        n=16,
        month=month
    )
    # continue
    # df_labeled = prepare_labels(stock_price_data, options_chain_data)
    # print(f"labels created for month {month}")
    # X_month, y_month = create_option_dataset_full(df_labeled, n=16)
    # X_month = add_datetime_features(X_month)
    # X_month = add_advanced_features(X_month, n=16)
    print(f"dataset completed for month {month}")

    # print(f"💵 Final Capital for {month}: ${final_capital:.2f}")
    # fold_metrics, fold_capitals, trades_df, _, _ = time_series_cv(
    #     X_month,
    #     y_month,
    #     n_splits=5,
    #     threshold=0.8,
    #     starting_capital=current_capital,
    #     nb_contracts=2,
    # )
    fold_metrics, fold_capitals, trades_df, _, (_, best_threshold, best_precision, y_pred_proba) = time_series_real_eval(
        X_month,
        y_month,
        # n_splits=5,
        threshold=0.8,
        starting_capital=current_capital,
        nb_contracts=2,
    )

    # if best_precision is not None and best_precision > 0.7: # model having a low precision on best thr is not used for inference
    if fold_metrics is not None:
        if not np.isnan(fold_metrics).item():
            current_capital = np.mean(fold_metrics).item()
    else:
        fold_metrics = [current_capital]

    results[month] = {
        "fold_metrics": fold_metrics,
        "fold_capitals": fold_capitals,
        "trades_df": trades_df,
        "best_threshold": best_threshold,
        "best_precision": best_precision,
        "y_pred_proba": y_pred_proba,
    }
    print(f"current capital: {current_capital}")
    print(f"month {month} processed.")

    # save results (save on each iteration to have information before full run)
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    save_experiment_results(results, EXPERIMENT_MOVING_COMPOUND_RESULTS_PATH.format(today=now))
    print("tmp results saved. \n")

    # del df_labeled, X_month, y_month
    del X_month, y_month
    gc.collect()
