import gc
from datetime import datetime
import pandas as pd
from preprocessing import (
    prepare_labels,
    create_option_dataset_full,
    add_datetime_features,
    add_advanced_features,
)
from constants import (
    MONTHS_NUMBERS,
    EXPERIMENT_MOVING_RESULTS_PATH,
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
    X_month, y_month = create_option_dataset_full(df_labeled, n=16)
    X_month = add_datetime_features(X_month)
    X_month = add_advanced_features(X_month, n=16)

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
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    save_experiment_results(results, EXPERIMENT_MOVING_RESULTS_PATH.format(today=now))
    print("tmp results saved. \n")

    del df_labeled, X_month, y_month
    gc.collect()
