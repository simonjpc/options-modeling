import optuna
from exp2.optim_utils import time_series_real_eval
from exp2.data import load_and_preprocess_df, save_experiment_results
from exp2.constants import FEATURES, MONTHS
from exp2.evaluate import money_evaluation
from exp2.constants import EXPERIMENT_MOVING_COMPOUND_RESULTS_PATH

def create_results(final_capital, capital_history, trade_log_df, df, best_val_threshold, best_test_precision, month, results):

    results[month] = {
        "final_capital": final_capital,
        "capital_history": capital_history,
        "trade_log_df": trade_log_df,
        "df": df,
        "best_val_threshold": best_val_threshold,
        "best_test_precision": best_test_precision,
    }


def main():

    features = FEATURES
    results = {}

    intial_capital = 1000
    current_capital = intial_capital

    for month in MONTHS:
        if month in ("07", "11"):
            continue
        print(f"starting execution for month {month}...")
        df_labeled = load_and_preprocess_df(month=month, window=16)
        # Run Optuna
        objective, train_set, val_set, test_set, df_orig_bid_ask = time_series_real_eval(df_labeled, features, month)
        print("Starting Optuna optimization...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=10)
        args = money_evaluation(
            study, features, train_set, val_set, test_set, df_orig_bid_ask,
            starting_capital=current_capital, nb_contracts=2,
        )
        args = tuple(list(args) + [month, results])
        create_results(*args)
        save_experiment_results(results, EXPERIMENT_MOVING_COMPOUND_RESULTS_PATH.format(today="latest"))
        print(f"finished execution for month {month}.\n")

        end_of_month_capital = args[0]
        current_capital = end_of_month_capital
        if current_capital < 40:
            quit()


if __name__ == '__main__':
    main()

