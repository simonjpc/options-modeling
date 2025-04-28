LIST_OF_FLOAT_KEYS = [
    "quote_time_hours",
    "underlying_last",
    "dte",
    "c_delta",
    "c_gamma",
    "c_vega",
    "c_theta",
    "c_rho",
    "c_iv",
    "c_volume",
    "c_last",
    "c_bid",
    "c_ask",
    "strike",
    "p_bid",
    "p_ask",
    "p_last",
    "p_delta",
    "p_gamma",
    "p_vega",
    "p_theta",
    "p_rho",
    "p_iv",
    "p_volume",
    "strike_distance",
    "strike_distance_pct",
]

COMMOM_KEYS = [
    # "quote_unixtime",
    "quote_readtime",
    # "quote_date",
    "quote_time_hours",
    "underlying_last",
    "expire_date",
    # "expire_unix",
    "dte",
    "strike",
    "strike_distance",
    "strike_distance_pct",
]

CALL_KEYS = [
    "c_delta",
    "c_gamma",
    "c_vega",
    "c_theta",
    "c_rho",
    "c_iv",
    "c_volume",
    "c_last",
    "c_size",
    "c_bid",
    "c_ask",
    "type",
]

PUT_KEYS = [
    "p_bid",
    "p_ask",
    "p_size",
    "p_last",
    "p_delta",
    "p_gamma",
    "p_vega",
    "p_theta",
    "p_rho",
    "p_iv",
    "p_volume",
    "type",
]

KEYS_MAP = {
    "c_delta": "delta",
    "c_gamma": "gamma",
    "c_vega": "vega",
    "c_theta": "theta",
    "c_rho": "rho",
    "c_iv": "iv",
    "c_volume": "volume",
    "c_last": "last",
    "c_size": "size",
    "c_bid": "bid",
    "c_ask": "ask",
    "p_bid": "bid",
    "p_ask": "ask",
    "p_size": "size",
    "p_last": "last",
    "p_delta": "delta",
    "p_gamma": "gamma",
    "p_vega": "vega",
    "p_theta": "theta",
    "p_rho": "rho",
    "p_iv": "iv",
    "p_volume": "volume",
}

CATEGORICAL_COLS = ["day_of_week"]
MONTHS_NUMBERS = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

BASE_PATH = "/Users/simon/Documents/perso/repos/options-modeling/data/"
BASE_EXTERNAL_PATH = "/Volumes/T7/backup/Documents/perso/repos_perso/options-modeling/data/"
STOCK_PRICE_ABS_PATH = BASE_EXTERNAL_PATH + "5min_hist_SPY_2023.csv"
OPTIONS_CHAIN_ABS_PATH = BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q1/spy_15x_202301.txt"
OPTIONS_CHAIN_ABS_PATH_MAP = {
    "01": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q1/spy_15x_202301.txt",
    "02": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q1/spy_15x_202302.txt",
    "03": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q1/spy_15x_202303.txt",
    "04": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q2/spy_15x_202304.txt",
    "05": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q2/spy_15x_202305.txt",
    "06": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q2/spy_15x_202306.txt",
    "07": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q3/spy_15x_202307.txt",
    "08": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q3/spy_15x_202308.txt",
    "09": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q3/spy_15x_202309.txt",
    "10": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q4/spy_15x_202310.txt",
    "11": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q4/spy_15x_202311.txt",
    "12": BASE_EXTERNAL_PATH + "spy_options_15mins/spy_15x_2023q4/spy_15x_202312.txt",
}
MODELS_FOLDER_NAME = BASE_PATH + "models_january"
MODELS_FILENAME_BASE = "model_fold"
EXPERIMENT_FIXED_RESULTS_PATH = BASE_PATH + "experiments_fixed_training.pkl"
EXPERIMENT_MOVING_RESULTS_PATH = BASE_PATH + "experiments_moving_training.pkl"
