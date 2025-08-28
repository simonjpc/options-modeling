FEATURES = [
    'iv',
    'time_to_expiry',
    'bid',
    'ask',
    'delta_volume',
    'delta',
    'gamma',
    'vega',
    'theta',
    'rho',
    'open',
    'close',
    'low',
    'high',
    'underlying_volume',
    'intra_bar_return',
    'intra_bar_volatility',
    'relative_close_position',
    'up_down_gap',
    'strike',
    'moneyness',
]

FEATURES_TO_SCALE = [
        'iv', 'time_to_expiry', 'moneyness', 'strike',
        'bid', 'ask', 'delta_volume',
        'delta', 'gamma', 'vega', 'theta', 'rho',
        'open', 'close', 'low', 'high', 'underlying_volume',
        'intra_bar_return', 'intra_bar_volatility',
        'relative_close_position', 'up_down_gap',
    ]

BATCH_SIZE = 512
BASE_LOCAL_PATH = "/Users/simon/Documents/perso/repos/options-modeling/exp2/experiments/"
EXPERIMENT_MOVING_COMPOUND_RESULTS_PATH = BASE_LOCAL_PATH + "main_{today}.pkl"

MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]