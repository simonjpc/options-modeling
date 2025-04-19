import numpy as np
from constants import LIST_OF_FLOAT_KEYS, COMMOM_KEYS, CALL_KEYS, PUT_KEYS, KEYS_MAP

def build_dict(raw_keys, raw_values):
    element = dict(zip(
        raw_keys.lower().replace("\n", "").replace("[", "").replace("]", "").split(", "),
        raw_values.replace("\n", "").split(", ")
    ))
    for key in LIST_OF_FLOAT_KEYS:
        if len(element[key]) < 1:
            # assume empty string means zero (strong assumption)
            element[key] = 0
        else:
            element[key] = float(element[key])

    return element

def isolate_option_types(element):
    call_element, put_element = {}, {}
    for key in element:
        if key in COMMOM_KEYS:
            call_element[key] = element[key]
            put_element[key] = element[key]
        if key in CALL_KEYS:
            call_element[KEYS_MAP[key]] = element[key]
        if key in PUT_KEYS:
            put_element[KEYS_MAP[key]] = element[key]
    call_element["type"] = "call"
    put_element["type"] = "put"

    return call_element, put_element

def correct_outliers(x, y, window=5):
    y_corrected = y.copy()
    outliers_idxs = []
    half_window = window // 2
    for i in range(half_window, len(x) - half_window):
        x_window = np.concatenate((x[i - half_window:i], x[i+1:i + half_window + 1]))
        y_window = np.concatenate((y[i - half_window:i], y[i+1:i + half_window + 1]))
        coeffs = np.polyfit(x_window, y_window, deg=2)
        pred = np.polyval(coeffs, np.array([x[i]]))[0]
        residuals = np.abs(y_window - np.polyval(coeffs, x_window))
        mad = np.median(residuals)
        # Compare the point's residual to MAD
        if abs(y[i] - pred) > 6 * mad:
            outliers_idxs.append(i)
            y_corrected[i] = pred  # Replace with predicted value
        if i == 13:
            break
    return y_corrected