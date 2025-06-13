from scipy.stats import ks_2samp
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity


# 1. Kolmogorov–Smirnov Test
def compute_ks_drift(train_df, test_df, feature_cols):
    drift_scores = {}
    for col in feature_cols:
        stat, p_value = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        drift_scores[col] = {'ks_stat': stat, 'p_value': p_value}
    return drift_scores

# 2. Population Stability Index (PSI)
def compute_psi(expected_array, actual_array, buckets=10):
    def scale_range(array, buckets):
        return np.percentile(array, np.linspace(0, 100, buckets + 1))

    breakpoints = scale_range(expected_array, buckets)
    expected_percents = np.histogram(expected_array, bins=breakpoints)[0] / len(expected_array)
    actual_percents = np.histogram(actual_array, bins=breakpoints)[0] / len(actual_array)

    psi_values = []
    for e, a in zip(expected_percents, actual_percents):
        if e == 0:
            e = 0.0001
        if a == 0:
            a = 0.0001
        psi_values.append((a - e) * np.log(a / e))
    return np.sum(psi_values)

def compute_psi_drift(train_df, test_df, feature_cols):
    psi_scores = {}
    for col in feature_cols:
        psi = compute_psi(train_df[col].dropna().values, test_df[col].dropna().values)
        psi_scores[col] = psi
    return psi_scores

# 3. Maximum Mean Discrepancy (MMD) with RBF kernel
def compute_mmd(train_array, test_array, gamma=1.0):
    X = train_array.reshape(-1, 1)
    Y = test_array.reshape(-1, 1)
    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd

def compute_mmd_drift(train_df, test_df, feature_cols, gamma=1.0):
    mmd_scores = {}
    for col in feature_cols:
        train_array = train_df[col].dropna().values
        test_array = test_df[col].dropna().values
        mmd_scores[col] = compute_mmd(train_array, test_array, gamma)
    return mmd_scores


def prediction_entropy(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def filter_low_support_confident_samples(sub_df, X_pos_ref, feature_cols, k=3, sim_thresh=0.9):
    sub_feats = sub_df[feature_cols].values
    ref_feats = X_pos_ref[feature_cols].values

    sims = cosine_similarity(sub_feats, ref_feats)
    support_counts = (sims >= sim_thresh).sum(axis=1)
    return sub_df[support_counts >= k]


def filter_far_from_centroid(sub_df, X_pos_ref, feature_cols, distance_thresh=3.0):
    centroid = X_pos_ref[feature_cols].mean().values.reshape(1, -1)
    distances = cdist(sub_df[feature_cols].values, centroid)
    return sub_df[distances[:, 0] <= distance_thresh]



def train_isolation_forest(X_pos_ref, feature_cols):
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_pos_ref[feature_cols])
    return iso
