import numpy as np

def predict_with_ensemble(models, X, invalid_mask_cols=["time_to_expiry", "strike", "open_t0"]):
    preds = []

    for model in models:
        pred_proba = model.predict(X.drop(columns=["datetime", "expire_date"], errors='ignore'))
        
        # Apply invalid mask like before
        invalid_mask = (X[invalid_mask_cols[0]] >= 1500) | (X[invalid_mask_cols[1]] >= 1.11 * X[invalid_mask_cols[2]])
        pred_proba[invalid_mask.values] = 0.0

        preds.append(pred_proba)
    
    preds = np.vstack(preds)  # shape: (n_folds, n_samples)
    avg_preds = preds.mean(axis=0)  # average over folds
    return avg_preds