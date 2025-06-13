import os
import pickle
import lightgbm as lgb
from typing import List, Any

def load_models(folder_path: str) -> List[Any]:
    models = []
    if not os.path.exists(folder_path):
        return models
    files = os.listdir(folder_path)
    for f in files:
        if "DS_Store" in f:
            continue
        booster = lgb.Booster(model_file=os.path.join(folder_path, f))
        models.append(booster)
    return models

def save_model(folder_path: str, filename: str, model) -> None:
    os.makedirs(folder_path, exist_ok=True)
    model.save_model(f"{folder_path}/{filename}.txt")

def save_experiment_results(results, filepath, save_mode="wb") -> None:
    with open(filepath, save_mode) as f:
        pickle.dump(results, f)
