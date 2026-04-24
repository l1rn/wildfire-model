from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from pathlib import Path
from src.config import PROCESSED_DIR

from sklearn.metrics import (
    precision_recall_curve, 
    roc_auc_score, 
    classification_report, 
)

import numpy as np
import optuna
from optuna.samplers import TPESampler

def get_xgboost(scale_pos_weight):
    return XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42
    )
    
def get_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        n_jobs=-1,  
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
def optimize_xgboost(
    X_train, 
    y_train, 
    scale_weight
):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    base_model = XGBClassifier(
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=10,        
        scoring='f1',     
        cv=3,             
        verbose=1,        
        n_jobs=-1,        
        random_state=42
    )
    
    print("\n[+] Starting Randomized Search CV...")
    search.fit(X_train, y_train)
    
    print(f"[+] Optimization Complete!")
    print(f"[+] Best Parameters: {search.best_params_}")
    
    return search.best_estimator_

def lightgbm_factory():
    return LGBMClassifier(random_state=42, verbosity=-1)

def tuning(scale_pos_weight, X_train, y_train, X_val, y_val, model_type):
    def objective(trial):    
        base_gradient_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0,                    
        }
        if model_type == "xgb":
            params = {
                'gamma': trial.suggest_float('gamma', 0, 5),
            }
            params = params | base_gradient_params
            model = XGBClassifier(**params)
        elif model_type == "lgbm":
            params = {
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            params = params | base_gradient_params 
            model = LGBMClassifier(**params)
        elif model_type == "rf":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': 'balanced',  
                'random_state': 42,
                'n_jobs': -1,
            }
            model = RandomForestClassifier(**params)
        else:
            raise ValueError("Unsupported model type")
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        return best_f1
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    best_params = study.best_params
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    import json
    if model_type == "xgb":
        best_params['scale_pos_weight'] = scale_pos_weight
        best_params['random_state'] = 42
        best_params['verbosity'] = 0
        model = XGBClassifier(**best_params)
        with open(Path(PROCESSED_DIR) / "best_xgboost_params.json", 'w') as f:
            json.dump(study.best_params, f)
    elif model_type == "lgbm":
        best_params['scale_pos_weight'] = scale_pos_weight
        best_params['random_state'] = 42
        best_params['verbosity'] = -1
        model = LGBMClassifier(**best_params)  
        with open(Path(PROCESSED_DIR) / "best_lightgbm_params.json", 'w') as f:
            json.dump(study.best_params, f)  
    elif model_type == "rf":
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        model = RandomForestClassifier(**best_params)
        with open(Path(PROCESSED_DIR) / "best_rf_params.json", 'w') as f:
            json.dump(study.best_params, f)
    return model