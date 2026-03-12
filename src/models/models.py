from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

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

def get_random_forest_search():
    model = RandomForestClassifier(
        class_weight="balanced",
        n_jobs=1,
        random_state=42
    )   
    
    param_dist = {
        "n_estimators": randint(100, 250),
        "max_depth": randint(3, 10),
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 10)
    }
    
    return RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=5,
        cv=2,
        scoring="f1",
        random_state=42,
        n_jobs=1
    )