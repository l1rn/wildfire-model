from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def get_xgboost(scale_pos_weight):
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
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