from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier

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
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )