from sklearn.model_selection import train_test_split
import pandas as pd

def temporal_split(
    df: pd.DataFrame
):
    X = df.drop(columns=['fire'])
    y = df['fire']
    
    stratify_col = df['fire'].astype(str) + "_" + df['is_extreme_fire'].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_col
    )
    
    X_test_hot = X_test[X_test['is_extreme_year'] == 1]
    y_test_hot = y_test[X_test['is_extreme_year'] == 1]
    
    X_test_cold = X_test[X_test['is_extreme_year'] == 0]
    y_test_cold = y_test[X_test['is_extreme_year'] == 0]
    
    return X_train, X_test, y_train, y_test, X_test_hot, y_test_hot, X_test_cold, y_test_cold