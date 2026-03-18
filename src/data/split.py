from sklearn.model_selection import train_test_split
import pandas as pd

def temporal_split(
    df: pd.DataFrame,
    test_size: int = 0.2
):
    df = df.sort_values('valid_time')
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df.drop(columns=["fire"])
    y_train = train_df["fire"]
    
    X_test = test_df.drop(columns=["fire"])
    y_test = test_df["fire"]
    
    X_test_hot = X_test[X_test['is_extreme_year'] == 1]
    y_test_hot = y_test[X_test['is_extreme_year'] == 1]
    X_test_cold = X_test[X_test['is_extreme_year'] == 0]
    y_test_cold = y_test[X_test['is_extreme_year'] == 0]
    return X_train, X_test, y_train, y_test