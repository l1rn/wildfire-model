from sklearn.model_selection import train_test_split
import pandas as pd

def temporal_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15
):
    df = df.sort_values('valid_time')
    val_idx = int(len(df) * (1  - test_size - val_size))
    test_idx = int(len(df) * (1 - test_size))
    
    train = df.iloc[:val_idx]
    val = df.iloc[val_idx:test_idx]
    test = df.iloc[test_idx:]
    
    logger
    return train, val, test