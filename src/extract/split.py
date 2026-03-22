from sklearn.model_selection import train_test_split
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def temporal_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15
):
    df = df.sort_values('valid_time')
    extreme_years = sorted(df[df['is_extreme_year'] == 1]['year'].unique())
    normal_years = sorted(df[df['is_extreme_year'] == 0]['year'].unique())
    
    val_idx = int(len(df) * (1  - test_size - val_size))
    test_idx = int(len(df) * (1 - test_size))
    
    train_years = df.iloc[:val_idx]
    val_years = df.iloc[val_idx:test_idx]
    test_years = df.iloc[test_idx:]

    train = df[df['year'].isin(train_years)]
    val = df[df['year'].isin(val_years)]
    test = df[df['year'].isin(test_years)]
    
    logger.info(train_years['year'].unique())
    
    logger.info(test_years['year'].unique())
    return train_years, val_years, test_years