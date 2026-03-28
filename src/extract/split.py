from sklearn.model_selection import train_test_split
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def temporal_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.2
):
    df = df.sort_values('valid_time')
    
    val_idx = int(len(df) * (1  - test_size - val_size))
    test_idx = int(len(df) * (1 - test_size))
    
    train_years = df.iloc[:val_idx]
    val_years = df.iloc[val_idx:test_idx]
    test_years = df.iloc[test_idx:]

    logger.info(train_years['year'].unique())
    
    logger.info(test_years['year'].unique())
    return train_years, val_years, test_years