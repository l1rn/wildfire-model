from sklearn.model_selection import train_test_split
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def temporal_split(df, val_size=0.1, test_size=0.15):
    years = sorted(df['year'].unique())
    n_years = len(years)
    n_test = max(1, int(n_years * test_size))
    n_val = max(1, int(n_years * val_size))
    n_train = n_years - n_test - n_val
    if n_train < 1:
        n_train = 1
        n_val = max(0, n_val - 1)  
        n_test = max(0, n_test - 1)
    train_years = years[:n_train]
    val_years = years[n_train:n_train+n_val]
    test_years = years[n_train+n_val:]
    
    logger.info(f"Train years: {train_years}")
    logger.info(f"Validation years: {val_years}")
    logger.info(f"Test years: {test_years}")
    train = df[df['year'].isin(train_years)]
    val = df[df['year'].isin(val_years)]
    test = df[df['year'].isin(test_years)]
    return train, val, test