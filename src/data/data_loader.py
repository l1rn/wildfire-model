import xarray as xr
import rioxarray
import pandas as pd
import geopandas as gpd
import numpy as np
from src.config import Config


from typing import Optional
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

cfg = Config()

def load_meterological(path: str) -> Optional[xr.Dataset]:
    """Loads ERA5 NetCDF and ensure coordinates are standard."""
    try:
        with xr.open_dataset(path) as ds:
            logger.info(f"Loaded meteorological data from {path}")
            return ds.load()
    except Exception as e:
        print(f"Failed to open NetCDF4: {e}")
        return None
    
def load_static_raster(path: str) -> Optional[xr.DataArray]:
    """Loads GeoTIFFs using rioxarray"""
    try:
        with rioxarray.open_rasterio(path) as rst:
            logger.info(f"Loaded a raster image from {path}")
            return rst.load()
    except Exception as e:
        print(f"Failed to open TIFF: {e}")
        return None
        
def load_firms(path: str) -> Optional[gpd.GeoDataFrame]:
    """Loads FIRMS CSV and converts to a GeoDataFrame"""
    try:
        df = pd.read_csv(path)
        df["acq_date"] = pd.to_datetime(df["acq_date"])
        df["year_month"] = df["acq_date"].dt.to_period("M")
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )
        logger.info(f"Loaded fire data from {path}")
        return gdf
    except Exception as e:
        print(f"Failed to Open CSV: {e}")
        return None
    
def load_master_dataset():
    df = pd.read_parquet(cfg.processed_table)
    df = df.reset_index()
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df['month'] = df['valid_time'].dt.month
    
    df = df[df['month'].isin(cfg.WILDFIRE_SEASON_MONTHS)]
    
    df = df[~df['landcover'].isin(cfg.NON_BURNABLE_CLASSES_LC)]
    return df

def create_new_features(df: pd.DataFrame):
    df["vpd_ghm_interaction"] = df["vpd"] * df["ghm"]
    df["month"] = df["valid_time"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df['month'] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df['month'] / 12)
    df["temp_precip_interaction"] = df["temp"] * df["precip"]
    df["ghm_windspeed_interaction"] = df["ghm"] * df["wind_speed"]
    return df
    

def create_lag_features(df: pd.DataFrame):
    df["vpd_lag1"] = df.groupby(["y", "x"])["vpd"].shift(1)
    df["temp_lag1"] = df.groupby(["y", "x"])["temp"].shift(1)
    df["precip_lag1"] = df.groupby(["y", "x"])["precip"].shift(1)
    df["vpd_ghm_interaction_lag1"] = df["vpd_lag1"] * df["ghm"]
    
    return df

def prepare_features(df: pd.DataFrame):
    df = df.sort_values(["y", "x", "valid_time"])        
    df = create_lag_features(df)
    df = create_new_features(df)
    
    df = df.dropna(subset=[
        "temp_lag1",
        "vpd_lag1",
        "precip_lag1"
    ])
    
    return df

def validate_dataset(df: pd.DataFrame):
    """Run sanity checks on the final dataframe."""
    assert df['fire'].sum() > 0, "No fire events in dataset!"
    assert not df[['temp', 'vpd', 'precip', 'ghm']].isnull().any().any(), "Missing values in core features"
    assert df['x'].between(58, 87).all(), "Longitudes out of expected range"
    assert df['y'].between(57, 67).all(), "Latitudes out of expected range"
    logger.info(f"Validation passed: {len(df)} rows, {df['fire'].sum()} fires")