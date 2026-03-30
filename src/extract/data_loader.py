import xarray as xr
import rioxarray
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

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
    
    df = df[df['year'].isin(range(2010, 2025))]
    df = df[df['month'].isin(cfg.WILDFIRE_SEASON_MONTHS)]
    
    df = df[~df['landcover'].isin(cfg.NON_BURNABLE_CLASSES_LC)]
    return df

def load_russian_fires(filepath: str, use_start_date: bool = True):
    df = pd.read_csv(filepath, sep=';')  
    df['date_beginning'] = pd.to_datetime(df['date_beginning'])
    df['acq_date'] = df['date_beginning']
    df['type'] = df['type'].map({'Лесные': 0, 'Нелесные': 2})
    
    cols = ['geometry', 'acq_date', 'type']
    if 'area_beginning' in df.columns:
        cols.append('area_beginning')
    
    if 'code' in df.columns:
        df = df.sort_values('date_beginning').groupby('code').first().reset_index()
        
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
    return gdf[cols].copy()

def create_new_features(df: pd.DataFrame):
    df["vpd_ghm_interaction"] = df["vpd"] * df["ghm"]
    df["month"] = df["valid_time"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df['month'] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df['month'] / 12)
    df["vpd_pop_density_interaction"] = df["vpd"] * df["pop_density"]
    df["vpd_3m_avg"] = (
        df.groupby(['x', 'y'])['vpd']
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )
    df["vpd_3m_avg_ghm_interaction"] = df["vpd_3m_avg"] * df["ghm"]
    df["temp_ghm_interaction"] = df["temp"] * df["ghm"]
    df["temp_infrastructure_interaction"] = df["temp"] * df["dist_oil_gas"]
    df["temp_precip_interaction"] = df["temp"] * df["precip"]
    df["dew_ghm_interaction"] = df["dew"] * df["ghm"]
    df["dew_infrastructure_interaction"] = df["dew"] * df["dist_oil_gas"]
    df['vpd_14p_max'] = (
        df.groupby(['y', 'x'])['vpd']
        .transform(lambda x: x.rolling(window=14, min_periods=1).max())
    )
    df['precip_30p_sum'] = (
        df.groupby(['y', 'x'])['precip']
        .transform(lambda x: x.rolling(window=30, min_periods=1).sum())
    )
    df['synergy_vpd_ghm'] = df['vpd_14p_max'] * df['ghm']
    df['synergy_vpd_infrastructure'] = df['vpd_14p_max'] * (1 / (df['dist_oil_gas'] + 1))
    return df

def create_lag_features(df: pd.DataFrame, lag_vars=['vpd','temp','precip','fire'], lags=1):
    for var in lag_vars :
        if var in df.columns:
            for lag in range(1, lags+1):
                df[f'{var}_lag{lag}'] = df.groupby(['y', 'x'])[var].shift(lag)
    if 'vpd_lag1' in df.columns and 'fire_lag1' in df.columns:
        df["vpd_fire_lag1_interaction"] = df["vpd_lag1"] * df["fire_lag1"]    
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