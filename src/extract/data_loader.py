import xarray as xr
import rioxarray
import pandas as pd
import geopandas as gpd

from typing import Optional
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

_cfg = None

def get_cfg():
    global _cfg
    if _cfg is None:
        from src.config import Config
        _cfg = Config()
    return _cfg

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

def load_gee_ndvi(tif_path: str, start_year: int = 2010, end_year: int = 2025):
    import numpy as np
    da = rioxarray.open_rasterio(tif_path)
    da = da.where(da != 9999, np.nan)
    
    time_index = pd.date_range(start=f'{start_year}-01-01',
                               end=f'{end_year}-12-31',
                               freq='MS')
    
    time_index = time_index + pd.offsets.MonthEnd(0)
    da = da.rename({'band': 'valid_time'})
    da['valid_time'] = time_index
    return da
    
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
    cfg = get_cfg()
    df = pd.read_parquet(cfg.processed_table)
    df = df.reset_index()
    df["valid_time"] = pd.to_datetime(df["valid_time"], format="%Y-%m-%d")
    df['month'] = df['valid_time'].dt.month
    
    # df = df[df['month'].isin(cfg.WILDFIRE_SEASON_MONTHS)]
    mask = (
        df['year'].between(2010, 2023) &
        ~df['landcover'].isin(cfg.NON_BURNABLE_CLASSES_LC)
    )
    df = df.loc[mask]
    return df

def load_russian_fires(filepath: str, use_start_date: bool = True):
    df = pd.read_csv(filepath, sep=';')  
    df['date_beginning'] = pd.to_datetime(df['date_beginning'])
    df['acq_date'] = df['date_beginning']
    
    cols = ['geometry', 'acq_date']
    if 'area_beginning' in df.columns:
        cols.append('area_beginning')
    cols.append('area_total')
    
    if 'code' in df.columns:
        df = df.sort_values('date_beginning').groupby('code').first().reset_index()
    
    from shapely.geometry import Point

    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
    return gdf[cols].copy()

def create_new_features(df: pd.DataFrame):
    
    # df["month"] = df["valid_time"].dt.month
    # df["month_sin"] = np.sin(2 * np.pi * df['month'] / 12)
    # df["month_cos"] = np.cos(2 * np.pi * df['month'] / 12)
    
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
    # df = create_lag_features(df)
    # df = create_new_features(df)
    
    # df = df.dropna(subset=[
    #     "temp_lag1",
    #     "vpd_lag1",
    #     "precip_lag1"
    # ])
    
    return df

def validate_dataset(df: pd.DataFrame):
    """Run sanity checks on the final dataframe."""
    assert df['fire'].sum() > 0, "No fire events in dataset!"
    assert not df[['temp', 'vpd', 'precip', 'ghm']].isnull().any().any(), "Missing values in core features"
    assert df['x'].between(58, 87).all(), "Longitudes out of expected range"
    assert df['y'].between(57, 67).all(), "Latitudes out of expected range"
    logger.info(f"Validation passed: {len(df)} rows, {df['fire'].sum()} fires")