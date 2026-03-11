import xarray as xr
import rioxarray
import pandas as pd
import geopandas as gpd
from src.config import PROCESSED_DIR

from typing import Optional

def load_meterological(path: str) -> Optional[xr.Dataset]:
    """Loads ERA5 NetCDF and ensure coordinates are standard."""
    try:
        with xr.open_dataset(path) as ds:
            return ds.load()
    except Exception as e:
        print(f"Failed to open NetCDF4: {e}")
        return None
    
def load_static_raster(path: str) -> Optional[xr.DataArray]:
    """Loads GeoTIFFs using rioxarray"""
    try:
        with rioxarray.open_rasterio(path) as rst:
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
        return gdf
    except Exception as e:
        print(f"Failed to Open CSV: {e}")
        return None
    
def load_master_dataset():
    df = pd.read_parquet(f"{PROCESSED_DIR}/khmao_master.parquet")
    df = df.reset_index()
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    return df

def create_lag_features(df: pd.DataFrame):
    df["vpd_lag1"] = df.groupby(["y", "x"])["vpd"].shift(1)
    df["temp_lag1"] = df.groupby(["y", "x"])["temp"].shift(1)
    df["precip_lag1"] = df.groupby(["y", "x"])["precip"].shift(1)
    df["vpd_ghm_interaction_lag1"] = df["vpd_lag1"] * df["ghm"]
    df["vpd_ghm_interaction"] = df["vpd"] * df["ghm"]
    df["month"] = df["valid_time"].dt.month
    return df

def prepare_features(df: pd.DataFrame):
    df = df.sort_values(["y", "x", "valid_time"])        
    df = create_lag_features(df)
    
    df = df.dropna(subset=[
        "temp_lag1",
        "vpd_lag1",
        "precip_lag1"
    ])
    
    return df