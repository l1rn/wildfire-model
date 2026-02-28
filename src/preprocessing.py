from src.data import data_loader
from src.config import RAW_DIR, PROCESSED_DIR
import numpy as np
import pandas as pd
import geopandas as gpd

from rioxarray.raster_array import RasterArray
from rasterio.features import rasterize
import xarray as xr

from tqdm.auto import tqdm

KELVIN = 273.15

def calculate_vpd(t2m_k, d2m_k):
    t_c = t2m_k - KELVIN
    d_c = d2m_k - KELVIN

    es = 610.78 * np.exp((17.2694 * t_c) / (t_c + 237.3))
    ea = 610.78 * np.exp((17.2694 * d_c) / (d_c + 237.3))
    return es - ea

def dimension_unify_xy(
    t2m: xr.DataArray,
    d2p: xr.DataArray,
    tp: xr.DataArray,
    vpd: xr.DataArray,
    sm1: xr.DataArray
):
    rename_dict = {"latitude": "y", "longitude": "x"}
    t2m = t2m.rename(rename_dict)
    d2p = d2p.rename(rename_dict)
    tp = tp.rename(rename_dict)
    vpd = vpd.rename(rename_dict)
    sm1 = sm1.rename(sm1)
    
    return t2m, d2p, tp, vpd, sm1

def broadcast_static_layers(
    main_dim: xr.DataArray,
    dem: xr.DataArray,
    slope: xr.DataArray,
    lc: xr.DataArray,
    ghm: xr.DataArray
):
    dem = dem.expand_dims(valid_time=main_dim.valid_time)
    lc = lc.expand_dims(valid_time=main_dim.valid_time)
    ghm = ghm.expand_dims(valid_time=main_dim.valid_time)
    slope = slope.expand_dims(valid_time=main_dim.valid_time)
    
    dem = dem.dropna("valid_time", how="all")
    lc = lc.dropna("valid_time", how="all")
    ghm = ghm.dropna("valid_time", how="all")
    slope = slope.dropna("valid_time", how="all")
    
    return dem, lc, ghm, slope

def rasterize_monthly_fire(
    firms_gdf: pd.DataFrame, 
    climate_da: xr.DataArray
):
    template = climate_da.isel(valid_time=0)
    fire_rasters = []
    template_rio: RasterArray = template.rio
    
    for time in tqdm(climate_da.valid_time.values, desc="Rasterizing Fire Data"):
        month = pd.to_datetime(time).to_period("M")
        monthly_fires = firms_gdf[firms_gdf["year_month"] == month]
        if len(monthly_fires) == 0:
            fire_array = np.zeros(template.shape, dtype=np.uint8)
        else:
            shapes = [(geom, 1) for geom in monthly_fires.geometry]
            
            fire_array = rasterize(
                shapes,
                out_shape=template.shape,
                transform=template_rio.transform(),
                fill=0,
                dtype=np.uint8
            )
            
        fire_rasters.append(fire_array)
        
    fire_stack = np.stack(fire_rasters)
    
    return xr.DataArray (
        fire_stack,
        dims=("valid_time", "y", "x"),
        coords={
            "valid_time": climate_da.valid_time,
            "y": template.y,
            "x": template.x,
        }
    )

def process_data():
    """ Data Integration """
    topo_ds = data_loader.load_static_raster(f"{RAW_DIR}/khmao_terrain_1km.tif")
    lc_ds = data_loader.load_static_raster(f"{RAW_DIR}/khmao_lc_1km.tif")
    human_mod_ds = data_loader.load_static_raster(f"{RAW_DIR}/khmao_human_mod_1km.tif")
    
    ds: xr.Dataset = data_loader.load_meterological(f"{RAW_DIR}/khmao_era5.nc")
    firms = data_loader.load_firms(f"{RAW_DIR}/khmao_fire_archive.csv")
            
    t2m_monthly: xr.Dataset = ds["t2m"].resample(valid_time="1ME").mean()
    d2p_monthly: xr.Dataset = ds["d2m"].resample(valid_time="1ME").mean()
    sm1_monthly: xr.Dataset = ds["swvl1"].resample(valid_time="1ME").mean()
    
    tp_monthly = ds["tp"].resample(valid_time="1ME").sum()
    tp_monthly_mm = tp_monthly * 1000
    
    t2m_rio: RasterArray = t2m_monthly.rio
    t2m_monthly = t2m_rio.write_crs("EPSG:4326")
    
    d2p_rio: RasterArray = d2p_monthly.rio
    d2p_monthly = d2p_rio.write_crs("EPSG:4326")
    
    sm1_rio: RasterArray = sm1_monthly.rio
    sm1_monthly = sm1_rio.write_crs("EPSG:4326")
    
    lc_rio: RasterArray = lc_ds.rio
    lc_matched = lc_rio.reproject_match(t2m_monthly).squeeze("band", drop=True)
    
    human_mod_rio: RasterArray = human_mod_ds.rio
    human_mod_matched = human_mod_rio.reproject_match(t2m_monthly).squeeze("band", drop=True)
    
    dem_matched = topo_ds.sel(band=1).rio.reproject_match(t2m_monthly)
    slope_matched = topo_ds.sel(band=2).rio.reproject_match(t2m_monthly)

    vpd_monthly: RasterArray = calculate_vpd(t2m_monthly, d2p_monthly)

    t2m_monthly, d2p_monthly, tp_monthly_mm, vpd_monthly, sm1_monthly = dimension_unify_xy(
        t2m=t2m_monthly, 
        d2p=d2p_monthly, 
        tp=tp_monthly_mm, 
        vpd=vpd_monthly, 
        sm1=sm1_monthly
    )
    
    dem_matched, lc_matched, human_mod_matched, slope_matched = broadcast_static_layers(
        main_dim=t2m_monthly, 
        dem=dem_matched, 
        lc=lc_matched, 
        ghm=human_mod_matched,
        slope=slope_matched
    )
    
    fire_monthly = rasterize_monthly_fire(
        firms_gdf=firms, climate_da=t2m_monthly
    )
    
    dataset = xr.Dataset({
        "temp": t2m_monthly,
        "vpd": vpd_monthly,
        "precip": tp_monthly_mm,
        "dem": dem_matched,
        "slope": slope_matched,
        "sm1": sm1_monthly,
        "landcover": lc_matched,
        "ghm": human_mod_matched,
        "fire": fire_monthly
    })
    
    khmao_boundary = gpd.read_file(f"{RAW_DIR}/khmao.geojson")
    dataset: RasterArray = dataset.rio
    
    dataset = dataset.write_crs("EPSG:4326")
    
    dataset = dataset.clip(khmao_boundary, khmao_boundary.crs, drop=True)
    
    dataset = dataset.to_dataframe()
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=['number', 'spatial_ref'], errors='ignore')
    dataset = dataset.reset_index()
    
    return dataset

def upload_dataset_to_parquet(
    ds: pd.DataFrame
):
    ds["valid_time"] = pd.to_datetime(ds["valid_time"])
    ds["year"] = ds["valid_time"].dt.year
    ds.to_parquet(f"{PROCESSED_DIR}/khmao_master.parquet", index=True)
    
