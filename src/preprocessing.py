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

def unify_xy(*arrays):
    return [da.rename({"latitude": "y", "longitude": "x"}) for da in arrays]

def write_crs_all(crs, *arrays):
    return [da.rio.write_crs(crs) for da in arrays]

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
    transform = template.rio.transform()    
    
    grouped = firms_gdf.groupby("year_month")
    fire_rasters = []
    for time in tqdm(climate_da.valid_time.values, desc="Rasterizing Fire Data"):
        month = pd.to_datetime(time).to_period("M")
        if month in grouped.groups:
            monthly_fires = grouped.get_group(month)
            shapes = [(geom, 1) for geom in monthly_fires.geometry]
            
            fire_array = rasterize(
                shapes,
                out_shape=template.shape,
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
        else:
            fire_array = np.zeros(template.shape, dtype=np.uint8)
            
        fire_rasters.append(fire_array)
        
    fire_stack = np.stack(fire_rasters)
    
    return xr.DataArray (
        fire_stack,
        dims=("valid_time", "y", "x"),
        coords={
            "valid_time": climate_da.valid_time,
            "y": climate_da.y,
            "x": climate_da.x,
        },
        name="file"
    )

def process_data():
    """ Data Integration """
    topo = data_loader.load_static_raster(f"{RAW_DIR}/khmao_terrain_1km.tif")
    lc = data_loader.load_static_raster(f"{RAW_DIR}/khmao_lc_1km.tif")
    ghm = data_loader.load_static_raster(f"{RAW_DIR}/khmao_human_mod_1km.tif")
    ds: xr.Dataset = data_loader.load_meterological(f"{RAW_DIR}/khmao_era5.nc")
    firms = data_loader.load_firms(f"{RAW_DIR}/khmao_fire_archive.csv")
    
    monthly = ds.resample(valid_time="1ME").mean()
    
    t2m = monthly["t2m"]
    d2m = monthly["d2m"]
    u10 = monthly["u10"]
    v10 = monthly["v10"]
    sm1 = monthly["swvl1"]
    tp = ds["tp"].resample(valid_time="1ME").sum() * 1000
    
    t2m, d2m, u10, v10, sm1, tp = write_crs_all(
        "EPSG:4326",
        t2m,
        d2m,
        u10,
        v10,
        sm1,
        tp
    )
    
    t2m = t2m.rio.write_crs("EPSG:4326")
    
    vpd = calculate_vpd(t2m, d2m)
    
    dem = topo.sel(band=1).rio.reproject_match(t2m).drop_vars("band", errors="ignore")
    slope = topo.sel(band=2).rio.reproject_match(t2m).drop_vars("band", errors="ignore")
    
    lc = lc.rio.reproject_match(t2m).squeeze("band", drop=True)
    ghm = ghm.rio.reproject_match(t2m).squeeze("band", drop=True)

    t2m, d2m, tp, vpd, \
    sm1, u10, v10 = unify_xy(
        t2m, d2m, tp, vpd, sm1, u10, v10,
    )
    
    dem, lc, ghm, slope = broadcast_static_layers(
        main_dim=t2m, 
        dem=dem, 
        lc=lc, 
        ghm=ghm,
        slope=slope
    )
    
    fire_monthly = rasterize_monthly_fire(
        firms_gdf=firms, climate_da=t2m
    )
    
    def align(da, target):
        return da.assign_coords(y=target.y, x = target.x)
    
    dataset = xr.Dataset({
        "temp": t2m,
        "vpd": align(vpd, t2m),
        "precip": align(tp, t2m),
        "u10": align(u10, t2m),
        "v10": align(v10, t2m),
        "dem": align(dem, t2m),
        "slope": align(slope, t2m),
        "sm1": align(sm1, t2m),
        "landcover": align(lc, t2m),
        "ghm": align(ghm, t2m),
        "fire": align(fire_monthly, t2m)
    })
    
    khmao_boundary = gpd.read_file(f"{RAW_DIR}/khmao.geojson")
    
    dataset = dataset.rio.write_crs("EPSG:4326")
    dataset = dataset.rio.clip(khmao_boundary.geometry, khmao_boundary.crs, drop=True)
    
    df = (
        dataset
        .stack(points=("x", "y"))
        .dropna("points")
        .to_dataframe()
        .reset_index()
    )
    
    return df

def upload_dataset_to_parquet(
    ds: pd.DataFrame
):
    ds["valid_time"] = pd.to_datetime(ds["valid_time"])
    ds["year"] = ds["valid_time"].dt.year
    ds.to_parquet(f"{PROCESSED_DIR}/khmao_master.parquet", index=True)
    
