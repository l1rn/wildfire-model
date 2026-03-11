from src.data import data_loader
from src.config import RAW_DIR, PROCESSED_DIR
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

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

def broadcast_static_layers(
    main_dim: xr.DataArray, **static_layers
):
    broadcasted = {}
    for name, layer in static_layers.items():
        layer_expanded = layer.expand_dims(valid_time=main_dim.valid_time)
        broadcasted[name] = layer_expanded
    return broadcasted

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
    oil_gas = data_loader.load_static_raster(f"{RAW_DIR}/khmao_oil_gas.tif")
    peat = data_loader.load_static_raster(f"{RAW_DIR}/khmao_peatland.tif")
    pop = data_loader.load_static_raster(f"{RAW_DIR}/khmao_pop.tif")
    ds = data_loader.load_meterological(f"{RAW_DIR}/khmao_era5.nc")
    firms = data_loader.load_firms(f"{RAW_DIR}/khmao_fire_archive.csv")
    
    monthly = ds.resample(valid_time="1ME").mean()
    
    t2m = monthly["t2m"]
    d2m = monthly["d2m"]
    u10 = monthly["u10"]
    v10 = monthly["v10"]
    sm1 = monthly["swvl1"]
    tp = ds["tp"].resample(valid_time="1ME").sum() * 1000
    
    static_stack = {
        "dem": topo.sel(band=1),
        "slope": topo.sel(band=2),
        "landcover": lc,
        "ghm": ghm,
        "pop_density": pop,
        "dist_oil_gas": oil_gas,
        "peatland": peat
    }
    
    t2m = t2m.rio.write_crs("EPSG:4326")
    
    vpd = calculate_vpd(t2m, d2m)


    t2m, d2m, tp, vpd, \
    sm1, u10, v10 = unify_xy(
        t2m, d2m, tp, vpd, sm1, u10, v10,
    )
    
    processed_static = {}
    for name, da in static_stack.items():
        da_matched = da.rio.reproject_match(t2m)
        if 'band' in da_matched.dims:
            da_matched = da_matched.squeeze("band", drop=True)  
        da_matched = da_matched.drop_vars("band", errors="ignore")
        processed_static[name] = da_matched
        
    for name, da in list(processed_static.items()):
        nans_before = da.isnull().sum().item()
        if nans_before > 0:
            if name in ["slope", "ghm", "dist_oil_gas"]:
                fill_value = float(da.median())
                processed_static[name] = da.fillna(fill_value)
                print(f"  {name}: filled {nans_before} NaNs with median = {fill_value:.2f}")
            elif name in ["landcover", "peatland"]:
                fill_value = int(da.mode(dim=["x", "y"].isel(mode=0)))
                processed_static[name] = da.fillna(fill_value)
                print(f"  {name}: filled {nans_before} NaNs with median = {fill_value}")
            else:
                processed_static[name] = da.fillna(0)

    processed_static["pop_density"] = processed_static["pop_density"].fillna(0)
    processed_static["dist_oil_gas"] = processed_static["dist_oil_gas"].fillna(processed_static["dist_oil_gas"].max())
    
    t2m = t2m.interpolate_na(dim="x", method="nearest")
    sm1 = sm1.interpolate_na(dim="x", method="nearest")
    
    broadcast_layers = broadcast_static_layers(t2m, **processed_static)
    
    wildfire = firms[firms['type'] == 0]
    industrial_heat = firms[firms['type'] == 2]
    
    fire_monthly = rasterize_monthly_fire(
        firms_gdf=wildfire, climate_da=t2m
    )
    
    def align(da, target):
        return (
            da.assign_coords(y=target.y, x = target.x)
            .rio.write_crs("EPSG:4326")
            .drop_vars("band", errors="ignore")
        )
    
    dataset_dict = {
        "temp": t2m,
        "vpd": align(vpd, t2m),
        "precip": align(tp, t2m),
        "sm1": align(sm1, t2m),
        "u10": align(u10, t2m),
        "v10": align(v10, t2m),
        "fire": align(fire_monthly, t2m)
    }
    
    for name, data in broadcast_layers.items():
        dataset_dict[name] = align(data, t2m)
    
    dataset = xr.Dataset(dataset_dict)
    
    khmao_boundary = gpd.read_file(f"{RAW_DIR}/khmao.geojson")
    dataset = dataset.rio.clip(
        khmao_boundary.geometry, 
        khmao_boundary.crs, 
        drop=True,
        all_touched=True
    )
    
    df = (
        dataset
        .stack(points=("x", "y", "valid_time"))
        .dropna("points", how="all")
    )
    
    df_final = df.to_dataframe().reset_index().fillna(0)
    
    return df_final

def upload_dataset_to_parquet(
    ds: pd.DataFrame
):
    ds["valid_time"] = pd.to_datetime(ds["valid_time"])
    ds["year"] = ds["valid_time"].dt.year
    ds.to_parquet(f"{PROCESSED_DIR}/khmao_master.parquet", index=True)
    
