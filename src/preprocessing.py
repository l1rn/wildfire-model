from src.extract import data_loader
from src.config import RAW_DIR, PROCESSED_DIR, Config
import numpy as np
import pandas as pd
import geopandas as gpd

from rasterio.features import rasterize
import xarray as xr

from tqdm.auto import tqdm

KELVIN = 273.15
cfg = Config()

def calculate_vpd(t2m_k, d2m_k):
    """
    Calculate Vapor Pressure Deficit (VPD) from temperature and dewpoint.
    
    Parameters
    ----------
    t2m_k : xr.DataArray
        Air temperature at 2m (Kelvin).
    d2m_k : xr.DataArray
        Dewpoint temperature at 2m (Kelvin).
    
    Returns
    -------
    xr.DataArray
        VPD in hPa (hectopascals).
    """
    t_c = t2m_k - KELVIN
    d_c = d2m_k - KELVIN

    es = 610.78 * np.exp((17.2694 * t_c) / (t_c + 237.3))
    ea = 610.78 * np.exp((17.2694 * d_c) / (d_c + 237.3))
    
    vpd_pa = es - ea
    return vpd_pa / 100.0

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

def process_data(target_resolution=0.25, time_agg='monthly', use_area=True, min_area=0):
    """ Data Integration with resampling to coarser resolution """
    topo = data_loader.load_static_raster(cfg.raw_dem)
    if topo is None:
        raise FileNotFoundError(f"Could not load DEM from {cfg.raw_dem}")
    
    lc = data_loader.load_static_raster(cfg.raw_landcover)
    ghm = data_loader.load_static_raster(cfg.raw_human_mod)
    oil_gas = data_loader.load_static_raster(cfg.raw_oil_gas)
    peat = data_loader.load_static_raster(cfg.raw_peatland)
    pop = data_loader.load_static_raster(cfg.raw_pop_density)
    ds = data_loader.load_meterological(cfg.raw_weather)
    firms = data_loader.load_russian_fires(cfg.raw_fire_data)   

    if time_agg == 'monthly':
        climate_freq = '1ME'
        fire_period_col = 'acq_date'
    elif time_agg == 'quarterly':
        climate_freq = '1QE'
        firms['period'] = firms['acq_date'].dt.to_period('Q')
        fire_period_col = 'period'
    elif time_agg == 'yearly':
        climate_freq = '1YE'
        firms['period'] = firms['acq_date'].dt.to_period('Y')
        fire_period_col = 'period'
    monthly = ds.resample(valid_time=climate_freq).mean()
    
    t2m = monthly["t2m"]
    d2m = monthly["d2m"]
    u10 = monthly["u10"]
    v10 = monthly["v10"]
    sm1 = monthly["swvl1"]
    tp = ds["tp"].resample(valid_time=climate_freq).sum() * 1000 
    
    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_speed.name = "wind_speed"
    vpd = calculate_vpd(t2m, d2m)

    lat_orig = t2m.latitude.values
    lon_orig = t2m.longitude.values
    lat_min, lat_max = lat_orig.min(), lat_orig.max()
    lon_min, lon_max = lon_orig.min(), lon_orig.max()
    
    new_lat = np.arange(lat_min + target_resolution/2, lat_max + target_resolution/2, target_resolution)
    new_lon = np.arange(lon_min + target_resolution/2, lon_max + target_resolution/2, target_resolution)
    new_lat = new_lat[(new_lat >= lat_min) & (new_lat <= lat_max)]
    new_lon = new_lon[(new_lon >= lon_min) & (new_lon <= lon_max)]
    new_lat = new_lat[::-1]
    
    template = xr.DataArray(
        np.zeros((len(new_lat), len(new_lon))),
        dims=("y", "x"),
        coords={"y": new_lat, "x": new_lon},
    ).rio.write_crs("EPSG:4326")
    
    t2m_coarse = t2m.interp(latitude=new_lat, longitude=new_lon, method="linear")
    t2m_coarse = t2m_coarse.rename({'latitude': 'y', 'longitude': 'x'})
    t2m_coarse = t2m_coarse.rio.write_crs("EPSG:4326")
    
    d2m_coarse = d2m.interp(latitude=new_lat, longitude=new_lon, method="linear")
    d2m_coarse = d2m_coarse.rename({'latitude': 'y', 'longitude': 'x'})
    
    vpd_coarse = vpd.interp(latitude=new_lat, longitude=new_lon, method="linear")
    vpd_coarse = vpd_coarse.rename({'latitude': 'y', 'longitude': 'x'})
    
    tp_coarse = tp.interp(latitude=new_lat, longitude=new_lon, method="linear")
    tp_coarse = tp_coarse.rename({'latitude': 'y', 'longitude': 'x'})
    
    sm1_coarse = sm1.interp(latitude=new_lat, longitude=new_lon, method="linear")
    sm1_coarse = sm1_coarse.rename({'latitude': 'y', 'longitude': 'x'})
    
    wind_speed_coarse = wind_speed.interp(latitude=new_lat, longitude=new_lon, method="linear")
    wind_speed_coarse = wind_speed_coarse.rename({'latitude': 'y', 'longitude': 'x'})
    
    static_stack = {
        "dem": topo.sel(band=1),
        "slope": topo.sel(band=2),
        "landcover": lc,
        "ghm": ghm,
        "pop_density": pop,
        "dist_oil_gas": oil_gas,
        "peatland": peat
    }
    
    processed_static = {}
    for name, da in static_stack.items():
        da_matched = da.rio.reproject_match(template)
        if 'band' in da_matched.dims:
            da_matched = da_matched.squeeze("band", drop=True)
        da_matched = da_matched.drop_vars("band", errors="ignore")
        processed_static[name] = da_matched
        
    for name, da in list(processed_static.items()):
        nans_before = da.isnull().sum().item()
        if nans_before > 0:
            if name in ["slope", "ghm"]:
                fill_value = float(da.median())
                processed_static[name] = da.fillna(fill_value)
            elif name == "dist_oil_gas":
                fill_value = float(da.max())
                processed_static[name] = da.fillna(fill_value)
            elif name in ["landcover", "peatland"]:
                fill_value = int(da.mode(dim=["x", "y"]).isel(mode=0).compute())
                processed_static[name] = da.fillna(fill_value).astype(int)
            else:
                processed_static[name] = da.fillna(0)
    
    if use_area and 'area_total' in firms.columns:
        firms = firms[firms['area_total'] > min_area].copy()
    
    firms['period_key'] = firms[fire_period_col]
    grouped = firms.groupby('period_key')    


    if time_agg == 'monthly':
        period_str = 'M'
    elif time_agg == 'quarterly':
        period_str = 'Q'
    elif time_agg == 'yearly':
        period_str = 'Y'
    fire_rasters = []
    for time in tqdm(t2m_coarse.valid_time.values, desc="Rasterizing Fire Data"):
        period = pd.to_datetime(time).to_period(period_str)
        if period in grouped.groups:
            monthly_fires = grouped.get_group(period)
            if use_area:
                shapes = [(row.geometry, row['area_total']) for _, row in monthly_fires.iterrows()]
                fire_array = rasterize(shapes, out_shape=template.shape,
                                       transform=template.rio.transform(),
                                       fill=0, dtype=np.float32)
                fire_array = (fire_array > 0).astype(np.uint8)
            else:
                shapes = [(geom, 1) for geom in monthly_fires.geometry]
            
                fire_array = rasterize(shapes, out_shape=template.shape,
                    transform=template.rio.transform(), fill=0, dtype=np.uint8
                )
        else:
            fire_array = np.zeros(template.shape, dtype=np.uint8)
            
        fire_rasters.append(fire_array)
        
    fire_stack = np.stack(fire_rasters)
    fire_coarse = xr.DataArray(fire_stack, dims=("valid_time", "y", "x"),
                               coords={"valid_time": t2m_coarse.valid_time,
                                       "y": t2m_coarse.y, "x": t2m_coarse.x},
                               name="fire")
    
    dataset_dict = {
        "temp": t2m_coarse,
        "dew": d2m_coarse,
        "vpd": vpd_coarse,
        "precip": tp_coarse,
        "sm1": sm1_coarse,
        "wind_speed": wind_speed_coarse,
        "fire": fire_coarse
    }
    
    dataset_dict.update(processed_static)
    dataset = xr.Dataset(dataset_dict)
    
    khmao_boundary = gpd.read_file(cfg.khmao_geojson)
    dataset = dataset.rio.clip(
        khmao_boundary.geometry,
        khmao_boundary.crs,
        drop=True,
        all_touched=True
    )
    
    df = (
        dataset
        .stack(points=("x", "y", "valid_time"))
        .dropna("points", subset=["temp", "dem"])
    )
    
    df_final = df.to_dataframe().reset_index()
    if time_agg == 'quarterly':
        df_final['quarter'] = df_final['valid_time'].dt.quarter
    elif time_agg == 'yearly':
        df_final['year'] = df_final['valid_time'].dt.year

    return df_final

def upload_dataset_to_parquet(
    ds: pd.DataFrame
):
    ds["valid_time"] = pd.to_datetime(ds["valid_time"])
    ds["year"] = ds["valid_time"].dt.year
    
    ds.to_parquet(cfg.processed_table, index=True)
    
