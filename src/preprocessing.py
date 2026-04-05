from src.extract import data_loader
from src.config import Config
import numpy as np
import pandas as pd
import geopandas as gpd
import math

from rasterio.features import rasterize
import xarray as xr

from tqdm.auto import tqdm

import logging

KELVIN = 273.15
cfg = Config()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

def harmonize_fire_records(
    off_df: gpd.GeoDataFrame,
    v_df: gpd.GeoDataFrame,
    spatial_radius_m: float = 5000,
    temporal_window_days: int = 7
):
    off_df = off_df.copy()
    v_df = v_df.copy()
    
    off_df['acq_date'] = pd.to_datetime(off_df['acq_date'])
    v_df['acq_date'] = pd.to_datetime(v_df['acq_date'])
    
    original_crs = off_df.crs
    off_df['fire_id'] = off_df.index
    
    off_proj = off_df.to_crs("EPSG:3857")
    v_proj = v_df.to_crs("EPSG:3857")

    v_proj['viirs_date'] = v_proj['acq_date']
    v_proj_isolated = v_proj[['viirs_date', 'geometry']].copy()

    joined = gpd.sjoin_nearest(
        off_proj,
        v_proj_isolated,
        how='left',
        max_distance=spatial_radius_m,
        distance_col='spatial_distance'
    )
    
    joined['time_diff'] = joined['acq_date'] - joined['viirs_date']
    
    valid_temporal_mask = (
        (joined['time_diff'] >= pd.Timedelta(days=0)) &
        (joined['time_diff'] <= pd.Timedelta(days=temporal_window_days))
    )
    
    matched = joined[valid_temporal_mask].copy()
    
    if not matched.empty:
        matched = matched.sort_values(by=['fire_id', 'viirs_date'], ascending=[True, True])        
        matched = matched.drop_duplicates(subset=['fire_id'], keep='first')
        off_proj.loc[off_proj['fire_id'].isin(matched['fire_id']), 'acq_date'] = matched.set_index('fire_id')['viirs_date']
        match_count = len(matched)
        logger.info(f"Successfully harmonized {match_count} records ({match_count/len(off_proj)*100:.2f}%).")
    else:
        logger.warning("No records matched within the specified spatiotemporal parameters.")
    
    off_proj = off_proj.drop(columns=['fire_id'])
    harmonized_gdf = off_proj.to_crs(original_crs)
    return harmonized_gdf

def process_data(target_resolution=0.25, time_agg='monthly', use_area=True, min_area=10):
    """ Data Integration with resampling to coarser resolution """
    topo = data_loader.load_static_raster(cfg.raw_dem)
    if topo is None:
        raise FileNotFoundError(f"Could not load DEM from {cfg.raw_dem}")
    
    lc = data_loader.load_static_raster(cfg.raw_landcover)
    ghm = data_loader.load_static_raster(cfg.raw_human_mod)
    cisi = data_loader.load_static_raster("/home/lirn/geo_env/data/raw/khmao_cisi_1km.tif")
    oil_gas = data_loader.load_static_raster(cfg.raw_oil_gas)
    peat = data_loader.load_static_raster(cfg.raw_peatland)
    pop = data_loader.load_static_raster(cfg.raw_pop_density)
    ds = data_loader.load_meterological(cfg.raw_weather)
    fire_data = data_loader.load_russian_fires("data/raw/fires_inside_borders.csv")   
    viirs_firms = data_loader.load_firms("/home/lirn/geo_env/data/raw/fire_archive_modis.csv")
    ndvi = data_loader.load_gee_ndvi("/home/lirn/geo_env/data/raw/khmao_ndvi_monthly_2010_2025.tif")
    lai = data_loader.load_gee_ndvi("/home/lirn/geo_env/data/raw/khmao_lai_monthly_2010_2024.tif", end_year=2024)

    if viirs_firms is not None and not viirs_firms.empty:
        fire_data = harmonize_fire_records(
            off_df=fire_data, 
            v_df=viirs_firms, 
            spatial_radius_m=5000, 
            temporal_window_days=7
        )
    if time_agg == 'monthly':
        climate_freq = '1ME'
        fire_data['period'] = fire_data['acq_date'].dt.to_period('M')
        fire_period_col = 'period'
    elif time_agg == 'quarterly':
        climate_freq = '1QE'
        fire_data['period'] = fire_data['acq_date'].dt.to_period('Q')
        fire_period_col = 'period'
    elif time_agg == 'yearly':
        climate_freq = '1YE'
        fire_data['period'] = fire_data['acq_date'].dt.to_period('Y')
        fire_period_col = 'period'
        
        
    monthly = ds.resample(valid_time=climate_freq).mean()
    
    t2m = monthly["t2m"]
    d2m = monthly["d2m"]
    u10 = monthly["u10"]
    v10 = monthly["v10"]
    sm1 = monthly["swvl1"]
    vpd = calculate_vpd(t2m, d2m)
    
    tp = ds["tp"].resample(valid_time=climate_freq).sum() * 1000 
    
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
    
    u10_coarse = u10.interp(latitude=new_lat, longitude=new_lon, method="linear")
    u10_coarse = u10_coarse.rename({'latitude': 'y', 'longitude': 'x'})
    
    v10_coarse = v10.interp(latitude=new_lat, longitude=new_lon, method="linear")
    v10_coarse = v10_coarse.rename({'latitude': 'y', 'longitude': 'x'})
    
    ndvi_coarse = ndvi.interp(y=new_lat, x=new_lon, method="linear")
    ndvi_coarse = ndvi_coarse.fillna(0)
    ndvi_coarse.name = "ndvi"
    
    lai_coarse = lai.interp(y=new_lat, x=new_lon, method="linear")
    lai_coarse = lai_coarse.fillna(0)
    lai_coarse.name = "lai"
    
    static_stack = {
        "dem": topo.sel(band=1),
        "slope": topo.sel(band=2),
        "landcover": lc,
        "ghm": ghm,
        "cisi": cisi,
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
            if name in ["slope", "ghm", "cisi"]:
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
    
    if use_area and 'area_total' in fire_data.columns:
        fire_data = fire_data[fire_data['area_total'] > min_area].copy()
    
    fire_data['period_key'] = fire_data[fire_period_col]
    grouped = fire_data.groupby('period_key')    


    if time_agg == 'monthly':
        period_str = 'M'
    elif time_agg == 'quarterly':
        period_str = 'Q'
    elif time_agg == 'yearly':
        period_str = 'Y'
    fire_rasters = []
    def buffer_point(point, area_ha):
        area_m2 = area_ha * 1000
        radius = math.sqrt(area_m2 / math.pi)
        return point.buffer(radius)
    
    for time in tqdm(t2m_coarse.valid_time.values, desc="Rasterizing Fire Data"):
        period = pd.to_datetime(time).to_period(period_str)
        if period in grouped.groups:
            monthly_fires = grouped.get_group(period).copy()
            if use_area:
                metric_fires = gpd.GeoDataFrame(
                    monthly_fires,
                    geometry='geometry',
                    crs='EPSG:4326').to_crs('EPSG:3857')
                
                areas_m2 = metric_fires['area_total'].fillna(10) * 10000 
                radii = np.sqrt(areas_m2 / np.pi)
                metric_fires['geometry'] = metric_fires.geometry.buffer(radii)
                
                deg_fires = metric_fires.to_crs("EPSG:4326")
                shapes = [(geom, 1) for geom in deg_fires.geometry]
                
                fire_array = rasterize(shapes, out_shape=template.shape,
                                       transform=template.rio.transform(),
                                       fill=0, dtype=np.uint8, all_touched=True)
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
        "u10": u10_coarse,
        "v10": v10_coarse,
        "ndvi": ndvi_coarse,
        "lai": lai_coarse,
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
    correlation = ds['ghm'].corr(ds['cisi'])
    print(f"Correlation between GHM and CISI: {correlation}")
    ds.to_parquet(cfg.processed_table, index=True)
    
