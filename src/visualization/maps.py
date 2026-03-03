import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from src.config import RAW_DIR

def plot_month_map(
    df: pd.DataFrame,
    year: int,
    month: int,
    title: str,
):
    subset = df[
        (df["valid_time"].dt.year == year) &
        (df["valid_time"].dt.month == month)
    ]
    
    df = df.dropna(subset=["fire_probability"])    
    
    risk_map = subset.pivot(
        index="y",
        columns="x",
        values="fire_probability"
    )
    
    xmin, xmax = risk_map.columns.min(), risk_map.columns.max()
    ymin, ymax = risk_map.index.min(), risk_map.index.max()

    plt.figure(figsize=(14, 8))
    plt.imshow(
        risk_map.values, 
        origin="lower",
        extent=[xmin, xmax, ymin, ymax], 
        vmin=0, 
        vmax=1,
        cmap="plasma"
    )
    plt.colorbar(label="Wildfire Probability")
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()
    
def save_to_geotiff(
    df: pd.DataFrame,
    year: int,
    month: int,
    filename: str
):
    subset = df[
        (df["valid_time"].dt.year == year) &
        (df["valid_time"].dt.month == month)
    ]

    risk_grid = subset.pivot(index="y", columns="x", values="fire_probability")
    da = xr.DataArray(
        data=risk_grid.values,
        dims=("y", "x"),
        coords={"y":risk_grid.index, "x":risk_grid.columns}
    )
    mean_risk = da.mean().item()
    da = da.fillna(mean_risk)
    da.rio.write_crs("EPSG:4326", inplace=True)
    # y_min, y_max = da.y.min().item(), da.y.max().item()
    # x_min, x_max = da.x.min().item(), da.x.max().item()
    
    # new_y = np.linspace(y_min, y_max, 1000)
    # new_x = np.linspace(x_min, x_max, 1000)
    
    # da_smooth = da.interp(y=new_y, x=new_x, method="linear")
    khmao_boundary = gpd.read_file(f"{RAW_DIR}/khmao.geojson")
    da_smooth = da.rio.write_crs("EPSG:4326")
    da_smooth = da_smooth.rio.clip(khmao_boundary.geometry, khmao_boundary.crs, drop=True)
    da_smooth = da_smooth.rio.write_nodata(-9999, inplace=True)
    da_smooth.astype("float32").rio.to_raster(filename)
    
    print(f"Saved georeferenced TIF to {filename}")