import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
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
    sigma: float = 1.5
):
    subset = df[
        (df["valid_time"].dt.year == year) &
        (df["valid_time"].dt.month == month)
    ].copy()
    
    subset["x_rounded"] = subset["x"].round(2)
    subset["y_rounded"] = subset["y"].round(2)   
    
    risk_map = subset.pivot(
        index="y_rounded",
        columns="x_rounded",
        values="fire_probability"
    ).fillna(0)
    
    data_mask = risk_map.notnull()
    
    smoothed_values = gaussian_filter(risk_map.values, sigma=sigma)

    colors = ["#228b22", "#ffff00", "#ff8c00", "#ff0000"]
    levels = [0, 0.2, 0.5, 0.8, 1.0]
    
    cmap = mcolors.ListedColormap(colors)
    cmap.set_bad(color='white', alpha=0)
    
    norm = mcolors.BoundaryNorm(levels, cmap.N) 
    xmin, xmax = risk_map.columns.min(), risk_map.columns.max()
    ymin, ymax = risk_map.index.min(), risk_map.index.max()
    plt.figure(figsize=(14, 8), facecolor='white')
    im = plt.imshow(
        smoothed_values, 
        origin="lower",
        extent=[xmin, xmax, ymin, ymax], 
        cmap=cmap,
        norm=norm,
        interpolation='bilinear'
    )
    
    plt.imshow(
        risk_map.values, 
        origin="lower",
        extent=[xmin, xmax, ymin, ymax], 
        vmin=0, 
        vmax=1,
        cmap="plasma"
    )
    cbar = plt.colorbar(im, spacing='proportional', shrink=0.7)
    cbar.set_label("Wildfire Risk Level", fontsize=12, fontweight='bold')
    cbar.set_ticks([0.1, 0.35, 0.65, 0.9])
    cbar.set_ticklabels(["Low", "Moderate", "High", "Extreme"])

    plt.title(f"{title}\n(Spatially Smoothed, $\sigma={sigma}$)", fontsize=16, pad=20)
    plt.xlabel("Longitude", fontsize=10)
    plt.ylabel("Latitude", fontsize=10)
    
    plt.grid(color='black', linestyle='--', linewidth=0.2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"fire_risk_{year}_{month}.png", dpi=300)
    print(f"Map saved as fire_risk_{year}_{month}.png")
    
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