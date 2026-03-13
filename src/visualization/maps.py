import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from src.config import RAW_DIR, Config
from matplotlib.colors import ListedColormap
from shapely.geometry import Point 

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
    
    if subset.empty:
        print(f"Error: No geographic data available for {year}-{month:02d}. Cannot generate map.")
        return
    
    subset["x_rounded"] = subset["x"].round(2)
    subset["y_rounded"] = subset["y"].round(2)   
    
    risk_map = subset.pivot(
        index="y_rounded",
        columns="x_rounded",
        values="fire_probability"
    ).fillna(0)
    
    xmin, xmax = risk_map.columns.min(), risk_map.columns.max()
    ymin, ymax = risk_map.index.min(), risk_map.index.max()
    
    plt.figure(figsize=(14, 8), facecolor='white')

    plt.imshow(
        risk_map.values, 
        origin="lower",
        extent=[xmin, xmax, ymin, ymax], 
        vmin=0, 
        vmax=1,
        cmap="plasma"
    )
    
    plt.title(f"{title}\n(Spatially Smoothed, $\sigma={sigma}$)", fontsize=16, pad=20)
    plt.xlabel("Longitude", fontsize=10)
    plt.ylabel("Latitude", fontsize=10)
    
    plt.grid(color='black', linestyle='--', linewidth=0.2, alpha=0.5)
    
    plt.tight_layout()
    plt.show()
 
def plot_historical_fires(
    csv_path: str,
    geojson_path: str,
    target_year: int,
    target_month: int
):
    df = pd.read_csv(csv_path)
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    
    subset = df[
        (df['acq_date'].dt.year == target_year) &
        (df['acq_date'].dt.month == target_month)
    ]
    
    if subset.empty:
        print("No thermal anomalies")
        return
    
    wildfires = subset[subset['type'] == 0]
    
    if wildfires.empty:
        print("No wildfires")
        return
    
    gdf_fires = gpd.GeoDataFrame(
        wildfires,
        geometry=gpd.points_from_xy(wildfires.longitude, wildfires.latitude),
        crs="EPSG:4326"
    ) 
    
    khmao_boundary = gpd.read_file(geojson_path)
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    khmao_boundary.plot(ax=ax, facecolor="#e8f4f8", edgecolor="black", linewidth=1.5)
    
    gdf_fires.plot(
        ax=ax, 
        color="red", 
        markersize=15, 
        alpha=0.7, 
        edgecolor="darkred",
        label=f"Wildfires (n={len(gdf_fires)})"
    )
    
    plt.title(f"Observed Wildfire Ignitions (FIRMS) – KhMAO ({target_month:02d}/{target_year})", fontsize=16, pad=15)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
def plot_landcover_map(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    subset = df[
        (df['valid_time'].dt.year == 2022) & 
        (df['valid_time'].dt.month == 7)
    ].copy()
    
    if subset.empty:
        print("Error: Could not extract spatial grid for the specified timeframe.")
        return

    subset["x_rounded"] = subset["x"].round(2)
    subset["y_rounded"] = subset["y"].round(2)   
    
    lc_map = subset.pivot(
        index="y_rounded",
        columns="x_rounded",
        values="landcover"
    ).fillna(-1) 
    
    esa_colors = {
        10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff",
        50: "#fa0000", 60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8",
        90: "#0096a0", 95: "#00cf75", 100: "#fae6a0"
    }   

    colors = ["#000000"] * 101
    for val, hex_code in esa_colors.items():
        colors[val] = hex_code
        
    custom_cmap = ListedColormap(colors)
    
    plt.figure(figsize=(14, 8), facecolor='white')
    
    xmin, xmax = lc_map.columns.min(), lc_map.columns.max()
    ymin, ymax = lc_map.index.min(), lc_map.index.max()
    
    im = plt.imshow(
        lc_map.values, 
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=custom_cmap,
        vmin=0, vmax=100,
        interpolation='nearest'
    )
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("ESA WorldCover Classification Code", fontsize=12)
    
    plt.title("KhMAO Landcover Classification Grid", fontsize=16, pad=15)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.show()
  
def create_bivariate_map(
    df, var1='vpd', var2='ghm', output_path='docs/bivariate.png'
):
    subset = df[
        (df["valid_time"].dt.year == 2022) & 
        (df["valid_time"].dt.month == 7)
    ].copy()
    
    if subset.empty:
        print("⚠️ No data for July 2022. Using the entire dataset instead.")
        subset = df.copy()

    print("⚙️ Converting grid points to spatial geometries...")
    geometry = [Point(xy) for xy in zip(subset['x'], subset['y'])]
    gdf = gpd.GeoDataFrame(subset, geometry=geometry, crs="EPSG:4326")
    
    gdf['vpd_class'] = pd.qcut(gdf[var1].rank(method='first'), 3, labels=[0, 1, 2])
    gdf['ghm_class'] = pd.qcut(gdf[var2].rank(method='first'), 3, labels=[0, 1, 2])
    
    gdf['bivariate_class'] = gdf['vpd_class'].astype(str) + gdf['ghm_class'].astype(str)
    
    color_dict = {
        '00': '#e8e8e8', '10': '#e4acac', '20': '#c85a5a',
        '01': '#b0d5df', '11': '#ad9ea5', '21': '#985356', 
        '02': '#64acbe', '12': '#627f8c', '22': '#574249' 
    }
    
    gdf['color'] = gdf['bivariate_class'].map(color_dict)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf.plot(color=gdf['color'], edgecolor='none', ax=ax)
    
    ax.set_title("Synergistic Wildfire Drivers: VPD and Human Modification", fontsize=16)
    ax.set_axis_off()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    khmao_boundary = gpd.read_file(Config().khmao_geojson)
    da_smooth = da.rio.write_crs("EPSG:4326")
    da_smooth = da_smooth.rio.clip(khmao_boundary.geometry, khmao_boundary.crs, drop=True)
    da_smooth = da_smooth.rio.write_nodata(-9999, inplace=True)
    da_smooth.astype("float32").rio.to_raster(filename)
    
    print(f"Saved georeferenced TIF to {filename}")