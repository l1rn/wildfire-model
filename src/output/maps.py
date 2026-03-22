import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from src.config import Config
from matplotlib.colors import ListedColormap
from shapely.geometry import Point 

import matplotlib.animation as animation

cfg = Config()
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
        print("No data for July 2022. Using the entire dataset instead.")
        subset = df.copy()

    print("Converting grid points to spatial geometries...")
    res = 0.05
    subset['x_grid'] = (subset['x'] / res).round() * res
    subset['y_grid'] = (subset['y'] / res).round() * res
    
    grid = subset.groupby(['x_grid', 'y_grid']).agg({
        var1: 'mean',
        var2: 'mean'
    }).reset_index()
    
    
    grid[f'{var1}_class'] = pd.qcut(grid[var1].rank(method='first'), 3, labels=[0, 1, 2])
    grid[f'{var2}_class'] = pd.qcut(grid[var2].rank(method='first'), 3, labels=[0, 1, 2])
    grid['bivariate_class'] = grid[f'{var1}_class'].astype(str) + grid[f'{var2}_class'].astype(str)
    
    color_dict = {
        '00': '#e8e8e8', '10': '#e4acac', '20': '#c85a5a',
        '01': '#b0d5df', '11': '#ad9ea5', '21': '#985356', 
        '02': '#64acbe', '12': '#627f8c', '22': '#574249' 
    }
    
    grid['color'] = grid['bivariate_class'].map(color_dict)
    gdf = gpd.GeoDataFrame(
        grid, 
        geometry=gpd.points_from_xy(grid.x_grid, grid.y_grid), 
        crs="EPSG:4326"
    )
    
    boundary = gpd.read_file(cfg.khmao_geojson)
    gdf = gdf.clip(boundary)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf.plot(color=gdf['color'], edgecolor='none', ax=ax)    
    ax.set_title("Synergistic Wildfire Drivers: VPD and Human Modification", fontsize=16)
    ax.set_axis_off()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def animate_risk_over_time(
    df, years=None, output_file=cfg.risk_map_animation_output, fps=2, boundary_geojson = cfg.khmao_geojson
    ):
    if 'valid_time' not in df.columns:
        print("DataFrame must have a 'valid_time' column.")
        return
    df['year_month'] = df['valid_time'].dt.to_period('M')
    time_points = sorted(df['year_month'].unique())
    if len(time_points) == 0:
        print("No time data available.")
        return

    if years is not None:
        df = df[df['valid_time'].dt.year.isin(years)].copy()
        if df.empty:
            print(f"No data for years {years}. Cannot create animation.")
            return

    df['year_month'] = df['valid_time'].dt.to_period('M')
    time_points = sorted(df['year_month'].unique())
    if len(time_points) == 0:
        print("No time data available.")
        return
    
    df['x_round'] = df['x'].round(2)
    df['y_round'] = df['y'].round(2)
    
    xs = sorted(df['x_round'].unique())
    ys = sorted(df['y_round'].unique())

    boundary = gpd.read_file(boundary_geojson)
    if boundary.crs is None:
        boundary = boundary.set_crs('EPSG:4326')
    elif boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs('EPSG:4326')
    boundary = boundary.dissolve()
    geom = boundary.geometry.iloc[0]
    
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if geom.contains(Point(x, y)):
                mask[j, i] = True 
    
    mask_df = pd.DataFrame(mask, index=ys, columns=xs)
    mask_df = mask_df.sort_index(axis=0).sort_index(axis=1)
    
    xmin, xmax = xs[0], xs[-1]
    ymin, ymax = ys[0], ys[-1]

    cmap = plt.cm.plasma
    cmap.set_bad(alpha=0) 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(t):
        ax.clear()
        subset = df[df['year_month'] == t]
        
        if subset.empty:
            dummy_grid = np.zeros((10, 10))
            im = ax.imshow(dummy_grid, origin='lower',
                           extent=[xmin, xmax, ymin, ymax],
                           cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"Fire probability – {t} (no data)")
        else:
            risk_grid = subset.pivot_table(
                index='y_round', columns='x_round', values='fire_probability', aggfunc='mean'
            ).sort_index(axis=0).sort_index(axis=1)
            risk_grid = risk_grid.fillna(0)
            risk_grid = risk_grid.where(mask_df, np.nan)
            risk_grid = risk_grid.sort_index(axis=0).sort_index(axis=1)
            
            im = ax.imshow(risk_grid.values, origin='lower',
                            extent=[risk_grid.columns.min(), risk_grid.columns.max(),
                                    risk_grid.index.min(), risk_grid.index.max()],
                            cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"Fire probability – {t}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle='--', alpha=0.3)
        return im
        
    ani = animation.FuncAnimation(fig, update, frames=time_points, repeat=False, interval=1000/fps)
    ani.save(output_file, writer='pillow', fps=fps)
    plt.close()
    print(f"Animation saved to {output_file}")
        
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
    
def plot_time_series_risk(
    df, output_file='time_series_risk.png', freq='M', tick_interval=6
):
    df = df.copy()
    df['date'] = pd.to_datetime(df['valid_time'])
    
    if freq == 'M':
        df['period'] = df['date'].dt.to_period('M')
        xlabel = 'Month'
        all_periods = sorted(df['period'].unique())
        tick_indices = range(0, len(all_periods), tick_interval)
        tick_labels = [str(p) for i, p in enumerate(all_periods) if i in tick_indices]
        all_periods_str = [str(p) for p in all_periods]
    else:
        df['period'] = df['date'].dt.to_period('Y')
        xlabel = 'Year'
    
    grouped = df.groupby('period').agg({
        'fire_probability': 'sum',
        'fire': 'sum'
    }).reset_index()
    
    grouped = grouped.set_index('period').reindex(all_periods).reset_index()
    grouped['period_str'] = [str(p) for p in grouped['period']]
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    color = 'tab:red'
    
    ax1.bar(grouped['period_str'], grouped['fire'], color=color, alpha = 0.6, label='Observed Fires')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Observed fire count', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(grouped['period_str'], grouped['fire_probability'], color=color, marker='o', linewidth=2, label='Sum of predicted probabilities')
    ax2.set_ylabel('Sum of predicted probabilities', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.set_xticks([grouped['period_str'][i] for i in tick_indices])
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    for tick in ax1.get_yticklabels():
        tick.set_fontsize(9)
    
    plt.title('Time Series of Fire Risk: Predicted vs. Observed')
    fig.tight_layout()
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f'Time Series plot saved to {output_file}')