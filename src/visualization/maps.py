import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np

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

    plt.figure(figsize=(12, 6))
    plt.imshow(risk_map, origin="lower", vmin=0, vmax=1)
    plt.colorbar(label="Wildfire Probability")
    plt.title(title)
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
    
    da.rio.write_crs("EPSG:4326", inplace=True)
    y_min, y_max = da.y.min().item(), da.y.max().item()
    x_min, x_max = da.x.min().item(), da.x.max().item()
    
    new_y = np.linspace(y_min, y_max, 1000)
    new_x = np.linspace(x_min, x_max, 1000)
    
    da_smooth = da.interp(y=new_y, x=new_x, method="linear")
    da_smooth = da_smooth.astype("float32")
    da_smooth.rio.to_raster(
        filename, 
        compress='DEFLATE', 
        zlevel=9
    )
    
    print(f"Saved georeferenced TIF to {filename}")