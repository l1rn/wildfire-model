from enum import Enum
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from matplotlib.colors import ListedColormap
from src.config import PROCESSED_DIR

class Risk(Enum):
    GREEN = 1
    YELLOW = 2
    ORANGE = 3
    RED = 4

class VPDThresholdPipeline:
    def __init__(self, df): 
        self.df = df
        self.original_df = df.copy()
    
    def get_month_data(self, year, month):
        """Filters and calculates risk indices dynamically for a target month."""
        month_df = self.original_df[
            (self.original_df["year"] == year) &
            (self.original_df["month"] == month)
        ].copy()
        
        month_df["vpd_risk"] = np.select(
            [
                month_df["vpd"] < 2,
                (month_df["vpd"] >= 2) & (month_df["vpd"] < 3.6),
                (month_df["vpd"] >= 3.6) & (month_df["vpd"] < 5),
                month_df["vpd"] >= 5,
            ],
            [1, 2, 3, 4],
            default=1
        )

        month_df["vpd_ghm_ndvi_risk"] = np.select(
            [
                month_df["vpd"] < 2,
                (month_df["vpd"] >= 2) & (month_df["vpd"] < 3.6),
                (month_df["vpd"] >= 3.6) & (month_df["vpd"] < 5),
                month_df["vpd"] >= 5,
            ],
            [1, 2, 3, 4],
            default=1
        )
        return month_df

    def plot_vpd_collage(self, year=2022):
        khmao = gpd.read_file("data/processed/khmao.geojson").to_crs("EPSG:4326")

        cmap = ListedColormap([
            "#4CAF50", 
            "#FFD54F",  
            "#FB8C00",  
            "#D32F2F", 
        ])  

        months = [5, 6, 8]
        month_names = {5: "May", 6: "June", 8: "August"}

        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
        
        im = None  

        for i, month in enumerate(months):
            ax = axes[i]
            month_df = self.get_month_data(year, month)
            
            if month_df.empty:
                ax.text(0.5, 0.5, f"No Data for {month_names[month]}", 
                        ha='center', va='center', transform=ax.transAxes)
                continue

            grid = month_df.pivot_table(index="y", columns="x", values="vpd_risk")
            data = grid.values
            lon = grid.columns.values
            lat = grid.index.values

            im = ax.imshow(
                data,
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                origin="lower",
                cmap=cmap,
                vmin=1,
                vmax=4,
            )

            bbox = box(lon.min(), lat.min(), lon.max(), lat.max())
            khmao.clip(bbox).boundary.plot(ax=ax, color="black", linewidth=1)

            ax.set_title(f"{month_names[month]} {year}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Longitude")
            if i == 0:
                ax.set_ylabel("Latitude")

        fig.tight_layout()
        if im is not None:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=[1, 2, 3, 4], 
                                shrink=0.7, pad=0.02, location="right")
            cbar.ax.set_yticklabels(['Low', 'Moderate', 'High', 'Critical'])

        output_path = Path(PROCESSED_DIR) / "vpd/vpd_seasonal_collage.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Seasonal spatial collage successfully saved to: {output_path}")

    def get_interaction_month_data(self, year, month):
        """Filters and calculates risk indices by evaluating the three-way interaction."""
        month_df = self.original_df[
            (self.original_df["year"] == year) &
            (self.original_df["month"] == month)
        ].copy()
        
        if month_df.empty:
            return month_df

        vpd_conds = [
            month_df["vpd"] < 2,
            (month_df["vpd"] >= 2) & (month_df["vpd"] < 3.6),
            (month_df["vpd"] >= 3.6) & (month_df["vpd"] < 5),
            month_df["vpd"] >= 5,
        ]
        month_df["vpd_tier"] = np.select(vpd_conds, [1, 2, 3, 4], default=1)

        month_df["interaction_score"] = month_df["ghm"] * (month_df["ndvi"] - 1)
        high_interaction_cutoff = month_df["interaction_score"].quantile(0.75)

        interaction_condition = month_df["interaction_score"] >= high_interaction_cutoff
        
        month_df["risk"] = np.where(
            interaction_condition & (month_df["vpd_tier"] < 4) & (month_df["vpd_tier"] > 1),
            month_df["vpd_tier"] + 1,
            month_df["vpd_tier"]
        )
        
        return month_df
    
    def plot_interaction_collage(self, year=2022):
        khmao = gpd.read_file("data/processed/khmao.geojson").to_crs("EPSG:4326")

        cmap = ListedColormap([
            "#4CAF50",  
            "#FFD54F",
            "#FB8C00", 
            "#D32F2F", 
        ])  

        months = [5, 6, 8]
        month_names = {5: "May", 6: "June", 8: "August"}

        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
        im = None 

        for i, month in enumerate(months):
            ax = axes[i]
            month_df = self.get_interaction_month_data(year, month)
            
            if month_df.empty:
                ax.text(0.5, 0.5, f"No Data for {month_names[month]}", 
                        ha='center', va='center', transform=ax.transAxes)
                continue

            grid = month_df.pivot_table(index="y", columns="x", values="risk")
            data = grid.values
            lon = grid.columns.values
            lat = grid.index.values

            im = ax.imshow(
                data,
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                origin="lower",  
                cmap=cmap,
                vmin=1,
                vmax=4,
            )
            bbox = box(lon.min(), lat.min(), lon.max(), lat.max())
            khmao.clip(bbox).boundary.plot(ax=ax, color="black", linewidth=1.2)

            ax.set_title(f"{month_names[month]} {year}\n(VPD × GHM × NDVI)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Longitude")
            if i == 0:
                ax.set_ylabel("Latitude")

        fig.tight_layout()

        if im is not None:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=[1, 2, 3, 4], 
                                shrink=0.7, pad=0.02, location="right")
            cbar.ax.set_yticklabels(['Low', 'Moderate', 'High', 'Critical (Synergistic)'])

        output_path = Path(PROCESSED_DIR) / "vpd/vpd_ghm_ndvi_interaction_collage.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Three-way interaction spatial collage successfully saved to: {output_path}")
        plt.show()

    def run(self):
        self.plot_vpd_collage()
        self.plot_interaction_collage()