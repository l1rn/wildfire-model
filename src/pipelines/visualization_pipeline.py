
_cfg = None

def get_cfg():
    global _cfg
    if _cfg is None:
        from src.config import Config
        _cfg = Config()
    return _cfg

class PreprocessingVisualizationPipeline:
    def __init__(self):
        self.bbox = (59.0, 58.0, 86.0, 65.8)

    def load_df(self):
        from src.extract import data_loader
        df = data_loader.load_master_dataset()
        self.df = df

    def load_bbox(self):
        from shapely.geometry import box
        import geopandas as gpd 
        self.bbox_geom = gpd.GeoDataFrame(
            geometry=[box(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])], 
            crs="EPSG:4326"
        )

    def load_environmental_variables(self):
        from src.extract import data_loader
        cfg = get_cfg()
        self.land_cover = data_loader.load_static_raster(cfg.raw_landcover).squeeze().rio.clip_box(*self.bbox)
        topography = data_loader.load_static_raster(cfg.raw_dem)
        self.elevation = topography.sel(band=1).squeeze().rio.clip_box(*self.bbox)
        self.slope = topography.sel(band=2).squeeze().rio.clip_box(*self.bbox)

    def load_natural_variables(self):
        from src.extract import data_loader
        cfg = get_cfg()
        era5 = data_loader.load_meterological(cfg.raw_weather)
        era5 = era5.rio.write_crs("EPSG:4326")
        era5 = era5.rio.clip_box(*self.bbox)

        july_2020 = era5.sel(valid_time="2020-07")
        self.t2m = july_2020["t2m"] - 273.15
        self.d2m = july_2020["d2m"] - 273.15
        self.u10 = july_2020["u10"]
        self.v10 = july_2020["v10"]
        self.swvl1 = july_2020["swvl1"]
        self.tp = july_2020["tp"] * 1000
        
    def create_features(self):
        self.human_related_features = [
            "ghm", "dist_oil_gas", "pop_density", "cisi"
        ]
        self.nature_related_features = [
            "temp", "dew", "ndvi", "lai", "fpar", "precip", "sm1", "u10", "v10", "peatland"
        ]

        self.environment_related_features = [
            "dem", "slope", "landcover"
        ]

        self.engineered_features = [
            "vpd",
            "vpd_ghm_interaction",
            "vpd_cisi_interaction",
            "vpd_3m_avg",
            "temp_ghm_interaction",
            "temp_cisi_interaction",
            "temp_precip_interaction",
            "precip_30p_sum",
            "wind_slope_synergy",
            "dew_ghm_interaction",
            "ndvi_ghm_interaction",
            "ndvi_vpd_interaction"
        ]
    
    def load_borders(self):
        import geopandas as gpd
        self.region = gpd.read_file("data/processed/khmao.geojson")
        self.region = self.region.to_crs("EPSG:4326")

    def to_grid(self, column):
        grid = self.df.pivot_table(
            index='y',
            columns='x',
            values=column
        )

        grid = grid.sort_index()
        grid = grid.sort_index(axis=1)

        return grid.values, grid.columns.values, grid.index.values
    
    def plot_enviromental_variables(self):
        import matplotlib.pyplot as plt
        import geopandas as gpd
        from shapely.geometry import box
        from pathlib import Path
        fig, axes = plt.subplots(1, 3, figsize=(20, 4))
        
        khmao = gpd.read_file("data/processed/khmao.geojson")
        khmao = khmao.to_crs("EPSG:4326")

        features = [
            (self.elevation, "Elevation (m)"),
            (self.slope, "Slope (degrees)"),
            (self.land_cover, "Land Cover Class")
        ]
        
        for ax, (data_array, title) in zip(axes.flat, features):
            data = data_array.values
            lon = data_array.x.values
            lat = data_array.y.values
            
            im = ax.imshow(
                data,
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                origin='upper',
                aspect='auto',
                cmap='viridis'
            )

            bbox = box(lon.min(), lat.min(), lon.max(), lat.max())
            gdf_clipped = khmao.clip(bbox)

            gdf_clipped.boundary.plot(
                ax=ax,
                color='red',
                linewidth=1
            )
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.subplots_adjust(wspace=0.05)
        from src.config import PREPROCESSED_DIR
        from pathlib import Path
        plt.savefig(Path(PREPROCESSED_DIR) / "preprocessed_environment_raw.png", dpi=300)

    def plot_natural_variables(self):
        import matplotlib.pyplot as plt
        import geopandas as gpd
        from shapely.geometry import box

        fig, axes = plt.subplots(2, 3, figsize=(22, 10))
        khmao = gpd.read_file("data/processed/khmao.geojson")
        khmao = khmao.to_crs("EPSG:4326")

        features = [
            (self.t2m, "Air Temperature at 2m (°C)"),
            (self.d2m, "Dew Point Temperature at 2m (°C)"),
            (self.tp, "Total Precipitation (mm)"),
            (self.u10, "Zonal Wind at 10m (m/s)"),
            (self.v10, "Meridional Wind at 10m (m/s)"),
            (self.swvl1, "Soil Moisture Layer 1 (m³/m³)"),
        ]

        for ax, (data_array, title) in zip(axes.flat, features):
            data = data_array.squeeze().values
            lon = data_array.longitude.values
            lat = data_array.latitude.values
            
            im = ax.imshow(
                data,
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                origin='upper',
                aspect='auto',
                cmap='viridis'
            )

            bbox = box(lon.min(), lat.min(), lon.max(), lat.max())
            gdf_clipped = khmao.clip(bbox)

            gdf_clipped.boundary.plot(
                ax=ax,
                color='red',
                linewidth=1
            )
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        from src.config import PREPROCESSED_DIR
        from pathlib import Path
        plt.savefig(Path(PREPROCESSED_DIR) / "preprocessed_nature_raw.png", dpi=300)

    def plot_human_variables(self):
        pass
    def plot_engineered_variables(self):
        pass
        
    def run(self):
        import questionary
        # self.load_df()
        self.load_bbox()
        self.create_features()
        self.load_borders()
        
        
        plot_list = questionary.checkbox(
            "Which categories to plot", 
            choices=["nature", "human", "environment", "engineered"]
        ).ask()

        if "nature" in plot_list:
            self.load_natural_variables()
            self.plot_natural_variables()
        if "human" in plot_list:
            self.plot_human_variables()
        if "environment" in plot_list:
            self.load_environmental_variables()
            self.plot_enviromental_variables()
        if "engineered" in plot_list:
            self.plot_engineered_variables()