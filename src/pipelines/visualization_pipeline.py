
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

    def load_environmental_variables(self):
        from src.extract import data_loader
        cfg = get_cfg()
        self.land_cover = data_loader.load_static_raster(cfg.raw_landcover).rio.clip_box(*self.bbox)
        topography = data_loader.load_static_raster(cfg.raw_dem)
        self.elevation = topography.sel(band=1).rio.clip_box(*self.bbox)
        self.slope = topography.sel(band=2).rio.clip_box(*self.bbox)
        from shapely.geometry import box
        import geopandas as gpd 
        self.bbox_geom = gpd.GeoDataFrame(
            geometry=[box(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])], 
            crs="EPSG:4326"
        )
    def load_df(self):
        from src.extract import data_loader
        df = data_loader.load_master_dataset()
        self.df = df
    
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
        fig, axes = plt.subplots(1, 3, figsize=(24, 4))
        
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
            
            self.bbox_geom.boundary.plot(ax=ax, color='red', linewidth=2, label='Bounding Box')
            
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        from src.config import PREPROCESSED_DIR
        from pathlib import Path
        plt.savefig(Path(PREPROCESSED_DIR) / "preprocessed_environment_raw.png", dpi=300)
        plt.show()

    def plot_natural_variables(self):
        pass
    def plot_human_variables(self):
        pass
    def plot_engineered_variables(self):
        pass
        
    def run(self):
        import questionary
        self.load_df()
        self.create_features()
        self.load_borders()
        
        plot_list = questionary.checkbox(
            "Which categories to plot", 
            choices=["nature", "human", "environment", "engineered"]
        ).ask()

        if "nature" in plot_list:
            self.plot_natural_variables()
        if "human" in plot_list:
            self.plot_human_variables()
        if "environment" in plot_list:
            self.plot_enviromental_variables()
        if "engineered" in plot_list:
            self.plot_engineered_variables()