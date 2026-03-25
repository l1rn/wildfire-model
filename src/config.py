from pathlib import Path
import yaml

THIS_FILE = Path(__file__).resolve()

SRC_DIR = THIS_FILE.parent
BASE_DIR = SRC_DIR.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            
        self.full_weather = data["data_paths"]["full_era5"]
        self.full_firms = data["data_paths"]["full_firms"]
        
        self.raw_weather = data["data_paths"]["raw_weather"]
        self.raw_dem = data["data_paths"]["raw_dem"]
        self.raw_landcover = data["data_paths"]["raw_landcover"]
        self.raw_pop_density = data["data_paths"]["raw_pop_density"]
        self.raw_firms = data["data_paths"]["raw_firms"]
        self.raw_human_mod = data["data_paths"]["raw_ghm"]
        self.raw_peatland = data["data_paths"]["raw_peatland"]
        self.raw_oil_gas = data["data_paths"]["raw_oil_gas"]
        self.processed_table = data["data_paths"]["processed_table"]
        
        self.khmao_geojson = data["data_paths"]["khmao_geojson"]
        self.xgboost_params = data["data_paths"]["xgboost_params"]
        self.lightgbm_params = data["data_paths"]["lightgbm_params"]
        self.risk_map_animation_output = data["data_paths"]["risk_map_animation_output"]

        self.config_file = data
        self.production_mode = data["production"]
    WILDFIRE_SEASON_MONTHS = list(range(5, 10))
    NON_BURNABLE_CLASSES_LC = [50, 70, 80]
    RANDOM_SEED = 42
    BALANCED_RATIO = 10
        
    def get_study_years(self) -> dict:
        return {
            'anomalous_hot': [2016, 2020, 2022, 2023],
            'baseline_cold': [2017, 2018, 2019, 2021, 2024, 2025]
        }