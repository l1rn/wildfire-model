from src.data import data_loader
from src.config import PROCESSED_DIR, RAW_DIR
from src.collection import GeeExtractor
from src.models import cross_validation, models
from src.cli import menu
from src.pipelines import WildfirePipeline

import src.preprocessing as preprocessing

def build_xgb(train):
    scale_pos_weight = len(train) / train["fire"].sum()
    return models.get_xgboost(scale_pos_weight) 

dict = {1: "raw", 2: "preprocessing", 3: "extract data", 4: "testing the model", 0: "exit"}

def show_era5_head():
    print("Showing era5 head...")
    df = data_loader.load_meterological(f"{RAW_DIR}/khmao_era5.nc")
    print(df.head())
    
def show_master_table():
    import pandas as pd
    df = data_loader.load_master_dataset()
    df = df.loc[:, ~df.columns.str.contains("^index")]
    df: pd.DataFrame = df.loc[:, ~df.columns.str.contains("^level_0")]
    print(df.columns)
    
def show_fire_archive_head():
    print("Showing fire archive head...")
    firms = data_loader.load_firms(f"{RAW_DIR}/khmao_fire_archive.csv")
    print(firms.head())
    
def show_ghm_info():
    print("Showing Global Human Modification head...")
    human_mod = data_loader.load_static_raster(f"{RAW_DIR}/khmao_human_mod_90m.tif")
    print(human_mod.head())
    
def show_topography_info():
    print("Showing Topography head...")
    dem = data_loader.load_static_raster(f"{RAW_DIR}/khmao_topography.tif")
    print(dem.head())
    
def show_landcover_info():
    print("Showing Land Cover head...")
    lc = data_loader.load_static_raster(f"{RAW_DIR}/khmao_lc_90m.tif")
    print(lc.head())
    
def process_and_upload():
    ds = preprocessing.process_data()
    preprocessing.upload_dataset_to_parquet(ds)
    
def summarize_cv():
    df = data_loader.load_master_dataset()
    print("Loaded: ", df.shape)
    
    df = data_loader.prepare_features(df)
    
    features= [
        "temp_lag2",
        "vpd_lag2",
        "precip_lag2",
        "vpd_ghm_interaction",
        "dem",
        "soil1",
        "soil2",
        "landcover",
        "ghm",
    ]
    
    cross_validation.temporal_cross_validation(
        df,
        features,
        build_xgb
    )

def wildfire_pipeline():
    model_name, factory = menu.choose_model()
    use_lag = int(input("Use lag features(yes - 1 / no - 0): "))
        
    pipeline = WildfirePipeline(factory, use_lag)
    pipeline.run()

def execute_modis_pipeline():
    collection = GeeExtractor()
    collection.run()
    
options = {
1: {
    1: show_era5_head,
    2: show_fire_archive_head,
    3: show_ghm_info,
    4: show_topography_info,
    5: show_landcover_info,
    6: show_master_table
},
2: {
    1: process_and_upload,
},
3: {
    1: execute_modis_pipeline,
},
4: {
    1: summarize_cv,
    2: wildfire_pipeline,
}}
    
def choose_option():
    print("=== Choose the option ===")
    for i, j in dict.items():
        print(f"{i}: {j}")
        
    return int(input("ans: "))
    
def choose_sub_option(ans):
    sub_options = options.get(ans)
    if not sub_options:
        print("Invalid option")
        return
    
    print("=== Choose what to execute ===")
    for key in sub_options:
        print(f"{key}: {sub_options[key].__name__}")

    sub_ans = int(input("ans: "))

    action = sub_options.get(sub_ans)
    if action:
        action()
    else:
        print("Invalid sub-option")            
