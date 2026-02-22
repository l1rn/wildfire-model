from src.data import data_loader, split
from src.config import PROCESSED_DIR, RAW_DIR
from src.collection import gee_extractor
from src.models import cross_validation, models
from src.features import features as ft
from src.models import train as tr
import src.preprocessing as preprocessing

from xgboost import XGBClassifier

def build_xgb(train):
    scale_pos_weight = len(train) / train["fire"].sum()
    return models.get_xgboost(scale_pos_weight) 

dict = {1: "raw", 2: "preprocessing", 3: "extract data", 4: "testing the model" }

def show_era5_head():
    print("Showing era5 head...")
    df = data_loader.load_meterological(f"{RAW_DIR}/khmao_era5.nc")
    print(df.head())
    
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
    
def check_cross_validation():
    df = data_loader.load_master_dataset()
    print("Loaded: ", df.shape)
    
    df = ft.prepare_features(df)
    
    features= [
        "temp",
        "vpd",
        "precip",
        "vpd_ghm_interaction",
        "dem",
        "landcover",
        "ghm",
    ]
    
    cross_validation.temporal_cross_validation(
        df,
        features,
        build_xgb
    )

def visualize_map():
    df = data_loader.load_master_dataset()
    print("Loaded: ", df.shape)
    
    df = ft.prepare_features(df)
    train, test, future = split.temporal_split(df)
    
    features= [
        "temp",
        "vpd",
        "precip",
        "vpd_ghm_interaction",
        "dem",
        "landcover",
        "ghm",
    ]
    X_train = train[features]
    y_train = train["fire"]
    
    X_test = test[features]
    y_test = test["fire"]
    
    rf = models.get_random_forest()
    model = tr.train_model(rf, X_train, y_train)
    probs = tr.evaluate_model(model, X_test, y_test, features)
    print(probs)
    
options = {
1: {
    1: show_era5_head,
    2: show_fire_archive_head,
    3: show_ghm_info,
    4: show_topography_info,
    5: show_landcover_info
},
2: {
    1: process_and_upload,
},
3: {
    1: gee_extractor.run_gee_pipeline
},
4: {
    1: check_cross_validation,
    2: visualize_map
}}
    
def choose_option():
    print("=== Choose the option ===")
    for i, j in dict.items():
        print(i, j)
        
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
