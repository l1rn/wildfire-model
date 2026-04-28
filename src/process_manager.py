_cfg = None

def get_cfg():
    global _cfg
    if _cfg is None:
        from src.config import Config
        _cfg = Config()
    return _cfg

def build_xgb(train):
    from src.models import models
    scale_pos_weight = len(train) / train["fire"].sum()
    return models.get_xgboost(scale_pos_weight) 

dict = {1: "raw", 2: "preprocessing", 3: "extract data", 4: "testing the model", 0: "exit"}

def show_era5_head():
    from src.config import RAW_DIR
    from src.extract import data_loader
    print("Showing era5 head...")
    df = data_loader.load_meterological(f"{RAW_DIR}/khmao_era5.nc")
    print(df.head())
    
def show_master_table():
    from src.extract import data_loader
    import pandas as pd
    df = data_loader.load_master_dataset()
    df = df.loc[:, ~df.columns.str.contains("^index")]
    df: pd.DataFrame = df.loc[:, ~df.columns.str.contains("^level_0")]
    print(df.columns)
    print(df['year'].unique())
    
def show_fire_archive_head():
    from src.extract import data_loader
    cfg = get_cfg()
    print("Showing fire archive head...")
    firms = data_loader.load_firms(cfg.raw_fire_data)
    print(firms.head())
    
def show_ghm_info():
    from src.extract import data_loader
    cfg = get_cfg()
    print("Showing Global Human Modification head...")
    human_mod = data_loader.load_static_raster(cfg.raw_human_mod)
    print(human_mod.head())
    
def show_topography_info():
    from src.extract import data_loader
    print("Showing Topography head...")
    cfg = get_cfg()
    dem = data_loader.load_static_raster(cfg.raw_dem)
    print(dem.head())
    
def show_landcover_info():
    from src.extract import data_loader
    cfg = get_cfg()
    print("Showing Land Cover head...")
    lc = data_loader.load_static_raster(cfg.raw_landcover)
    print(lc.head())
    
def process_and_upload():
    import src.preprocessing as preprocessing
    ds = preprocessing.process_data()
    preprocessing.upload_dataset_to_parquet(ds)
    
def summarize_cv():
    from src.extract import data_loader
    from src.models import cross_validation, models
    df = data_loader.load_master_dataset()
    print("Loaded: ", df.shape)
    
    df = data_loader.prepare_features(df)
    
    features= [
        "temp",
        "vpd",
        "precip",
        "dem", 
        "landcover", 
        "ghm", 
        "slope", 
        "sm1", 
        "u10", 
        "v10", 
        "pop_density", 
        "dist_oil_gas", 
        "peatland"
    ]
    
    cross_validation.temporal_cross_validation(
        df,
        features,
        build_xgb
    )

def wildfire_pipeline():
    from src.cli import menu
    from src.pipelines import WildfirePipeline
    import questionary
    cfg = get_cfg()
    name, factory = menu.choose_model()
    use_lag = questionary.confirm("Use lag variables in data model", default=False).ask()
    use_tune = questionary.confirm("Use tune to seek for hyperparameters in the model", default=False).ask()
    groups = ["all", 
            #   "natural", "anthropogenic", "compounding"
              ]
    results = {}
    for group in groups:
        import json
        with open(cfg.xgboost_params, 'r') as f:
            best_params = json.load(f)
            
        pipeline = WildfirePipeline(
            factory, 
            use_lag=use_lag, 
            tune=use_tune, 
            params=best_params,
            feature_group=group,
            downsample_ratio=10,
            # use_smote=True,
            # smote_ratio=0.2,
        )
        pipeline.run()
        metrics_answers = questionary.checkbox("which metrics to execute?", choices=[
            "imbalanced test set",
            "balanced test set"
        ]).ask()
        if "imbalanced test set" in metrics_answers:
            results[f"{group}_imbalanced"] = pipeline.get_metrics(parameter="imbalanced")
        if "balanced test set" in metrics_answers:
            results[f"{group}_balanced"] = pipeline.get_metrics(parameter="balanced")
    import pandas as pd
    df_table = pd.concat(results, axis=0)
    
    print(df_table)

def eda_execution():
    import src.output.data_analysis as da
    from src.extract import data_loader
    df = data_loader.load_master_dataset()
    da.execute_eda_pipeline(df)

def temperature_pipeline():
    from src.pipelines import TemperaturePipeline
    pipeline = TemperaturePipeline()
    pipeline.run()
    
def plot_data():
    from src.pipelines import PreprocessingVisualizationPipeline
    pipeline = PreprocessingVisualizationPipeline()
    pipeline.run()

def execute_modis_pipeline():
    from src.collection import GeeExtractor
    collection = GeeExtractor()
    collection.run()

def execute_validation():
    from src.config import RAW_DIR
    from src.collection import GeeExtractor
    collection = GeeExtractor()
    collection.validate_with_sentinel2(f"{RAW_DIR}/validation_sample.csv")
    
def execute_cds_era5():
    from src.collection import cds_extractor
    cds_extractor.extract_era5()
    
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
    2: plot_data,
},
3: {
    1: execute_modis_pipeline,
    2: execute_validation,
    3: execute_cds_era5
},
4: {
    1: summarize_cv,
    2: wildfire_pipeline,
    3: temperature_pipeline,
    4: eda_execution
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
