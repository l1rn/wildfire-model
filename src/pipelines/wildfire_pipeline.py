from src.data import data_loader, split
from src.models import train as tr
from src.models import models
from src.visualization import maps 
import pandas as pd

class WildfirePipeline:
    def __init__(self, model_factory, use_lag: bool):
        self.model_factory = model_factory
        self.use_lag = use_lag
        self.features = None
        self.model = None
        
    def load_data(self):
        df = data_loader.load_master_dataset()
        df = df.loc[:, ~df.columns.str.contains("^index|level_0")]
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        df['month'] = df['valid_time'].dt.month
        df = df[df['month'].between(4, 10)]
        return data_loader.prepare_features(df)
    
    def build_features(self):
        base = [
            "dem", 
            "landcover", 
            "ghm", 
            "slope", 
            "sm1", 
            "u10", 
            "v10", 
            "pop_density", 
            "dist_oil_gas", 
            "peatland",
            "month"
        ]
        
        if self.use_lag:
            extra = [
                "temp_lag1",
                "vpd_lag1",
                "precip_lag1",
                "vpd_ghm_interaction_lag1",
            ]
        else:
            extra = [
                "temp",
                "vpd",
                "precip",
                "vpd_ghm_interaction"
            ]
            
        self.features = base + extra
        
    def train(self, df):
        X_train_full, X_test_full, y_train_full, y_test, _, _, _, _ = split.temporal_split(df)       
        
        train_df = X_train_full.copy()
        train_df['fire'] = y_train_full        
        
        ones = train_df[train_df['fire'] == 1]
        zeros = train_df[train_df['fire'] == 0]
        
        n_zeros = min(len(ones) * 10, len(zeros))
        train_zeros_sampled = zeros.sample(n=n_zeros, random_state=42)
        train_balanced = pd.concat([ones, train_zeros_sampled]).sample(frac=1)
        
        X_train = train_balanced[self.features]
        y_train = train_balanced['fire']
        
        X_test = X_test_full[self.features]
        
        if "xgboost" in self.model_factory.__name__:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            self.model = models.optimize_xgboost(
                X_train,
                y_train,
                scale_pos_weight
            )
        else:
            self.model = self.model_factory()
            
        model = tr.train_model(self.model, X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1] 
        tr.evaluate_model(model, X_test, y_test, self.features)
        
        test_full = X_test_full.copy()
        test_full["fire"] = y_test
        test_full["fire_probability"] = probs
        
        return model, test_full
    
    def visualize(self, model, df_full):
        target_month = df_full[
            (df_full["valid_time"].dt.year == 2022) & 
            (df_full["valid_time"].dt.month == 7)
        ].copy()
        tr.explain_model_with_shap(model, df_full[self.features])
        X_viz = target_month[self.features]
        target_month["fire_probability"] = model.predict_proba(X_viz)[:, 1]
        
        maps.plot_month_map(
            target_month,
            year=2022,
            month=7,
            title="Wildfire Forecast – July 2022",
        )
        
    def save(self, test):
        maps.save_to_geotiff(
            test,
            year=2022,
            month=7,
            filename="khmao.tif"
        )
                
    def run(self):
        df = self.load_data()
        self.build_features()
        model, test = self.train(df)
        self.visualize(model, test)
        # self.save(test)