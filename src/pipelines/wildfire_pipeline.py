from src.data import data_loader, split
from src.models import train as tr
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
                "temp_lag2",
                "vpd_lag2",
                "precip_lag2",
                "vpd_ghm_interaction_lag2",
            ]
        else:
            extra = [
                "temp",
                "vpd",
                "precip",
            ]
            
        self.features = base + extra
        
    def train(self, df):
        train_full, test_full, _ = split.temporal_split(df)
        
        ones = train_full[train_full['fire'] == 1]
        zeros = train_full[train_full['fire'] == 0]
        
        train_zeros_sampled = zeros.sample(n=len(ones) * 20, random_state=42)
        train_balanced = pd.concat([ones, train_zeros_sampled]).sample(frac=1)
        
        X_train = train_balanced[self.features]
        y_train = train_balanced["fire"]
        
        X_test = test_full[self.features]
        y_test = test_full["fire"]
        
        if "xgboost" in self.model_factory.__name__:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            self.model = self.model_factory(scale_pos_weight)
        else:
            self.model = self.model_factory()
        
        model = tr.train_model(self.model, X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1] 
        tr.evaluate_model(model, X_test, y_test, self.features)
        test_full = test_full.copy()
        test_full["fire_probability"] = probs
        return model, test_full
    
    def visualize(self, model, df_full):
        target_month = df_full[
            (df_full["valid_time"].dt.year == 2025) & 
            (df_full["valid_time"].dt.month == 7)
        ].copy()
        # tr.explain_model_with_shap(model, test[self.features])
        X_viz = target_month[self.features]
        target_month["fire_probability"] = model.predict_proba(X_viz)[:, 1]
        
        maps.plot_month_map(
            target_month,
            year=2025,
            month=7,
            title="Wildfire Forecast – July 2025",
        )
        
    def save(self, test):
        maps.save_to_geotiff(
            test,
            year=2025,
            month=7,
            filename="khmao.tif"
        )
                
    def run(self):
        df = self.load_data()
        self.build_features()
        model, test = self.train(df)
        self.visualize(model, test)
        self.save(test)