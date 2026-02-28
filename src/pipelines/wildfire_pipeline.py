from src.data import data_loader, split
from src.models import train as tr
from src.visualization import maps 


class WildfirePipeline:
    def __init__(self, model_factory, use_lag: bool):
        self.model_factory = model_factory
        self.use_lag = use_lag
        self.features = None
        self.model = None
        
    def load_data(self):
        df = data_loader.load_master_dataset()
        df = df.loc[:, ~df.columns.str.contains("^index")]
        df = df.loc[:, ~df.columns.str.contains("^level_0")]
        return data_loader.prepare_features(df)
    
    def build_features(self):
        base = ["dem", "landcover", "ghm", "slope", "sm1"]
        
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
                "vpd_ghm_interaction",
            ]
            
        self.features = base + extra
        
    def train(self, df):
        train, test, _ = split.temporal_split(df)
        
        X_train = train[self.features]
        y_train = train["fire"]
        
        X_test = test[self.features]
        y_test = test["fire"]
        
        if "xgboost" in self.model_factory.__name__:
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            scale_pos_weight = neg / pos if pos > 0 else 1
            
            self.model = self.model_factory(scale_pos_weight)
        else:
            self.model = self.model_factory()
        
        model = tr.train_model(self.model, X_train, y_train)
        probs = tr.evaluate_model(model, X_test, y_test, self.features)
        test = test.copy()
        test["fire_probability"] = probs
        return model, test
    
    def visualize(self, model, test):
        tr.explain_model_with_shap(model, test[self.features])
        maps.plot_month_map(
            test,
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
        # self.visualize(model, test)
        self.save(test)