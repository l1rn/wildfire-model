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
        return data_loader.prepare_features(df)
    
    def build_features(self):
        base = ["dem", "landcover", "ghm", "soil1", "soil2"]
        
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
        
        self.model = self.model_factory(train)
        
        model = tr.train_model(self.model, X_train, y_train)
        probs = tr.evaluate_model(model, X_test, y_test, self.features)
        
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
                
    def run(self):
        df = self.load_data()
        self.build_features()
        model, test = self.train(df)
        self.visualize(model, test)