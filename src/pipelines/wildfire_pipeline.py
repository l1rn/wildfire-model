from src.data import data_loader, split
from src.models import train as tr
from src.visualization import maps
from src.config import Config
 
import pandas as pd
import questionary
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

cfg = Config()
class WildfirePipeline:
    def __init__(self, model_factory, use_lag: bool, tune: bool, params: dict = None):
        self.model_factory = model_factory
        self.use_lag = use_lag
        self.tune = tune
        self.params = params or {}
        self.features = None
        self.model = None
        
    def load_data(self):
        df = data_loader.load_master_dataset()
        df = df.loc[:, ~df.columns.str.contains("^index|level_0")]
        data_loader.validate_dataset(df)
        return data_loader.prepare_features(df)
    
    def build_features(self):
        base = [
            "dem", 
            "landcover", 
            "ghm", 
            "slope", 
            "sm1", 
            "wind_speed", 
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
        X_train, X_test, y_train, y_test  = split.temporal_split(df)       
        
        test_full = X_test.copy()
        test_full["fire"] = y_test
        
        train_df = X_train.copy()
        train_df['fire'] = y_train        
        
        ones = train_df[train_df['fire'] == 1]
        zeros = train_df[train_df['fire'] == 0]
        
        n_zeros = min(len(ones) * 10, len(zeros))
        train_zeros_sampled = zeros.sample(n=n_zeros, random_state=42)
        train_balanced = pd.concat([ones, train_zeros_sampled]).sample(frac=1)
        
        X_train = train_balanced[self.features]
        y_train = train_balanced['fire']
        X_test = X_test[self.features]
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.fit_transform(X_test)
        if "xgboost" in self.model_factory.__name__:
            if self.tune:
                base_model = XGBClassifier(scale_pos_weight=scale_pos_weight, 
                                        random_state=42, eval_metrics='logloss')
                
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
                
                tscv = TimeSeriesSplit(n_splits=3)
                
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv = tscv, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                print("Best parameters:", grid_search.best_params_)
                best_params = grid_search.best_params_                
                import json
                with open(cfg.xgboost_params, 'w') as f:
                    json.dump(best_params, f)
                
                self.model = XGBClassifier(
                    **best_params,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    eval_metric='logloss'
                )
            else:
                model_params = self.params.copy()
                model_params['scale_pos_weight'] = scale_pos_weight
                model_params['random_state'] = 42
                model_params['eval_metric'] = 'logloss'
                self.model = XGBClassifier(**model_params)
        else:
            self.model = self.model_factory()
        self.model.fit(X_train, y_train)
        probs = self.model.predict_proba(X_test)[:, 1] 
        optimal_threshold = tr.evaluate_model(
            self.model, X_test, y_test, self.features
        )

        test_full["fire_probability"] = probs
        return self.model, test_full
    
    def visualize(self, model, df_full):
        target_month = df_full[
            (df_full["valid_time"].dt.year == 2022) & 
            (df_full["valid_time"].dt.month == 7)
        ].copy()
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
        options = questionary.checkbox(
            "Select options:",
            choices=[
                "SHAP Explanation Bar & Summary",
                "Visualize Risk-map",
                "Generate Partial Dependence Plots (PDP)",
                "Save the map in TIFF format for QGIS",
                "Create Bivarite Map GHM & VPD",
                "Animate Risk Over Time"
            ]
        ).ask()
        
        if "Visualize Risk-map" in options:
            self.visualize(model, test)
        if "SHAP Explanation Bar & Summary" in options:
            tr.explain_model_with_shap(model, test[self.features])
        if "Generate Partial Dependence Plots (PDP)" in options:
            tr.generate_partial_dependence_plots(model, test[self.features])
        if "Save the map in TIFF format for QGIS" in options:
            self.save(test)
        if "Create Bivarite Map GHM & VPD" in options:
            maps.create_bivariate_map(test)
        if "Animate Risk Over Time" in options:
            maps.animate_risk_over_time(test, year=2022)