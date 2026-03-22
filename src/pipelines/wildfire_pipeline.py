from src.data import data_loader, split
from src.models import train as tr
from src.visualization import maps
from src.config import Config, PROCESSED_DIR
 
from pathlib import Path
import pandas as pd
import questionary
from sklearn.metrics import (
    precision_recall_curve, 
    roc_auc_score, 
    classification_report, 
    precision_score,
    f1_score,
    recall_score
)

from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import numpy as np

cfg = Config()
class WildfirePipeline:
    def __init__(
        self, model_factory, use_lag: bool, 
        tune: bool, params: dict = None, downsample_ratio: int = 10,
        use_smote: bool = False, smote_ratio: float = 0.5, feature_group: str = 'all',
    ):
        self.model_factory = model_factory
        self.use_lag = use_lag
        self.tune = tune
        self.params = params or {}
        self.use_smote = use_smote
        
        self.downsample_ratio = downsample_ratio
        self.smote_ratio = smote_ratio
        
        self.features = None
        self.model = None
        self.feature_group = feature_group
        
    def load_data(self):
        df = data_loader.load_master_dataset()
        df = df.loc[:, ~df.columns.str.contains("^index|level_0")]
        data_loader.validate_dataset(df)
        return data_loader.prepare_features(df)
    
    def build_features(self):
        base = [
            "dem", "landcover", "slope", "sm1", 
            "wind_speed", "peatland", "month"
        ]

        if self.use_lag:
            extra = [
                "temp_lag1", "vpd_lag1", "precip_lag1",
                "vpd_ghm_interaction_lag1"
            ]
        else:
            extra = [
                "temp", "vpd", "precip"            
            ]

        engineered = [
            "month_sin", "month_cos",
            "vpd_ghm_interaction",       
            "ghm_windspeed_interaction",
            "temp_precip_interaction" 
        ]

        anthropogenic = [
            "dist_oil_gas", "pop_density", "ghm"
        ]

        all_candidates = base + extra + engineered + anthropogenic

        all_candidates = list(dict.fromkeys(all_candidates))

        if self.feature_group == "compounding":
            self.features = [f for f in all_candidates if f in engineered]
        elif self.feature_group == "natural":
            natural = base + extra
            self.features = [f for f in all_candidates if f in natural]
        elif self.feature_group == "anthropogenic":
            self.features = [f for f in all_candidates if f in {"dist_oil_gas", "pop_density", "ghm"}]
        else: 
            self.features = all_candidates
        
    def train(self, df):
        train, val, test  = split.temporal_split(df)       
        
        ones = train[train['fire'] == 1]
        zeros = train[train['fire'] == 0]
        
        if self.downsample_ratio is not None:
            n_zeros = min(len(ones) * self.downsample_ratio, len(zeros))
            zeros_sampled = zeros.sample(n=n_zeros, random_state=42)
            train_balanced = pd.concat([ones, zeros_sampled])
        else:
            train_balanced = train 
            
        X_train = train_balanced[self.features]
        y_train = train_balanced['fire']
        
        if self.use_smote:
            smote = SMOTE(sampling_strategy=self.smote_ratio, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
        scale_pos_weight = min(50, (y_train == 0).sum() / (y_train == 1).sum())
        
        X_val = val[self.features]
        y_val = val["fire"]
        if "xgboost" in self.model_factory.__name__:
            if self.tune:
                base_model = XGBClassifier(scale_pos_weight=scale_pos_weight, 
                                        random_state=42, eval_metrics='logloss')
                
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                }
                
                X_combined =  pd.concat([X_train, X_val])
                y_combined = pd.concat([y_train, y_val])
                test_fold = np.array([-1]*len(X_train) + [0]*len(X_val))
                ps = PredefinedSplit(test_fold)
                
                random_search = RandomizedSearchCV (
                    estimator=base_model,
                    param_distributions=param_grid,
                    cv = ps, 
                    scoring='average_precision',
                    n_jobs=-1,
                    verbose=1,
                    random_state=cfg.RANDOM_SEED
                )
                
                random_search.fit(X_combined, y_combined)
                print("Best parameters:", random_search.best_params_)
                best_params = random_search.best_params_                
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
        class CalibratedXGB:
            def __init__(self, base_model, calibrator):
                self.base_model = base_model
                self.calibrator = calibrator
                self.feature_importances_ = base_model.feature_importances_

            def predict_proba(self, X):
                return self.calibrator.predict_proba(X)
        self.base_model = self.model
        from sklearn.calibration import CalibratedClassifierCV
        
        self.base_model.fit(X_train, y_train)
        calibrator = CalibratedClassifierCV(
            estimator=self.base_model,
            method='isotonic',
            cv=3
        )
        calibrator.fit(X_val, y_val)

        self.model = CalibratedXGB(self.base_model, calibrator)
        
        X_test = test[self.features]
        y_test = test['fire']

        probs = self.model.predict_proba(X_test)[:, 1]

        optimal_threshold = tr.evaluate_model(self.model, X_test, y_test, self.features)
        test_probs = self.model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= optimal_threshold).astype(int)        
        
        self._evaluate_by_year_type(test, test_probs, optimal_threshold)
        tr.generate_evaluation_artifacts(
            model=self.model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            optimal_threshold=optimal_threshold,
            primary_probs=test_probs,          
            primary_preds=test_preds
        )
    
        test_full = test.copy()
        test_full["fire_probability"] = test_probs
        
        self.metrics = {
            "precision": precision_score(y_test, test_preds),
            "recall": recall_score(y_test, test_preds),
            "f1": f1_score(y_test, test_preds),
            "roc_auc": roc_auc_score(y_test, test_preds),
            "threshold": optimal_threshold
        }
        return self.model, test_full, optimal_threshold
    
    def _evaluate_by_year_type(self, test_df, probs, threshold):
        extreme = test_df[test_df['is_extreme_year'] == 1]
        normal = test_df[test_df['is_extreme_year'] == 0]
        
        print("extreme:", len(test_df['is_extreme_year']))
        if len(extreme) > 0:
            probs_ext = probs[test_df['is_extreme_year'] == 1]
            y_ext = test_df.loc[test_df['is_extreme_year'] == 1, 'fire']
            preds_ext = (probs_ext >= threshold).astype(int)
            print("\n=== Extreme Years Performance ===")
            print(classification_report(y_ext, preds_ext))
            print(f"ROC-AUC (extreme): {roc_auc_score(y_ext, probs_ext):.4f}")

        if len(normal) > 0:
            probs_norm = probs[test_df['is_extreme_year'] == 0]
            y_norm = test_df.loc[test_df['is_extreme_year'] == 0, 'fire']
            preds_norm = (probs_norm >= threshold).astype(int)
            print("\n=== Normal Years Performance ===")
            print(classification_report(y_norm, preds_norm))
            print(f"ROC-AUC (normal): {roc_auc_score(y_norm, probs_norm):.4f}")
    
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
        
    def get_metrics(self):
        return self.metrics
                
    def run(self):
        df = self.load_data()
        self.build_features()
        model, test, optimal_threshold = self.train(df)
        df_full = df.copy() 
        X_full = df_full[self.features]

        probs = model.predict_proba(X_full)[:, 1]
        print(f"Probability array shape: {probs.shape}")

        df_full['fire_probability'] = probs

        if 'fire_probability' not in df_full.columns:
            print("ERROR: fire_probability column was not added!")
        else:
            print("fire_probability column added successfully.")
            print(df_full[['valid_time', 'fire_probability', 'x', 'y']].head())
        
        options = questionary.checkbox(
            "Select options:",
            choices=[
                "SHAP Explanation Bar & Summary",
                "Visualize Risk-map",
                "Generate Partial Dependence Plots (PDP)",
                "Spatial Reliability Map",
                "Save the map in TIFF format for QGIS",
                "Create Bivarite Map GHM & VPD",
                "Animate Risk Over Time",
                "Calibration Plot",
                "Time Series of Predicted vs. Observed Fire Counts"
            ]
        ).ask()
        
        if "Visualize Risk-map" in options:
            self.visualize(model.base_model, df_full)
        if "SHAP Explanation Bar & Summary" in options:
            tr.explain_model_with_shap(model.base_model, test[self.features])
        if "Generate Partial Dependence Plots (PDP)" in options:
            tr.generate_partial_dependence_plots(model.base_model, test[self.features])
        if "Save the map in TIFF format for QGIS" in options:
            self.save(test)
        if "Create Bivarite Map GHM & VPD" in options:
            maps.create_bivariate_map(df_full, var1='vpd', var2='ghm')
        if "Spatial Reliability Map" in options:
            tr.generate_spatial_reliability_map(
                original_df=test,
                resolution=0.05,
                n_classes=4
            )
        if "Calibration Plot" in options:
            y_test = test['fire']
            probs = test['fire_probability']
            tr.plot_calibration_curve(
                y_test, probs, output_file=Path(PROCESSED_DIR) / "calibration_plot.png"
            )
        if "Time Series of Predicted vs. Observed Fire Counts" in options:
            maps.plot_time_series_risk(df_full, output_file=Path(PROCESSED_DIR) / "time_series_risk.png", freq='M')
        if "Animate Risk Over Time" in options:
            maps.animate_risk_over_time(df_full, years=None, output_file=cfg.risk_map_animation_output)