from src.extract import data_loader, split
from src.output import train as tr
from src.output import maps
from src.config import Config, PROCESSED_DIR

import optuna
from optuna.samplers import TPESampler

from pathlib import Path
import pandas as pd
import questionary
from sklearn.metrics import (
    precision_recall_curve, 
    roc_auc_score, 
    classification_report, 
    precision_score,
    f1_score,
    recall_score,
    accuracy_score,
    fbeta_score,
    make_scorer
)

from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
            "u10", "v10", "peatland", "ndvi", "lai", "fpar"
        ]

        if self.use_lag:
            extra = [
                "temp_lag1", "vpd_lag1", "precip_lag1",
                "vpd_fire_lag1_interaction"
            ]
        else:
            extra = [
                "temp", "vpd", "precip"       
            ]

        engineered = [
            "vpd_ghm_interaction", 
            "vpd_3m_avg",
            "temp_ghm_interaction",
            "temp_precip_interaction",
            "temp_cisi_interaction",
            "precip_30p_sum",
            "wind_slope_synergy",
        ]

        anthropogenic = [
            "dist_oil_gas", "pop_density", "ghm", "cisi"
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
            
    def tuning(self, scale_pos_weight, X_train, X_val, y_train, y_val, model_type):
        def objective(trial):    
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('float', 0, 5),
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'logloss',
                'random_state': 42,
                'verbosity': 0,                    
            }
            if model_type == "xgb":
                model = XGBClassifier(**params)
            elif model_type == "lgbm":
                model = LGBMClassifier(**params)
            else:
                raise ValueError("Unsupported model type")
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred_probs = model.predict_proba(X_val)[:, 1]
            f1 = f1_score(y_val, y_pred_probs > 0.5)
            return f1
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        best_params = study.best_params
        print(f"Best F1: {study.best_value:.4f}")
        print(f"Best params: {best_params}")

        import json
        if model_type == "xgb":
            best_params['scale_pos_weight'] = scale_pos_weight
            best_params['random_state'] = 42
            best_params['verbosity'] = 0
            self.model = XGBClassifier(**best_params)
            with open(Path(PROCESSED_DIR) / "best_xgboost_params.json") as f:
                json.dump(study.best_params, f)
        elif model_type == "lgbm":
            best_params['scale_pos_weight'] = scale_pos_weight
            best_params['random_state'] = 42
            best_params['verbosity'] = -1
            self.model = LGBMClassifier(**best_params)  
            with open(Path(PROCESSED_DIR) / "best_lightgbm_params.json") as f:
                json.dump(study.best_params, f)  
        
    def find_optimal_threshold(self, model, X_val, y_val):
        probs = model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx]
    
    def balance_test_set(self, test_df, random_state=42):
        
        fire = test_df[test_df['fire'] == 1]
        non_fire = test_df[test_df['fire'] == 0]
        
        n_fire = len(fire)
        if n_fire == 0:
            print("No fire point in test dataset")
            return test_df
        
        non_fire_balance = non_fire.sample(n=n_fire, random_state=random_state)
        balanced = pd.concat([fire, non_fire_balance])
        return balanced.sample(frac=1, random_state=random_state)
    
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
            
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        X_val = val[self.features]
        y_val = val["fire"]
        if "xgboost" in self.model_factory.__name__:
            if self.tune:
                self.tuning(scale_pos_weight, X_train, X_val, y_train, y_val, "xgb")
            else:
                model_params = self.params.copy()
                model_params['scale_pos_weight'] = scale_pos_weight
                model_params['random_state'] = 42
                model_params['eval_metric'] = 'logloss'
                self.model = XGBClassifier(**model_params)
        elif "lightgbm" in self.model_factory.__name__:
            if self.tune:
                self.tuning(scale_pos_weight, X_train, X_val, y_train, y_val, "lgbm")
            else:
                model_params = self.params.copy()
                model_params['scale_pos_weight'] = scale_pos_weight
                model_params['random_state'] = 42
                model_params['verbosity'] = -1
                self.model = LGBMClassifier(**model_params)
        else:
            self.model = self.model_factory()
        self.model.fit(X_train, y_train)
        
        X_test = test[self.features]
        y_test = test['fire']

        balanced_test_df = self.balance_test_set(test)
        print("balanced test df")
        tr.evaluate_model(self.model, balanced_test_df[self.features], balanced_test_df['fire'], self.features)
        print("normal test df")
        optimal_threshold, test_probs, test_preds = tr.evaluate_model(self.model, X_test, y_test, self.features)
        K = 1000
        top_k_indices = np.argsort(test_probs)[-K:]
        
        actual_fires_in_top_k = y_test.iloc[top_k_indices].sum()
        precision_at_k = actual_fires_in_top_k / K
        # self._evaluate_by_year_type(test, test_probs, optimal_threshold)
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
            f"p@{K}": precision_at_k,            
            "roc_auc": roc_auc_score(y_test, test_probs),
            "threshold": optimal_threshold,
            "accuracy": accuracy_score(y_test, test_preds),
            "f2": fbeta_score(y_test, test_preds, beta=2),
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
        print("\n === Risk Percentile Analysis ===")
        percentile_results = tr.evaluate_risk_percentiles(test)
        print(percentile_results.to_string(index=False))
        
        maps.plot_cumulative_gains(
            test, output_file=Path(PROCESSED_DIR) / "cumulative_gains_chart.png"
        )
        if 'fire_probability' not in df_full.columns:
            print("ERROR: fire_probability column was not added!")
        else:
            print("fire_probability column added successfully.")
            print(df_full[['valid_time', 'fire_probability', 'x', 'y']].head())
        
        options = questionary.checkbox(
            "Select options:",
            choices=[
                "Feature Importance from the model",
                "SHAP Explanation Bar & Summary",
                "Visualize Risk-map",
                "Generate Partial Dependence Plots (PDP)",
                "Spatial Reliability Map",
                "Save the map in TIFF format for QGIS",
                "Create Bivarite Map GHM & VPD",
                "Animate Risk Over Time",
                "Calibration Plot",
                "Time Series of Predicted vs. Observed Fire Counts",
                "Map of Top Driver",
                "Threshold Performance Plot"
            ]
        ).ask()
        
        if "Feature Importance from the model" in options:
            maps.plot_feature_importance(
                model, 
                features=self.features, 
                top_n=15, 
                output_file=Path(PROCESSED_DIR) / "feature_importance.png" 
            )
        if "Visualize Risk-map" in options:
            self.visualize(model, df_full)
        if "SHAP Explanation Bar & Summary" in options:
            tr.explain_model_with_shap(model, test[self.features])
        if "Generate Partial Dependence Plots (PDP)" in options:
            tr.generate_partial_dependence_plots(model, test[self.features])
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
        if "Map of Top Driver" in options:
            maps.map_top_driver(df_full, output_file=Path(PROCESSED_DIR) / "top_driver_map.png")
        if "Threshold Performance Plot" in options:
            test_probs = model.predict_proba(test[self.features])[:, 1]
            tr.plot_threshold_analysis(test['fire'], test_probs, output_file=Path(PROCESSED_DIR) / "threshold_analysis.png")