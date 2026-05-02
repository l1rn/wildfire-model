_cfg = None

def get_cfg():
    global _cfg
    if _cfg is None:
        from src.config import Config
        _cfg = Config()
    return _cfg

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
        from src.extract import data_loader
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
            "vpd_ghm_ndvi_interaction",
            "temp_ghm_interaction",
            "temp_precip_interaction",
            "temp_cisi_interaction",
            "precip_30p_sum",
            "wind_slope_synergy",
            "ndvi_vpd_interaction",
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

    def evaluation_every_model(self, test, X_train, y_train, K=1000):
        import questionary
        from src.output import train as tr
        print("normal test df")
        
        X_test = test[self.features]
        y_test = test['fire']
        
        optimal_threshold, test_probs, test_preds = tr.evaluate_model(self.model, X_test, y_test, self.features)
        baseline_preds, baseline_probs, baseline_optimal_threshold = tr.\
            evaluate_logistic_regression(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        precision_at_k = tr.calculate_p_at_k(K=K, y_test=y_test, test_probs=test_probs)
        base_precision_at_k = tr.calculate_p_at_k(K=K, y_test=y_test, test_probs=baseline_probs)

        self.metrics = {
            "xgboost": tr.\
                generate_metrics_model(y_test, test_preds, test_probs, optimal_threshold, K, precision_at_k),
            "logistic_regression": tr.\
                generate_metrics_model(y_test, baseline_preds, baseline_probs, baseline_optimal_threshold, K, base_precision_at_k),
            "rf": {}
        }

        ans = questionary.confirm("Generate Evaluation Artifacts?").ask()
        if ans:
            tr.generate_evaluation_artifacts(
                y_test=y_test,
                optimal_threshold=optimal_threshold,
                primary_probs=test_probs,
                primary_preds=test_preds,
                baseline_preds=baseline_preds,
                baseline_probs=baseline_probs
            )

        balanced_test_df = self.balance_test_set(test)
        
        balanced_X_test = balanced_test_df[self.features]
        balanced_y_test = balanced_test_df['fire']

        optimal_threshold, balanced_probs, balanced_preds  = tr.evaluate_model(self.model, balanced_X_test, balanced_y_test, self.features)
        balanced_baseline_preds, balanced_baseline_probs, balanced_baseline_optimal_threshold = tr.\
                    evaluate_logistic_regression(X_train=X_train, y_train=y_train, X_test=balanced_X_test, y_test=balanced_y_test)
       
        balanced_precision_at_k = tr.calculate_p_at_k(K=K, y_test=balanced_y_test, test_probs=balanced_probs)
        balanced_baseline_precision_at_k = tr.calculate_p_at_k(K=K, y_test=balanced_y_test, test_probs=balanced_baseline_probs)
        self.balanced_metrics = {
            "xgboost": tr.\
                generate_metrics_model(balanced_y_test, balanced_preds, balanced_probs, optimal_threshold, K, balanced_precision_at_k),
            "logistic_regression": tr.\
                generate_metrics_model(balanced_y_test, balanced_baseline_preds, balanced_baseline_probs, balanced_baseline_optimal_threshold, K, balanced_baseline_precision_at_k)
        }
        return test_probs
    
    def balance_test_set(self, test_df, random_state=42):
        import pandas as pd
        
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
        from imblearn.over_sampling import SMOTE
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import RandomForestClassifier
        from src.extract import split
        train, val, test  = split.temporal_split(df)       
        
        ones = train[train['fire'] == 1]
        zeros = train[train['fire'] == 0]
        
        if self.downsample_ratio is not None:
            import pandas as pd
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

        from src.models import models as mod
        if "xgboost" in self.model_factory.__name__:
            if self.tune:
                self.model = mod.tuning(scale_pos_weight, X_train, y_train, X_val, y_val, model_type="xgb")
            else:
                model_params = self.params.copy()
                model_params['scale_pos_weight'] = scale_pos_weight
                model_params['random_state'] = 42
                model_params['eval_metric'] = 'logloss'
                self.model = XGBClassifier(**model_params)
        elif "lightgbm" in self.model_factory.__name__:
            if self.tune:
                self.model = mod.tuning(scale_pos_weight, X_train, y_train,X_val, y_val, model_type="lgbm",)
            else:
                model_params = self.params.copy()
                model_params['scale_pos_weight'] = scale_pos_weight
                model_params['random_state'] = 42
                model_params['verbosity'] = -1
                self.model = LGBMClassifier(**model_params)
        elif "get_random_forest" in self.model_factory.__name__:
            if self.tune:
                self.model = mod.tuning(scale_pos_weight, X_train, y_train, X_val, y_val, "rf")
            else:
                model_params = self.params.copy()
                model_params['class_weight'] = 'balanced'
                model_params['random_state'] = 42
                model_params['n_jobs'] = -1
                self.model = RandomForestClassifier(**model_params)
        else:
            self.model = self.model_factory()

        self.model.fit(X_train, y_train)
        K = 1000
        test_probs = self.evaluation_every_model(test=test, X_train=X_train, y_train=y_train, K=K)
        
        # self._evaluate_by_year_type(test, test_probs, optimal_threshold)
    
        test_full = test.copy()
        test_full["fire_probability"] = test_probs
        
        return self.model, test_full
    
    def _evaluate_by_year_type(self, test_df, probs, threshold):
        extreme = test_df[test_df['is_extreme_year'] == 1]
        normal = test_df[test_df['is_extreme_year'] == 0]
        
        print("extreme:", len(test_df['is_extreme_year']))
        from sklearn.metrics import (
            roc_auc_score, 
            classification_report, 
        )
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
        
    def get_metrics(self, parameter="imbalanced"):
        import pandas as pd
        if parameter == "imbalanced":
            metrics_data = self.metrics
        elif parameter == "balanced":
            metrics_data = self.balanced_metrics
        else:
            metrics_data = self.metrics
        
        if not metrics_data:
            return pd.DataFrame()
        
        records = []
        for model_name, model_metrics in metrics_data.items():
            if model_metrics and isinstance(model_metrics, dict):
                record = {'model': model_name}
                record.update(model_metrics)
                records.append(record)
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df.set_index('model', inplace=True)

        numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        return df
                
    def run(self):
        from src.output import train as tr, evaluation
        df = self.load_data()
        self.build_features()
        model, test = self.train(df)
        df_full = df.copy()
        X_full = df_full[self.features]

        probs = model.predict_proba(X_full)[:, 1]
        print(f"Probability array shape: {probs.shape}")

        df_full['fire_probability'] = probs
        print("\n === Risk Percentile Analysis ===")
        percentile_results = tr.evaluate_risk_percentiles(test)
        print(percentile_results.to_string(index=False))
        from src.config import PROCESSED_DIR
        
        if 'fire_probability' not in df_full.columns:
            print("ERROR: fire_probability column was not added!")
        else:
            print("fire_probability column added successfully.")
            print(df_full[['valid_time', 'fire_probability', 'x', 'y']].head())
        
        import questionary
        options = questionary.checkbox(
            "Select options:",
            choices=[
                "Plot Cumulative Gains",
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

        from pathlib import Path
        if "Plot Cumulative Gains" in options:
            evaluation.plot_cumulative_gains(
                test, output_file=Path(PROCESSED_DIR) / "cumulative_gains_chart.png"
            )
        if "Feature Importance from the model" in options:
            evaluation.plot_feature_importance(
                model, 
                features=self.features, 
                top_n=10, 
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
            evaluation.create_bivariate_map(df_full, var1='vpd', var2='ghm')
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
            evaluation.plot_time_series_risk(df_full, output_file=Path(PROCESSED_DIR) / "time_series_risk.png", freq='M')
        if "Animate Risk Over Time" in options:
            cfg = get_cfg()
            evaluation.animate_risk_over_time(df_full, years=None, output_file=cfg.risk_map_animation_output)
        if "Map of Top Driver" in options:
            evaluation.map_top_driver(df_full, output_file=Path(PROCESSED_DIR) / "top_driver_map.png")
        if "Threshold Performance Plot" in options:
            test_probs = model.predict_proba(test[self.features])[:, 1]
            tr.plot_threshold_analysis(test['fire'], test_probs, output_file=Path(PROCESSED_DIR) / "threshold_analysis.png")