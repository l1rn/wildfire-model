import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    precision_score,
    f1_score,
    recall_score,
    accuracy_score,
    fbeta_score,
    make_scorer,
    matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.calibration import calibration_curve

from src.config import PROCESSED_DIR, Config
import matplotlib.pyplot as plt
import shap
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as colors

cfg = Config()

def generate_metrics_model(y_test, preds, probs, threshold, K, k):
    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "f1": f1_score(y_test, preds),
        "f2": fbeta_score(y_test, preds, beta=2),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "threshold": threshold,
        "mcc": matthews_corrcoef(y_test, preds),
        "accuracy": accuracy_score(y_test, preds),
        "p@{K}": k,
    }

def evaluate_logistic_regression(X_train, y_train, X_test, y_test, optimal_threshold=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    baseline_model = LogisticRegression(
        class_weight='balanced', 
        max_iter=2000, 
        random_state=42, 
        solver='lbfgs'
    )
    baseline_model.fit(X_train_scaled, y_train)
        
    baseline_probs = baseline_model.predict_proba(X_test_scaled)[:, 1]
    if optimal_threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y_test, baseline_probs)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]

    baseline_preds = (baseline_probs >= optimal_threshold).astype(int)
    return baseline_preds, baseline_probs, optimal_threshold

def generate_evaluation_artifacts(
    y_test,
    optimal_threshold,
    primary_probs=None,
    primary_preds=None,
    baseline_preds=None,
    baseline_probs=None,
):
    print("\n=== GENERATING VISUAL ARTIFACTS ===")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        primary_preds,
        ax=ax[0],
        cmap='Blues',
        display_labels=['Non-Fire (0)', 'Fire (1)']
    )
    ax[0].set_title(f"Confusion Matrix\n(Threshold = {optimal_threshold:.4f})", fontsize=14)
    
    PrecisionRecallDisplay.from_predictions(
        y_test,
        primary_probs,
        ax=ax[1],
        name="XGBoost Ensemble"
    )
    
    PrecisionRecallDisplay.from_predictions(
        y_test,
        baseline_probs,
        ax=ax[1],
        name="Logistic Regression Baseline",
        color="gray",
        linestyle="--"
    )
    ax[1].set_title("Precision-Recall Curve Comparison", fontsize=14)
    plt.tight_layout()
    output_filename = Path(PROCESSED_DIR) / "evaluation_artifacts.png"
    plt.savefig(output_filename, dpi=300)

def generate_spatial_reliability_map(
    original_df, resolution=0.05, n_classes=4
):  
    df = original_df.copy()
    df['x_grid'] = (df['x'] / resolution).round() * resolution
    df['y_grid'] = (df['y'] / resolution).round() * resolution
    
    grid = df.groupby(['x_grid', 'y_grid']).agg({
        'fire_probability': 'median',
        'fire': 'max'
    }).reset_index()
        
    try:
        grid['class'], _ = pd.qcut(
            grid['fire_probability'],
            n_classes,
            labels=False,
            retbins=True,
            duplicates='drop'
        )
    except ValueError:
        grid['prob_rank'] = grid['fire_probability'].rank(method='first')
        grid['class'], _ = pd.qcut(
            grid['prob_rank'],
            n_classes,
            labels=False,
            retbins=True,
            duplicates='drop'
        )
    colors = plt.cm.plasma(np.linspace(0, 1, n_classes))
    grid['color'] = [colors[c] for c in grid['class']]

    gdf = gpd.GeoDataFrame(
        grid,
        geometry=gpd.points_from_xy(grid.x_grid, grid.y_grid),
        crs="EPSG:4326"
    )
    if hasattr(cfg, 'khmao_geojson') and cfg.khmao_geojson:
        boundary = gpd.read_file(cfg.khmao_geojson)
        gdf = gdf.clip(boundary)

    fig, ax = plt.subplots(figsize=(14, 8))
    gdf.plot(ax=ax, color=gdf['color'], edgecolor='none', markersize=20, alpha=0.7)
    fires = df[df['fire'] == 1]
    ax.scatter(fires['x'], fires['y'],
               marker='x', s=80, color='red', linewidths=2,
               label='Observed fires (test set)')
    
    from matplotlib.lines import Line2D
    legend_elements = []
    for i in range(n_classes):
        label = f'Risk {i+1}'
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                       label=label, markerfacecolor=colors[i],
                                       markersize=8))
    legend_elements.append(Line2D([0], [0], marker='x', color='red',
                                  label='Observed fires', markersize=8))
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title('Spatial Reliability: Predicted Risk (Quantile Classes) vs. Observed Ignitions', fontsize= 16)   
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    output_filename = Path(PROCESSED_DIR) / "spatial_reliability_map.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Spatial reliability map successfully exported as {output_filename}")

def generate_partial_dependence_plots(model, X_test, sample_size=10000, random_state=42):
    """
    Generates 1D and 2D Partial Dependence Plots to visualize feature thresholds 
    and interaction effects.
    """
    output_dir = Path(PROCESSED_DIR)
    
    print("Generating Partial Dependence Plots...")
    
    X_eval = X_test
        
    features_1d = [
        'ghm', 
        'vpd', 
        'ndvi',
    ]

    display_1d = PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_eval,
        features=features_1d,
        kind='average',
        grid_resolution=40, 
        n_jobs=-1,
        n_cols=3,
    )
    
    fig_1d = display_1d.figure_
    fig_1d.set_size_inches(16, 6)
    fig_1d.suptitle(
        'Partial Dependence (1D): Main Effects of GHM, VPD, and NDVI',
        fontsize=16
    )

    output_1d = output_dir / "pdp_1d_main_effects.png"
    fig_1d.savefig(output_1d, dpi=300, bbox_inches='tight')
    plt.close(fig_1d)

    print(f"Saved 1D PDPs to {output_1d}")
    
    features_2d = [
        ('ghm', 'vpd'),
        ('ghm', 'ndvi'),
        ('vpd', 'ndvi'),
    ]
    
    
    display_2d = PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_eval,
        features=features_2d,
        kind='average',
        grid_resolution=40, 
        n_jobs=-1,
        n_cols=3,
    )
    fig_2d = display_2d.figure_
    fig_2d.set_size_inches(16, 6)
    fig_2d.suptitle(
        'Partial Dependence (2D): Interaction Effects Between GHM, VPD, and NDVI',
        fontsize=16
    )
    output_2d = output_dir / "pdp_2d_interactions.png"
    fig_2d.savefig(output_2d, dpi=300, bbox_inches='tight')
    plt.close(fig_2d)
    
def generate_forecast(
    model, 
    df, 
    features
):
    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    
    df = df.copy()
    df["fire_probability"] = probs
    return df

def evaluate_model(model, X, y, features, threshold=None):
    probs = model.predict_proba(X)[:, 1]
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y, probs)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx]
        print(f"\nOptimal Probability Threshold (Max F1): {threshold:.4f} ({f1_scores[best_idx]})")
    preds = (probs >= threshold).astype(int)

    print(classification_report(y, preds))
    print(f"ROC-AUC: {roc_auc_score(y, probs):.4f}")
    
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print(importance)

    return threshold, probs, preds

def evaluate_risk_percentiles(
    test_df: pd.DataFrame,
    prob_col: str = 'fire_probability',
    target_col: str = 'fire',
):
    sorted_df = test_df.sort_values(by=prob_col, ascending=False).reset_index(drop=True)
    
    total_grid_cells = len(sorted_df)
    total_actual_fires = sorted_df[target_col].sum()
    
    percentiles = [1, 5, 10, 20, 30]
    results = []
    
    for p in percentiles:
        top_k_count = int(total_grid_cells * (p / 100.0))

        top_k_cells = sorted_df.head(top_k_count)
        
        fires_captured = top_k_cells[target_col].sum()
        capture_rate = (fires_captured / total_actual_fires) * 100
        
        results.append({
            "Risk Tier": f"Top {p}%",
            "Grid Cells Flagged": top_k_count,
            "Actual Fires Captured": int(fires_captured),
            "Total Fires": int(total_actual_fires),
            "Capture Rate (%)": round(capture_rate, 2)
        })
    evaluation_table = pd.DataFrame(results)
    return evaluation_table

def calculate_p_at_k(K = 1000, test_probs=None, y_test=None):
    top_k_indices = np.argsort(test_probs)[-K:]
    
    actual_fires_in_top_k = y_test.iloc[top_k_indices].sum()
    precision_at_k = actual_fires_in_top_k / K
    return precision_at_k

def explain_model_with_shap(model, X_test):
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    if hasattr(model, "base_estimator"):
        model = model.base_estimator
    if hasattr(model, "estimator"):
        model = model.estimator
        
    X_test = X_test.copy()
    explainer = shap.TreeExplainer(model)
    
    X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer(X_sample)
    
    if len(shap_values.values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance (Impact on Model Output)")
    shap_summary_path = Path(PROCESSED_DIR) / "shap" / "shap_summary_plot.png"
    plt.savefig(shap_summary_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    shap_bar_path = Path(PROCESSED_DIR) / "shap" / "shap_bar_plot.png"
    plt.savefig(shap_bar_path, bbox_inches='tight', dpi=300)
    plt.close()

    print("SHAP plots saved as 'shap_summary_plot.png' and 'shap_bar_plot.png'")
    
def plot_calibration_curve(
    y_true, y_proba, n_bins=10, 
    output_file="calibration_plot.png"
):
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='XGBoost (calibrated)')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    
    ax.set_title('Calibration plot (reliability diagram)')
    ax.legend(loc='best')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Calibration plot saved to {output_file}")
    
def plot_threshold_analysis(y_true, y_proba, output_file='threshold_analysis.png'):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions[:-1]) * (recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
    ax.plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1', linewidth=2)
    
    ax.axvline(best_thresh, color='red', linestyle=':', label=f'Optimal threshold = {best_thresh:.4f}')
    
    ax.set_xlabel('Probability threshold')
    ax.set_ylabel('Score')
    ax.legend()
    ax.set_title('Precision, Recall, and F1 vs. Decision Threshold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Threshold analysis plot saved to {output_file}")
    