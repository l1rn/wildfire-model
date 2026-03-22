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
)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import PartialDependenceDisplay
from src.config import PROCESSED_DIR, Config
import matplotlib.pyplot as plt
import shap
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as colors

cfg = Config()

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def generate_evaluation_artifacts(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    optimal_threshold,
    primary_probs=None,
    primary_preds=None
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    baseline_model = LogisticRegression(
        class_weight='balanced', max_iter=2000, random_state=42,solver='lbfgs'
    )
    baseline_model.fit(X_train_scaled, y_train)
    
    baseline_probs = baseline_model.predict_proba(X_test_scaled)[:, 1]
    baseline_preds = (baseline_probs >= 0.5).astype(int)
    
    print("Logistic Regression Baseline Classification Report:")
    print(classification_report(y_test, baseline_preds))
    
    print("\n=== GENERATING VISUAL ARTIFACTS ===")
    if primary_probs is None:
        primary_probs = model.predict_proba(X_test)[:, 1]
    if primary_preds is None:
        primary_preds = (primary_probs >= optimal_threshold).astype(int)
    
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
    output_filename = "evaluation_artifacts.png"
    plt.savefig(output_filename, dpi=300)
    
def generate_spatial_reliability_map(
    original_df, resolution=0.05, n_classes=4
):  
    df = original_df.copy()
    df = df[df['year'] == 2022]
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
    print("Generating Partial Dependence Plots...")
    if len(X_test) > sample_size:
        print(f"Subsampling test set from {len(X_test)} to {sample_size} rows for rapid PDP generation.")
        X_eval = X_test.sample(n=sample_size, random_state=random_state)
    else:
        X_eval = X_test
        
    features_to_plot = [
        'ghm', 
        'vpd', 
        ('ghm', 'vpd')
    ]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    display = PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_eval,
        features=features_to_plot,
        kind='average',
        grid_resolution=40, 
        ax=ax,
        n_jobs=-1
    )
    
    fig.suptitle('Partial Dependence: Infrastructure Proximity vs. Synergistic Climate Effects', fontsize=16)
    plt.subplots_adjust(top=0.9)  
    
    output_filename = Path(PROCESSED_DIR) / "pdp_infrastructure_climate.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Partial Dependence Plots saved successfully to {output_filename}")
    
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

def evaluate_model(model, X_test, y_test, features):
    probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(
        y_test,
        probs
    )
    
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0
    )
    
    optimal_idx = np.argmax(f1_scores)
    
    if optimal_idx < len(thresholds):
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5
    
    print(f"\nOptimal Probability Threshold (Max F1): {optimal_threshold:.4f}")
    
    preds_optimized = (probs >= optimal_threshold).astype(int)
    
    K = 1000
    top_k_indices = np.argsort(probs)[-K:]
    
    actual_fires_in_top_k = y_test.iloc[top_k_indices].sum()
    precision_at_k = actual_fires_in_top_k / K
    
    print(f"Precision@{K}: {precision_at_k:.4f}")
    
    auc = roc_auc_score(y_test, probs)
    print("ROC-AUC: ", auc)
    print(classification_report(y_test, preds_optimized))
    
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)
    
    print(importance)
    
    return optimal_threshold

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