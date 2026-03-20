import pandas as pd

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
from src.config import PROCESSED_DIR
import matplotlib.pyplot as plt
import shap
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    X_test, y_test, probs, optimal_threshold, original_df
):  
    x_coords = original_df.loc[X_test.index, 'x'].values
    y_coords = original_df.loc[X_test.index, 'y'].values
    results_df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'probability': probs,
        'observed_fire': y_test.values
    })
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    sc = ax.scatter(
        results_df['x'], 
        results_df['y'], 
        c=results_df['probability'], 
        cmap='YlOrRd', 
        s=15, 
        alpha=0.4,
        edgecolors='none',
        label='Predicted Probability Surface'
    )
    
    observed_fires = results_df[results_df['observed_fire'] == 1]
    ax.scatter(
        observed_fires['x'],
        observed_fires['y'], 
        color='#000000',
        marker='+', 
        s=40, 
        linewidths=1.5,
        label='Observed Fire Hotspot (Ground Truth)'
    )
    
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Algorithm Ignition Probability', fontsize=12)
    cbar.ax.axhline(optimal_threshold, color='black', linestyle='--', linewidth=2)
    cbar.ax.text(1.2, optimal_threshold, f'Threshold\n({optimal_threshold:.4f})', 
                 va='center', ha='left', fontsize=10)
    
    ax.set_title('Spatial Reliability Analysis: Predicted Risk vs. Observed Ignitions', fontsize=16, pad=15)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-2:], labels[-2:], loc='upper right', framealpha=0.9)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    output_filename = "spatial_reliability_map.png"
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
    
    output_filename = "pdp_infrastructure_climate.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Partial Dependence Plots saved successfully to {output_filename}")
    
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

def explain_model_with_shap(model, X_test):
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    
    X_test = X_test.copy()
    explainer = shap.TreeExplainer(model)
    
    X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer(X_sample)
    
    if len(shap_values.values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance (Impact on Model Output)")
    plt.savefig("shap_summary_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.savefig("shap_bar_plot.png", bbox_inches='tight', dpi=300)
    plt.close()

    print("SHAP plots saved as 'shap_summary_plot.png' and 'shap_bar_plot.png'")