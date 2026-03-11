import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, roc_auc_score, precision_recall_curve
from src.config import PROCESSED_DIR
import matplotlib.pyplot as plt
import shap
import numpy as np

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

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
    
    auc = roc_auc_score(y_test, probs)
    
    print("ROC-AUC: ", auc)
    print(classification_report(y_test, preds_optimized))
    
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)
    
    print(importance)
    
    return probs

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