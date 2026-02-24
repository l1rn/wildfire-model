import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, roc_auc_score
from src.config import PROCESSED_DIR
import matplotlib.pyplot as plt
import shap

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, features):
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    auc = roc_auc_score(y_test, probs)
    
    print("ROC-AUC: ", auc)
    print(classification_report(y_test, preds))
    
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
    explainer = shap.TreeExplainer(model)
    
    X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance (Impact on Model Output)")
    plt.savefig("shap_summary_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    shap.plots.bar(explainer(X_sample), show=False)
    plt.savefig("shap_bar_plot.png", bbox_inches='tight', dpi=300)
    plt.close()

    print("SHAP plots saved as 'shap_summary_plot.png' and 'shap_bar_plot.png'")