from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd

def temporal_cross_validation(
    df,
    features,
    model_builder
):
    print(features)
    years = sorted(df["year"].unique())
    roc_scores = []
    pr_scores = []
    
    for test_year in years[2:-1]:
        train = df[df["year"] < test_year]
        test = df[df["year"] == test_year]
        
        model = model_builder(train)
        model.fit(train[features], train["fire"])
        
        probs = model.predict_proba(test[features])[:, 1]
        
        auc = roc_auc_score(test["fire"], probs)
        pr = average_precision_score(test["fire"], probs)
        
        roc_scores.append(auc)
        pr_scores.append(pr)

        print(f"Year {test_year} ROC-AUC: {auc:.4f} | PR_AUC: {pr:.4f}")
        importance = pd.Series(
            model.feature_importances_,
            index=features
        ).sort_values(ascending=False)
        
        print(importance)
        
    print("\n=== CV Summary ===")
    print(f"Mean ROC-AUC: {np.mean(roc_scores):.4f} +- {np.std(roc_scores):.4f}")
    print(f"Mean PR-AUC: {np.mean(pr_scores):.4f} +- {np.std(pr_scores):.4f}", )

    return roc_scores, pr_scores