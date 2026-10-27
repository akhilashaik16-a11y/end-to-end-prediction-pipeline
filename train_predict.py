"""
End-to-end prediction pipeline for the Breast Cancer Wisconsin dataset.

Run from the project folder:
    python train_predict.py

Creates:
    outputs/metrics.json
    outputs/predictions.csv
    outputs/misclassified_cases.csv
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "breast_cancer_raw.csv"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Raw data -> separate labels -> train/test split before fitting any transforms.
    raw = pd.read_csv(DATA_PATH)
    if "target" not in raw.columns:
        raise ValueError("Expected a 'target' column in data/breast_cancer_raw.csv")
    X = raw.drop(columns=["target"])
    y = raw["target"].astype(int)

    # Dataset convention: 0 = malignant, 1 = benign.
    # Treat malignant (0) as the positive class for precision/recall/F1.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced",
                                          random_state=RANDOM_STATE))
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    # Probability for malignant class (target 0), not benign class (target 1).
    probabilities = model.predict_proba(X_test)
    classes = list(model.named_steps["classifier"].classes_)
    malignant_col = classes.index(0)
    malignant_probability = probabilities[:, malignant_col]

    metrics = {
        "dataset": "Breast Cancer Wisconsin (scikit-learn copy)",
        "target_mapping": {"0": "malignant", "1": "benign"},
        "positive_class_for_precision_recall_f1": "malignant (target=0)",
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_malignant": float(precision_score(y_test, pred, pos_label=0, zero_division=0)),
        "recall_malignant": float(recall_score(y_test, pred, pos_label=0, zero_division=0)),
        "f1_malignant": float(f1_score(y_test, pred, pos_label=0, zero_division=0)),
        "roc_auc_malignant": float(roc_auc_score((y_test == 0).astype(int), malignant_probability)),
        "confusion_matrix_labels_order": ["malignant (0)", "benign (1)"],
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, pred, labels=[0, 1], target_names=["malignant", "benign"],
            output_dict=True, zero_division=0
        )
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    results = X_test.copy()
    results.insert(0, "row_id", X_test.index)
    results["actual_target"] = y_test
    results["actual_label"] = y_test.map({0: "malignant", 1: "benign"})
    results["predicted_target"] = pred
    results["predicted_label"] = pd.Series(pred, index=X_test.index).map({0: "malignant", 1: "benign"})
    results["predicted_malignant_probability"] = malignant_probability
    results["correct"] = (y_test == pred)
    results.sort_values("row_id").to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    results.loc[~results["correct"]].sort_values("row_id").to_csv(
        OUTPUT_DIR / "misclassified_cases.csv", index=False
    )

    print("End-to-end pipeline finished.")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Malignant precision: {metrics['precision_malignant']:.3f}")
    print(f"Malignant recall: {metrics['recall_malignant']:.3f}")
    print(f"Malignant F1: {metrics['f1_malignant']:.3f}")
    print(f"Malignant ROC-AUC: {metrics['roc_auc_malignant']:.3f}")
    print(f"Confusion matrix [malignant, benign]: {metrics['confusion_matrix']}")
    print(f"Saved metrics and predictions under: {OUTPUT_DIR}")
    print(f"Misclassified cases: {int((~results['correct']).sum())}")

if __name__ == "__main__":
    main()
