# End-to-End Prediction Pipeline: Breast Cancer Classification

## Goal
Build a small, reproducible classification pipeline that takes a raw CSV dataset, trains a model, makes held-out predictions, and inspects the cases it gets wrong.

**Important:** this is an educational machine-learning project, not a medical device. It must not be used to diagnose or guide treatment.

## What's included
- `data/breast_cancer_raw.csv` — raw tabular dataset (569 rows; 30 numeric measurements plus target).
- `train_predict.py` — complete pipeline from CSV loading through model training, test predictions, metrics, and error analysis.
- `requirements.txt` — Python dependencies.
- `outputs/` — generated when the script runs:
  - `metrics.json` — test metrics and confusion matrix.
  - `predictions.csv` — each held-out row, actual label, predicted label, and malignant probability.
  - `misclassified_cases.csv` — only the wrong predictions for error analysis.

## Dataset and target
This CSV is the Breast Cancer Wisconsin dataset distributed with scikit-learn. The target is encoded as:
- `0` = malignant
- `1` = benign

For precision, recall, and F1, this project treats **malignant (0)** as the positive class because missing a malignant case is particularly important to examine. The code explicitly calculates the probability for class 0 rather than assuming the positive class is 1.

## Run locally
Use Python 3.10 or newer if possible.

1. Extract/download this project folder.
2. Open Command Prompt or a terminal in the `prediction_pipeline_project` folder.
3. Install packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the full pipeline:

   ```bash
   python train_predict.py
   ```

5. Open the generated files inside `outputs/`.

The CSV is included, so the script does not need to download the dataset at runtime.

## Pipeline decisions
1. Read the CSV and separate features from the target.
2. Create a stratified 80/20 train/test split with a fixed random seed.
3. Fit median imputation, feature scaling, and logistic regression together in a scikit-learn `Pipeline`. Imputation and scaling are learned from training data only, avoiding test-data leakage.
4. Evaluate once on the held-out test set.
5. Save predictions and export misclassified rows so errors can be reviewed.

## Error analysis
Open `outputs/misclassified_cases.csv` after running the script. Each row includes the true class, predicted class, predicted probability of malignancy, and all measurements. Review:
- false negatives for malignancy: actual malignant, predicted benign (especially important to investigate);
- false positives for malignancy: actual benign, predicted malignant;
- whether wrong cases have probabilities near the decision boundary.

The file is generated from the actual run, so the number and identities of errors are not hard-coded in this README.

## Assumptions and limitations
- The included dataset is a small, curated benchmark, not a representative sample of all patients or clinical settings.
- The target labels and measurements are assumed to be correct and consistently collected.
- A single random hold-out test split gives only one estimate of performance; results can vary with the split and library versions.
- Logistic regression is a simple, interpretable baseline, not necessarily the best model.
- The default 0.5 probability threshold is not clinically validated. A real deployment would need prospective external validation, calibration, threshold selection with domain experts, bias assessment, and privacy/security review.
- Feature values alone do not replace clinical judgment. Do not use this model for diagnosis or treatment.

## Reproducibility
A fixed random seed (`42`) is used for the split. The script prints the key scores and writes machine-readable results to `outputs/`.
