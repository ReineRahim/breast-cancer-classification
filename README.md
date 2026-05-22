# Breast Cancer Diagnosis — ML Classification Pipeline

Binary classification model to detect malignant vs benign breast tumours 
using the UCI Breast Cancer Wisconsin Diagnostic dataset.

Built as an AI/ML course final project at the University of Balamand (2025).

---

## Results

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Tuned L2 Logistic Regression (Final) | 97.4% | 96.8% | 99.5% |
| Tuned L1 Logistic Regression | 96.0% | 94.7% | 99.1% |
| Tuned SVM | 97.6% | 96.7% | 99.4% |
| Tuned Random Forest | 96.5% | 95.2% | 98.8% |

Validated with 5-fold stratified cross-validation.  
Final model: **Tuned L2 Logistic Regression without PCA** — strongest 
overall performance with highest interpretability.

---

## Dataset

- Source: UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)
- 569 instances, 30 features, 2 classes (M = malignant, B = benign)
- No missing values

---

## Project Structure

- Phase 1: Data loading, EDA, feature analysis, baseline models
- Phase 2: Hyperparameter tuning (GridSearchCV), PCA analysis, 
  cross-validation, final model selection

---

## Key Findings

- "Worst" measurements (concavity, concave points, radius, area) 
  are the strongest separators between malignant and benign cases
- PCA did not improve L2 performance and reduced interpretability — 
  excluded from final model
- Tuning helped L2 most; SVM and Random Forest changed little

---

## Tech Stack

- Python, scikit-learn, pandas, NumPy
- GridSearchCV, StratifiedKFold, PCA
- ucimlrepo (dataset loader)
- Google Colab

---

## Authors

Reine Rahim & Jane Ibrahim — University of Balamand, CSIS290
