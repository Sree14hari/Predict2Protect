<p align="center">
  <a href="https://sree14hari.vercel.app/">
    <img src="https://d8it4huxumps7.cloudfront.net/uploads/images/opportunity/banner/68fcddbc6c45a_predict2protect.png" alt="Portfolio Preview" width="100%" />
  </a>
</p>


# Predicting Company Failure: Ensemble Models on Imbalanced Tabular Data

## 📌 Project Overview

This project focuses on developing a robust machine learning model to predict corporate failure using imbalanced tabular financial data. The primary goal is to accurately identify at-risk companies by maximizing the **F1-score for the minority class** ("failed" companies), rather than relying on traditional accuracy metrics.

---

## 🎯 Key Objectives

- Build a binary classifier to predict whether a company is "alive" or "failed"
- Handle severe class imbalance (≈ 14:1 ratio)
- Compare data-level vs. model-level imbalance-handling techniques
- Evaluate advanced models including tree-based boosters and neural networks
- Optimize model performance via ensemble learning and threshold tuning

---

## 🧩 Methodology

### Data Preprocessing
- Dropped non-predictive columns (`company_name`, `fyear`)
- Handled missing values using `SimpleImputer`
- Scaled numerical features (`StandardScaler`)
- Encoded categorical features (`OneHotEncoder`)

### Models Evaluated
- **Baseline**: Random Forest
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Deep Learning**: PyTorch-based Feed-Forward Neural Network

### Imbalance Handling Strategies
- **Data-Level**: SMOTE, SMOTETomek, ADASYN
- **Model-Level**: Internal class weighting (`scale_pos_weight`, `class_weight`)

### Ensemble Technique
- **Stacking Classifier**: Combines XGBoost and LightGBM with a Logistic Regression meta-model

### Optimization
- Hyperparameter tuning via `GridSearchCV`
- Optimal threshold tuning using precision-recall curves

---

## 🏆 Results

### Champion Model: Stacked Ensemble (XGBoost + LightGBM)
- **F1-Score (Failed)**: 0.45
- **ROC-AUC**: 0.868
- **Precision (Failed)**: 0.44
- **Recall (Failed)**: 0.46
- **Optimal Threshold**: 0.2858

### Key Insights
- ✅ Model-level weighting outperformed data-level sampling
- ✅ Ensemble models (stacking) improved performance over single models
- ✅ Threshold tuning was critical for balancing precision and recall

---

## 📊 Performance Summary

| Model / Strategy                 | Imbalance Handling     | F1-Score (Failed) | ROC-AUC |
|----------------------------------|------------------------|-------------------|---------|
| Random Forest (Baseline)         | class_weight           | 0.29              | 0.800   |
| Random Forest + SMOTE            | SMOTE                  | 0.22              | 0.711   |
| PyTorch Neural Network           | class_weight           | 0.32              | 0.842   |
| Tuned XGBoost                    | scale_pos_weight       | 0.42              | 0.887   |
| Tuned LightGBM                   | class_weight           | 0.45              | 0.861   |
| Stacked (XGB+LGBM) + SMOTE       | SMOTE                  | 0.37              | 0.823   |
| **Stacked (XGB+LGBM)**           | **Internal Weighting** | **0.45**          | **0.868** |

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- PyTorch
- Imbalanced-learn
- Pandas, NumPy, Matplotlib

---

## 👥 Authors

- **Sreehari R** – [sreeharil4shr@gmail.com](mailto:sreeharil4shr@gmail.com)
- **Avanthika T**

**Affiliation**: Sree Buddha College of Engineering, Alappuzha & Predict2Protect, IIT (ISM) Dhanbad
