# Breast Cancer Diagnosis – Classification Models & Streamlit App

## a. Problem statement
Doctors examine cells taken from a breast mass under a microscope to decide whether a tumour is benign or malignant. This project uses the measurements computed from those cell images to build a binary classifier that predicts the diagnosis automatically. Five classical machine-learning models (Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM) are trained on the same data, compared using six evaluation metrics, and exposed through an interactive Streamlit web app where a test CSV can be uploaded and every model evaluated on it.

## b. Dataset description
# Breast Cancer Wisconsin (Diagnostic) – Binary Classification

## 1. Dataset Information

- **Source:** UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)
- **Dataset file:** `wdbc.data`
- **Total instances:** 569
- **Input features:** 30 numeric features
- **Additional columns:** ID and Diagnosis (target)
- **Task:** Binary Classification

## 2. Target Variable

The target variable is `Diagnosis`, encoded as follows:

- **Malignant (M):** 1
- **Benign (B):** 0

## 3. Feature Description

The dataset contains 10 nucleus measurements:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave Points
9. Symmetry
10. Fractal Dimension

Each measurement is represented by three values:

- Mean
- Standard Error
- Worst (largest value)

This results in a total of 30 numeric input features.

## 4. Class Distribution

| Diagnosis | Count |
|---|---:|
| Benign (0) | 357 |
| Malignant (1) | 212 |
| **Total** | **569** |

- No missing values are present in the dataset.

## 5. Train-Test Split

The dataset is divided into training and testing sets using the following configuration:

- **Training size:** 80%
- **Testing size:** 20%
- **Random state:** 42
- **Stratification:** Enabled

| Dataset | Number of Samples |
|---|---:|
| Training set | 455 |
| Testing set | 114 |
| **Total** | **569** |

The 114 test samples are saved in `test_data.csv`.

## 6. Data Preprocessing

The following preprocessing steps are performed:

1. The ID column is dropped because it does not contribute to classification.
2. The Diagnosis column is encoded:
   - Malignant (M) → 1
   - Benign (B) → 0
3. StandardScaler is applied within the pipelines of:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
4. Decision Tree and Naive Bayes use the raw input features without standardization.


## c. GitHub repository
GitHub Link : https://github.com/Janeni-03/ml-assignment

Live app: [https://<your-app>.streamlit.app](https://ml-assignment-d45dbx9guboehjjwz3p7ha.streamlit.app/)


## d. Models Used

Evaluated on the 114-row held-out test set (positive class = malignant).

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

### Model Settings

- Logistic Regression: `max_iter=1000`
- Decision Tree: `max_depth=5`
- KNN: `k=7`
- Gaussian Naive Bayes: Default parameters
- Support Vector Machine: RBF kernel, `C=1.0`

### Observations on Model Performance

| ML Model Name | Observation About Model Performance |
|---|---|
| **Logistic Regression** | Very strong for a simple linear model, achieving 96.5% accuracy and the highest AUC (0.996). It ranks malignant and benign cases almost perfectly. It produced one false positive and missed 3 of the 42 malignant cases. The classes are close to linearly separable once the features are scaled. |
| **Decision Tree** | The weakest model overall, with 92.1% accuracy and an MCC of 0.83. With a depth limit of 5, it still misses 7 of 42 malignant cases (recall 0.833). Its AUC of 0.945 is lower than the other models. A single tree is also unstable and sensitive to the data split. |
| **KNN** | Good and well-balanced performance, achieving 95.6% accuracy and an MCC of 0.906 after feature standardisation. Without scaling, large-valued features like area would dominate distance calculations. It misses 4 malignant cases and produces 1 false positive, performing slightly below Logistic Regression and SVM. |
| **Naive Bayes** | Achieves perfect precision (no false positives) and a high AUC (0.993), but its recall is only 0.833, meaning it misses 7 malignant cases. The independence assumption is unrealistic because features such as radius, perimeter and area are strongly correlated, which may make its predictions overconfident. |
| **Support Vector Machine** | Best overall performance, achieving the highest accuracy (97.4%), F1-score (0.963) and MCC (0.944). It produces no false positives and misses only 3 malignant cases. The RBF kernel with scaled features fits the decision boundary between the classes effectively. |
| **Overall Winner for Your Dataset** | **Support Vector Machine (SVM)** is the overall winner, achieving the best accuracy (0.9737), F1-score (0.9630) and MCC (0.9442). Logistic Regression is a close second, achieving the highest AUC (0.9960). Since missing a malignant tumour is particularly costly, recall is important. Both SVM and Logistic Regression achieve the highest recall of 0.9286. Therefore, SVM is selected as the best-performing model, while Logistic Regression remains a strong alternative due to its simplicity and interpretability. |

## Project structure

```
├── app.py              Streamlit app
├── train.py            trains and saves the five models, writes test_data.csv
├── utils.py            shared metric helpers
├── requirements.txt    Python dependencies
├── test_data.csv       held-out test rows (features + target)
└── model/              saved .joblib models, metrics_summary.csv, results_table.md
```

## Run locally

```bash
pip install -r requirements.txt
python train.py --data breast_cancer_data/wdbc.data
streamlit run app.py
```
```
