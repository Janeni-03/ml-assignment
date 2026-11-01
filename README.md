# Breast Cancer Diagnosis – Classification Models & Streamlit App

## a. Problem statement
<!-- Write this in your own words (3-5 sentences). Ideas to cover:
     - what is being predicted (benign vs malignant tumour from cell-nucleus measurements)
     - why it matters (early, consistent screening support)
     - what you did: trained 5 classifiers, compared 6 metrics, deployed an app -->
TODO

## b. Dataset description
- **Source:** UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic), `wdbc.data`
- **Instances:** 569  |  **Features:** 30 numeric (+ ID and Diagnosis columns)
- **Task:** binary classification – `M` (malignant) = 1, `B` (benign) = 0
- **Features:** 10 measurements of the cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given as mean, standard error and "worst" value.
- **Class balance:** 357 benign / 212 malignant (check against your own `value_counts()` output)
- **Split:** 80/20 stratified, `random_state=42`; the 20% test split is saved as `test_data.csv`.
- **Preprocessing:** ID dropped, target encoded, StandardScaler inside the pipelines of Logistic Regression, KNN and SVM.

## c. GitHub repository
<!-- paste your link -->
https://github.com/<your-username>/<your-repo>

Live app: https://<your-app>.streamlit.app

## d. Models used

Paste the table produced by `python train.py` (file `model/results_table.md`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations
<!-- Write these yourself after looking at your numbers and confusion matrices.
     Think about: scaling needs, overfitting, false negatives (missed malignant cases),
     probability calibration, which metric you would trust most and why. -->

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner** | |

## Project structure
```
├── app.py              Streamlit front-end
├── train.py            trains + saves the five models, writes test_data.csv
├── utils.py            shared metric helpers
├── requirements.txt
├── test_data.csv       held-out rows (features + target) for the app
└── model/              *.joblib files, metrics_summary.csv, results_table.md
```

## Run locally
```bash
pip install -r requirements.txt
python train.py --data breast_cancer_data/wdbc.data
streamlit run app.py
```
