"""Train the five classifiers on the WDBC breast-cancer data and save everything
the Streamlit app needs (models, test_data.csv, metrics summary).

Usage:
    python train.py       # expects breast_cancer_data/wdbc.data
    python train.py --data path/to/wdbc.data
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from utils import TARGET_COL, MODEL_FILES, compute_metrics, positive_scores

SEED = 42
BASE_FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry",
    "fractal_dimension",
]
COLUMNS = ["ID", "Diagnosis"] + [
    f"{name}_{kind}" for kind in ("mean", "se", "worst") for name in BASE_FEATURES
]


def load_data(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    X = df.drop(columns=["ID", "Diagnosis"])
    y = df["Diagnosis"].map({"B": 0, "M": 1}).rename(TARGET_COL)
    return X, y


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, random_state=SEED)),
        ]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="breast_cancer_data/wdbc.data")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    model_dir = os.path.join(args.out, "model")
    os.makedirs(model_dir, exist_ok=True)

    X, y = load_data(args.data)
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # test_data.csv = unseen rows + target column (this is what the app uploads)
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(os.path.join(args.out, "test_data.csv"), index=False)
    print(f"Saved test_data.csv with {len(test_df)} rows")

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(model_dir, MODEL_FILES[name]))

        pred = model.predict(X_test)
        score = positive_scores(model, X_test)
        rows.append({"ML Model Name": name, **compute_metrics(y_test, pred, score)})
        print(f"trained + saved: {name}")

    results = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    results.to_csv(os.path.join(model_dir, "metrics_summary.csv"))

    # Markdown version, ready to paste into the README
    header = "| ML Model Name | " + " | ".join(results.columns) + " |"
    divider = "|" + "---|" * (len(results.columns) + 1)
    lines = [header, divider]
    for name, r in results.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in r) + " |")
    md = "\n".join(lines)
    with open(os.path.join(model_dir, "results_table.md"), "w") as f:
        f.write(md + "\n")

    print("\n" + results.to_string())
    print("\nMarkdown table written to model/results_table.md")


if __name__ == "__main__":
    main()
