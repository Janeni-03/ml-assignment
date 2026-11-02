import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from utils import CLASS_NAMES, MODEL_FILES, TARGET_COL, compute_metrics, positive_scores

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
SAMPLE_CSV = os.path.join(HERE, "test_data.csv")

st.set_page_config(page_title="Tumour Classifier Lab", page_icon="🔬", layout="wide")


@st.cache_resource
def load_models():
    return {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }


def split_features_target(df, feature_names):
    """Validate the uploaded frame and return (X, y)."""
    target_col = TARGET_COL if TARGET_COL in df.columns else (
        "Diagnosis" if "Diagnosis" in df.columns else None
    )
    if target_col is None:
        raise ValueError(f"Target column '{TARGET_COL}' (or 'Diagnosis') not found.")

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature column(s): {', '.join(missing[:6])} ...")

    y = df[target_col]
    if y.dtype == object:
        y = y.astype(str).str.strip().str.upper().map({"B": 0, "M": 1})
    if y.isna().any():
        raise ValueError("Target column has values other than 0/1 or B/M.")
    return df[feature_names], y.astype(int)


models = load_models()
feature_names = list(next(iter(models.values())).feature_names_in_)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("1 · Data")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
use_sample = st.sidebar.checkbox(
    "Use the bundled test_data.csv", value=uploaded is None, disabled=uploaded is not None
)

st.sidebar.header("2 · Model")
choice = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES))

st.sidebar.caption("Models are pre-trained; nothing is retrained on the uploaded file.")

# ---------------------------------------------------------------- main
st.title("🔬 Tumour Classifier Lab")
st.write(
    "Evaluate five pre-trained classifiers on the Wisconsin Diagnostic Breast Cancer "
    "features. Upload a test CSV (features + `target`, where 1 = malignant) and compare."
)

if uploaded is not None:
    data = pd.read_csv(uploaded)
elif use_sample and os.path.exists(SAMPLE_CSV):
    data = pd.read_csv(SAMPLE_CSV)
else:
    st.info("Upload a CSV in the sidebar to begin.")
    st.stop()

try:
    X, y = split_features_target(data, feature_names)
except ValueError as err:
    st.error(str(err))
    st.stop()

st.success(f"Loaded {len(X)} rows · {len(feature_names)} features · "
           f"{int(y.sum())} malignant / {int((y == 0).sum())} benign")
with st.expander("Preview data"):
    st.dataframe(data.head(10), use_container_width=True)

# -------- selected model
model = models[choice]
pred = model.predict(X)
score = positive_scores(model, X)
m = compute_metrics(y, pred, score)

st.subheader(f"Results · {choice}")
cols = st.columns(6)
for col, (label, value) in zip(cols, m.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="PuBuGn", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(y, score)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(fpr, tpr, label=f"AUC = {m['AUC']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)

st.markdown("**Classification report**")
report = classification_report(y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# -------- all models side by side
st.subheader("All five models on this data")
rows = []
for name, mdl in models.items():
    p = mdl.predict(X)
    s = positive_scores(mdl, X)
    rows.append({"Model": name, **compute_metrics(y, p, s)})
board = pd.DataFrame(rows).set_index("Model")
st.dataframe(board.style.format("{:.4f}").highlight_max(axis=0, color="#cde8d4"),
             use_container_width=True)
st.caption(f"Best MCC on this file: **{board['MCC'].idxmax()}**")
