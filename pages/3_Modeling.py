import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
)
import plotly.graph_objects as go

st.title("🤖 Modeling & Evaluation")

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

X = df.drop(columns=["stroke", "id"])
y = df["stroke"]

num = ["age", "avg_glucose_level", "bmi"]
cat = [c for c in X.columns if c not in num]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num),

    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)
])

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced")
}

choice = st.selectbox("Pilih Model:", list(models.keys()))
model = models[choice]

test_size = st.slider("Test Size (Holdout)", 0.1, 0.5, 0.2)
k = st.selectbox("K-Fold:", [5, 10])

if st.button("Run"):
    pipe = Pipeline([("prep", pre), ("model", model)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)[:, 1]

    acc = accuracy_score(yte, pred)
    sens = recall_score(yte, pred)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    spec = tn/(tn+fp)
    auc = roc_auc_score(yte, proba)

    st.subheader("Holdout Result")
    st.write({
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "AUC": auc
    })

    # ROC Curve
    fpr, tpr, _ = roc_curve(yte, proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline", line=dict(dash="dash")))
    st.plotly_chart(fig)

    # K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    res = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr2, Xte2 = X.iloc[train_idx], X.iloc[test_idx]
        ytr2, yte2 = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(Xtr2, ytr2)
        pred2 = pipe.predict(Xte2)
        proba2 = pipe.predict_proba(Xte2)[:, 1]

        res.append([
            accuracy_score(yte2, pred2),
            recall_score(yte2, pred2),
            roc_auc_score(yte2, proba2)
        ])

    st.subheader("K-Fold Summary")
    st.dataframe(pd.DataFrame(res, columns=["Accuracy", "Sensitivity", "AUC"]).describe())
