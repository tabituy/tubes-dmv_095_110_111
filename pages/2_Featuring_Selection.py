import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

st.title("🔍 Feature Selection / Importances")

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

model = Pipeline([
    ("prep", pre),
    ("rf", RandomForestClassifier(class_weight="balanced", n_estimators=200))
])

model.fit(X, y)

ohe = model.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
cat_names = list(ohe.get_feature_names_out(cat))
feature_names = num + cat_names

importances = model.named_steps["rf"].feature_importances_

imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).head(20)

st.subheader("Top 20 Feature Importances")
fig = px.bar(imp_df, x="importance", y="feature", orientation="h")
st.plotly_chart(fig)
