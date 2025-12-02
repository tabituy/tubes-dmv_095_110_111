import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Exploratory Data Analysis (EDA)")

# --- LOAD DATASET ---
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

st.subheader("Preview Dataset")
st.dataframe(df.head())

st.subheader("Distribusi Target (stroke)")
fig1 = px.bar(df["stroke"].value_counts(), text=df["stroke"].value_counts())
st.plotly_chart(fig1)

st.subheader("Statistik Numerik")
num_cols = ["age", "avg_glucose_level", "bmi"]
st.dataframe(df[num_cols].describe())
