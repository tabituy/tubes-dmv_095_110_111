import streamlit as st

st.set_page_config(page_title="Stroke App", page_icon="🧠")

st.title("🧠 Stroke Prediction Dashboard")
st.markdown("""
Selamat datang di aplikasi analisis Stroke Prediction.

Gunakan menu di sebelah kiri untuk membuka:

- 📊 EDA
- 🔍 Feature Selection
- 🤖 Modeling & Evaluation

Semua halaman sudah disiapkan otomatis via folder `pages/`.
""")
